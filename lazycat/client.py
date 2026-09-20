from __future__ import annotations

import json
import logging
import os
import asyncio
from typing import Any, AsyncIterator, Optional
import httpx
from pydantic import ValidationError
from lazycat.generated_runtime_models import RunEvent as GeneratedRunEvent, RunResult as GeneratedRunResult

from lazycat.models import (
    CreateRunRequest,
    DecisionReceipt,
    TypedDecisionRequest,
    RunEvent,
    RunResult,
    StructuredError,
)

logger = logging.getLogger(__name__)


class RuntimeClientError(Exception):
    """Base exception for RuntimeClient errors."""
    def __init__(self, message: str, status_code: Optional[int] = None, structured_error: Optional[StructuredError] = None, details: Any = None):
        super().__init__(message)
        self.status_code = status_code
        self.structured_error = structured_error
        self.details = details


class RunNotFoundError(RuntimeClientError):
    """Raised when the requested run ID is not found on the runtime server."""
    pass


class RunTimeoutError(RuntimeClientError):
    """Raised when an operation on a run times out."""
    pass


class RunEventDecodeError(RuntimeClientError):
    """Raised when an incoming Server-Sent Event cannot be strictly decoded according to the contract."""
    pass


class RuntimeClient:
    """
    Canonical typed client for lazy-agent-service v1 agent execution API (/v1/runs).
    
    Provides non-streaming execution, SSE streaming, status inspection, and cancellation.
    No provider transport or internal harness logic is embedded here.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: float = 600.0,
        project: str = "trading",
        username: str = "lazycat",
        client: Optional[httpx.AsyncClient] = None,
        authorization_bearer: Optional[str] = None,
        bearer: Optional[str] = None,
    ):
        raw_url = base_url or os.environ.get("AGENT_SERVICE_URL") or os.environ.get("LAZY_AGENT_URL") or "http://localhost:8080/v1/runs"
        raw_url = raw_url.rstrip("/")
        if not raw_url.endswith("/v1/runs") and not raw_url.endswith("/runs"):
            raw_url = f"{raw_url}/v1/runs"
        self.base_url = raw_url
        self.timeout = timeout
        self.project = project
        self.username = username
        self._external_client = client
        # A scoped bearer always wins over environment backend credentials.
        self._authorization_bearer = (authorization_bearer or bearer or "").strip() or None

    def _get_headers(self, idempotency_key: Optional[str] = None) -> dict[str, str]:
        headers = {
            "x-project": self.project,
            "x-username": self.username,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self._authorization_bearer:
            headers["Authorization"] = f"Bearer {self._authorization_bearer}"
        else:
            credential = os.environ.get("RUNTIME_API_TOKEN") or os.environ.get("RUNTIME_AUTH_SECRET") or os.environ.get("INTERNAL_EXECUTE_TOKEN")
        if not self._authorization_bearer and credential:
            headers["x-runtime-token"] = credential
        if idempotency_key:
            headers["x-idempotency-key"] = idempotency_key
        return headers

    def _get_http_client(self) -> httpx.AsyncClient:
        if self._external_client is not None:
            return self._external_client
        return httpx.AsyncClient(timeout=self.timeout)

    @staticmethod
    def _validate_wire_event(raw: dict[str, Any]) -> RunEvent:
        # Additive extension events remain accepted by the compatibility model;
        # every canonical event is checked against the generated neutral schema.
        generated_raw = {**raw}
        if "run_id" not in generated_raw and "runId" in generated_raw:
            generated_raw["run_id"] = generated_raw["runId"]
        if raw.get("type") in {
            "run.admitted", "run.started", "run.created", "message.delta", "message.completed",
            "tool.invoked", "tool.called", "tool.completed", "tool.result", "tool.failed",
            "approval.required", "approval.resolved", "worker.dispatched", "worker.completed",
            "run.completed", "run.failed", "run.cancelled",
        }:
            GeneratedRunEvent.model_validate(generated_raw)
        return RunEvent.model_validate(raw)

    @staticmethod
    def _validate_wire_result(raw: dict[str, Any]) -> RunResult:
        result = RunResult.model_validate(raw)
        if result.status in {"admitted", "running", "waiting_for_tool", "waiting_for_worker", "waiting_for_approval", "completed", "failed", "cancelled", "timed_out"}:
            generated_raw = {**raw}
            if "run_id" not in generated_raw and "id" in generated_raw:
                generated_raw["run_id"] = generated_raw["id"]
            GeneratedRunResult.model_validate(generated_raw)
        return result

    @property
    def wire_contract_url(self) -> str:
        """Return the service's neutral wire-contract endpoint."""
        return self.base_url.rsplit("/v1/runs", 1)[0] + "/v1/contracts/wire-schema"

    async def check_wire_compatibility(self, expected_version: str, expected_digest: Optional[str] = None) -> dict[str, Any]:
        """Fetch and validate the runtime wire contract before starting work."""
        client = self._get_http_client()
        should_close = self._external_client is None
        try:
            resp = await client.get(self.wire_contract_url, headers=self._get_headers())
            if resp.status_code != 200:
                raise RuntimeClientError(f"Failed to fetch runtime wire contract (HTTP {resp.status_code})", status_code=resp.status_code)
            document = resp.json()
            if document.get("contract_version") != expected_version:
                raise RuntimeClientError("Runtime wire contract version mismatch", details=document)
            if expected_digest is not None and document.get("digest") != expected_digest:
                raise RuntimeClientError("Runtime wire contract digest mismatch", details=document)
            return document
        finally:
            if should_close:
                await client.aclose()

    async def create_run(self, request: CreateRunRequest) -> RunResult:
        """
        Execute a run non-streaming to completion.
        
        POST /v1/runs with stream=False.
        Returns the terminal RunResult.
        """
        # Ensure non-streaming flag
        req_data = request.model_dump(by_alias=False, exclude_none=True)
        req_data["stream"] = False
        headers = self._get_headers(idempotency_key=request.idempotency_key)

        client = self._get_http_client()
        should_close = self._external_client is None
        try:
            resp = await client.post(self.base_url, json=req_data, headers=headers)
            if resp.status_code in (200, 201):
                return self._validate_wire_result(resp.json())

            # Parse structured error if present
            err_data = None
            structured_err = None
            try:
                err_data = resp.json()
                if "error" in err_data and isinstance(err_data["error"], dict):
                    structured_err = StructuredError.model_validate(err_data["error"])
            except Exception:
                pass

            msg = f"Failed to create run (HTTP {resp.status_code}): {resp.text}"
            if structured_err:
                msg = f"Runtime error [{structured_err.code}]: {structured_err.message}"

            if resp.status_code == 404:
                raise RunNotFoundError(msg, status_code=resp.status_code, structured_error=structured_err, details=err_data)
            raise RuntimeClientError(msg, status_code=resp.status_code, structured_error=structured_err, details=err_data)
        finally:
            if should_close:
                await client.aclose()

    async def stream_run(self, request: CreateRunRequest) -> AsyncIterator[RunEvent]:
        """
        Execute a run with streaming Server-Sent Events.
        
        POST /v1/runs with stream=True.
        Strictly decodes incoming SSE events into canonical RunEvent objects.
        """
        req_data = request.model_dump(by_alias=False, exclude_none=True)
        req_data["stream"] = True
        headers = self._get_headers(idempotency_key=request.idempotency_key)
        headers["Accept"] = "text/event-stream"

        client = self._get_http_client()
        should_close = self._external_client is None
        run_id: Optional[str] = None
        terminal_seen = False
        try:
            async with client.stream("POST", self.base_url, json=req_data, headers=headers) as resp:
                if resp.status_code != 200:
                    err_text = ""
                    async for chunk in resp.aiter_text():
                        err_text += chunk
                    raise RuntimeClientError(
                        f"Failed to initiate run stream (HTTP {resp.status_code}): {err_text}",
                        status_code=resp.status_code
                    )

                data_lines = []
                seen = set()
                async for line in resp.aiter_lines():
                    if line.startswith(":"):
                        continue
                    if line:
                        if line.startswith("data:"):
                            data_lines.append(line[5:].lstrip(" "))
                        continue
                    if not data_lines:
                        continue
                    raw_payload = "\n".join(data_lines)
                    data_lines.clear()
                    if raw_payload == "[DONE]":
                        raise RunEventDecodeError("Stream ended without a terminal run outcome")
                    try:
                        event = self._validate_wire_event(json.loads(raw_payload))
                    except (json.JSONDecodeError, ValidationError) as exc:
                        raise RunEventDecodeError("Invalid runtime SSE event", details=raw_payload) from exc
                    if event.id in seen:
                        continue
                    seen.add(event.id)
                    run_id = event.run_id or run_id
                    yield event
                    if event.type in ("run.completed", "run.failed", "run.cancelled"):
                        terminal_seen = True
                        return
                raise RunEventDecodeError("Runtime stream ended before a terminal run outcome")
        finally:
            if run_id and not terminal_seen:
                # Closing an SSE response does not cancel the server-side run.
                # Use a bounded child task so cancellation/transport failures
                # never mask the original stream exception.
                cancel_task = asyncio.create_task(self.cancel_run(run_id))
                try:
                    await asyncio.wait_for(asyncio.shield(cancel_task), timeout=2.0)
                except BaseException:
                    cancel_task.cancel()
            if should_close:
                await client.aclose()

    async def steer_run(self, run_id: str, instruction: str) -> dict[str, Any]:
        """Queue a bounded instruction for the run's next model turn."""
        if not isinstance(instruction, str) or not instruction.strip() or len(instruction) > 4096:
            raise ValueError("instruction must be 1-4096 non-whitespace characters")
        url = f"{self.base_url}/{run_id}/steer"
        client = self._get_http_client()
        should_close = self._external_client is None
        try:
            resp = await client.post(url, headers=self._get_headers(), json={"instruction": instruction})
            if resp.status_code == 202:
                return resp.json()
            if resp.status_code == 404:
                raise RunNotFoundError(f"Run '{run_id}' not found for steering", status_code=404)
            raise RuntimeClientError(f"Failed to steer run '{run_id}' (HTTP {resp.status_code}): {resp.text}", status_code=resp.status_code)
        finally:
            if should_close:
                await client.aclose()

    async def submit_tool_result(self, run_id: str, call_id: str, *, result: Any,
                                 authorization_receipt: dict, is_error: bool = False) -> dict:
        """Return a real local observation using the runtime's scoped signed receipt."""
        from urllib.parse import quote
        client = self._get_http_client()
        try:
            resp = await client.post(
                f"{self.base_url}/{quote(run_id, safe='')}/tools/{quote(call_id, safe='')}/result",
                headers=self._get_headers(),
                json={"result": result, "is_error": is_error, "authorization_receipt": authorization_receipt},
            )
            if resp.status_code != 200:
                raise RuntimeClientError("Tool result rejected", status_code=resp.status_code)
            return resp.json()
        finally:
            if self._external_client is None:
                await client.aclose()

    async def decide(self, run_id: str, request: TypedDecisionRequest) -> DecisionReceipt:
        """Submit a typed, shadow-only decision request for an existing run."""
        from urllib.parse import quote
        if request.run_id != run_id:
            raise ValueError("Decision request run_id must match the path run_id")
        client = self._get_http_client()
        try:
            resp = await client.post(
                f"{self.base_url}/{quote(run_id, safe='')}/decisions",
                headers=self._get_headers(),
                json=request.model_dump(by_alias=True, exclude_none=True),
            )
            if resp.status_code != 200:
                raise RuntimeClientError("Decision request rejected", status_code=resp.status_code, details=resp.text)
            return DecisionReceipt.model_validate(resp.json())
        finally:
            if self._external_client is None:
                await client.aclose()

    async def replay_events(self, run_id: str, after: Optional[str] = None) -> AsyncIterator[RunEvent]:
        """Replay recorded events after a cursor; never starts or resumes execution."""
        from urllib.parse import quote
        client = self._get_http_client()
        headers = self._get_headers()
        headers["Accept"] = "text/event-stream"
        if after:
            headers["Last-Event-ID"] = after
        try:
            async with client.stream("GET", f"{self.base_url}/{quote(run_id, safe='')}/events", headers=headers) as resp:
                if resp.status_code != 200:
                    body = ""
                    async for chunk in resp.aiter_text():
                        body += chunk
                    raise RuntimeClientError(f"Failed to replay run events (HTTP {resp.status_code}): {body}", status_code=resp.status_code)
                data_lines: list[str] = []
                seen: set[str] = set()
                async for line in resp.aiter_lines():
                    if line.startswith(":"):
                        continue
                    if line.startswith("data:"):
                        data_lines.append(line[5:].lstrip(" "))
                        continue
                    if line:
                        continue
                    if not data_lines:
                        continue
                    raw_payload = "\n".join(data_lines)
                    data_lines.clear()
                    try:
                        event = self._validate_wire_event(json.loads(raw_payload))
                    except (json.JSONDecodeError, ValidationError) as exc:
                        raise RunEventDecodeError("Invalid replay SSE event", details=raw_payload) from exc
                    if event.id in seen:
                        continue
                    seen.add(event.id)
                    yield event
        finally:
            if self._external_client is None:
                await client.aclose()

    async def resolve_approval(self, run_id: str, approval_id: str, *, approved: bool) -> dict[str, Any]:
        """Resolve one persisted approval request for a scoped run."""
        from urllib.parse import quote
        client = self._get_http_client()
        try:
            resp = await client.post(
                f"{self.base_url}/{quote(run_id, safe='')}/approvals/{quote(approval_id, safe='')}",
                headers=self._get_headers(),
                json={"approved": approved},
            )
            if resp.status_code not in (200, 202):
                raise RuntimeClientError("Approval resolution rejected", status_code=resp.status_code, details=resp.text)
            return resp.json()
        finally:
            if self._external_client is None:
                await client.aclose()

    async def get_run(self, run_id: str) -> RunResult:
        """
        Fetch the current status and result of an existing run.
        
        GET /v1/runs/{run_id}.
        """
        url = f"{self.base_url}/{run_id}"
        headers = self._get_headers()

        client = self._get_http_client()
        should_close = self._external_client is None
        try:
            resp = await client.get(url, headers=headers)
            if resp.status_code == 200:
                return self._validate_wire_result(resp.json())
            if resp.status_code == 404:
                raise RunNotFoundError(f"Run '{run_id}' not found", status_code=404)
            raise RuntimeClientError(f"Failed to fetch run '{run_id}' (HTTP {resp.status_code}): {resp.text}", status_code=resp.status_code)
        finally:
            if should_close:
                await client.aclose()

    async def cancel_run(self, run_id: str) -> bool:
        """
        Explicitly cancel an active run.
        
        POST /v1/runs/{run_id}/cancel.
        """
        url = f"{self.base_url}/{run_id}/cancel"
        headers = self._get_headers()

        client = self._get_http_client()
        should_close = self._external_client is None
        try:
            resp = await client.post(url, headers=headers)
            if resp.status_code in (200, 202):
                data = resp.json()
                return data.get("cancelled", True)
            if resp.status_code == 404:
                raise RunNotFoundError(f"Run '{run_id}' not found for cancellation", status_code=404)
            raise RuntimeClientError(f"Failed to cancel run '{run_id}' (HTTP {resp.status_code}): {resp.text}", status_code=resp.status_code)
        finally:
            if should_close:
                await client.aclose()
