import asyncio
import json
import logging
from typing import AsyncIterator, Any
import httpx

from lazycat.models import (
    RunRequest, RunResult, StreamEvent, StreamChunk, StreamToolCall,
    StreamToolExecution, StreamUsage, StreamDone, StreamError, RunError
)
from lazycat.sse import iter_sse_lines, iter_sse_json_lines

logger = logging.getLogger(__name__)

class RunClientError(Exception):
    def __init__(self, message: str, status_code: int | None = None, details: Any = None):
        super().__init__(message)
        self.status_code = status_code
        self.details = details

class RunClient:
    """Client for the lazy-agent-service v1 run contract."""
    def __init__(self, base_url: str, project: str = "lazycat-sdk-app", username: str = "lazycat-sdk", timeout: float = 600.0):
        self.base_url = base_url.rstrip("/")
        self.project = project
        self.username = username
        self.timeout = timeout
        
    def _get_headers(self) -> dict:
        return {
            "x-project": self.project,
            "x-username": self.username,
            "Content-Type": "application/json"
        }

    async def create_run(self, request: RunRequest) -> str:
        """Submit a run request, returning the run_id."""
        url = f"{self.base_url}/runs"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(url, json=request.model_dump(exclude_none=True), headers=self._get_headers())
            if resp.status_code not in (200, 201, 202):
                raise RunClientError(f"Failed to create run: {resp.text}", status_code=resp.status_code)
            
            data = resp.json()
            return data["run_id"]

    async def stream_run(self, run_id: str, last_event_id: str | None = None) -> AsyncIterator[StreamEvent]:
        """Stream events from a run, handling reconnection automatically."""
        url = f"{self.base_url}/runs/{run_id}/stream"
        
        headers = self._get_headers()
        headers["Accept"] = "text/event-stream"
        
        current_event_id = last_event_id
        retry_count = 0
        max_retries = 3
        
        while retry_count <= max_retries:
            if current_event_id:
                headers["Last-Event-ID"] = str(current_event_id)
                
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    async with client.stream("GET", url, headers=headers) as resp:
                        if resp.status_code == 404:
                            raise RunClientError(f"Run {run_id} not found", status_code=404)
                        if resp.status_code != 200:
                            raise RunClientError(f"Failed to stream run: {resp.status_code}", status_code=resp.status_code)
                        
                        retry_count = 0 # reset on successful connection
                        async for data in iter_sse_json_lines(resp.aiter_lines(), done_sentinel="[DONE]"):
                            # The sse helper would need to pass back the event ID. Assuming the server sends seq in data
                            if "seq" in data:
                                current_event_id = str(data["seq"])
                            
                            event_type = data.get("type")
                            if event_type == "chunk":
                                yield StreamChunk(**data)
                            elif event_type in ("tool_calls", "toolCalls"):
                                data["type"] = "tool_calls"
                                if "toolCalls" in data:
                                    data["tool_calls"] = data.pop("toolCalls")
                                yield StreamToolCall(**data)
                            elif event_type == "tool_execution":
                                yield StreamToolExecution(**data)
                            elif event_type == "usage_update":
                                yield StreamUsage(**data)
                            elif event_type == "done":
                                yield StreamDone(**data)
                                return # Stream completes
                            elif event_type == "error":
                                yield StreamError(**data)
                                return # Terminate on stream-level error
            except httpx.RequestError as e:
                logger.warning(f"Stream dropped for {run_id}: {e}")
                retry_count += 1
                if retry_count > max_retries:
                    raise RunClientError(f"Failed to stream after {max_retries} retries", details=str(e))
                await asyncio.sleep(2 ** retry_count)

    async def get_run_status(self, run_id: str) -> RunResult:
        """Fetch the current status and/or result of a run."""
        url = f"{self.base_url}/runs/{run_id}"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.get(url, headers=self._get_headers())
            if resp.status_code == 404:
                raise RunClientError(f"Run {run_id} not found", status_code=404)
            if resp.status_code != 200:
                raise RunClientError(f"Failed to fetch run status: {resp.text}", status_code=resp.status_code)
            
            data = resp.json()
            return RunResult(**data)

    async def cancel_run(self, run_id: str) -> None:
        """Cancel an ongoing run."""
        url = f"{self.base_url}/runs/{run_id}/cancel"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(url, headers=self._get_headers())
            if resp.status_code == 404:
                raise RunClientError(f"Run {run_id} not found", status_code=404)
            if resp.status_code not in (200, 202):
                raise RunClientError(f"Failed to cancel run: {resp.text}", status_code=resp.status_code)
