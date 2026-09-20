import json
import secrets
import pytest
import respx
import httpx
from pydantic import ValidationError

from lazycat.models import (
    CreateRunRequest,
    DecisionConstraints,
    DecisionQuestion,
    TypedDecisionRequest,
    RunBudget,
    RunEvent,
    RunEventType,
    RunResult,
    RunUsage,
    StructuredError,
)
from lazycat.client import (
    RuntimeClient,
    RuntimeClientError,
    RunNotFoundError,
    RunEventDecodeError,
)


def test_models_contract_validation():
    # Test valid request with snake_case
    req = CreateRunRequest(
        profile_id="v3_junior_analyst",
        input="Analyze AAPL",
        budget=RunBudget(max_tokens=4096, max_tool_calls=7),
        stream=False,
        idempotency_key="test-key-123",
    )
    assert req.profile_id == "v3_junior_analyst"
    assert req.budget.max_tool_calls == 7

    # Test camelCase aliases
    req_camel = CreateRunRequest.model_validate({
        "profileId": "v3_junior_analyst",
        "input": "Analyze AAPL",
        "budget": {"maxTokens": 4096, "maxToolCalls": 7},
        "idempotencyKey": "key-456",
    })
    assert req_camel.profile_id == "v3_junior_analyst"
    assert req_camel.budget.max_tokens == 4096
    assert req_camel.idempotency_key == "key-456"

    # Test invalid budget exceeding limits
    with pytest.raises(ValidationError):
        RunBudget(max_tokens=50000)  # max is 32768

    with pytest.raises(ValidationError):
        RunBudget(max_tool_calls=100)  # max is 50


def test_run_event_strict_decoding():
    # Valid canonical event
    valid_event = {
        "id": "evt-1",
        "runId": "run-100",
        "type": "run.started",
        "timestamp": "2026-09-19T10:00:00Z",
        "data": {"status": "in_progress"},
    }
    event = RunEvent.model_validate(valid_event)
    assert event.id == "evt-1"
    assert event.run_id == "run-100"
    assert event.type == RunEventType.RUN_STARTED

    # Missing required field 'run_id'/'runId' must fail validation visibly
    invalid_event = {
        "id": "evt-2",
        "type": "run.started",
        "timestamp": "2026-09-19T10:00:00Z",
        "data": {},
    }
    with pytest.raises(ValidationError):
        RunEvent.model_validate(invalid_event)

    # Unknown additive event type is safely preserved without crash
    additive_event = {
        "id": "evt-3",
        "run_id": "run-100",
        "type": "custom.extension.notification",
        "timestamp": "2026-09-19T10:00:00Z",
        "data": {"foo": "bar"},
    }
    parsed = RunEvent.model_validate(additive_event)
    assert parsed.type == "custom.extension.notification"
    assert parsed.data["foo"] == "bar"


@pytest.mark.asyncio
@respx.mock
async def test_runtime_client_create_run_success():
    client = RuntimeClient(base_url="http://agent-test/v1/runs")

    mock_result = {
        "run_id": "run-test-1",
        "status": "completed",
        "messages": [{"role": "assistant", "content": '{"summary": "All good"}'}],
        "usage": {
            "prompt_tokens": 150,
            "completion_tokens": 50,
            "total_tokens": 200,
            "tool_calls_count": 1,
            "retry_count": 0,
            "duration_ms": 350,
        },
    }

    route = respx.post("http://agent-test/v1/runs").respond(
        status_code=201, json=mock_result
    )

    req = CreateRunRequest(
        profile_id="v3_junior_analyst",
        input="Check earnings",
        idempotency_key="idemp-999",
    )

    result = await client.create_run(req)
    assert route.called
    sent_request = route.calls.last.request
    assert sent_request.headers["x-idempotency-key"] == "idemp-999"
    sent_json = json.loads(sent_request.content)
    assert sent_json["stream"] is False
    assert sent_json["profile_id"] == "v3_junior_analyst"

    assert isinstance(result, RunResult)
    assert result.run_id == "run-test-1"
    assert result.status == "completed"
    assert result.usage.total_tokens == 200
    assert result.usage.tool_calls_count == 1


@pytest.mark.asyncio
@respx.mock
async def test_runtime_client_create_run_structured_error():
    client = RuntimeClient(base_url="http://agent-test/v1/runs")

    error_payload = {
        "error": {
            "code": "BUDGET_EXCEEDED",
            "message": "Maximum turns exceeded",
            "retryable": False,
            "category": "POLICY",
        }
    }

    respx.post("http://agent-test/v1/runs").respond(
        status_code=400, json=error_payload
    )

    req = CreateRunRequest(profile_id="v3_junior_analyst", input="Fail now")

    with pytest.raises(RuntimeClientError) as exc_info:
        await client.create_run(req)

    assert "BUDGET_EXCEEDED" in str(exc_info.value)
    assert exc_info.value.structured_error.code == "BUDGET_EXCEEDED"
    assert exc_info.value.structured_error.retryable is False


@pytest.mark.asyncio
@respx.mock
async def test_runtime_client_stream_run_success():
    client = RuntimeClient(base_url="http://agent-test/v1/runs")

    sse_body = (
        'data: {"id": "1", "runId": "run-stream", "type": "run.started", "timestamp": "2026-09-19T10:00:00Z", "data": {"status": "in_progress"}}\n\n'
        'data: {"id": "2", "runId": "run-stream", "type": "message.delta", "timestamp": "2026-09-19T10:00:01Z", "data": {"text": "hello"}}\n\n'
        'data: {"id": "3", "runId": "run-stream", "type": "run.completed", "timestamp": "2026-09-19T10:00:02Z", "data": {"status": "completed"}}\n\n'
        'data: [DONE]\n\n'
    )

    respx.post("http://agent-test/v1/runs").respond(
        status_code=200,
        content=sse_body.encode("utf-8"),
        headers={"Content-Type": "text/event-stream"},
    )

    req = CreateRunRequest(profile_id="v3_junior_analyst", input="Stream me")

    events = []
    async for evt in client.stream_run(req):
        events.append(evt)

    assert len(events) == 3
    assert events[0].type == "run.started"
    assert events[1].type == "message.delta"
    assert events[2].type == "run.completed"


@pytest.mark.asyncio
@respx.mock
async def test_runtime_client_stream_run_malformed_event_fails_visibly():
    client = RuntimeClient(base_url="http://agent-test/v1/runs")

    # Second event missing required 'id' and 'timestamp'
    sse_body = (
        'data: {"id": "1", "runId": "run-stream", "type": "run.started", "timestamp": "2026-09-19T10:00:00Z", "data": {}}\n\n'
        'data: {"broken": "no required fields"}\n\n'
    )

    respx.post("http://agent-test/v1/runs").respond(
        status_code=200,
        content=sse_body.encode("utf-8"),
        headers={"Content-Type": "text/event-stream"},
    )

    req = CreateRunRequest(profile_id="v3_junior_analyst", input="Stream me")

    with pytest.raises(RunEventDecodeError):
        async for _ in client.stream_run(req):
            pass


@pytest.mark.asyncio
@respx.mock
async def test_runtime_client_get_and_cancel_run():
    client = RuntimeClient(base_url="http://agent-test/v1/runs")

    respx.get("http://agent-test/v1/runs/run-123").respond(
        status_code=200,
        json={"id": "run-123", "status": "completed", "messages": []},
    )
    respx.get("http://agent-test/v1/runs/run-not-found").respond(
        status_code=404, text="Not Found"
    )

    run = await client.get_run("run-123")
    assert run.run_id == "run-123"
    assert run.status == "completed"

    with pytest.raises(RunNotFoundError):
        await client.get_run("run-not-found")

    respx.post("http://agent-test/v1/runs/run-123/cancel").respond(
        status_code=200, json={"cancelled": True}
    )
    cancelled = await client.cancel_run("run-123")
    assert cancelled is True


@pytest.mark.asyncio
@respx.mock
async def test_runtime_client_scoped_bearer_precedes_environment(monkeypatch):
    env_credential = "auth_" + secrets.token_hex(12)
    monkeypatch.setenv("INTERNAL_EXECUTE_TOKEN", env_credential)
    route = respx.get("http://agent-test/v1/runs/run-scoped").respond(
        status_code=200, json={"id": "run-scoped", "status": "completed", "messages": []}
    )
    client = RuntimeClient(base_url="http://agent-test/v1/runs", authorization_bearer="scoped-session")
    await client.get_run("run-scoped")
    request = route.calls.last.request
    assert request.headers["authorization"] == "Bearer scoped-session"
    assert "x-runtime-token" not in request.headers


@pytest.mark.asyncio
@respx.mock
async def test_runtime_client_decide_validates_and_posts_typed_request():
    route = respx.post("http://agent-test/v1/runs/run-decision/decisions").respond(
        status_code=200,
        json={
            "receipt_id": "decision-1", "request_id": "00000000-0000-4000-8000-000000000001",
            "run_id": "run-decision", "agent_profile": "trading-analyst-v1", "policy_version": "shadow.v1",
            "input_hash": "sha256-" + "a" * 64, "created_at": "2026-09-19T10:00:00Z",
            "latency_ms": 3, "policy_outcome": "shadow_only", "authorizes_actions": False,
            "fallback": "primary_llm"
        }
    )
    request = TypedDecisionRequest(
        request_id="00000000-0000-4000-8000-000000000001", run_id="run-decision",
        question_id="agent.evidence_sufficiency.v1", state="{}",
        questions={"insufficient_evidence": DecisionQuestion(instructions="Choose", criteria={"insufficient_evidence": "No", "ready": "Yes"})},
        constraints=DecisionConstraints(max_latency_ms=100),
    )
    receipt = await RuntimeClient(base_url="http://agent-test/v1/runs").decide("run-decision", request)
    assert receipt.fallback == "primary_llm"
    payload = json.loads(route.calls.last.request.content)
    assert payload["requestId"] == request.request_id
    assert payload["constraints"]["shadowOnly"] is True


@pytest.mark.asyncio
@respx.mock
async def test_runtime_client_replays_events_without_requiring_terminal():
    body = (
        'id: evt-1\nevent: run.started\ndata: {"id":"evt-1","run_id":"run-replay","type":"run.started","timestamp":"2026-09-19T10:00:00Z","data":{}}\n\n'
        'id: evt-2\nevent: message.delta\ndata: {"id":"evt-2","run_id":"run-replay","type":"message.delta","timestamp":"2026-09-19T10:00:01Z","data":{"delta":"hello"}}\n\n'
    )
    route = respx.get("http://agent-test/v1/runs/run-replay/events").respond(
        status_code=200, content=body.encode(), headers={"Content-Type": "text/event-stream"}
    )
    events = [event async for event in RuntimeClient(base_url="http://agent-test/v1/runs").replay_events("run-replay", after="evt-0")]
    assert [event.id for event in events] == ["evt-1", "evt-2"]
    assert route.calls.last.request.headers["last-event-id"] == "evt-0"


@pytest.mark.asyncio
@respx.mock
async def test_runtime_client_resolves_scoped_approval():
    route = respx.post("http://agent-test/v1/runs/run-approval/approvals/approval-1").respond(
        status_code=200, json={"ok": True, "status": "running"}
    )
    result = await RuntimeClient(base_url="http://agent-test/v1/runs").resolve_approval(
        "run-approval", "approval-1", approved=True
    )
    assert result["ok"] is True
    assert json.loads(route.calls.last.request.content) == {"approved": True}


@pytest.mark.asyncio
@respx.mock
async def test_runtime_client_checks_wire_contract_version_and_digest():
    route = respx.get("http://agent-test/v1/contracts/wire-schema").respond(
        status_code=200, json={"contract_version": "runtime-wire.v1.0.0", "digest": "sha256-wire"}
    )
    document = await RuntimeClient(base_url="http://agent-test/v1/runs").check_wire_compatibility(
        "runtime-wire.v1.0.0", "sha256-wire"
    )
    assert document["digest"] == "sha256-wire"
    assert route.called
