"""The harness must expose the service's loop honesty fields to the caller.

lazy-agent-service decorates the final `done` SSE event with
answerSource/finalizeReason/providerAttempts/providerRetries
(LoopHonestyFields.ts). Without them a loop-written digest ("synthetic")
and a real model answer are indistinguishable from the returned text —
trading-service was booking digests as answers. The harness already
publishes per-call metadata (last_usage, last_model, last_provider) as
instance attributes filled from the stream; the honesty fields ride the
same path under snake_case names. An older service omits them: None,
never an error, and the returned text is untouched either way.

There is no non-stream JSON body path in the SDK to cover: the
non-streaming call_agent() returns the RAW httpx.Response for callers to
.json() themselves (see the NOTE in AgentClient.call_agent), and
AgentHarness always requests stream=True.
"""
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from lazycat.agent import AgentHarness, BaseAgent
from lazycat.session import ConversationSession


def _harness(*responses):
    """A harness whose llm_client replays the given SSE event lists."""
    agent = BaseAgent(model="test-model", name="t", system_prompt="sys")
    agent.llm_client = MagicMock()
    calls = list(responses)

    async def _call_agent(**kwargs):
        events = calls.pop(0)

        async def aiter_lines():
            for ev in events:
                yield "data: " + json.dumps(ev)
            yield "data: [DONE]"

        resp = MagicMock()
        resp.aiter_lines = aiter_lines
        resp.aclose = AsyncMock()
        return resp

    agent.llm_client.call_agent = _call_agent
    return AgentHarness(agent=agent, session=ConversationSession(session_id="s"))


def _done(**honesty):
    ev = {"type": "done", "model": "m", "provider": "p"}
    ev.update(honesty)
    return [{"type": "chunk", "content": "the answer"}, ev]


@pytest.mark.asyncio
async def test_fields_on_the_done_event_are_exposed_snake_case():
    h = _harness(_done(
        answerSource="synthetic",
        finalizeReason="CUT_OFF_TWICE",
        providerAttempts=3,
        providerRetries=2,
    ))
    text = await h.run("go")
    assert h.answer_source == "synthetic"
    assert h.finalize_reason == "CUT_OFF_TWICE"
    assert h.provider_attempts == 3
    assert h.provider_retries == 2
    # returned text untouched
    assert text == "the answer"


@pytest.mark.asyncio
async def test_model_answer_reports_its_honesty_too():
    h = _harness(_done(
        answerSource="model",
        finalizeReason="COMPLETED",
        providerAttempts=1,
        providerRetries=0,
    ))
    text = await h.run("go")
    assert (h.answer_source, h.finalize_reason) == ("model", "COMPLETED")
    assert (h.provider_attempts, h.provider_retries) == (1, 0)
    assert text == "the answer"


@pytest.mark.asyncio
async def test_older_service_without_the_fields_is_none_not_an_error():
    h = _harness(_done())  # done without any honesty keys
    text = await h.run("go")
    assert h.answer_source is None
    assert h.finalize_reason is None
    assert h.provider_attempts is None
    assert h.provider_retries is None
    assert text == "the answer"


@pytest.mark.asyncio
async def test_later_done_wins_within_one_stream():
    """Last-wins like last_usage: the fields describe the FINAL request."""
    events = [
        {"type": "chunk", "content": "the answer"},
        {"type": "done", "answerSource": "partial",
         "finalizeReason": "CUT_OFF", "providerAttempts": 2, "providerRetries": 1},
        {"type": "done", "answerSource": "model",
         "finalizeReason": "COMPLETED", "providerAttempts": 4, "providerRetries": 3},
    ]
    h = _harness(events)
    text = await h.run("go")
    assert (h.answer_source, h.finalize_reason) == ("model", "COMPLETED")
    assert (h.provider_attempts, h.provider_retries) == (4, 3)
    assert text == "the answer"


@pytest.mark.asyncio
async def test_reused_harness_does_not_leak_the_previous_run():
    h = _harness(
        _done(answerSource="synthetic", finalizeReason="STEP_LIMIT",
              providerAttempts=5, providerRetries=4),
        _done(),  # second run against a service without the fields
    )
    assert await h.run("go") == "the answer"
    assert h.answer_source == "synthetic"
    assert await h.run("again") == "the answer"
    assert h.answer_source is None
    assert h.finalize_reason is None
    assert h.provider_attempts is None
    assert h.provider_retries is None


@pytest.mark.asyncio
async def test_sdk_max_iterations_fallback_is_marked_synthetic(monkeypatch):
    """The fallback string is SDK-written, so honesty says synthetic /
    SDK_MAX_ITERATIONS; attempts/retries still report the last done event."""
    events = [
        {"type": "chunk", "content": "working..."},
        {"toolCalls": [{"id": "c1", "name": "dummy_tool", "arguments": "{}"}]},
        {"type": "done", "answerSource": "model", "finalizeReason": "COMPLETED",
         "providerAttempts": 2, "providerRetries": 1},
    ]
    h = _harness(events)
    h.max_iterations = 1

    async def _fake_invoke(func_name, arguments):
        return {"ok": True}, 5

    monkeypatch.setattr(h, "_invoke_remote_tool", _fake_invoke)
    text = await h.run("go")
    assert text == "Max iterations reached without a final answer."
    assert h.answer_source == "synthetic"
    assert h.finalize_reason == "SDK_MAX_ITERATIONS"
    # untouched by the fallback branch — they describe the last real request
    assert (h.provider_attempts, h.provider_retries) == (2, 1)
