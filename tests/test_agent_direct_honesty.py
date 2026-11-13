"""The harness must fill the honesty fields on the DIRECT path itself.

On the direct-openai path (provider "vllm") the agentic loop runs in this
process (llm.py:844); no service decorates the final done event with
answerSource/finalizeReason/providerAttempts/providerRetries, so without a
SDK-side derivation every harness seat would report all four as None and
trading-service could not tell a model answer from a cut-off one. The done
event the LLMStreamWrapper synthesises carries only type/model/provider/
usage (+finishReason now), so who/why must be derived here. A done event
that DOES carry the keys (newer shim, or the /agent path) stays
authoritative: the derivation must never overwrite service values.

Style mirrors tests/test_agent_honesty_fields.py: a MagicMock llm_client
replaying SSE dicts, pytest.mark.asyncio per test. provider="vllm" makes
the scenario explicit: these are direct-path runs.
"""
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from lazycat.agent import AgentHarness, BaseAgent
from lazycat.session import ConversationSession


def _harness(*responses, **harness_kwargs):
    """A direct-path harness whose llm_client replays the given SSE lists."""
    agent = BaseAgent(model="test-model", name="t", system_prompt="sys", provider="vllm")
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
    return AgentHarness(
        agent=agent, session=ConversationSession(session_id="s"), **harness_kwargs
    )


def _done(**extra):
    """What the shim's done event looks like: no honesty keys."""
    ev = {"type": "done", "model": "m", "provider": "p"}
    ev.update(extra)
    return ev


def _text_turn(text, **done_extra):
    return [{"type": "chunk", "content": text}, _done(**done_extra)]


def _tool_turn(name="dummy_tool", args='{"q": "1"}'):
    return [
        {
            "type": "tool_calls",
            "toolCalls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": name, "arguments": args},
                }
            ],
        },
        _done(),
    ]


def _tool_results(h, *results):
    """Execute dispatched calls through agent.tool_registry, replaying results.

    The SDK test env has no app.tools.registry, so _try_inprocess_tool falls
    through to the agent's own registry — the hermetic execution seam.
    """
    queue = list(results)

    async def _execute_tool_call(call_payload):
        return queue.pop(0)

    h.agent.tool_registry = MagicMock()
    h.agent.tool_registry.execute_tool_call = _execute_tool_call


@pytest.mark.asyncio
async def test_normal_stop_is_model_completed():
    h = _harness(_text_turn("the answer", finishReason="stop"))
    text = await h.run("go")
    assert text == "the answer"
    assert (h.answer_source, h.finalize_reason, h.provider_attempts, h.provider_retries) == (
        "model",
        "COMPLETED",
        1,
        0,
    )


@pytest.mark.asyncio
async def test_cut_off_reply_is_partial():
    h = _harness(_text_turn("the answer", finishReason="length"))
    text = await h.run("go")
    assert text == "the answer"
    assert (h.answer_source, h.finalize_reason, h.provider_attempts, h.provider_retries) == (
        "partial",
        "CUT_OFF",
        1,
        0,
    )


@pytest.mark.asyncio
async def test_stream_without_done_event_is_partial_cut_off():
    """Chunk only, then the [DONE] sentinel: the cut text is not an answer."""
    h = _harness([{"type": "chunk", "content": "half a sentence"}])
    with pytest.raises(RuntimeError):
        await h.run("go")
    assert (h.answer_source, h.finalize_reason) == ("partial", "CUT_OFF")


@pytest.mark.asyncio
async def test_forced_final_turn_answer_is_model_completed():
    h = _harness(_tool_turn(), _text_turn("final", finishReason="stop"), max_iterations=2)
    _tool_results(h, {"ok": True})
    text = await h.run("go")
    assert text == "final"
    assert (h.answer_source, h.finalize_reason) == ("model", "COMPLETED")
    assert h.provider_attempts == 2


@pytest.mark.asyncio
async def test_max_iterations_fallback_stays_synthetic():
    h = _harness(_tool_turn(), max_iterations=1)
    _tool_results(h, {"ok": True})
    text = await h.run("go")
    assert text == "Max iterations reached without a final answer."
    assert (h.answer_source, h.finalize_reason, h.provider_attempts, h.provider_retries) == (
        "synthetic",
        "SDK_MAX_ITERATIONS",
        1,
        0,
    )


@pytest.mark.asyncio
async def test_loop_detector_escalation_does_not_abort_the_run():
    """Escalation only injects an override; the model still answers on its own.

    Identical failing call #3 gets the warning, #4 the escalation override
    (the tool is never executed again), then a text answer closes the run.
    """
    h = _harness(
        *[_tool_turn() for _ in range(4)],
        _text_turn("done now", finishReason="stop"),
        max_iterations=6,
    )
    _tool_results(h, {"error": "boom"}, {"error": "boom"})
    text = await h.run("go")
    assert text == "done now"
    assert (h.answer_source, h.finalize_reason) == ("model", "COMPLETED")
    assert h.loop_detector.escalation_triggered is True


@pytest.mark.asyncio
async def test_guard_abort_is_named_as_repeated_tool_call():
    """A guard that aborts (abort_agent_run) leaves no answer: name the tool."""

    class DoomLoop(Exception):
        abort_agent_run = True

    def _hook(func_name, arguments, result, was_blocked, elapsed_ms):
        raise DoomLoop("guard says stop")

    h = _harness(_tool_turn(name="dummy_tool"), on_tool_result=_hook)
    _tool_results(h, {"ok": True})
    with pytest.raises(DoomLoop):
        await h.run("go")
    assert h.answer_source is None
    assert h.finalize_reason == "repeated_tool_call:dummy_tool"


@pytest.mark.asyncio
async def test_provider_exception_is_named_without_an_answer():
    h = _harness()

    async def _boom(**kwargs):
        raise RuntimeError("connection reset")

    h.agent.llm_client.call_agent = _boom
    with pytest.raises(RuntimeError):
        await h.run("go")
    assert (h.answer_source, h.finalize_reason, h.provider_attempts, h.provider_retries) == (
        None,
        "PROVIDER_ERROR",
        1,
        0,
    )


@pytest.mark.asyncio
async def test_service_values_win_over_the_derivation():
    """A done event carrying the keys is authoritative, even on provider vllm."""
    h = _harness(
        _text_turn(
            "svc",
            answerSource="synthetic",
            finalizeReason="STEP_LIMIT",
            providerAttempts=9,
            providerRetries=3,
            finishReason="length",
        )
    )
    text = await h.run("go")
    assert text == "svc"
    assert (h.answer_source, h.finalize_reason, h.provider_attempts, h.provider_retries) == (
        "synthetic",
        "STEP_LIMIT",
        9,
        3,
    )
