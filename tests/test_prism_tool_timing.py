"""Prism-internal tool calls must report a real duration.

`tool_execution` events carry `status: "calling"` before `done`/`error`, so the
round trip is measurable — but the handler hardcoded `elapsed_ms = 0` with the
comment "Fallback for prism-internal tools". Since essentially every MCP tool
executes inside Prism, that was nearly every call.

A trading-service audit on 2026-07-27 found all 8 lazy_web_search failures of
cycle-v3-1785137616 recorded at 0 ms, which made a fast refusal
indistinguishable from the 20-second connect timeout that was actually
happening — the single most useful number for diagnosing it was the one being
discarded.
"""

import time
from unittest.mock import MagicMock, patch

import pytest

from lazycat.agent import AgentHarness, BaseAgent
from lazycat.session import ConversationSession


def _harness(on_tool_result):
    agent = BaseAgent(name="t", system_prompt="sys")
    return AgentHarness(
        agent=agent,
        session=ConversationSession(session_id="s"),
        on_tool_result=on_tool_result,
        max_iterations=1,
    )


def _sse(events):
    """Build the SSE line stream AgentHarness.run consumes."""
    import json as _json
    lines = [f"data: {_json.dumps(e)}" for e in events]
    lines.append("data: [DONE]")
    return lines


def _stub_client(lines):
    """Stub llm_client.call_agent — run() consumes resp.aiter_lines()."""
    class _Resp:
        status_code = 200

        async def aiter_lines(self):
            for line in lines:
                yield line

        async def aclose(self):
            return None

    class _Client:
        async def call_agent(self, **kw):
            return _Resp()

    return _Client()


@pytest.mark.asyncio
async def test_calling_event_produces_a_nonzero_elapsed_ms():
    captured = []

    def hook(name, args, result, blocked, elapsed_ms):
        captured.append((name, elapsed_ms))

    h = _harness(hook)

    real_sleep_marker = {"t": None}

    def _fake_time():
        # First call = the "calling" event, second = "done".
        if real_sleep_marker["t"] is None:
            real_sleep_marker["t"] = 1000.0
            return 1000.0
        return 1000.75  # 750 ms later

    events = [
        {"type": "tool_execution", "status": "calling",
         "tool": {"name": "lazy_web_search", "args": {}}},
        {"type": "tool_execution", "status": "error",
         "tool": {"name": "lazy_web_search", "args": {}, "result": {"error": "ConnectTimeout"}}},
    ]

    h.agent.llm_client = _stub_client(_sse(events))
    with patch("lazycat.agent.time.time", _fake_time):
        await h.run("go")

    assert captured, "on_tool_result was never invoked"
    name, elapsed = captured[0]
    assert name == "lazy_web_search"
    assert elapsed == 750, f"expected a measured 750ms, got {elapsed}"


@pytest.mark.asyncio
async def test_missing_calling_event_still_reports_zero():
    """0 must keep meaning "not measurable", not be faked from an unrelated
    clock reading — a wrong duration is worse than an admitted unknown."""
    captured = []

    def hook(name, args, result, blocked, elapsed_ms):
        captured.append(elapsed_ms)

    h = _harness(hook)
    events = [
        {"type": "tool_execution", "status": "done",
         "tool": {"name": "get_market_data", "args": {}, "result": {"ok": 1}}},
    ]

    h.agent.llm_client = _stub_client(_sse(events))
    await h.run("go")

    assert captured == [0]


@pytest.mark.asyncio
async def test_two_tools_do_not_swap_timings():
    """Timings are keyed per tool, so an interleaved pair keeps its own clock."""
    captured = {}

    def hook(name, args, result, blocked, elapsed_ms):
        captured[name] = elapsed_ms

    h = _harness(hook)
    ticks = iter([100.0, 200.0, 100.5, 260.0])

    events = [
        {"type": "tool_execution", "status": "calling", "tool": {"name": "fast", "args": {}}},
        {"type": "tool_execution", "status": "calling", "tool": {"name": "slow", "args": {}}},
        {"type": "tool_execution", "status": "done",
         "tool": {"name": "fast", "args": {}, "result": {"ok": 1}}},
        {"type": "tool_execution", "status": "done",
         "tool": {"name": "slow", "args": {}, "result": {"ok": 1}}},
    ]

    h.agent.llm_client = _stub_client(_sse(events))
    with patch("lazycat.agent.time.time", lambda: next(ticks)):
        await h.run("go")

    assert captured["fast"] == 500       # 100.5 - 100.0
    assert captured["slow"] == 60_000    # 260.0 - 200.0
