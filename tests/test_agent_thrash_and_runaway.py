import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lazycat.agent import AgentHarness, BaseAgent
from lazycat.session import ConversationSession


def _make_stream(chunks: list[str] | None = None, tool_calls: list[dict] | None = None):
    resp = MagicMock()
    resp.aclose = AsyncMock()

    async def _aiter_lines():
        for c in chunks or []:
            yield f"data: {json.dumps({'type': 'chunk', 'content': c})}"
        if tool_calls:
            yield f"data: {json.dumps({'toolCalls': tool_calls})}"
        yield 'data: {"type": "done", "finishReason": "stop"}'
        yield "data: [DONE]"

    resp.aiter_lines = _aiter_lines
    return resp


@pytest.mark.asyncio
async def test_duplicate_tool_thrash_short_circuits_and_forces_final_answer(monkeypatch):
    monkeypatch.setenv("V3_USE_PRISM", "0")
    agent = BaseAgent(model="test-model", name="thrash_agent", system_prompt="sys", provider="vllm")
    agent.add_tool({"name": "get_quote", "description": "quote"})
    session = ConversationSession(session_id="thrash_1")
    harness = AgentHarness(agent=agent, session=session, max_iterations=8, max_tool_calls=20)

    recorded_calls: list[dict] = []

    async def _stub_call(**kwargs):
        recorded_calls.append({"tools": kwargs.get("tools"), "messages": list(kwargs.get("messages", []))})
        if kwargs.get("tools") is None:
            return _make_stream(chunks=["Final answer after thrash stop."])
        return _make_stream(
            tool_calls=[
                {
                    "id": f"call_{len(recorded_calls)}",
                    "name": "get_quote",
                    "arguments": '{"ticker": "NVDA"}',
                }
            ]
        )

    with patch("lazycat.agent.agent_client.call_agent", side_effect=_stub_call), patch(
        "lazycat.agent.tool_executor.execute_tool", new_callable=AsyncMock
    ) as mock_exec:
        mock_exec.return_value = {"price": 142.85}
        result = await harness.run("Check NVDA")

    assert result == "Final answer after thrash stop."
    assert harness.stop_cause == "tool_thrash"
    # First 2 identical calls execute; 3rd identical call is blocked before execution.
    assert mock_exec.call_count == 2
    assert len(recorded_calls) == 4
    assert recorded_calls[-1]["tools"] is None
    blocked_msgs = [m for m in session.get_messages() if m.get("role") == "tool" and "TOOL_THRASH_GUARD" in m.get("content", "")]
    assert len(blocked_msgs) == 1


@pytest.mark.asyncio
async def test_max_tool_calls_cap_triggers_final_turn_nudge(monkeypatch):
    monkeypatch.setenv("V3_USE_PRISM", "0")
    agent = BaseAgent(model="test-model", name="budget_agent", system_prompt="sys", provider="vllm")
    agent.add_tool({"name": "get_quote", "description": "quote"})
    session = ConversationSession(session_id="budget_1")
    harness = AgentHarness(agent=agent, session=session, max_iterations=10, max_tool_calls=2)

    recorded_calls: list[dict] = []
    tickers = ["AAPL", "MSFT", "NVDA"]

    async def _stub_call(**kwargs):
        idx = len(recorded_calls)
        recorded_calls.append({"tools": kwargs.get("tools")})
        if kwargs.get("tools") is None:
            return _make_stream(chunks=["Budget summary done."])
        sym = tickers[min(idx, len(tickers) - 1)]
        return _make_stream(
            tool_calls=[
                {
                    "id": f"call_{idx}",
                    "name": "get_quote",
                    "arguments": json.dumps({"ticker": sym}),
                }
            ]
        )

    with patch("lazycat.agent.agent_client.call_agent", side_effect=_stub_call), patch(
        "lazycat.agent.tool_executor.execute_tool", new_callable=AsyncMock
    ) as mock_exec:
        mock_exec.return_value = {"price": 100.0}
        result = await harness.run("Check quotes")

    assert result == "Budget summary done."
    assert harness.stop_cause == "tool_thrash"
    assert mock_exec.call_count == 2
    assert recorded_calls[-1]["tools"] is None
    guard_msgs = [m for m in session.get_messages() if m.get("role") == "tool" and "TOOL_BUDGET_GUARD" in m.get("content", "")]
    assert len(guard_msgs) == 1


@pytest.mark.asyncio
async def test_runaway_repeated_text_block_sets_stop_cause(monkeypatch):
    monkeypatch.setenv("V3_USE_PRISM", "0")
    agent = BaseAgent(model="test-model", name="runaway_agent", system_prompt="sys", provider="vllm")
    session = ConversationSession(session_id="runaway_1")
    harness = AgentHarness(agent=agent, session=session, max_iterations=4)

    prefix = "NVDA thesis remains constructive on Blackwell demand. "
    repeated_block = "This analysis repeats the same thesis sentence indefinitely without adding new evidence. "
    chunks = [prefix] + [repeated_block] * 5

    with patch(
        "lazycat.agent.agent_client.call_agent",
        new_callable=AsyncMock,
        return_value=_make_stream(chunks=chunks),
    ):
        result = await harness.run("Summarize NVDA")

    assert harness.stop_cause == "runaway_stream"
    assert harness.finalize_reason == "RUNAWAY_STREAM"
    assert harness.answer_source == "partial"
    assert result.startswith(prefix)
    assert result.count(repeated_block.strip()) <= 2
