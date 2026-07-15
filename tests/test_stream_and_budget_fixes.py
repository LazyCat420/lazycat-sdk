"""
Tests for the 2026-07-15 harness-bridge fixes.

1. LLMStreamWrapper (direct-vLLM path) must ASSEMBLE fragmented OpenAI
   tool-call deltas into one complete toolCalls event — previously every
   partial delta was re-emitted as a full event and the consumer replaced
   its list each time, losing names and argument fragments.
2. AgentHarness must send its max_iterations over the wire — on the prism
   path the agentic loop runs server-side, so without maxIterations in the
   payload the caller's per-role turn budget never bound.
"""
import json

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from lazycat.agent import BaseAgent, AgentHarness
from lazycat.llm import LLMStreamWrapper
from lazycat.session import ConversationSession


def _fake_openai_response(sse_lines):
    resp = MagicMock()

    async def aiter_text():
        yield "\n".join(sse_lines) + "\n"

    resp.aiter_text = aiter_text
    resp.aclose = AsyncMock()
    return resp


def _delta_line(tool_calls=None, content=None, finish_reason=None):
    delta = {}
    if content is not None:
        delta["content"] = content
    if tool_calls is not None:
        delta["tool_calls"] = tool_calls
    choice = {"delta": delta}
    if finish_reason:
        choice["finish_reason"] = finish_reason
    return "data: " + json.dumps({"choices": [choice]})


@pytest.mark.asyncio
async def test_fragmented_tool_call_deltas_are_assembled():
    lines = [
        # First delta: id + name + start of args for tool 0
        _delta_line(tool_calls=[{"index": 0, "id": "call_a",
                                 "function": {"name": "get_market", "arguments": '{"tick'}}]),
        # Second delta: rest of tool 0's args
        _delta_line(tool_calls=[{"index": 0, "function": {"arguments": 'er": "NVDA"}'}}]),
        # Third delta: a SECOND tool call
        _delta_line(tool_calls=[{"index": 1, "id": "call_b",
                                 "function": {"name": "web_search", "arguments": '{"q": "news"}'}}]),
        _delta_line(finish_reason="tool_calls"),
        "data: [DONE]",
    ]
    wrapper = LLMStreamWrapper(_fake_openai_response(lines), is_openai=True)

    tool_call_events = []
    async for line in wrapper.aiter_lines():
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        data = json.loads(line[6:])
        if "toolCalls" in data:
            tool_call_events.append(data["toolCalls"])

    # Exactly ONE assembled event, not one per delta
    assert len(tool_call_events) == 1
    calls = tool_call_events[0]
    assert len(calls) == 2
    assert calls[0]["id"] == "call_a"
    assert calls[0]["function"]["name"] == "get_market"
    assert json.loads(calls[0]["function"]["arguments"]) == {"ticker": "NVDA"}
    assert calls[1]["function"]["name"] == "web_search"


@pytest.mark.asyncio
async def test_flush_on_done_without_finish_reason():
    lines = [
        _delta_line(tool_calls=[{"index": 0, "id": "call_a",
                                 "function": {"name": "t", "arguments": ""}}]),
        "data: [DONE]",
    ]
    wrapper = LLMStreamWrapper(_fake_openai_response(lines), is_openai=True)

    seen = [line async for line in wrapper.aiter_lines()]

    # toolCalls flushed before [DONE], empty args defaulted to {}
    assert any("toolCalls" in s for s in seen)
    flushed = json.loads([s for s in seen if "toolCalls" in s][0][6:])
    assert flushed["toolCalls"][0]["function"]["arguments"] == "{}"
    assert seen[-1] == "data: [DONE]"


@pytest.mark.asyncio
async def test_harness_sends_max_iterations_over_the_wire():
    agent = BaseAgent(name="test_agent", system_prompt="You are a test agent")
    session = ConversationSession(session_id="test_mi")
    harness = AgentHarness(agent=agent, session=session, max_iterations=5)

    mock_resp = MagicMock()
    mock_resp.aclose = AsyncMock()

    async def mock_aiter_lines():
        yield 'data: {"type": "chunk", "content": "done"}'
        yield "data: [DONE]"

    mock_resp.aiter_lines = mock_aiter_lines

    with patch("lazycat.agent.prism_client.call_agent", new_callable=AsyncMock) as mock_call:
        mock_call.return_value = mock_resp
        await harness.run("Hello")

    assert mock_call.call_args.kwargs["max_iterations"] == 5
