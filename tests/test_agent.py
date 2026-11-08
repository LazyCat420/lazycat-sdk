import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from lazycat.agent import BaseAgent, AgentHarness
from lazycat.session import ConversationSession

@pytest.mark.asyncio
async def test_agent_harness_terminates_cleanly():
    agent = BaseAgent(model="test-model", name="test_agent", system_prompt="You are a test agent")
    session = ConversationSession(session_id="test_123")
    harness = AgentHarness(agent=agent, session=session)
    
    # Mock the LLM call to return a text response with NO tool calls
    mock_resp = MagicMock()
    mock_resp.aclose = AsyncMock()
    
    async def mock_aiter_lines():
        yield "data: {\"type\": \"chunk\", \"content\": \"I am done.\"}"
        yield "data: {\"type\": \"done\"}"
        yield "data: [DONE]"
    
    mock_resp.aiter_lines = mock_aiter_lines
    
    with patch("lazycat.agent.agent_client.call_agent", new_callable=AsyncMock) as mock_call:
        mock_call.return_value = mock_resp
        result = await harness.run("Hello")
        
        # Should terminate immediately and return the LLM's text
        assert result == "I am done."
        
        # Check session history
        messages = session.get_messages()
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "I am done."


@pytest.mark.asyncio
async def test_agent_harness_captures_resolved_model_from_done_event():
    """The done event carries prism's SERVER-side resolved model — the value a
    per-model scorecard must attribute to, since it survives gateway-side model
    swaps the requested name knows nothing about."""
    agent = BaseAgent(model="test-model", name="test_agent", system_prompt="You are a test agent")
    session = ConversationSession(session_id="test_model")
    harness = AgentHarness(agent=agent, session=session)

    mock_resp = MagicMock()
    mock_resp.aclose = AsyncMock()

    async def mock_aiter_lines():
        yield "data: {\"type\": \"chunk\", \"content\": \"done.\"}"
        yield "data: {\"type\": \"done\", \"model\": \"deepseek-v4-flash-0731\", \"provider\": \"vllm-2\"}"
        yield "data: [DONE]"

    mock_resp.aiter_lines = mock_aiter_lines

    assert harness.last_model is None
    assert harness.last_provider is None
    with patch("lazycat.agent.agent_client.call_agent", new_callable=AsyncMock) as mock_call:
        mock_call.return_value = mock_resp
        await harness.run("Hello")

    assert harness.last_model == "deepseek-v4-flash-0731"
    assert harness.last_provider == "vllm-2"


@pytest.mark.asyncio
async def test_done_event_without_model_keeps_previous_value():
    """A done frame with a null/absent model must not clobber the last known
    resolution — partial frames happen on error paths."""
    agent = BaseAgent(model="test-model", name="test_agent", system_prompt="You are a test agent")
    session = ConversationSession(session_id="test_model_keep")
    harness = AgentHarness(agent=agent, session=session)
    harness.last_model = "prior-model"

    mock_resp = MagicMock()
    mock_resp.aclose = AsyncMock()

    async def mock_aiter_lines():
        yield "data: {\"type\": \"chunk\", \"content\": \"done.\"}"
        yield "data: {\"type\": \"done\", \"model\": null}"
        yield "data: [DONE]"

    mock_resp.aiter_lines = mock_aiter_lines

    with patch("lazycat.agent.agent_client.call_agent", new_callable=AsyncMock) as mock_call:
        mock_call.return_value = mock_resp
        await harness.run("Hello")

    assert harness.last_model == "prior-model"


@pytest.mark.asyncio
async def test_agent_harness_tool_call():
    agent = BaseAgent(model="test-model", name="test_agent", system_prompt="You are a test agent")
    # Add a mock tool to the agent
    agent.add_tool({"name": "dummy_tool", "description": "dummy"})
    
    session = ConversationSession(session_id="test_123")
    harness = AgentHarness(agent=agent, session=session)
    
    # Mock LLM to return a tool call chunk, then a text chunk
    mock_resp1 = MagicMock()
    mock_resp1.aclose = AsyncMock()
    
    async def mock_aiter_lines1():
        yield "data: {\"type\": \"chunk\", \"content\": \"\"}"
        yield "data: {\"toolCalls\": [{\"id\": \"call_1\", \"name\": \"dummy_tool\", \"arguments\": \"{}\"}]}"
        yield "data: {\"type\": \"done\"}"
        yield "data: [DONE]"
    mock_resp1.aiter_lines = mock_aiter_lines1

    mock_resp2 = MagicMock()
    mock_resp2.aclose = AsyncMock()
    async def mock_aiter_lines2():
        yield "data: {\"type\": \"chunk\", \"content\": \"Result processed.\"}"
        yield "data: {\"type\": \"done\"}"
        yield "data: [DONE]"
    mock_resp2.aiter_lines = mock_aiter_lines2
    
    # Mock execute_tool to return a simple dummy result
    with patch("lazycat.agent.agent_client.call_agent", new_callable=AsyncMock) as mock_call, \
         patch("lazycat.agent.tool_executor.execute_tool", new_callable=AsyncMock) as mock_exec:
        
        mock_call.side_effect = [mock_resp1, mock_resp2]
        mock_exec.return_value = {"status": "ok"}
        
        result = await harness.run("Hello")
        
        assert result == "Result processed."
        assert mock_exec.called
        assert mock_exec.call_args[0][0] == "dummy_tool"



@pytest.mark.asyncio
async def test_error_event_is_terminal_and_discards_partial_content():
    """A mid-stream `error` event must fail the turn: the partially
    accumulated content is not an answer (2026-10-20 cut-stream audit:
    'probeprobeprobe...' was committed and returned as a successful reply).
    """
    agent = BaseAgent(model="test-model", name="test_agent", system_prompt="You are a test agent")
    session = ConversationSession(session_id="test_error")
    harness = AgentHarness(agent=agent, session=session)

    mock_resp = MagicMock()
    mock_resp.aclose = AsyncMock()

    async def mock_aiter_lines():
        yield "data: {\"type\": \"chunk\", \"content\": \"probeprobe\"}"
        yield "data: {\"type\": \"error\", \"message\": \"upstream vLLM died\"}"
        yield "data: {\"type\": \"chunk\", \"content\": \"more partial text\"}"
        yield "data: [DONE]"

    mock_resp.aiter_lines = mock_aiter_lines

    with patch("lazycat.agent.agent_client.call_agent", new_callable=AsyncMock) as mock_call:
        mock_call.return_value = mock_resp
        with pytest.raises(RuntimeError, match="upstream vLLM died"):
            await harness.run("Hello")

    # The partial content must not be booked as the assistant's answer.
    messages = session.get_messages()
    assert [m["role"] for m in messages] == ["user"]
    assert not any(m["role"] == "assistant" for m in messages)


@pytest.mark.asyncio
async def test_stream_ending_without_done_event_is_not_returned_as_answer():
    """A stream cut mid-generation (no `done` event) must not come back as a
    normal answer; the partial content is discarded and the turn fails."""
    agent = BaseAgent(model="test-model", name="test_agent", system_prompt="You are a test agent")
    session = ConversationSession(session_id="test_cut")
    harness = AgentHarness(agent=agent, session=session)

    mock_resp = MagicMock()
    mock_resp.aclose = AsyncMock()

    async def mock_aiter_lines():
        yield "data: {\"type\": \"chunk\", \"content\": \"probeprobeprobe\"}"
        # connection dies here: no done event, no [DONE] sentinel

    mock_resp.aiter_lines = mock_aiter_lines

    with patch("lazycat.agent.agent_client.call_agent", new_callable=AsyncMock) as mock_call:
        mock_call.return_value = mock_resp
        with pytest.raises(RuntimeError, match="without a done event"):
            await harness.run("Hello")

    messages = session.get_messages()
    assert [m["role"] for m in messages] == ["user"]


@pytest.mark.asyncio
async def test_normal_stream_with_done_event_still_returns_text():
    agent = BaseAgent(model="test-model", name="test_agent", system_prompt="You are a test agent")
    session = ConversationSession(session_id="test_normal")
    harness = AgentHarness(agent=agent, session=session)

    mock_resp = MagicMock()
    mock_resp.aclose = AsyncMock()

    async def mock_aiter_lines():
        yield "data: {\"type\": \"chunk\", \"content\": \"OK\"}"
        yield "data: {\"type\": \"done\"}"
        yield "data: [DONE]"

    mock_resp.aiter_lines = mock_aiter_lines

    with patch("lazycat.agent.agent_client.call_agent", new_callable=AsyncMock) as mock_call:
        mock_call.return_value = mock_resp
        result = await harness.run("Hello")

    assert result == "OK"
    messages = session.get_messages()
    assert messages[-1] == {"role": "assistant", "content": "OK"}


def _make_stream_resp(content: str = "", tool_calls: list[dict] | None = None):
    import json

    resp = MagicMock()
    resp.aclose = AsyncMock()

    async def _aiter_lines():
        if content:
            yield f'data: {json.dumps({"type": "chunk", "content": content})}'
        if tool_calls:
            yield f'data: {json.dumps({"toolCalls": tool_calls})}'
        yield 'data: {"type": "done"}'
        yield "data: [DONE]"

    resp.aiter_lines = _aiter_lines
    return resp


@pytest.mark.asyncio
async def test_direct_path_final_turn_withholds_tools_and_nudges_final_answer(monkeypatch):
    monkeypatch.setenv("V3_USE_PRISM", "0")
    agent = BaseAgent(
        model="test-model", name="test_agent", system_prompt="sys", provider="vllm"
    )
    agent.add_tool({"name": "dummy_tool", "description": "dummy"})
    session = ConversationSession(session_id="test_final_turn")
    harness = AgentHarness(agent=agent, session=session, max_iterations=3)

    recorded_calls: list[dict] = []

    async def _stub_call_agent(**kwargs):
        recorded_calls.append(
            {"tools": kwargs.get("tools"), "messages": list(kwargs.get("messages", []))}
        )
        if kwargs.get("tools") is not None:
            return _make_stream_resp(
                tool_calls=[{"id": f"call_{len(recorded_calls)}", "name": "dummy_tool", "arguments": "{}"}]
            )
        return _make_stream_resp(content="Final synthesized answer.")

    with patch("lazycat.agent.agent_client.call_agent", side_effect=_stub_call_agent), \
         patch("lazycat.agent.tool_executor.execute_tool", new_callable=AsyncMock) as mock_exec:
        mock_exec.return_value = {"status": "ok"}
        result = await harness.run("Analyze NVDA")

    expected_nudge = (
        "Tool budget exhausted. Do not call tools. Reply now with your final "
        "answer in the required output format using the evidence gathered so far."
    )
    assert result == "Final synthesized answer."
    assert len(recorded_calls) == harness.max_iterations == 3
    assert recorded_calls[0]["tools"] == agent.tools
    assert recorded_calls[1]["tools"] == agent.tools
    assert recorded_calls[2]["tools"] is None
    assert recorded_calls[2]["messages"][-1] == {
        "role": "user",
        "content": expected_nudge,
    }
    assert all(
        m.get("content") != expected_nudge for m in session.get_messages()
    )


@pytest.mark.asyncio
async def test_prism_path_offers_tools_on_every_call(monkeypatch):
    monkeypatch.setenv("V3_USE_PRISM", "1")
    agent = BaseAgent(
        model="test-model", name="test_agent", system_prompt="sys", provider="vllm"
    )
    agent.add_tool({"name": "dummy_tool", "description": "dummy"})
    session = ConversationSession(session_id="test_prism_path")
    harness = AgentHarness(agent=agent, session=session, max_iterations=3)

    recorded_calls: list[dict] = []

    async def _stub_call_agent(**kwargs):
        recorded_calls.append(
            {"tools": kwargs.get("tools"), "messages": list(kwargs.get("messages", []))}
        )
        if kwargs.get("tools") is not None:
            return _make_stream_resp(
                tool_calls=[{"id": f"call_{len(recorded_calls)}", "name": "dummy_tool", "arguments": "{}"}]
            )
        return _make_stream_resp(content="Should not reach tools=None on prism")

    with patch("lazycat.agent.agent_client.call_agent", side_effect=_stub_call_agent), \
         patch("lazycat.agent.tool_executor.execute_tool", new_callable=AsyncMock) as mock_exec:
        mock_exec.return_value = {"status": "ok"}
        result = await harness.run("Analyze NVDA")

    assert result == "Max iterations reached without a final answer."
    assert len(recorded_calls) == 3
    assert all(call["tools"] == agent.tools for call in recorded_calls)


@pytest.mark.asyncio
async def test_max_iterations_one_direct_still_offers_tools(monkeypatch):
    monkeypatch.setenv("V3_USE_PRISM", "0")
    agent = BaseAgent(
        model="test-model", name="test_agent", system_prompt="sys", provider="vllm"
    )
    agent.add_tool({"name": "dummy_tool", "description": "dummy"})
    session = ConversationSession(session_id="test_max_iter_1")
    harness = AgentHarness(agent=agent, session=session, max_iterations=1)

    recorded_calls: list[dict] = []

    async def _stub_call_agent(**kwargs):
        recorded_calls.append(
            {"tools": kwargs.get("tools"), "messages": list(kwargs.get("messages", []))}
        )
        return _make_stream_resp(
            tool_calls=[{"id": "call_1", "name": "dummy_tool", "arguments": "{}"}]
        )

    with patch("lazycat.agent.agent_client.call_agent", side_effect=_stub_call_agent), \
         patch("lazycat.agent.tool_executor.execute_tool", new_callable=AsyncMock) as mock_exec:
        mock_exec.return_value = {"status": "ok"}
        result = await harness.run("Analyze NVDA")

    assert result == "Max iterations reached without a final answer."
    assert len(recorded_calls) == 1
    assert recorded_calls[0]["tools"] == agent.tools


@pytest.mark.asyncio
async def test_direct_path_final_turn_ignores_tool_calls_if_still_returned(monkeypatch):
    monkeypatch.setenv("V3_USE_PRISM", "0")
    agent = BaseAgent(
        model="test-model", name="test_agent", system_prompt="sys", provider="vllm"
    )
    agent.add_tool({"name": "dummy_tool", "description": "dummy"})

    # 1) Non-empty content alongside ignored tool_calls -> returns content without executing final tool
    session1 = ConversationSession(session_id="test_final_tc_content")
    harness1 = AgentHarness(agent=agent, session=session1, max_iterations=2)
    responses1 = [
        _make_stream_resp(tool_calls=[{"id": "c1", "name": "dummy_tool", "arguments": "{}"}]),
        _make_stream_resp(
            content="Answer despite stray tool call.",
            tool_calls=[{"id": "c2", "name": "dummy_tool", "arguments": "{}"}],
        ),
    ]
    with patch("lazycat.agent.agent_client.call_agent", new_callable=AsyncMock, side_effect=responses1), \
         patch("lazycat.agent.tool_executor.execute_tool", new_callable=AsyncMock) as mock_exec1:
        mock_exec1.return_value = {"status": "ok"}
        res1 = await harness1.run("Analyze NVDA")
    assert res1 == "Answer despite stray tool call."
    assert mock_exec1.call_count == 1

    # 2) Empty content alongside ignored tool_calls -> returns max-iterations fallback without executing final tool
    session2 = ConversationSession(session_id="test_final_tc_empty")
    harness2 = AgentHarness(agent=agent, session=session2, max_iterations=2)
    responses2 = [
        _make_stream_resp(tool_calls=[{"id": "c1", "name": "dummy_tool", "arguments": "{}"}]),
        _make_stream_resp(
            content="",
            tool_calls=[{"id": "c2", "name": "dummy_tool", "arguments": "{}"}],
        ),
    ]
    with patch("lazycat.agent.agent_client.call_agent", new_callable=AsyncMock, side_effect=responses2), \
         patch("lazycat.agent.tool_executor.execute_tool", new_callable=AsyncMock) as mock_exec2:
        mock_exec2.return_value = {"status": "ok"}
        res2 = await harness2.run("Analyze NVDA")
    assert res2 == "Max iterations reached without a final answer."
    assert mock_exec2.call_count == 1


@pytest.mark.asyncio
async def test_direct_path_final_turn_empty_reply_adds_no_assistant_message(monkeypatch):
    """An empty forced-final reply is not an answer: the loop must break to
    the max-iterations fallback without booking an empty assistant turn into
    the session (the old `and tool_calls` guard let this fall through to the
    normal history-append path)."""
    monkeypatch.setenv("V3_USE_PRISM", "0")
    agent = BaseAgent(
        model="test-model", name="test_agent", system_prompt="sys", provider="vllm"
    )
    agent.add_tool({"name": "dummy_tool", "description": "dummy"})
    session = ConversationSession(session_id="test_final_turn_no_content")
    harness = AgentHarness(agent=agent, session=session, max_iterations=2)

    responses = [
        _make_stream_resp(tool_calls=[{"id": "c1", "name": "dummy_tool", "arguments": "{}"}]),
        _make_stream_resp(content="", tool_calls=None),
    ]
    with patch("lazycat.agent.agent_client.call_agent", new_callable=AsyncMock, side_effect=responses), \
         patch("lazycat.agent.tool_executor.execute_tool", new_callable=AsyncMock) as mock_exec:
        mock_exec.return_value = {"status": "ok"}
        result = await harness.run("Analyze NVDA")

    assert result == "Max iterations reached without a final answer."
    assert mock_exec.call_count == 1
    last = session.get_messages()[-1]
    assert not (last["role"] == "assistant" and not last.get("content"))


@pytest.mark.asyncio
async def test_direct_path_final_turn_stray_tool_calls_never_recorded_or_run(monkeypatch):
    """A stray tool_calls on the forced final turn must not leak into the
    session as an orphaned call: every recorded assistant tool_calls id needs
    a following tool message, and the stray call must not be executed."""
    monkeypatch.setenv("V3_USE_PRISM", "0")
    agent = BaseAgent(
        model="test-model", name="test_agent", system_prompt="sys", provider="vllm"
    )
    agent.add_tool({"name": "dummy_tool", "description": "dummy"})
    session = ConversationSession(session_id="test_final_turn_stray_tc")
    harness = AgentHarness(agent=agent, session=session, max_iterations=2)

    responses = [
        _make_stream_resp(tool_calls=[{"id": "c1", "name": "dummy_tool", "arguments": "{}"}]),
        _make_stream_resp(
            content="Final answer despite stray tool call.",
            tool_calls=[{"id": "c2", "name": "dummy_tool", "arguments": "{}"}],
        ),
    ]
    with patch("lazycat.agent.agent_client.call_agent", new_callable=AsyncMock, side_effect=responses), \
         patch("lazycat.agent.tool_executor.execute_tool", new_callable=AsyncMock) as mock_exec:
        mock_exec.return_value = {"status": "ok"}
        result = await harness.run("Analyze NVDA")

    assert result == "Final answer despite stray tool call."
    # Only the iteration-1 call ran; the stray final-turn call was not executed.
    assert mock_exec.call_count == 1

    pending: set[str] = set()
    for message in session.get_messages():
        if message["role"] == "assistant":
            pending.update(tc["id"] for tc in message.get("tool_calls", []))
        elif message["role"] == "tool":
            pending.discard(message["tool_call_id"])
    assert pending == set()


@pytest.mark.asyncio
async def test_provider_none_offers_tools_on_every_call(monkeypatch):
    """provider=None keeps the prism path: force_final_answer never fires,
    so tools stay on the wire for every call up to max_iterations."""
    monkeypatch.setenv("V3_USE_PRISM", "1")
    agent = BaseAgent(
        model="test-model", name="test_agent", system_prompt="sys", provider=None
    )
    agent.add_tool({"name": "dummy_tool", "description": "dummy"})
    session = ConversationSession(session_id="test_provider_none_tools")
    harness = AgentHarness(agent=agent, session=session, max_iterations=3)

    recorded_calls: list[dict] = []

    async def _stub_call_agent(**kwargs):
        recorded_calls.append(
            {"tools": kwargs.get("tools"), "messages": list(kwargs.get("messages", []))}
        )
        return _make_stream_resp(
            tool_calls=[{"id": f"call_{len(recorded_calls)}", "name": "dummy_tool", "arguments": "{}"}]
        )

    with patch("lazycat.agent.agent_client.call_agent", side_effect=_stub_call_agent), \
         patch("lazycat.agent.tool_executor.execute_tool", new_callable=AsyncMock) as mock_exec:
        mock_exec.return_value = {"status": "ok"}
        result = await harness.run("Analyze NVDA")

    assert result == "Max iterations reached without a final answer."
    assert len(recorded_calls) == 3
    assert all(call["tools"] == agent.tools for call in recorded_calls)
