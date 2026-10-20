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
