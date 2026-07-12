import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from lazycat.agent import BaseAgent, AgentHarness
from lazycat.session import ConversationSession

@pytest.mark.asyncio
async def test_agent_harness_terminates_cleanly():
    agent = BaseAgent(name="test_agent", system_prompt="You are a test agent")
    session = ConversationSession(session_id="test_123")
    harness = AgentHarness(agent=agent, session=session)
    
    # Mock the LLM call to return a text response with NO tool calls
    mock_resp = MagicMock()
    mock_resp.aclose = AsyncMock()
    
    async def mock_aiter_lines():
        yield "data: {\"type\": \"chunk\", \"content\": \"I am done.\"}"
        yield "data: [DONE]"
    
    mock_resp.aiter_lines = mock_aiter_lines
    
    with patch("lazycat.agent.prism_client.call_agent", new_callable=AsyncMock) as mock_call:
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
async def test_agent_harness_tool_call():
    agent = BaseAgent(name="test_agent", system_prompt="You are a test agent")
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
        yield "data: [DONE]"
    mock_resp1.aiter_lines = mock_aiter_lines1

    mock_resp2 = MagicMock()
    mock_resp2.aclose = AsyncMock()
    async def mock_aiter_lines2():
        yield "data: {\"type\": \"chunk\", \"content\": \"Result processed.\"}"
        yield "data: [DONE]"
    mock_resp2.aiter_lines = mock_aiter_lines2
    
    # Mock execute_tool to return a simple dummy result
    with patch("lazycat.agent.prism_client.call_agent", new_callable=AsyncMock) as mock_call, \
         patch("lazycat.agent.tool_executor.execute_tool", new_callable=AsyncMock) as mock_exec:
        
        mock_call.side_effect = [mock_resp1, mock_resp2]
        mock_exec.return_value = {"status": "ok"}
        
        result = await harness.run("Hello")
        
        assert result == "Result processed."
        assert mock_exec.called
        assert mock_exec.call_args[0][0] == "dummy_tool"

