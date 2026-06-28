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
