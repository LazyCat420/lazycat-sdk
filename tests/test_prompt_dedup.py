"""Unit tests for system prompt deduplication across the /agent wire.

Ensures system_prompt is passed in payload['systemPrompt'] without being duplicated
as an inlined role:'system' entry in payload['messages'].
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from lazycat.agent import BaseAgent, AgentHarness
from lazycat.session import ConversationSession
from lazycat.llm import PrismClient


@pytest.mark.asyncio
async def test_call_agent_does_not_duplicate_system_prompt_in_messages():
    client = PrismClient()
    client.url = "http://fake-prism:8000"
    
    mock_post = MagicMock()
    mock_post.status_code = 200
    mock_post.headers = {"content-type": "text/event-stream"}
    
    async def aiter_lines():
        yield "data: [DONE]"
    
    mock_post.aiter_lines = aiter_lines
    mock_post.aclose = AsyncMock()
    
    with patch.object(client, "_get_client") as mock_get_client:
        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=mock_post)
        mock_get_client.return_value = mock_http
        
        await client.call_agent(
            model="test-model",
            messages=[{"role": "user", "content": "analyze ticker"}],
            system_prompt="You are a senior analyst with strict guidelines.",
            inline_system_prompt=False,
        )
        
        assert mock_http.post.called
        sent_json = mock_http.post.call_args.kwargs.get("json", {})
        
        # systemPrompt must be present at top level of payload
        assert sent_json.get("systemPrompt") == "You are a senior analyst with strict guidelines."
        # messages array must NOT contain the inlined system prompt
        assert not any(m.get("role") == "system" for m in sent_json.get("messages", []))
        assert len(sent_json.get("messages", [])) == 1
        assert sent_json["messages"][0]["role"] == "user"


@pytest.mark.asyncio
async def test_agent_harness_passes_inline_system_prompt_false():
    agent = BaseAgent(name="test_agent", system_prompt="You are a test agent")
    session = ConversationSession(session_id="test_dedup")
    session.add_user_message("evaluate")
    harness = AgentHarness(agent=agent, session=session)
    
    mock_resp = MagicMock()
    mock_resp.aclose = AsyncMock()
    async def mock_aiter_lines():
        yield "data: {\"type\": \"chunk\", \"content\": \"done\"}"
        yield "data: [DONE]"
    mock_resp.aiter_lines = mock_aiter_lines
    
    with patch("lazycat.agent.prism_client.call_agent", new_callable=AsyncMock) as mock_call:
        mock_call.return_value = mock_resp
        await harness.run("evaluate")
        
        assert mock_call.called
        call_kwargs = mock_call.call_args.kwargs
        assert call_kwargs.get("inline_system_prompt") is False
