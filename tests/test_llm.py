import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from lazycat.llm import PrismClient

@pytest.mark.asyncio
async def test_prism_client_sends_correct_shape():
    client = PrismClient()
    client.url = "http://prism"
    
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"choices": []}
    
    with patch("httpx.AsyncClient.post", return_value=mock_resp) as mock_post:
        await client.call_agent(
            model="test-model",
            messages=[{"role": "user", "content": "hello"}],
            system_prompt="system test",
            agent_name="test_agent",
            tools=[{"type": "function", "function": {"name": "test_tool"}}]
        )
        
        # Verify the call shape
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        
        assert args[0] == "http://prism/agent?stream=false"
        
        payload = kwargs["json"]
        assert payload["model"] == "test-model"
        assert payload["messages"] == [{"role": "user", "content": "hello"}]
        assert payload["systemPrompt"] == "system test"
        assert payload["agent"] == "test_agent"
        assert payload["functionCallingEnabled"] is True
        assert len(payload["tools"]) == 1
        assert payload["tools"][0]["function"]["name"] == "test_tool"
