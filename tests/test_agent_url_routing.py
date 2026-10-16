"""Agent URL routing: the SDK talks only to lazy-agent-service (LAZY_AGENT_URL)."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from lazycat.config import Config
from lazycat.llm import AgentClient


def _mock_post(url_capture: dict):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"choices": []}

    async def _post(url, *args, **kwargs):
        url_capture["url"] = url
        return mock_resp

    return _post


def test_default_url_points_at_native_gateway(monkeypatch):
    monkeypatch.delenv("LAZY_AGENT_URL", raising=False)
    assert Config().LAZY_AGENT_URL == "http://10.0.0.16:5591"


def test_url_comes_from_lazy_agent_url_env(monkeypatch):
    monkeypatch.setenv("LAZY_AGENT_URL", "http://agent.example:1234")
    assert AgentClient().url == "http://agent.example:1234"


@pytest.mark.asyncio
async def test_call_agent_without_explicit_url_posts_to_configured_agent(monkeypatch):
    monkeypatch.setenv("LAZY_AGENT_URL", "http://agent.example:1234")
    client = AgentClient()  # no url set
    captured: dict = {}
    with patch.object(
        AgentClient, "_get_client", new=AsyncMock(return_value=MagicMock(post=_mock_post(captured)))
    ):
        await client.call_agent(model="m", system_prompt="s", messages=[{"role": "user", "content": "hi"}])

    assert captured["url"] == "http://agent.example:1234/agent?stream=false"
