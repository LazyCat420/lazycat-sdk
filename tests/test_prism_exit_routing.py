"""Prism-exit routing (2026-10-15).

With ``PRISM_ENABLED=false`` the SDK must never reach prism (:7777 or the
/prism-proxy). Two contracts:

* A caller that EXPLICITLY set ``prism_client.url`` (trading-service points it
  at lazy-agent-service's native :5591) still gets the agentic harness — the
  call goes to ``{url}/agent``.
* A caller that never set a URL keeps the old standalone bypass: direct vLLM.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from lazycat.config import config
from lazycat.llm import PrismClient


def _mock_post(url_capture: dict):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"choices": []}

    async def _post(url, *args, **kwargs):
        url_capture["url"] = url
        return mock_resp

    return _post


@pytest.mark.asyncio
async def test_prism_disabled_with_explicit_url_still_uses_native_harness(monkeypatch):
    monkeypatch.setenv("PRISM_ENABLED", "False")
    client = PrismClient()
    client.url = "http://10.0.0.16:5591"

    captured: dict = {}
    with patch.object(
        PrismClient, "_get_client", new=AsyncMock(return_value=MagicMock(post=_mock_post(captured)))
    ), patch.object(
        PrismClient, "_call_vllm_direct", new=AsyncMock(side_effect=AssertionError("must not bypass the harness"))
    ):
        await client.call_agent(model="m", system_prompt="s", messages=[{"role": "user", "content": "hi"}])

    assert captured["url"] == "http://10.0.0.16:5591/agent?stream=false"
    assert ":7777" not in captured["url"]
    assert "prism-proxy" not in captured["url"]


@pytest.mark.asyncio
async def test_prism_disabled_without_explicit_url_falls_back_to_vllm(monkeypatch):
    monkeypatch.setenv("PRISM_ENABLED", "False")
    client = PrismClient()  # no url set

    async def _direct(**kwargs):
        return "direct-vllm"

    with patch.object(PrismClient, "_call_vllm_direct", new=AsyncMock(side_effect=_direct)):
        result = await client.call_agent(model="m", system_prompt="s", messages=[{"role": "user", "content": "hi"}])

    assert result == "direct-vllm"


def test_default_prism_url_points_at_native_gateway(monkeypatch):
    monkeypatch.delenv("PRISM_URL", raising=False)
    from lazycat.config import Config

    assert Config().PRISM_URL == "http://10.0.0.16:5591"
    assert config is not None  # singleton exists for callers
