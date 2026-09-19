import asyncio
import json
from unittest.mock import AsyncMock, MagicMock
import pytest
from lazycat.agent import BaseAgent, AgentHarness
from lazycat.llm import PrismClient
from lazycat.session import ConversationSession


@pytest.mark.asyncio
async def test_explicit_provider_wins_over_same_model_on_another_box():
    client = PrismClient()
    client._model_to_provider_cache["same-model"] = "vllm-2"
    assert await client._resolve_provider_instance("same-model", "vllm") == "vllm"
    assert await client._resolve_provider_instance("same-model", "vllm-2") == "vllm-2"


@pytest.mark.asyncio
async def test_default_provider_resolves_via_model_cache():
    """When provider/base_provider is omitted or None, dynamic resolution via cache/config takes effect."""
    client = PrismClient()
    import time
    client._last_config_fetch = time.time()
    client._model_to_provider_cache["GLM-5.3-Flash-EXL3"] = "vllm-2"
    assert await client._resolve_provider_instance("GLM-5.3-Flash-EXL3") == "vllm-2"
    assert await client._resolve_provider_instance("GLM-5.3-Flash-EXL3", None) == "vllm-2"


@pytest.mark.asyncio
async def test_default_provider_falls_back_to_vllm_if_unknown():
    client = PrismClient()
    import time
    client._last_config_fetch = time.time()
    assert await client._resolve_provider_instance("unknown-model", None) == "vllm"


def test_signatures_default_provider_to_none():
    import inspect
    sig_call = inspect.signature(PrismClient.call_agent)
    assert sig_call.parameters["provider"].default is None
    sig_stream = inspect.signature(PrismClient.agent_chat_stream)
    assert sig_stream.parameters["provider"].default is None



def test_harness_has_no_hardcoded_model_default():
    with pytest.raises(TypeError):
        BaseAgent(name="test", system_prompt="test")


@pytest.mark.asyncio
async def test_cancellation_keeps_last_cumulative_usage_snapshot_once():
    agent = BaseAgent(name="test", system_prompt="test", model="discovered-model")
    response = MagicMock()
    response.aclose = AsyncMock()
    async def lines():
        for output in (10, 20):
            yield 'data: ' + json.dumps({"type": "usage_update", "usage": {"inputTokens": 100, "outputTokens": output}})
        raise asyncio.CancelledError()
    response.aiter_lines = lines
    agent.llm_client = MagicMock()
    agent.llm_client.call_agent = AsyncMock(return_value=response)
    harness = AgentHarness(agent, ConversationSession(session_id="test"))
    with pytest.raises(asyncio.CancelledError):
        await harness.run("hello")
    assert harness.completion_tokens == 20
    assert harness.total_tokens == 120
    assert harness.usage_requests == 1
    assert harness.total_requests == 1
    assert harness.prompt_tokens == 100
    response.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_expired_discovery_replaces_moved_and_removed_models():
    client = PrismClient()
    client._model_to_provider_cache = {"moved": "old-endpoint", "removed": "old-endpoint"}
    response = MagicMock(status_code=200)
    response.json.return_value = {"textToText": {"models": {"new-endpoint": [{"name": "moved"}]}}}
    transport = MagicMock()
    transport.get = AsyncMock(return_value=response)
    client._get_client = AsyncMock(return_value=transport)
    assert await client._resolve_provider_instance("moved") == "new-endpoint"
    assert "removed" not in client._model_to_provider_cache
    transport.get.assert_awaited_once()
