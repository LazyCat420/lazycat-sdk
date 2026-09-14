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
    response.aclose.assert_awaited_once()
