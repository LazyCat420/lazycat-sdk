"""The harness must accumulate the OUTPUT side of usage, not only the input.

`last_usage` is the LAST request's snapshot and `total_tokens` is one fused
sum of input+output+reasoning, so neither can answer "what did this agent
GENERATE" — a caller costing a decision was left reading the prompt side only.
`outputTokens` was parsed off the stream and discarded.

`usage_requests` counts the requests that actually REPORTED a usage block.
That count is what keeps a RECORDED zero (an all-zero usage block marks a
TRUNCATED generation, not a cheap one) distinct from NOT RECORDED — both of
which otherwise present as a completion count of 0.
"""
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from lazycat.agent import AgentHarness, BaseAgent
from lazycat.session import ConversationSession


def _harness(*responses):
    """A harness whose llm_client replays the given SSE event lists."""
    agent = BaseAgent(name="t", system_prompt="sys")
    agent.llm_client = MagicMock()
    calls = list(responses)

    async def _call_agent(**kwargs):
        events = calls.pop(0)

        async def aiter_lines():
            for ev in events:
                yield "data: " + json.dumps(ev)
            yield "data: [DONE]"

        resp = MagicMock()
        resp.aiter_lines = aiter_lines
        resp.aclose = AsyncMock()
        return resp

    agent.llm_client.call_agent = _call_agent
    return AgentHarness(agent=agent, session=ConversationSession(session_id="s"))


def _final(text, usage=None):
    events = [{"type": "chunk", "content": text}]
    if usage is not None:
        events.append({"type": "usage_update", "usage": usage})
    events.append({"type": "done", "model": "m", "provider": "p"})
    return events


@pytest.mark.asyncio
async def test_the_accumulators_start_at_zero():
    h = _harness()
    assert h.completion_tokens == 0
    assert h.usage_requests == 0


@pytest.mark.asyncio
async def test_output_and_reasoning_tokens_are_summed():
    h = _harness(_final("hi", {
        "inputTokens": 1000, "outputTokens": 300, "reasoningOutputTokens": 45,
    }))
    await h.run("go")
    assert h.completion_tokens == 345
    assert h.usage_requests == 1
    # existing behaviour untouched
    assert h.total_tokens == 1345
    assert h.last_usage["outputTokens"] == 300


@pytest.mark.asyncio
async def test_an_all_zero_usage_block_still_counts_as_reported():
    """A truncated generation reports zeros — we must know it was MEASURED."""
    h = _harness(_final("cut off", {
        "inputTokens": 0, "outputTokens": 0, "reasoningOutputTokens": 0,
    }))
    await h.run("go")
    assert h.completion_tokens == 0
    assert h.usage_requests == 1, "a reported zero was mistaken for no report"


@pytest.mark.asyncio
async def test_no_usage_block_leaves_the_request_count_at_zero():
    h = _harness(_final("no usage block"))
    await h.run("go")
    assert h.completion_tokens == 0
    assert h.usage_requests == 0
