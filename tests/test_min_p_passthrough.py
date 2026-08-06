"""AgentHarness must forward BaseAgent.min_p to call_agent.

WHY THIS EXISTS. `PrismClient.call_agent` has always accepted `min_p`, but
`AgentHarness.run` never passed it, so no BaseAgent-based caller could set it.
Prism's ParameterRegistry then filled its own `minP: agentDefault 0.05`, and a
vLLM box running speculative decoding REFUSES any `min_p > 0`:

    ValueError: The min_p and logit_bias sampling parameters are not yet
    supported with speculative decoding

vLLM raises that INSIDE the stream generator after already answering HTTP 200,
so prism sees an empty stream, not an error, and the caller gets a successful
call with zero content. Measured 2026-08-06 against the Jetson with one
variable changed and the same prompt: default -> 0 chars, min_p=0.0 -> 2,534.

The bug was a dropped argument, so the test asserts on the argument — a test
that only checked "agent produced text" passes with or without the fix,
because the empty stream is indistinguishable from a terse model.
"""
from unittest.mock import AsyncMock, patch

import pytest

from lazycat.agent import AgentHarness, BaseAgent
from lazycat.session import ConversationSession


def _harness(**agent_kwargs):
    agent = BaseAgent(name="t", system_prompt="sys", **agent_kwargs)
    return AgentHarness(agent=agent, session=ConversationSession(session_id="s"))


class _EmptyStream:
    """Minimal stand-in for LLMStreamWrapper: yields no SSE lines.

    This is also exactly the shape prism returns on the failure being fixed —
    HTTP 200, zero events — so the fixture doubles as the bug's signature.
    """

    async def aiter_lines(self):
        return
        yield  # pragma: no cover — makes this an async generator

    async def aclose(self):
        return None


async def _call_kwargs(harness) -> dict:
    with patch.object(
        harness.agent.llm_client, "call_agent", new=AsyncMock(return_value=_EmptyStream())
    ) as mock:
        await harness.run("hello")
    assert mock.await_count >= 1
    return mock.await_args_list[0].kwargs


@pytest.mark.asyncio
async def test_min_p_is_forwarded_when_set():
    kwargs = await _call_kwargs(_harness(min_p=0.0))
    assert "min_p" in kwargs, "harness dropped min_p — prism will inject 0.05"
    assert kwargs["min_p"] == 0.0


@pytest.mark.asyncio
async def test_min_p_defaults_to_none_so_gateway_default_still_applies():
    """Unset must stay None, NOT 0.0.

    Silently defaulting every agent to 0.0 would be a behaviour change for
    cloud providers that never had the spec-decoding problem. Opt in.
    """
    kwargs = await _call_kwargs(_harness())
    assert kwargs.get("min_p") is None


@pytest.mark.asyncio
async def test_non_zero_min_p_survives_the_hop():
    """The passthrough carries the caller's value, it does not clamp to 0."""
    kwargs = await _call_kwargs(_harness(min_p=0.05))
    assert kwargs["min_p"] == 0.05
