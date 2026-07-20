"""
Tests for AgentHarness.stream_run's raw SSE pass-through.

Regression: the non-200 error branch built its frame with an escaped '\\n\\n'
inside a single-quoted f-string, so it emitted the two-character sequence
backslash-n twice instead of a real SSE terminator. Every consumer parsing on
blank lines saw one unterminated frame and silently stalled on the error path.
"""
import sys
from unittest.mock import MagicMock, patch

import pytest

from lazycat.agent import AgentHarness, BaseAgent
from lazycat.session import ConversationSession


def _harness():
    agent = BaseAgent(name="t", system_prompt="sys")
    return AgentHarness(agent=agent, session=ConversationSession(session_id="s"))


def _patched_client(resp):
    """Patch the httpx.AsyncClient that stream_run imports at call time."""

    class _Ctx:
        async def __aenter__(self):
            return resp

        async def __aexit__(self, *a):
            return False

    class _Client:
        def __init__(self, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        def stream(self, *a, **kw):
            return _Ctx()

    return patch.object(sys.modules["httpx"], "AsyncClient", _Client)


@pytest.mark.asyncio
async def test_error_frame_uses_real_newlines():
    resp = MagicMock()
    resp.status_code = 500

    async def aiter_text():
        yield "upstream exploded"

    resp.aiter_text = aiter_text

    with _patched_client(resp):
        frames = [f async for f in _harness().stream_run()]

    assert len(frames) == 1
    assert frames[0].endswith("\n\n")
    assert "\\n" not in frames[0]  # not the literal backslash-n sequence
    assert "upstream exploded" in frames[0]


@pytest.mark.asyncio
async def test_lines_are_passed_through_with_sse_terminators():
    resp = MagicMock()
    resp.status_code = 200

    async def aiter_text():
        # Split mid-line to exercise chunk reassembly.
        yield 'data: {"type": "chu'
        yield 'nk"}\ndata: {"type": "done"}\n'

    resp.aiter_text = aiter_text

    with _patched_client(resp):
        frames = [f async for f in _harness().stream_run()]

    assert frames == [
        'data: {"type": "chunk"}\n\n',
        'data: {"type": "done"}\n\n',
    ]
