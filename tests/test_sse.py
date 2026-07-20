"""
Tests for lazycat.sse — the shared SSE line/event parser.

The parsing loop these helpers replace was copy-pasted across llm.py,
streaming.py, research.py and agent.py. The tests below pin the exact
semantics of those originals (strip, skip empties, tolerate garbage, drop a
trailing partial line) plus characterization tests asserting the frames
emitted by stream_prism_events are unchanged by the consolidation.
"""
import json

import pytest

from lazycat.sse import (
    format_sse,
    iter_sse_json,
    iter_sse_json_lines,
    iter_sse_lines,
)


async def _chunks(*items):
    for item in items:
        yield item


async def _collect(aiter):
    return [x async for x in aiter]


# ── iter_sse_lines ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_line_split_across_chunk_boundary():
    # The defining case: one logical line arrives split across chunks.
    src = _chunks('data: {"a"', ': 1}\n', 'data: {"b": 2}\n')
    assert await _collect(iter_sse_lines(src)) == [
        'data: {"a": 1}',
        'data: {"b": 2}',
    ]


@pytest.mark.asyncio
async def test_multiple_events_in_one_chunk():
    out = await _collect(iter_sse_lines(_chunks("a\nb\nc\n")))
    assert out == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_blank_lines_skipped_and_lines_stripped():
    out = await _collect(iter_sse_lines(_chunks("  spaced  \n\n\n   \n next \n")))
    assert out == ["spaced", "next"]


@pytest.mark.asyncio
async def test_trailing_partial_line_is_not_emitted():
    # An event cut off by end-of-stream is incomplete and must be dropped,
    # not handed downstream as if it were whole.
    out = await _collect(iter_sse_lines(_chunks("complete\n", "partial-no-newline")))
    assert out == ["complete"]


# ── iter_sse_json_lines / iter_sse_json ─────────────────────────────────


@pytest.mark.asyncio
async def test_json_lines_decode_and_skip_non_data():
    src = _chunks(
        ": keepalive comment\n",
        "event: ping\n",
        'data: {"type": "chunk", "content": "hi"}\n',
    )
    assert await _collect(iter_sse_json(src)) == [{"type": "chunk", "content": "hi"}]


@pytest.mark.asyncio
async def test_malformed_json_payload_is_skipped_not_raised():
    src = _chunks("data: {not json}\n", 'data: {"ok": true}\n')
    assert await _collect(iter_sse_json(src)) == [{"ok": True}]


@pytest.mark.asyncio
async def test_done_sentinel_stops_iteration():
    src = _chunks('data: {"n": 1}\n', "data: [DONE]\n", 'data: {"n": 2}\n')
    assert await _collect(iter_sse_json(src, done_sentinel="[DONE]")) == [{"n": 1}]


@pytest.mark.asyncio
async def test_without_sentinel_done_is_just_unparseable_and_skipped():
    src = _chunks('data: {"n": 1}\n', "data: [DONE]\n", 'data: {"n": 2}\n')
    assert await _collect(iter_sse_json(src)) == [{"n": 1}, {"n": 2}]


@pytest.mark.asyncio
async def test_json_lines_accepts_a_presplit_line_source():
    # agent.py feeds this from httpx/LLMStreamWrapper aiter_lines(), which
    # already yields whole lines rather than raw text chunks.
    async def lines():
        yield 'data: {"type": "chunk"}'
        yield "data: [DONE]"

    assert await _collect(
        iter_sse_json_lines(lines(), done_sentinel="[DONE]")
    ) == [{"type": "chunk"}]


# ── format_sse ──────────────────────────────────────────────────────────


def test_format_sse_frames():
    assert format_sse({"type": "done"}) == 'data: {"type": "done"}\n\n'
    assert format_sse("raw") == "data: raw\n\n"
    # Real newlines, not the literal backslash-n that agent.py's error path
    # used to emit.
    assert format_sse({"a": 1}).endswith("\n\n")
    assert "\\n" not in format_sse({"a": 1})


# ── characterization: stream_prism_events output is unchanged ───────────


@pytest.mark.asyncio
async def test_stream_prism_events_frames_unchanged():
    """Pins the exact frames the SSE proxy emits for a representative stream."""
    from unittest.mock import MagicMock, patch

    import lazycat.streaming as streaming

    upstream = [
        'data: {"type": "chunk", "content": "Hello"}',
        'data: {"type": "tool_execution", "status": "calling", "tool": {"name": "search"}}',
        'data: {"type": "tool_execution", "status": "done", "tool": {"name": "search", "result": "{\\"k\\": 1}"}}',
        'data: {"type": "thinking"}',
        'data: {"type": "done"}',
    ]

    resp = MagicMock()
    resp.status_code = 200

    async def aiter_text():
        # Deliberately split mid-line to exercise the buffering path.
        blob = "\n".join(upstream) + "\n"
        yield blob[:40]
        yield blob[40:]

    resp.aiter_text = aiter_text

    class _Ctx:
        async def __aenter__(self):
            return resp

        async def __aexit__(self, *a):
            return False

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        def stream(self, *a, **kw):
            return _Ctx()

    with patch.object(streaming.httpx, "AsyncClient", lambda **kw: _Client()):
        frames = [
            f
            async for f in streaming.stream_prism_events("http://x/agent", {}, {})
        ]

    assert frames == [
        'data: {"type": "chunk", "content": "Hello"}\n\n',
        'data: {"type": "tool_call", "tool": "search"}\n\n',
        'data: {"type": "status", "message": "executing search..."}\n\n',
        'data: {"type": "tool_result", "tool": "search", "result": {"k": 1}}\n\n',
        'data: {"type": "status", "message": "reasoning..."}\n\n',
        'data: {"type": "done"}\n\n',
    ]
    for f in frames:
        assert f.endswith("\n\n") and "\\n\\n" not in f
