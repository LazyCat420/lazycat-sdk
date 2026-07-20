"""
Server-Sent Events parsing helpers.

Consuming an SSE stream over HTTP means reassembling lines from arbitrarily
chunked text: a single read can split an event mid-line, or deliver several
events at once. That buffer-and-split loop was copy-pasted across this package;
it lives here now.

Usage:
    from lazycat.sse import iter_sse_lines, iter_sse_json, format_sse

    async for line in iter_sse_lines(response.aiter_text()):
        ...

    async for event in iter_sse_json(response.aiter_text()):
        if event.get("type") == "chunk":
            ...

Semantics deliberately preserved from the hand-rolled loops these replaced:
lines are stripped, empty lines are skipped, unparseable payloads are skipped
rather than raised, and a trailing partial line left when the stream ends is
NOT emitted (an incomplete event is not a usable event).
"""

import json
import logging
from typing import Any, AsyncIterator

logger = logging.getLogger(__name__)

__all__ = [
    "iter_sse_lines",
    "iter_sse_json_lines",
    "iter_sse_json",
    "format_sse",
    "SSE_DATA_PREFIX",
]

SSE_DATA_PREFIX = "data: "


async def iter_sse_lines(text_chunks: AsyncIterator[str]) -> AsyncIterator[str]:
    """Reassemble raw text chunks into stripped, non-empty SSE lines.

    Args:
        text_chunks: Async iterator of text as it arrives (e.g.
            httpx.Response.aiter_text()). Chunk boundaries may fall anywhere,
            including mid-line.

    Yields:
        One stripped, non-empty line per event line received. A trailing
        partial line at end-of-stream is discarded.
    """
    buffer = ""
    async for chunk in text_chunks:
        buffer += chunk
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            line = line.strip()
            if not line:
                continue
            yield line


async def iter_sse_json_lines(
    lines: AsyncIterator[str],
    *,
    done_sentinel: str | None = None,
) -> AsyncIterator[dict]:
    """Decode `data:` lines into JSON objects.

    Args:
        lines: Async iterator of SSE lines (from iter_sse_lines, or any source
            that already yields whole lines such as httpx's aiter_lines()).
        done_sentinel: If given, a data payload equal to this string ends
            iteration (e.g. "[DONE]" for OpenAI-compatible streams).

    Yields:
        Decoded event objects. Lines without a `data:` prefix and payloads that
        fail to decode are skipped, matching the tolerant behavior these
        streams require — a malformed keepalive must not kill the response.
    """
    async for line in lines:
        line = line.strip()
        if not line.startswith(SSE_DATA_PREFIX):
            continue
        payload = line[len(SSE_DATA_PREFIX) :].strip()
        if done_sentinel is not None and payload == done_sentinel:
            return
        try:
            yield json.loads(payload)
        except json.JSONDecodeError:
            continue


async def iter_sse_json(
    text_chunks: AsyncIterator[str],
    *,
    done_sentinel: str | None = None,
) -> AsyncIterator[dict]:
    """Reassemble raw text chunks straight into decoded SSE event objects."""
    async for event in iter_sse_json_lines(
        iter_sse_lines(text_chunks), done_sentinel=done_sentinel
    ):
        yield event


def format_sse(obj: Any) -> str:
    """Render a payload as a complete SSE frame, terminator included.

    Strings are emitted verbatim; anything else is JSON-encoded.
    """
    payload = obj if isinstance(obj, str) else json.dumps(obj)
    return f"{SSE_DATA_PREFIX}{payload}\n\n"
