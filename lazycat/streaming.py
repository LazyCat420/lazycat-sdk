import json
import logging
from typing import AsyncGenerator
import httpx
from fastapi.responses import StreamingResponse

from lazycat.sse import format_sse, iter_sse_json

logger = logging.getLogger(__name__)

async def stream_prism_events(
    prism_url: str,
    payload: dict,
    headers: dict,
    timeout: float = 600.0,
) -> AsyncGenerator[str, None]:
    """
    Connect to Prism's SSE /agent endpoint, parse its raw JSON events, and
    yield formatted SSE strings for a frontend client to consume.
    """
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            async with client.stream(
                "POST",
                prism_url,
                json=payload,
                headers={**headers, "Accept": "text/event-stream"}
            ) as resp:
                if resp.status_code != 200:
                    error_body = ""
                    async for chunk in resp.aiter_text():
                        error_body += chunk
                    yield format_sse({"type": "error", "message": f"Prism error {resp.status_code}: {error_body[:500]}"})
                    return

                async for event in iter_sse_json(resp.aiter_text()):
                    event_type = event.get("type", "")

                    if event_type == "chunk":
                        yield format_sse({"type": "chunk", "content": event.get("content", "")})

                    elif event_type == "tool_execution":
                        status = event.get("status", "")
                        tool_info = event.get("tool", {})
                        tool_name = tool_info.get("name", "unknown")

                        if status == "calling":
                            yield format_sse({"type": "tool_call", "tool": tool_name})
                            yield format_sse({"type": "status", "message": f"executing {tool_name}..."})

                        elif status in ("done", "success"):
                            result = tool_info.get("result", {})
                            if isinstance(result, str):
                                try:
                                    result = json.loads(result)
                                except (json.JSONDecodeError, TypeError):
                                    pass

                            # Yield raw tool result back to the frontend in case it's needed
                            yield format_sse({"type": "tool_result", "tool": tool_name, "result": result})

                        elif status == "error":
                            error_msg = tool_info.get("result", "Unknown tool error")
                            yield format_sse({"type": "status", "message": f"tool error: {tool_name}: {str(error_msg)[:200]}"})

                    elif event_type == "thinking":
                        yield format_sse({"type": "status", "message": "reasoning..."})

                    elif event_type == "done":
                        yield 'data: {"type": "done"}\n\n'

                    elif event_type == "error":
                        yield format_sse({"type": "error", "message": event.get("message", "Agent error")})

        except Exception as e:
            logger.error(f"Prism SSE proxy error: {e}")
            yield format_sse({"type": "error", "message": f"Connection error: {str(e)}"})
            yield 'data: {"type": "done"}\n\n'

def create_streaming_response(
    prism_url: str,
    payload: dict,
    headers: dict,
    timeout: float = 600.0,
) -> StreamingResponse:
    """Helper to wrap the Prism stream generator in a FastAPI StreamingResponse."""
    return StreamingResponse(
        stream_prism_events(prism_url, payload, headers, timeout),
        media_type="text/event-stream"
    )
