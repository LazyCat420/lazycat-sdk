import json
import logging
from typing import AsyncGenerator
import httpx
from fastapi.responses import StreamingResponse

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
                    yield f'data: {json.dumps({"type": "error", "message": f"Prism error {resp.status_code}: {error_body[:500]}"})}\n\n'
                    return

                buffer = ""
                async for chunk in resp.aiter_text():
                    buffer += chunk
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        line = line.strip()

                        if not line.startswith("data: "):
                            continue

                        try:
                            event = json.loads(line[6:])
                        except json.JSONDecodeError:
                            continue

                        event_type = event.get("type", "")

                        if event_type == "chunk":
                            yield f'data: {json.dumps({"type": "chunk", "content": event.get("content", "")})}\n\n'

                        elif event_type == "tool_execution":
                            status = event.get("status", "")
                            tool_info = event.get("tool", {})
                            tool_name = tool_info.get("name", "unknown")

                            if status == "calling":
                                yield f'data: {json.dumps({"type": "tool_call", "tool": tool_name})}\n\n'
                                yield f'data: {json.dumps({"type": "status", "message": f"executing {tool_name}..."})}\n\n'

                            elif status in ("done", "success"):
                                result = tool_info.get("result", {})
                                if isinstance(result, str):
                                    try:
                                        result = json.loads(result)
                                    except (json.JSONDecodeError, TypeError):
                                        pass
                                
                                # Yield raw tool result back to the frontend in case it's needed
                                yield f'data: {json.dumps({"type": "tool_result", "tool": tool_name, "result": result})}\n\n'

                            elif status == "error":
                                error_msg = tool_info.get("result", "Unknown tool error")
                                yield f'data: {json.dumps({"type": "status", "message": f"tool error: {tool_name}: {str(error_msg)[:200]}"})}\n\n'

                        elif event_type == "thinking":
                            yield f'data: {json.dumps({"type": "status", "message": "reasoning..."})}\n\n'

                        elif event_type == "done":
                            yield 'data: {"type": "done"}\n\n'

                        elif event_type == "error":
                            yield f'data: {json.dumps({"type": "error", "message": event.get("message", "Agent error")})}\n\n'

        except Exception as e:
            logger.error(f"Prism SSE proxy error: {e}")
            yield f'data: {json.dumps({"type": "error", "message": f"Connection error: {str(e)}"})}\n\n'
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
