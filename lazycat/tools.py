import httpx
import logging
import asyncio
from typing import Any

from lazycat.config import config

logger = logging.getLogger(__name__)

#: Every MCP namespace a tool call may arrive under, longest-first.
#:
#: The service behind these tools was renamed `lazy-tool-service` ->
#: `lazy-agent-service` (2026-08-07). The prefix is minted by PRISM from its MCP
#: server registration name, so which spelling arrives depends on which
#: registration the caller's scope is connected to — and during the migration
#: both are live. A missed strip is silent: the name is forwarded verbatim to
#: `/execute/<name>`, which answers "Unknown tool" and reads to the model as a
#: missing capability rather than a routing bug.
_MCP_PREFIXES = (
    "mcp__lazy-agent-service__",
    "mcp__lazy-tool-service__",
)


def strip_mcp_prefix(tool_name: str) -> str:
    """Namespaced tool name -> bare name. Bare names pass through unchanged."""
    name = tool_name or ""
    for prefix in _MCP_PREFIXES:
        if name.startswith(prefix):
            return name[len(prefix):]
    return name

class ToolExecutor:
    """Standardized HTTP client for executing tools via lazy-tool-service."""

    def __init__(self):
        self._client: httpx.AsyncClient | None = None
    
    @property
    def url(self) -> str:
        import os
        env_url = os.getenv("LAZY_TOOL_SERVICE_URL")
        if env_url:
            return env_url
        port = config.LAZY_TOOL_SERVICE_PORT
        host = os.getenv("LAZY_TOOL_SERVICE_HOST", "127.0.0.1")
        return f"http://{host}:{port}"

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=60.0)
        return self._client

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Execute a tool by proxying to the lazy-tool-service.

        headers: identity headers for the proxy (x-conversation-id, x-agent,
        x-ticker). Without x-conversation-id the proxy's session whitelist
        cannot match the session and FAILS OPEN.
        """

        clean_name = strip_mcp_prefix(tool_name)

        client = await self._get_client()
        target_url = f"{self.url}/execute/{clean_name}"

        payload = {"arguments": arguments}
        request_headers = {k: v for k, v in (headers or {}).items() if v}

        max_retries = 3
        backoff = 1.0

        for attempt in range(max_retries):
            try:
                r = await client.post(target_url, json=payload, headers=request_headers or None)
                if r.status_code == 200:
                    return r.json()
                else:
                    return {"error": f"lazy-tool-service returned status code {r.status_code}: {r.text}"}
            except httpx.RequestError as e:
                if attempt == max_retries - 1:
                    return {"error": f"Tool service request failed: {str(e)}"}
                await asyncio.sleep(backoff)
                backoff *= 2
            except Exception as e:
                return {"error": f"Unexpected tool execution error: {str(e)}"}
                
        return {"error": "Max retries exceeded"}

tool_executor = ToolExecutor()
