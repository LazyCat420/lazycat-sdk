import asyncio
import logging
import uuid
from typing import Any

import httpx
from httpx import RequestError, HTTPStatusError

from lazycat.config import config

logger = logging.getLogger(__name__)

class PrismClient:
    """
    Standalone SDK client for routing LLM requests through Prism Gateway.
    Handles session tracking, timeouts, and payload enrichment for Prism's /agent endpoint.
    """

    def __init__(self):
        self._sessions: dict[str, str] = {}
        self._conversations: dict[str, str] = {}
        self._client: httpx.AsyncClient | None = None
        self._is_healthy = False
        self._last_health_check = 0.0
        self._url: str | None = None
        self._cycle_generation: int = 0
        self._custom_agent_locks: dict[str, asyncio.Lock] = {}

    @property
    def url(self) -> str:
        return self._url if self._url is not None else config.PRISM_URL

    @url.setter
    def url(self, value: str):
        self._url = value

    async def check_health(self) -> bool:
        """Dynamically check if Prism is available."""
        import time
        now = time.monotonic()

        # Return cached health if we checked within the last 5 seconds
        if now - self._last_health_check < 5.0:
            return self._is_healthy

        try:
            client = await self._get_client()
            r = await client.get(f"{self.url}/health", timeout=2.0)
            is_up = r.status_code == 200
        except Exception:
            is_up = False

        self._is_healthy = is_up
        self._last_health_check = now

        return is_up

    async def _get_client(self) -> httpx.AsyncClient:
        """Lazy-init a persistent async client for connection reuse."""
        current_loop = asyncio.get_running_loop()
        client_loop = getattr(self, "_client_loop", None)

        if self._client is not None and client_loop is not current_loop:
            await self._client.aclose()
            self._client = None

        if self._client is None or self._client.is_closed:
            limits = httpx.Limits(max_connections=100, max_keepalive_connections=20)
            self._client = httpx.AsyncClient(timeout=120.0, limits=limits)
            self._client_loop = current_loop
        return self._client

    def _get_or_create_session(self, group_key: str) -> tuple[str | None, bool]:
        if not group_key:
            return None, False
        if group_key in self._sessions:
            return self._sessions[group_key], False
        session_id = str(uuid.uuid4())
        self._sessions[group_key] = session_id
        return session_id, True

    def end_session(self, group_key: str):
        self._sessions.pop(group_key, None)
        self._conversations.pop(group_key, None)

    def cleanup_all_sessions(self):
        self._sessions.clear()
        self._conversations.clear()

    async def call_agent(
        self,
        model: str,
        messages: list[dict],
        system_prompt: str,
        agent_name: str = "default",
        tools: list[dict] | None = None,
        max_tokens: int = 8192,
        temperature: float = 0.0,
        provider: str = "vllm",
        project: str = "default-project",
        username: str = "lazycat-sdk",
        stream: bool = False,
    ) -> httpx.Response:
        """Execute a call to Prism's /agent endpoint."""
        client = await self._get_client()
        
        group_key = f"chat-{agent_name}"
        session_id, is_new = self._get_or_create_session(group_key)
        
        if group_key not in self._conversations:
            self._conversations[group_key] = str(uuid.uuid4())
        conversation_id = self._conversations[group_key]

        payload = {
            "provider": provider,
            "model": model,
            "messages": messages,
            "maxTokens": max_tokens,
            "temperature": temperature,
            "conversationId": conversation_id,
            "agentSessionId": session_id,
            "project": project,
            "username": username,
            "agent": agent_name,
            "systemPrompt": system_prompt[:15000],
            "functionCallingEnabled": bool(tools),
        }

        if tools:
            # Prism expects flat tools (name, description, parameters) 
            # instead of OpenAI's {"type": "function", "function": {...}} wrapper
            unwrapped_tools = []
            for t in tools:
                if "function" in t:
                    unwrapped_tools.append(t["function"])
                else:
                    unwrapped_tools.append(t)
            payload["tools"] = unwrapped_tools

        if is_new:
            payload["createSession"] = True
        elif session_id:
            payload["sessionId"] = session_id

        url = f"{self.url}/agent?stream={'true' if stream else 'false'}"
        headers = {
            "Content-Type": "application/json",
            "x-project": project,
            "x-username": username,
        }

        try:
            r = await client.post(url, json=payload, headers=headers, timeout=120.0)
            r.raise_for_status()
            return r
        except Exception as e:
            logger.error(f"Prism call failed: {e}")
            raise

prism_client = PrismClient()
