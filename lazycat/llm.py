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
            self._client = httpx.AsyncClient(timeout=600.0, limits=limits)
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
            if stream:
                req = client.build_request("POST", url, json=payload, headers=headers)
                r = await client.send(req, stream=True)
                r.raise_for_status()
                return r
            else:
                r = await client.post(url, json=payload, headers=headers, timeout=120.0)
                r.raise_for_status()
                return r
        except Exception as e:
            logger.error(f"Prism call failed: {e}")
            raise


    def _get_agent_lock(self, agent_id: str) -> asyncio.Lock:
        if agent_id not in self._custom_agent_locks:
            self._custom_agent_locks[agent_id] = asyncio.Lock()
        return self._custom_agent_locks[agent_id]

    async def register_or_update_custom_agent(
        self,
        name: str,
        identity: str,
        guidelines: str = "",
        enabled_tools: list[str] | None = None,
        project: str = "vllm-trading-bot",
    ) -> str:
        """Register a custom agent in Prism, or update it if it already exists.
        Returns the custom agent ID (e.g. 'CUSTOM_BEAR_MACRO_SENTIMENT_T2_AGENT').
        """
        slug = name.upper().replace(" ", "_").replace("-", "_").strip("_")
        while "__" in slug:
            slug = slug.replace("__", "_")
        agent_id = f"CUSTOM_{slug}" if not slug.startswith("CUSTOM_") else slug

        if not hasattr(self, "_registered_custom_agents"):
            self._registered_custom_agents = set()

        if agent_id in self._registered_custom_agents:
            return agent_id
            
        lock = self._get_agent_lock(agent_id)
        async with lock:
            if agent_id in self._registered_custom_agents:
                return agent_id

            client = await self._get_client()
            headers = {
                "Content-Type": "application/json",
                "x-project": project,
                "x-username": "lazycat-sdk",
            }

            agent_db_id = None
            try:
                r = await client.get(f"{self.url}/custom-agents", headers=headers, timeout=10.0)
                r.raise_for_status()
                existing_agents = r.json()
                for agent in existing_agents:
                    if agent.get("agentId") == agent_id:
                        agent_db_id = agent.get("_id")
                        break
            except Exception as e:
                logger.warning(f"[PRISM] Failed to query existing custom agents: {e}")

            display_name = name.replace("_", " ").title() if "_" in name else name

            _blocked_prism_tools = {"ask_user_question"}
            _filtered_tools = [
                t for t in (enabled_tools or [])
                if t not in _blocked_prism_tools
            ]

            payload = {
                "name": display_name,
                "identity": identity,
                "guidelines": guidelines,
                "enabledTools": _filtered_tools,
                "availableTools": _filtered_tools,
                "project": project,
                "usesDirectoryTree": False,
                "usesCodingGuidelines": False,
            }

            if agent_db_id:
                try:
                    logger.info(f"[PRISM] Updating existing custom agent {agent_id} (db_id: {agent_db_id})")
                    r = await client.put(f"{self.url}/custom-agents/{agent_db_id}", json=payload, headers=headers, timeout=10.0)
                    r.raise_for_status()
                except Exception as e:
                    logger.error(f"[PRISM] Failed to update custom agent {agent_id}: {e}")
                    raise
            else:
                try:
                    logger.info(f"[PRISM] Creating new custom agent {agent_id}")
                    r = await client.post(f"{self.url}/custom-agents", json=payload, headers=headers, timeout=10.0)
                    r.raise_for_status()
                except Exception as e:
                    logger.error(f"[PRISM] Failed to create custom agent {agent_id}: {e}")
                    raise

            self._registered_custom_agents.add(agent_id)
            return agent_id

prism_client = PrismClient()
