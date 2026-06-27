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
        self._kill_switch_armed: bool = False

    @property
    def url(self) -> str:
        return self._url if self._url is not None else config.PRISM_URL

    @url.setter
    def url(self, value: str):
        self._url = value

    def arm_kill_switch(self):
        """Immediately aborts all active and future LLM requests."""
        self._kill_switch_armed = True
        logger.warning("[PrismClient] Kill switch ARMED")

    def reset_kill_switch(self):
        """Re-enables LLM requests."""
        self._kill_switch_armed = False
        logger.info("[PrismClient] Kill switch RESET")

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
        if self._kill_switch_armed:
            raise asyncio.CancelledError("lazycat-sdk kill switch is armed")
            
        client = await self._get_client()
        
        group_key = f"chat-{agent_name}"
        session_id, is_new = self._get_or_create_session(group_key)
        
        if group_key not in self._conversations:
            self._conversations[group_key] = str(uuid.uuid4())
        conversation_id = self._conversations[group_key]

        # Prepend system prompt and dummy user message to align system message rewrite in prism-service.
        # This prevents double system prompt errors on Qwen/vLLM.
        new_messages = list(messages)
        if system_prompt and not any(m.get("role") == "system" for m in new_messages):
            new_messages.insert(0, {"role": "system", "content": system_prompt})
            if len(new_messages) > 1 and new_messages[1].get("role") == "user":
                new_messages.insert(1, {"role": "user", "content": "Acknowledged. I am ready to process the quantitative data."})

        payload = {
            "provider": provider,
            "model": model,
            "messages": new_messages,
            "maxTokens": max_tokens,
            "temperature": temperature,
            "conversationId": conversation_id,
            "agentSessionId": session_id,
            "project": project,
            "username": username,
            "agent": agent_name,
            "systemPrompt": system_prompt[:15000],
            "functionCallingEnabled": False,
            "autoApprove": True,
        }

        if tools:
            enabled_tools = []
            for t in tools:
                if "function" in t:
                    enabled_tools.append(t["function"]["name"])
                elif "name" in t:
                    enabled_tools.append(t["name"])
            payload["enabledTools"] = enabled_tools

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
                "agentId": agent_id,
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

    def get_stream_payload_and_url(
        self,
        model: str,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
        system_prompt: str,
        agent_name: str,
        conversation_id: str,
        session_id: str,
        project: str,
        username: str,
        is_new: bool,
        enable_thinking: bool,
        tools: list[dict] | None = None,
        is_qwen_model: bool = False,
        agentic_mode: bool = True,
        provider: str = "vllm-1",
    ) -> tuple[dict, str, dict]:
        """Returns (payload, url, headers) formatted for Prism /agent streaming."""
        payload: dict[str, Any] = {
            "provider": provider,
            "model": model,
            "messages": messages,
            "maxTokens": max_tokens,
            "temperature": temperature,
            "conversationId": conversation_id,
            "project": project,
            "username": username,
            "agent": agent_name,
            "functionCallingEnabled": agentic_mode,
            "agenticLoopEnabled": agentic_mode,
            "systemPrompt": system_prompt[:15000],
        }
        if is_qwen_model:
            payload["thinkingEnabled"] = enable_thinking

        if tools and agentic_mode:
            payload["tools"] = tools

        if is_new:
            payload["createSession"] = True
        elif session_id:
            payload["sessionId"] = session_id

        # target url for stream
        target_url = f"{self.url}/agent"
        headers = {
            "Content-Type": "application/json",
            "x-project": project,
            "x-username": username,
        }

        return payload, target_url, headers

    async def agent_chat_stream(
        self,
        model: str,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
        system_prompt: str,
        agent_name: str,
        project: str = "",
        username: str = "agent_runner",
        enable_thinking: bool = False,
        tools: list[dict] | None = None,
        is_qwen_model: bool = False,
        agentic_mode: bool = True,
        agentContext: dict | None = None,
        provider: str = "vllm-1",
    ):
        """High-level wrapper to stream Prism /agent response."""
        if self._kill_switch_armed:
            raise asyncio.CancelledError("lazycat-sdk kill switch is armed")

        group_key = f"chat-{agent_name}" if agent_name == "user_chat" else agent_name
        session_id, is_new = self._get_or_create_session(group_key)
        conversation_id = str(uuid.uuid4())

        payload, url, headers = self.get_stream_payload_and_url(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            system_prompt=system_prompt,
            agent_name=agent_name,
            conversation_id=conversation_id,
            session_id=session_id,
            project=project or config.PROJECT_NAME,
            username=username,
            is_new=is_new,
            enable_thinking=enable_thinking,
            tools=tools,
            is_qwen_model=is_qwen_model,
            agentic_mode=agentic_mode,
            provider=provider,
        )
        if agentContext:
            payload["agentContext"] = agentContext

        client = await self._get_client()
        try:
            async with client.stream("POST", url, json=payload, headers=headers, timeout=180.0) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line:
                        yield line
        except Exception as e:
            logger.error("[PRISM] Error in agent_chat_stream: %s", e)
            raise

prism_client = PrismClient()
