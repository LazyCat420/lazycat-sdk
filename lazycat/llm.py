import asyncio
import logging
import uuid
from typing import Any

import httpx
from httpx import RequestError, HTTPStatusError

from lazycat.config import config
import json

logger = logging.getLogger(__name__)

class LLMStreamWrapper:
    def __init__(self, response: httpx.Response, is_openai: bool = False):
        self.response = response
        self.is_openai = is_openai
        # OpenAI streams tool calls as PARTIAL deltas (first delta carries
        # id+name, later deltas carry argument fragments keyed by index).
        # Accumulate here and emit ONE complete toolCalls event at the end —
        # emitting per-delta made the consumer replace its list each time,
        # losing names and argument fragments (multi-tool calls broke too).
        self._pending_tool_calls: dict[int, dict] = {}

    def _accumulate_tool_deltas(self, deltas: list[dict]) -> None:
        for tc in deltas:
            idx = tc.get("index", 0)
            entry = self._pending_tool_calls.setdefault(
                idx, {"id": "", "function": {"name": "", "arguments": ""}}
            )
            if tc.get("id"):
                entry["id"] = tc["id"]
            fn = tc.get("function", {}) or {}
            if fn.get("name"):
                entry["function"]["name"] += fn["name"]
            if fn.get("arguments"):
                entry["function"]["arguments"] += fn["arguments"]

    def _flush_tool_calls_event(self) -> str | None:
        if not self._pending_tool_calls:
            return None
        tool_calls = []
        for idx in sorted(self._pending_tool_calls):
            entry = self._pending_tool_calls[idx]
            if not entry["function"]["arguments"]:
                entry["function"]["arguments"] = "{}"
            tool_calls.append(entry)
        self._pending_tool_calls = {}
        return f"data: {json.dumps({'toolCalls': tool_calls})}"

    async def aiter_lines(self):
        if not self.is_openai:
            async for line in self.response.aiter_lines():
                yield line
            return

        buffer = ""
        async for chunk in self.response.aiter_text():
            buffer += chunk
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.strip()
                if not line:
                    continue
                if line == "data: [DONE]":
                    flush = self._flush_tool_calls_event()
                    if flush:
                        yield flush
                    yield "data: [DONE]"
                    break
                if not line.startswith("data: "):
                    continue

                try:
                    data = json.loads(line[6:])
                    choices = data.get("choices", [])
                    if not choices:
                        continue
                    choice = choices[0]
                    delta = choice.get("delta", {})

                    if "content" in delta and delta["content"]:
                        prism_data = {"type": "chunk", "content": delta["content"]}
                        yield f"data: {json.dumps(prism_data)}"

                    if "tool_calls" in delta and delta["tool_calls"]:
                        self._accumulate_tool_deltas(delta["tool_calls"])

                    # vLLM/OpenAI signal completion of the tool-call block via
                    # finish_reason — flush the assembled calls as one event.
                    if choice.get("finish_reason"):
                        flush = self._flush_tool_calls_event()
                        if flush:
                            yield flush
                except Exception as e:
                    logger.debug("Failed decoding openai stream line: %s", e)
                    continue
                    
    async def aclose(self):
        await self.response.aclose()


class LLMResponseWrapper:
    def __init__(self, text: str, tool_calls: list | None = None):
        self._text = text
        self._tool_calls = tool_calls or []
        
    def json(self):
        return {
            "text": self._text,
            "toolCalls": self._tool_calls
        }
        
    @property
    def text(self):
        return self._text
        
    async def aclose(self):
        pass


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
        self._model_to_provider_cache: dict[str, str] = {}
        self._last_config_fetch: float = 0.0

    @property
    def url(self) -> str:
        return self._url if self._url is not None else config.PRISM_URL

    @url.setter
    def url(self, value: str):
        self._url = value

    def arm_kill_switch(self):
        """Immediately aborts all active and future LLM requests."""
        self._kill_switch_armed = True
        if self._client is not None and not self._client.is_closed:
            try:
                import asyncio
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(self._client.aclose())
                except RuntimeError:
                    # No running loop, let garbage collection handle it
                    pass
            except Exception as e:
                logger.warning(f"[PrismClient] Failed to schedule client close: {e}")
            self._client = None
        logger.warning("[PrismClient] Kill switch ARMED")

    def reset_kill_switch(self):
        """Re-enables LLM requests."""
        self._kill_switch_armed = False
        self._model_to_provider_cache.clear()
        self._last_config_fetch = 0.0
        logger.info("[PrismClient] Kill switch RESET")

    def _prepare_messages(self, messages: list[dict], system_prompt: str) -> list[dict]:
        """Prepends system prompt and dummy user message if system prompt is not already present."""
        new_messages = list(messages)
        if system_prompt and not any(m.get("role") == "system" for m in new_messages):
            new_messages.insert(0, {"role": "system", "content": system_prompt})
            if len(new_messages) > 1 and new_messages[1].get("role") == "user":
                new_messages.insert(1, {"role": "user", "content": "Acknowledged. I am ready to process the quantitative data."})
        return new_messages

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

    async def _resolve_provider_instance(self, model: str, base_provider: str = "vllm") -> str:
        """
        Query /config?includeLocal=true from Prism to discover which specific provider
        instance (e.g., 'vllm-2') holds the requested model.
        Falls back to base_provider if not found.
        """
        if not model:
            return base_provider

        # Check cache first
        if model in self._model_to_provider_cache:
            return self._model_to_provider_cache[model]

        # Fetch config
        try:
            import time
            now = time.time()
            # Fetch config at most once every 30 seconds
            if now - self._last_config_fetch > 30:
                client = await self._get_client()
                url = self.url.rstrip("/")
                config_url = f"{url}/config?includeLocal=true"
                
                logger.info(f"[PRISM] Fetching config to resolve model '{model}' from: {config_url}")
                response = await client.get(config_url, timeout=5.0)
                if response.status_code == 200:
                    data = response.json()
                    text_to_text = data.get("textToText", {})
                    models_map = text_to_text.get("models", {})
                    for inst_id, model_list in models_map.items():
                        for m_info in model_list:
                            m_name = m_info.get("name")
                            if m_name:
                                self._model_to_provider_cache[m_name] = inst_id
                    self._last_config_fetch = now
        except Exception as e:
            logger.warning(f"[PRISM] Failed to auto-resolve provider for model '{model}': {e}")

        resolved = self._model_to_provider_cache.get(model, base_provider)
        logger.info(f"[PRISM] Resolved model '{model}' to provider instance '{resolved}' (base: '{base_provider}')")
        return resolved

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

    def get_conversation_id(self, agent_name: str, session_id: str | None) -> str | None:
        """Return the conversationId call_agent registered for this agent/session.

        Mirrors call_agent's session-path group_key. Used by AgentHarness to
        stamp x-conversation-id on /execute so the proxy whitelist (keyed on
        the conversationId registered from the /agent body) can actually match.
        """
        suffix = f"-{session_id[-8:]}" if session_id else ""
        return self._conversations.get(f"chat-{agent_name}{suffix}")

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
        max_iterations: int | None = None,
        session_id: str | None = None,
        auto_approve: bool = True,
        thinking_enabled: bool | None = None,
    ) -> Any:
        """Execute a call to Prism's /agent endpoint, or directly to vLLM if Prism is disabled.

        thinking_enabled: None leaves the gateway default (thinking ON for
        local models on agent sessions); pass False on interactive paths where
        a user is watching the stream — the <think> block is most of the
        latency on Qwen-class models.
        """
        
        if self._kill_switch_armed:
            raise asyncio.CancelledError("lazycat-sdk kill switch is armed")
            
        if not config.PRISM_ENABLED:
            return await self._call_vllm_direct(
                model=model,
                messages=messages,
                system_prompt=system_prompt,
                tools=tools,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=stream
            )

        client = await self._get_client()
        
        session_suffix = ""
        if session_id:
            session_suffix = f"-{session_id[-8:]}"
        elif messages:
            # Find the first user message that is not a dummy acknowledgement to create a sticky session key unique to this conversation thread
            first_user_msg = next(
                (m for m in messages if m.get("role") == "user" and m.get("content") != "Acknowledged. I am ready to process the quantitative data."),
                None
            )
            if not first_user_msg:
                first_user_msg = next((m for m in messages if m.get("role") == "user"), None)
            if first_user_msg and isinstance(first_user_msg.get("content"), str):
                import hashlib
                content_hash = hashlib.md5(first_user_msg["content"].encode("utf-8")).hexdigest()[:8]
                session_suffix = f"-{content_hash}"
                
        group_key = f"chat-{agent_name}{session_suffix}"
        session_id, is_new = self._get_or_create_session(group_key)
        
        if group_key not in self._conversations:
            self._conversations[group_key] = str(uuid.uuid4())
        conversation_id = self._conversations[group_key]
 
        # Prepend system prompt and dummy user message to align system message rewrite in prism-service.
        # This prevents double system prompt errors on Qwen/vLLM.
        new_messages = self._prepare_messages(messages, system_prompt)
 
        # Resolve provider instance for local models to bypass load-balancer single-instance bug in Prism
        resolved_provider = await self._resolve_provider_instance(model, provider)
 
        payload = {
            "provider": resolved_provider,
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
            "autoApprove": auto_approve,
        }
        if max_iterations is not None:
            payload["maxIterations"] = max_iterations
        if thinking_enabled is not None:
            payload["thinkingEnabled"] = thinking_enabled

        if tools:
            enabled_tools = []
            for t in tools:
                if "function" in t:
                    enabled_tools.append(t["function"]["name"])
                elif "name" in t:
                    enabled_tools.append(t["name"])
            payload["enabledTools"] = enabled_tools

        # Always force a new session on Prism proxy to prevent it from accumulating
        # duplicate history (since lazycat-sdk manages the conversation history locally).
        payload["createSession"] = True

        url = f"{self.url}/agent?stream={'true' if stream else 'false'}"
        headers = {
            "Content-Type": "application/json",
            "x-project": project,
            "x-username": username,
        }

        logger.info(f"[INSTRUMENTATION] prism.call_agent attempting to connect to: {url}")
        try:
            if stream:
                req = client.build_request("POST", url, json=payload, headers=headers)
                r = await client.send(req, stream=True)
                r.raise_for_status()
                return LLMStreamWrapper(r, is_openai=False)
            else:
                r = await client.post(url, json=payload, headers=headers, timeout=600.0)
                r.raise_for_status()
                # NOTE: non-stream returns the RAW httpx.Response — callers
                # must .json() it themselves. AgentHarness always uses
                # stream=True; if you adopt this path, parse before reading
                # token usage or content.
                return r
        except Exception as e:
            logger.error(f"[INSTRUMENTATION] Prism call failed connecting to {url}. Error: {e.__class__.__name__} - {e}")
            raise

    async def _call_vllm_direct(
        self,
        model: str,
        messages: list[dict],
        system_prompt: str,
        tools: list[dict] | None = None,
        max_tokens: int = 8192,
        temperature: float = 0.0,
        stream: bool = False,
    ) -> Any:
        client = await self._get_client()
        
        is_qwen = "qwen" in model.lower()
        vllm_base = config.JETSON_VLLM_URL if is_qwen else config.DGX_SPARK_VLLM_URL
        url = f"{vllm_base}/v1/chat/completions"
        
        openai_messages = []
        if system_prompt:
            openai_messages.append({"role": "system", "content": system_prompt})
            
        for m in messages:
            if m.get("role") != "system":
                openai_messages.append({
                    "role": m.get("role"),
                    "content": m.get("content") or ""
                })
                
        payload = {
            "model": model,
            "messages": openai_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }
        
        if tools:
            openai_tools = []
            for t in tools:
                if "function" in t:
                    openai_tools.append(t)
                elif "name" in t:
                    openai_tools.append({
                        "type": "function",
                        "function": {
                            "name": t.get("name"),
                            "description": t.get("description", ""),
                            "parameters": t.get("parameters", {"type": "object", "properties": {}, "required": []})
                        }
                    })
            if openai_tools:
                payload["tools"] = openai_tools
                payload["tool_choice"] = "auto"
                
        headers = {"Content-Type": "application/json"}
        logger.info(f"[SDK direct-vLLM] completions endpoint: {url} (stream={stream})")
        
        try:
            if stream:
                req = client.build_request("POST", url, json=payload, headers=headers)
                r = await client.send(req, stream=True)
                r.raise_for_status()
                return LLMStreamWrapper(r, is_openai=True)
            else:
                r = await client.post(url, json=payload, headers=headers, timeout=600.0)
                r.raise_for_status()
                res_data = r.json()
                choice = res_data["choices"][0]
                msg = choice.get("message", {})
                text = msg.get("content") or ""
                
                tool_calls = []
                if "tool_calls" in msg and msg["tool_calls"]:
                    for tc in msg["tool_calls"]:
                        tool_calls.append({
                            "id": tc.get("id", ""),
                            "function": {
                                "name": tc.get("function", {}).get("name", ""),
                                "arguments": tc.get("function", {}).get("arguments", "{}")
                            }
                        })
                return LLMResponseWrapper(text, tool_calls)
        except Exception as e:
            logger.error(f"[SDK direct-vLLM] completions failed: {e}")
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
        core_tools_locked: bool = False,
        thinking_default: bool | None = None,
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
                # Every agent registered through this helper carries an
                # explicit tool list, so don't let the gateway force-document
                # its ~30 CORE_AGENTIC_TOOLS + orchestrator tools next to it
                # (honored by lazy-tool-service since 2026-07-14; real prism
                # ignores unknown fields). Callers that want the forced core
                # set pass core_tools_locked=True.
                "coreToolsLocked": core_tools_locked,
            }
            if thinking_default is not None:
                # Per-agent <think> default; an explicit thinkingEnabled on a
                # request always wins over this.
                payload["thinkingDefault"] = thinking_default

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
        provider: str = "vllm",
    ) -> tuple[dict, str, dict]:
        """Returns (payload, url, headers) formatted for Prism /agent streaming."""
        # Prepend system prompt and dummy user message to align system message rewrite in prism-service.
        # This prevents double system prompt errors on Qwen/vLLM.
        new_messages = self._prepare_messages(messages, system_prompt)

        payload: dict[str, Any] = {
            "provider": provider,
            "model": model,
            "messages": new_messages,
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
            enabled_tools = []
            for t in tools:
                if isinstance(t, dict):
                    if "function" in t:
                        enabled_tools.append(t["function"]["name"])
                    elif "name" in t:
                        enabled_tools.append(t["name"])
                elif isinstance(t, str):
                    enabled_tools.append(t)
            payload["enabledTools"] = enabled_tools

        # Always force a new session on Prism proxy to prevent it from accumulating
        # duplicate history (since lazycat-sdk manages the conversation history locally).
        payload["createSession"] = True

        # target url for stream
        target_url = f"{self.url}/agent?stream=true"
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
        provider: str = "vllm",
    ):
        """High-level wrapper to stream Prism /agent response."""
        
        if self._kill_switch_armed:
            raise asyncio.CancelledError("lazycat-sdk kill switch is armed")

        group_key = f"chat-{agent_name}" if agent_name == "user_chat" else agent_name
        session_id, is_new = self._get_or_create_session(group_key)
        conversation_id = str(uuid.uuid4())

        # Resolve provider instance for local models to bypass load-balancer single-instance bug in Prism
        resolved_provider = await self._resolve_provider_instance(model, provider)

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
            provider=resolved_provider,
        )
        if agentContext:
            payload["agentContext"] = agentContext

        client = await self._get_client()
        try:
            async with client.stream("POST", url, json=payload, headers=headers, timeout=600.0) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line:
                        yield line
        except Exception as e:
            logger.error("[PRISM] Error in agent_chat_stream: %s", e)
            raise

prism_client = PrismClient()
