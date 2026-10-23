import asyncio
import contextvars
import logging
import os
import uuid
from typing import Any

import httpx
from httpx import RequestError, HTTPStatusError

from lazycat.config import config
from lazycat.sse import iter_sse_lines
import json

logger = logging.getLogger(__name__)

#: The only limit on the `systemPrompt` field: lazy-agent-service accepts any
#: length (`systemPrompt: z.string()`). It has been here since the SDK's first
#: commit and cut SILENTLY until 2026-10-22, when the trading Board's
#: 19,598-char system prompt was found losing its last 4,598 characters (every
#: output example) on every call while 12 of 13 first answers failed the
#: schema (trading-client ch.47, plan item 1.1). It still cuts, but says so.
#: Override per process with LAZYCAT_SYSTEM_PROMPT_MAX_CHARS; <= 0 disables it.
SYSTEM_PROMPT_MAX_CHARS = int(os.environ.get("LAZYCAT_SYSTEM_PROMPT_MAX_CHARS", "15000"))

#: Agent names already warned about: a long prompt warns once per process,
#: not once per call.
_WARNED_SYSTEM_PROMPT_CUTS: set[str] = set()


def _cap_system_prompt(system_prompt: str | None, agent_name: str | None) -> str:
    """The payload's `systemPrompt`, cut to SYSTEM_PROMPT_MAX_CHARS, never silently.

    A cut logs a WARNING once per agent name per process, and adds an
    OpenTelemetry event to the caller's current span on every call when the
    caller runs OpenTelemetry (the SDK itself does not depend on it).
    """
    text = system_prompt or ""
    cap = SYSTEM_PROMPT_MAX_CHARS
    if cap <= 0 or len(text) <= cap:
        return text
    name = agent_name or "<unnamed>"
    dropped = len(text) - cap
    if name not in _WARNED_SYSTEM_PROMPT_CUTS:
        _WARNED_SYSTEM_PROMPT_CUTS.add(name)
        logger.warning(
            "[AgentClient] systemPrompt for agent %r is %d chars; the %d-char cap "
            "(LAZYCAT_SYSTEM_PROMPT_MAX_CHARS) drops its last %d chars, which the "
            "model never sees. Shorten it or move rules into the user message. "
            "(Logged once per agent per process; every cut is a span event.)",
            name, len(text), cap, dropped,
        )
    try:
        from opentelemetry import trace as _otel_trace

        span = _otel_trace.get_current_span()
        if span.is_recording():
            span.add_event("lazycat.system_prompt_truncated", {
                "lazycat.agent": name,
                "lazycat.system_prompt.chars": len(text),
                "lazycat.system_prompt.cap": cap,
                "lazycat.system_prompt.dropped_chars": dropped,
            })
    except Exception:  # noqa: BLE001 — tracing is best-effort; never block the call
        pass
    return text[:cap]

bench_harness_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("bench_harness", default=None)
bench_run_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("bench_run_id", default=None)
bench_task_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("bench_task", default=None)

def set_bench_context(run_id: str | None, harness: str = "trading-cycle", task: str | None = None) -> None:
    """Set benchmark context for DGX Spark monitoring (run_id, harness, task)."""
    bench_run_id_var.set(run_id)
    bench_harness_var.set(harness)
    bench_task_var.set(task)

def get_bench_context() -> dict[str, str | None]:
    """Retrieve the current benchmark context values."""
    return {
        "run_id": bench_run_id_var.get(),
        "harness": bench_harness_var.get(),
        "task": bench_task_var.get(),
    }

def clear_bench_context() -> None:
    """Clear active benchmark context."""
    bench_run_id_var.set(None)
    bench_harness_var.set(None)
    bench_task_var.set(None)

#: Field names a reasoning model may use INSTEAD of `content`. vLLM's OpenAI
#: layer sends `reasoning`; the OpenAI spelling is `reasoning_content`.
_REASONING_FIELDS = ("reasoning", "reasoning_content")


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

        async for line in iter_sse_lines(self.response.aiter_text()):
            if line == "data: [DONE]":
                flush = self._flush_tool_calls_event()
                if flush:
                    yield flush
                yield "data: [DONE]"
                # [DONE] terminates an OpenAI-compatible stream; stop reading
                # instead of draining the rest of the connection.
                return
            if not line.startswith("data: "):
                continue

            try:
                data = json.loads(line[6:])
                choices = data.get("choices", [])
                if not choices:
                    continue
                choice = choices[0]
                delta = choice.get("delta", {})

                # A reasoning model streams its generation in `reasoning`
                # deltas. Forwarding only `content` meant the consumer received
                # NOTHING while the engine was happily generating, and its
                # watchdog fired `Provider stream stalled: no data received for
                # 300s` — 44 times in seven days, retried up to 5x, which is
                # ~25 minutes burned per agent. Same root cause as the empty
                # non-streaming replies; two very different-looking symptoms.
                # `content` wins so a normal model emits exactly once.
                chunk = delta.get("content")
                if not (isinstance(chunk, str) and chunk):
                    chunk = next((delta[f] for f in _REASONING_FIELDS
                                  if isinstance(delta.get(f), str) and delta[f]), None)
                if chunk:
                    prism_data = {"type": "chunk", "content": chunk}
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


class AgentClient:
    """
    Standalone SDK client for routing LLM requests through lazy-agent-service.
    Handles session tracking, timeouts, and payload enrichment for the /agent endpoint.
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
        return self._url if self._url is not None else config.LAZY_AGENT_URL

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
                logger.warning(f"[AgentClient] Failed to schedule client close: {e}")
            self._client = None
        logger.warning("[AgentClient] Kill switch ARMED")

    def reset_kill_switch(self):
        """Re-enables LLM requests."""
        self._kill_switch_armed = False
        self._model_to_provider_cache.clear()
        self._last_config_fetch = 0.0
        logger.info("[AgentClient] Kill switch RESET")

    def _prepare_messages(
        self,
        messages: list[dict],
        system_prompt: str,
        inline_system_prompt: bool = True,
    ) -> list[dict]:
        """Prepends system prompt and dummy user message if system prompt is not already present.

        inline_system_prompt=False skips the inlined copy entirely. Every
        /agent payload we build ALSO carries the prompt as `systemPrompt`,
        and prism's context clamp counts the messages array and
        `systemPrompt` as two separate budget categories — so inlining bills
        the same text twice. Measured 2026-08-03 against an 8K-window model:
        3,082 duplicated tokens on a single chat request. Opt out on any
        caller tight enough on context to care; prism reads the payload
        field either way.
        """
        new_messages = list(messages)
        if not inline_system_prompt:
            return new_messages
        if system_prompt and not any(m.get("role") == "system" for m in new_messages):
            new_messages.insert(0, {"role": "system", "content": system_prompt})
            if len(new_messages) > 1 and new_messages[1].get("role") == "user":
                new_messages.insert(1, {"role": "user", "content": "Acknowledged. I am ready to process the quantitative data."})
        return new_messages

    async def check_health(self) -> bool:
        """Dynamically check if the agent service is available."""
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

    async def _resolve_provider_instance(self, model: str, base_provider: str | None = None) -> str:
        """
        Query /config?includeLocal=true from lazy-agent-service to discover which specific provider
        instance (e.g., 'vllm-2') holds the requested model.
        Falls back to base_provider if not found.
        """
        # A caller that discovered an endpoint/model pair owns that route.
        # An explicitly provided base_provider (not None) takes precedence.
        if base_provider is not None:
            return base_provider
        if not model:
            return "vllm"

        import time
        if time.time() - self._last_config_fetch <= 30 and model in self._model_to_provider_cache:
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
                
                logger.info(f"[agent] Fetching config to resolve model '{model}' from: {config_url}")
                response = await client.get(config_url, timeout=5.0)
                if response.status_code == 200:
                    data = response.json()
                    text_to_text = data.get("textToText", {})
                    models_map = text_to_text.get("models", {})
                    refreshed_mapping = {}
                    for inst_id, model_list in models_map.items():
                        for m_info in model_list:
                            m_name = m_info.get("name")
                            if m_name:
                                refreshed_mapping[m_name] = inst_id
                    self._model_to_provider_cache = refreshed_mapping
                    self._last_config_fetch = now
                else:
                    # A failed discovery must not route a model using stale
                    # provider data from an earlier deployment snapshot.
                    self._model_to_provider_cache = {}
                    self._last_config_fetch = now
            else:
                self._model_to_provider_cache = {}
                self._last_config_fetch = now
        except Exception as e:
            self._model_to_provider_cache = {}
            self._last_config_fetch = time.time()
            logger.warning(f"[agent] Failed to auto-resolve provider for model '{model}': {e}")

        fallback = base_provider or "vllm"
        resolved = self._model_to_provider_cache.get(model, fallback)
        logger.info(f"[agent] Resolved model '{model}' to provider instance '{resolved}' (fallback: '{fallback}')")
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
        provider: str | None = None,
        project: str = "default-project",
        username: str = "lazycat-sdk",
        stream: bool = False,
        max_iterations: int | None = None,
        session_id: str | None = None,
        auto_approve: bool = True,
        thinking_enabled: bool | None = None,
        min_p: float | None = None,
        disabled_tools: list[str] | None = None,
        workspace_enabled: bool | None = None,
        inline_system_prompt: bool = False,
        bench_harness: str | None = None,
        bench_run_id: str | None = None,
        bench_task: str | None = None,
    ) -> Any:
        """Execute a call to the agent service's /agent endpoint.

        thinking_enabled: None leaves the gateway default (thinking ON for
        local models on agent sessions); pass False on interactive paths where
        a user is watching the stream — the <think> block is most of the
        latency on Qwen-class models.

        min_p: pass 0.0 on any path that may hit a vLLM box running
        speculative decoding. 0.0 is vLLM's OWN default — not a tuning
        choice. Prism's ParameterRegistry gives minP an `agentDefault` of
        0.05, and vLLM refuses any min_p > 0 when speculative decoding is
        enabled ("The min_p and logit_bias sampling parameters are not yet
        supported with speculative decoding"). That refusal arrives AFTER a
        200 header, mid-stream, so prism sees an empty stream rather than an
        error and the caller gets silence. Sending an explicit 0 pre-empts
        the agentDefault (prism only fills keys that are undefined/null — a
        None here re-triggers 0.05). The default stays None so existing
        callers keep prism's sampling unchanged — at temperature > 0 a
        min_p switch is a real behaviour change, not a no-op.

        disabled_tools: a denylist of tool names. Prism only honours it when
        the payload carries NO `enabledTools` (AgenticToolResolver Mode 2 is
        gated on that), so passing this suppresses the enabledTools key.

        workspace_enabled: False drops the Core Workspace tool domain.
        """
        
        if self._kill_switch_armed:
            raise asyncio.CancelledError("lazycat-sdk kill switch is armed")

        effective_run_id = bench_run_id or bench_run_id_var.get()
        effective_harness = bench_harness or bench_harness_var.get() or "trading-cycle"
        effective_task = bench_task or bench_task_var.get()
            
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
        new_messages = self._prepare_messages(
            messages, system_prompt, inline_system_prompt=inline_system_prompt
        )

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
            "systemPrompt": _cap_system_prompt(system_prompt, agent_name),
            "functionCallingEnabled": False,
            "autoApprove": auto_approve,
        }
        if effective_run_id:
            payload["user"] = effective_run_id

        if max_iterations is not None:
            payload["maxIterations"] = max_iterations
        if thinking_enabled is not None:
            payload["thinkingEnabled"] = thinking_enabled
        if min_p is not None:
            payload["minP"] = min_p
        if workspace_enabled is not None:
            payload["workspaceEnabled"] = workspace_enabled

        if disabled_tools:
            # Mode 2 in prism's AgenticToolResolver: the denylist is read only
            # when `enabledTools` is absent, so these are mutually exclusive.
            payload["disabledTools"] = disabled_tools
        elif tools:
            enabled_tools = []
            for t in tools:
                if "function" in t:
                    enabled_tools.append(t["function"]["name"])
                elif "name" in t:
                    enabled_tools.append(t["name"])
            payload["enabledTools"] = enabled_tools

        # Always force a new session to prevent it from accumulating
        # duplicate history (since lazycat-sdk manages the conversation history locally).
        payload["createSession"] = True

        url = f"{self.url}/agent?stream={'true' if stream else 'false'}"
        headers = {
            "Content-Type": "application/json",
            "x-project": project,
            "x-username": username,
        }
        if effective_run_id:
            headers["X-Bench-Harness"] = effective_harness
            headers["X-Bench-Run"] = effective_run_id
            if effective_task:
                headers["X-Bench-Task"] = effective_task

        logger.debug(f"[INSTRUMENTATION] agent_client.call_agent attempting to connect to: {url}")
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
            logger.error(f"[INSTRUMENTATION] Agent call failed connecting to {url}. Error: {e.__class__.__name__} - {e}")
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
        policies: list[dict] | None = None,
    ) -> str:
        """Register a custom agent in lazy-agent-service, or update it if it already exists.
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
                logger.warning(f"[agent] Failed to query existing custom agents: {e}")

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
                # (honored by lazy-agent-service since 2026-07-14; real prism
                # ignores unknown fields). Callers that want the forced core
                # set pass core_tools_locked=True.
                "coreToolsLocked": core_tools_locked,
            }
            if policies:
                # Per-agent tool policy rules, each
                # `{"tool": <exact name>, "decision": "DENY"|"APPROVE"|
                #   "ASK_USER", "name": <label>}`.
                #
                # This is the ONLY tool restriction on a custom agent that
                # actually holds. `availableTools` does not: prism's
                # AgentPersonaRegistry.registerCustom() copies availableTools
                # out of the Mongo doc but NOT coreToolsLocked, so the
                # resolver's `persona?.coreToolsLocked ?? true` defaults a
                # custom agent to LOCKED and force-adds every CORE_AGENTIC /
                # system tool on top of the list we registered — which is how
                # trading-service's v3 analysts reached execute_command,
                # write_file and query_datastore despite an explicit whitelist.
                #
                # A DENY policy survives that: AutoApprovalEngine evaluates
                # policies BEFORE the tier system and before full-auto, and
                # documents DENY as "terminal rejection — never downgraded to
                # an approval prompt". Since these agents run auto_approve=True,
                # a DENY rule is the only thing that can stop a call.
                payload["policies"] = policies
            if thinking_default is not None:
                # Per-agent <think> default; an explicit thinkingEnabled on a
                # request always wins over this.
                payload["thinkingDefault"] = thinking_default

            if agent_db_id:
                try:
                    logger.info(f"[agent] Updating existing custom agent {agent_id} (db_id: {agent_db_id})")
                    r = await client.put(f"{self.url}/custom-agents/{agent_db_id}", json=payload, headers=headers, timeout=10.0)
                    r.raise_for_status()
                except Exception as e:
                    logger.error(f"[agent] Failed to update custom agent {agent_id}: {e}")
                    raise
            else:
                try:
                    logger.info(f"[agent] Creating new custom agent {agent_id}")
                    r = await client.post(f"{self.url}/custom-agents", json=payload, headers=headers, timeout=10.0)
                    r.raise_for_status()
                except Exception as e:
                    logger.error(f"[agent] Failed to create custom agent {agent_id}: {e}")
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
        enable_thinking: bool | None,
        tools: list[dict] | None = None,
        is_qwen_model: bool = False,
        agentic_mode: bool = True,
        provider: str = "vllm",
        min_p: float | None = None,
        disabled_tools: list[str] | None = None,
        workspace_enabled: bool | None = None,
        inline_system_prompt: bool = True,
        auto_approve: bool | None = None,
    ) -> tuple[dict, str, dict]:
        """Returns (payload, url, headers) formatted for /agent streaming.

        See `call_agent` for what min_p / disabled_tools / workspace_enabled /
        inline_system_prompt do — the semantics are identical on both paths.
        auto_approve mirrors call_agent's field: without it a tiered tool call
        sits in prism's approval queue until it times out as USER_REJECTED,
        which no streaming caller of this SDK can answer. None keeps the
        gateway default.
        """
        # Prepend system prompt and dummy user message to align system message rewrite in prism-service.
        # This prevents double system prompt errors on Qwen/vLLM.
        new_messages = self._prepare_messages(
            messages, system_prompt, inline_system_prompt=inline_system_prompt
        )

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
            "systemPrompt": _cap_system_prompt(system_prompt, agent_name),
        }
        # `thinkingEnabled` used to ship only when the caller also set
        # is_qwen_model, which trading-client never did — so an explicit
        # enable_thinking=False was silently dropped and local models
        # reasoned anyway. Honour the flag whenever it was actually given;
        # None still defers to the gateway default.
        if enable_thinking is not None:
            payload["thinkingEnabled"] = enable_thinking
        if min_p is not None:
            payload["minP"] = min_p
        if workspace_enabled is not None:
            payload["workspaceEnabled"] = workspace_enabled
        if auto_approve is not None:
            payload["autoApprove"] = auto_approve

        if disabled_tools and agentic_mode:
            # Mutually exclusive with enabledTools — see call_agent.
            payload["disabledTools"] = disabled_tools
        elif tools and agentic_mode:
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

        # Always force a new session to prevent it from accumulating
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
        enable_thinking: bool | None = None,
        tools: list[dict] | None = None,
        is_qwen_model: bool = False,
        agentic_mode: bool = True,
        agentContext: dict | None = None,
        provider: str | None = None,
        min_p: float | None = None,
        disabled_tools: list[str] | None = None,
        workspace_enabled: bool | None = None,
        inline_system_prompt: bool = True,
        auto_approve: bool | None = None,
        conversation_id: str | None = None,
    ):
        """High-level wrapper to stream /agent response.

        enable_thinking defaults to None (defer to the gateway) rather than
        False, so callers that never set it keep their existing behaviour now
        that the flag is no longer gated behind is_qwen_model.
        """
        
        if self._kill_switch_armed:
            raise asyncio.CancelledError("lazycat-sdk kill switch is armed")

        group_key = f"chat-{agent_name}" if agent_name == "user_chat" else agent_name
        session_id, is_new = self._get_or_create_session(group_key)
        conversation_id = conversation_id or str(uuid.uuid4())

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
            min_p=min_p,
            disabled_tools=disabled_tools,
            workspace_enabled=workspace_enabled,
            inline_system_prompt=inline_system_prompt,
            auto_approve=auto_approve,
        )
        if agentContext:
            payload["agentContext"] = agentContext

        from urllib.parse import urlsplit
        endpoint = urlsplit(url)
        logger.info("[agent] stream conversation=%s model=%s provider=%s endpoint=%s%s tools=%s max_output=%s",
                    conversation_id, model, resolved_provider, endpoint.hostname, endpoint.path,
                    len(tools or []), max_tokens)
        client = await self._get_client()
        try:
            async with client.stream("POST", url, json=payload, headers=headers, timeout=600.0) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line:
                        yield line
        except Exception as e:
            logger.error("[agent] Error in agent_chat_stream: %s", e)
            raise

agent_client = AgentClient()
