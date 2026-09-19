import hashlib
import json
import logging
import os
import time
from typing import Any, Callable

from lazycat.llm import prism_client
from lazycat.tools import tool_executor
from lazycat.session import ConversationSession
from lazycat.sse import format_sse, iter_sse_json_lines, iter_sse_lines
from lazycat.client import RunClient, RunClientError
from lazycat.models import RunRequest, AgentProfile, StreamChunk, StreamToolCall, StreamToolExecution, StreamDone, StreamError, StreamUsage

logger = logging.getLogger(__name__)

_STREAM_STDOUT = os.environ.get("LAZYCAT_STREAM_STDOUT", "").lower() in ("1", "true", "yes")
_MALFORMED_ARGS_ECHO_CHARS = 240

def decode_tool_arguments(raw: Any) -> tuple[dict, str | None]:
    if raw is None:
        return {}, None
    if isinstance(raw, dict):
        return raw, None
    if not isinstance(raw, str):
        return {}, f"Tool arguments must be a JSON object, got {type(raw).__name__}."
    text = raw.strip()
    if not text:
        return {}, None
    try:
        decoded = json.loads(text)
    except (ValueError, TypeError) as e:
        return {}, (
            f"Your tool arguments were not valid JSON and could not be "
            f"decoded ({e}). The call was NOT executed. Nothing was lost — "
            f"re-send the SAME call with correctly escaped JSON. You sent "
            f"{len(text)} characters starting: "
            f"{text[:_MALFORMED_ARGS_ECHO_CHARS]!r}"
        )
    if isinstance(decoded, dict):
        return decoded, None
    return {}, (
        f"Tool arguments must be a JSON object (e.g. {{\"ticker\": \"AAPL\"}}), "
        f"got a {type(decoded).__name__}. The call was NOT executed."
    )

class ToolLoopDetector:
    def __init__(self, max_identical_failures: int = 3, max_duplicate_queries: int = 2):
        self.max_identical_failures = max_identical_failures
        self.max_duplicate_queries = max_duplicate_queries
        self._history: dict[str, int] = {}
        self._warning_issued: set[str] = set()
        self.escalation_triggered: bool = False

    def _make_key(self, tool_name: str, args: Any, failed: bool) -> str:
        args_str = json.dumps(args, sort_keys=True, default=str) if args else ""
        args_hash = hashlib.sha256(args_str.encode()).hexdigest()[:12]
        status = "failed" if failed else "ok"
        return f"{tool_name}:{args_hash}:{status}"

    def record_call(self, tool_name: str, args: Any, failed: bool) -> str | None:
        key = self._make_key(tool_name, args, failed)
        self._history[key] = self._history.get(key, 0) + 1
        if failed and self._history[key] >= self.max_identical_failures:
            if key in self._warning_issued:
                self.escalation_triggered = True
                return f"[SYSTEM OVERRIDE — ESCALATION] The tool '{tool_name}' has now failed {self._history[key]} times. You MUST stop calling this tool immediately."
            self._warning_issued.add(key)
            return f"[SYSTEM OVERRIDE] The tool '{tool_name}' has failed {self._history[key]} times with the same arguments. STOP calling this tool."
        if not failed and self._history[key] > self.max_duplicate_queries:
            return f"[SYSTEM NOTICE] You have already called '{tool_name}' with these exact arguments {self._history[key]} times. Use the data you already have."
        return None

class BaseAgent:
    def __init__(
        self,
        name: str,
        system_prompt: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 8192,
        provider: str = "vllm",
        project: str = "lazycat-sdk-app",
        username: str = "lazycat-sdk",
        llm_client: Any = None,
        auto_approve: bool = True,
        min_p: float | None = None,
    ):
        self.name = name
        self.system_prompt = system_prompt
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.provider = provider
        self.project = project
        self.username = username
        self.tools: list[dict] = []
        self.llm_client = llm_client or prism_client
        self.auto_approve = auto_approve
        self.min_p = min_p
        
    def add_tool(self, tool_schema: dict):
        self.tools.append(tool_schema)

class AgentHarness:
    def __init__(
        self,
        agent: BaseAgent,
        session: ConversationSession,
        max_iterations: int = 15,
        on_tool_call: Callable[[str, dict], str | None] | None = None,
        on_tool_result: Callable[[str, dict, Any, bool, int], None] | None = None,
        max_tool_result_chars: int = 50_000,
        thinking_enabled: bool | None = None,
        bench_task: str | None = None,
    ):
        self.agent = agent
        self.session = session
        self.max_iterations = max_iterations
        self.thinking_enabled = thinking_enabled
        self.bench_task = bench_task
        self.on_tool_call = on_tool_call
        self.on_tool_result = on_tool_result
        self.max_tool_result_chars = max_tool_result_chars
        self.loop_detector = ToolLoopDetector(max_identical_failures=3)
        self.last_usage: dict = {}
        self.total_tokens: int = 0
        self.completion_tokens: int = 0
        self.usage_requests: int = 0
        self.total_requests: int = 0
        self.prompt_tokens: int = 0
        self.reasoning_tokens: int = 0
        self.last_model: str | None = None
        self.last_provider: str | None = None
        
        # New run client points to lazy-agent-service
        # Default assumes it is running locally or accessible via LAZY_AGENT_SERVICE_URL
        base_url = os.environ.get("LAZY_AGENT_SERVICE_URL", "http://lazy-agent-service:3000")
        self.run_client = RunClient(
            base_url=base_url,
            project=self.agent.project,
            username=self.agent.username
        )

    def _build_request(self) -> RunRequest:
        profile = AgentProfile(
            name=self.agent.name,
            system_prompt=self.agent.system_prompt,
            model=self.agent.model,
            provider=self.agent.provider,
            temperature=self.agent.temperature,
            max_tokens=self.agent.max_tokens,
            min_p=self.agent.min_p,
            tools=self.agent.tools
        )
        return RunRequest(
            profile=profile,
            messages=self.session.get_messages(),
            max_iterations=self.max_iterations,
            auto_approve=self.agent.auto_approve,
            thinking_enabled=self.thinking_enabled,
            bench_task=self.bench_task
        )

    async def run(self, user_input: str | None = None) -> str:
        """Run the agent loop via the shared run client instead of local loop."""
        if user_input:
            self.session.add_user_message(user_input)
            
        request = self._build_request()
        run_id = await self.run_client.create_run(request)
        
        content = ""
        try:
            async for event in self.run_client.stream_run(run_id):
                if isinstance(event, StreamChunk):
                    content += event.content
                    if _STREAM_STDOUT:
                        print(event.content, end="", flush=True)
                elif isinstance(event, StreamToolExecution):
                    # Hook compat
                    if event.status in ("done", "error") and self.on_tool_result:
                        args, _ = decode_tool_arguments(event.tool.get("args") or event.tool.get("arguments"))
                        self.on_tool_result(event.tool.get("name", ""), args, event.tool.get("result"), False, event.elapsed_ms)
                elif isinstance(event, StreamUsage):
                    self.last_usage = event.usage
                    completion = (int(event.usage.get("outputTokens") or 0) + int(event.usage.get("reasoningOutputTokens") or 0))
                    self.total_tokens += int(event.usage.get("inputTokens") or 0) + completion
                    self.completion_tokens += completion
                    self.usage_requests += 1
                    self.prompt_tokens += int(event.usage.get("inputTokens") or 0)
                    self.reasoning_tokens += int(event.usage.get("reasoningOutputTokens") or 0)
                elif isinstance(event, StreamDone):
                    self.last_model = event.model
                    self.last_provider = event.provider
                    if event.result and event.result.final_output:
                        content = event.result.final_output
                elif isinstance(event, StreamError):
                    logger.error(f"Stream error: {event.error.message}")
        finally:
            if _STREAM_STDOUT:
                print()
                
        self.session.add_assistant_message(content)
        return content

    async def stream_run(self, payload_override: dict | None = None):
        """Yields raw Server-Sent Events, wrapping the new RunClient stream."""
        request = self._build_request()
        if payload_override:
            # Quick conversion to model schema
            if "model" in payload_override:
                request.profile.model = payload_override["model"]
            if "systemPrompt" in payload_override:
                request.profile.system_prompt = payload_override["systemPrompt"]
            if "messages" in payload_override:
                request.messages = payload_override["messages"]
            
        run_id = await self.run_client.create_run(request)
        
        try:
            async for event in self.run_client.stream_run(run_id):
                # Emit raw SSE as consumers expect string frames
                yield format_sse(event.model_dump())
        except Exception as e:
            yield format_sse({"type": "error", "message": f"Client stream error: {e}"})

    async def _handle_pausing_tool(self, func_name: str, arguments: dict) -> Any:
        # Legacy fallback; handled by remote server now.
        pass
