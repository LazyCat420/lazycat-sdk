import hashlib
import json
import logging
from typing import Any, Callable

from lazycat.llm import prism_client
from lazycat.tools import tool_executor
from lazycat.session import ConversationSession

logger = logging.getLogger(__name__)

class ToolLoopDetector:
    """Detects and breaks tool call loops.

    Tracks (tool_name, args_hash, status) history per session.
    If the same combo fails N times, returns a stop injection message
    instructing the agent to reason from what it already has.
    """

    def __init__(self, max_identical_failures: int = 3, max_duplicate_queries: int = 2):
        self.max_identical_failures = max_identical_failures
        self.max_duplicate_queries = max_duplicate_queries
        self._history: dict[str, int] = {}  # "tool:args_hash:failed" -> count
        self._warning_issued: set[str] = set()  # keys that got a warning injection
        self.escalation_triggered: bool = False  # True = agent persisted after warning

    def _make_key(self, tool_name: str, args: Any, failed: bool) -> str:
        """Create a dedup key from tool name, args hash, and failure status."""
        args_str = json.dumps(args, sort_keys=True, default=str) if args else ""
        args_hash = hashlib.sha256(args_str.encode()).hexdigest()[:12]
        status = "failed" if failed else "ok"
        return f"{tool_name}:{args_hash}:{status}"

    def record_call(
        self, tool_name: str, args: Any, failed: bool
    ) -> str | None:
        """Record a tool call and check for loops.

        Returns:
            A stop injection message if a loop is detected, else None.
        """
        key = self._make_key(tool_name, args, failed)
        self._history[key] = self._history.get(key, 0) + 1

        # ── Failure loop detection ──
        if failed and self._history[key] >= self.max_identical_failures:
            if key in self._warning_issued:
                self.escalation_triggered = True
                logger.error(
                    "[ToolLoopDetector] ESCALATION: %s persisted after warning (%d failures).",
                    tool_name, self._history[key],
                )
                return (
                    f"[SYSTEM OVERRIDE — ESCALATION] The tool '{tool_name}' has now failed "
                    f"{self._history[key]} times. The previous warning was ignored. "
                    f"You MUST stop calling this tool immediately and produce your "
                    f"final artifact with the data you have. Mark missing data as "
                    f"'DataGap: [description]'."
                )

            self._warning_issued.add(key)
            logger.warning(
                "[ToolLoopDetector] Loop detected: %s failed %d times with same args",
                tool_name,
                self._history[key],
            )
            return (
                f"[SYSTEM OVERRIDE] The tool '{tool_name}' has failed "
                f"{self._history[key]} times with the same arguments. "
                f"STOP calling this tool. Instead, reason from the data you "
                f"already have and produce your final artifact. If critical "
                f"data is missing, mark it as 'DataGap: [description]' in "
                f"your output."
            )

        # ── Duplicate query detection (successful calls) ──
        if not failed and self._history[key] > self.max_duplicate_queries:
            logger.warning(
                "[ToolLoopDetector] Duplicate query: %s called %d times with same args (successful)",
                tool_name, self._history[key],
            )
            return (
                f"[SYSTEM NOTICE] You have already called '{tool_name}' with these "
                f"exact arguments {self._history[key]} times and received the same data. "
                f"Use the data you already have — do not re-request it."
            )

        return None

class BaseAgent:
    """Base class for all LazyCat SDK agents."""
    
    def __init__(
        self,
        name: str,
        system_prompt: str,
        model: str = "cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit",
        temperature: float = 0.0,
        max_tokens: int = 8192,
        provider: str = "vllm",
        project: str = "lazycat-sdk-app",
        llm_client: Any = None,
    ):
        self.name = name
        self.system_prompt = system_prompt
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.provider = provider
        self.project = project
        self.tools: list[dict] = []
        self.llm_client = llm_client or prism_client
        
    def add_tool(self, tool_schema: dict):
        self.tools.append(tool_schema)

class AgentHarness:
    """Standardized tool-call loop.
    
    send message -> check for tool calls -> dispatch -> loop until done.
    """
    
    def __init__(
        self,
        agent: BaseAgent,
        session: ConversationSession,
        max_iterations: int = 15,
        on_tool_call: Callable[[str, dict], str | None] | None = None,
        on_tool_result: Callable[[str, dict, Any, bool], None] | None = None,
    ):
        self.agent = agent
        self.session = session
        self.max_iterations = max_iterations
        # Hook: called before each tool execution with (tool_name, arguments).
        # Return None to proceed, or a string to inject as the tool result
        self.on_tool_call = on_tool_call
        # Hook: called after each tool execution with (tool_name, arguments, result, was_blocked).
        self.on_tool_result = on_tool_result
        self.loop_detector = ToolLoopDetector(max_identical_failures=3)

    async def run(self, user_input: str | None = None) -> str:
        """Run the agent loop until it completes or reaches max iterations."""
        if user_input:
            self.session.add_user_message(user_input)
            
        iterations = 0
        while iterations < self.max_iterations:
            iterations += 1
            
            # 1. Send messages to LLM
            resp = await self.agent.llm_client.call_agent(
                model=self.agent.model,
                messages=self.session.get_messages(),
                system_prompt=self.agent.system_prompt,
                agent_name=self.agent.name,
                project=self.agent.project,
                max_tokens=self.agent.max_tokens,
                tools=self.agent.tools if self.agent.tools else None,
                provider=self.agent.provider,
                stream=True
            )
            
            content = ""
            tool_calls = []
            
            import json
            try:
                async for line in resp.aiter_lines():
                    line = line.strip()
                    if not line or not line.startswith("data: "):
                        continue
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                        
                    event_type = data.get("type")
                    if event_type == "chunk":
                        chunk_text = data.get("content", "")
                        content += chunk_text
                        print(chunk_text, end="", flush=True)
                    elif event_type == "tool_calls" or "toolCalls" in data:
                        tool_calls = data.get("toolCalls", [])
                    elif event_type == "error":
                        logger.error(f"Prism stream error: {data.get('message')}")
                    elif "text" in data and not event_type:
                        content = data.get("text", content)
                        tool_calls = data.get("toolCalls", tool_calls)
                
                print() # flush newline after stream completes
            finally:
                await resp.aclose()
            
            
            # 2. Add LLM response to history
            self.session.add_assistant_message(content, tool_calls)
            
            # 3. If no tool calls, we're done
            if not tool_calls:
                return content or ""
                
            # 4. Dispatch tool calls
            for tc in tool_calls:
                tc_id = tc.get("id", "")
                if "function" in tc:
                    func = tc.get("function", {})
                    func_name = func.get("name", "")
                    try:
                        arguments = json.loads(func.get("arguments", "{}"))
                    except json.JSONDecodeError:
                        arguments = {}
                else:
                    func_name = tc.get("name", "")
                    arguments = tc.get("arguments") or tc.get("args") or {}
                    if isinstance(arguments, str):
                        try:
                            arguments = json.loads(arguments)
                        except json.JSONDecodeError:
                            arguments = {}
                
                logger.info(f"[{self.agent.name}] Executing tool: {func_name}")
                
                # Check for human-in-the-loop pauses
                if func_name in ("ask_user_question", "request_plan_approval"):
                    result = await self._handle_pausing_tool(func_name, arguments)
                    was_blocked = False
                else:
                    # Internal loop detection
                    loop_block_msg = self.loop_detector.record_call(func_name, arguments, failed=True)
                    if loop_block_msg:
                        override_result = loop_block_msg
                    else:
                        # Undo the speculative failure record
                        key = self.loop_detector._make_key(func_name, arguments, failed=True)
                        self.loop_detector._history[key] = max(0, self.loop_detector._history.get(key, 1) - 1)
                    
                    # External Hook
                    if override_result is None and self.on_tool_call is not None:
                        override_result = self.on_tool_call(func_name, arguments)
                    
                    if override_result is not None:
                        # Hook blocked this call — use override as result
                        logger.warning(f"[{self.agent.name}] Tool call blocked: {func_name}")
                        result = {"blocked": True, "message": override_result}
                        was_blocked = True
                    else:
                        # Execute via the tool service proxy
                        result = await tool_executor.execute_tool(func_name, arguments)
                        was_blocked = False
                        
                        # Record actual outcome
                        failed = False
                        if isinstance(result, dict):
                            if result.get("error") or result.get("is_error"):
                                failed = True
                            elif not result:
                                failed = True
                        elif result is None:
                            failed = True
                        
                        self.loop_detector.record_call(func_name, arguments, failed=failed)
                
                # Notify post-call hook (e.g. ToolLoopDetector records outcome)
                if self.on_tool_result is not None and func_name not in ("ask_user_question", "request_plan_approval"):
                    try:
                        self.on_tool_result(func_name, arguments, result, was_blocked)
                    except Exception as hook_err:
                        logger.warning(f"[{self.agent.name}] on_tool_result hook error: {hook_err}")
                
                # 5. Add result to history
                self.session.add_tool_message(
                    tool_call_id=tc_id,
                    name=func_name,
                    content=json.dumps(result, default=str)
                )
                
        logger.warning(f"[{self.agent.name}] Reached max iterations ({self.max_iterations})")
        return "Max iterations reached without a final answer."

    async def _handle_pausing_tool(self, func_name: str, arguments: dict) -> Any:
        import os
        import time
        import json
        import asyncio
        
        session_id = self.session.session_id
        if not session_id:
            session_id = f"session-{int(time.time())}"
            
        # Emit SSE status event to stdout for Express to forward to SSE client
        if func_name == "ask_user_question":
            question_text = arguments.get("question") or arguments.get("prompt") or ""
            choices = arguments.get("choices", [])
            print(f'data: {json.dumps({"type": "question", "question": question_text, "choices": choices})}', flush=True)
        else: # request_plan_approval
            plan_text = arguments.get("plan") or arguments.get("details") or ""
            print(f'data: {json.dumps({"type": "approval_required", "plan": plan_text})}', flush=True)
            
        # Poll file system for response file
        pause_file = os.path.join("data", "pauses", f"{session_id}.json")
        
        # Max wait: 10 minutes (600 seconds)
        max_wait = 600
        elapsed = 0
        while elapsed < max_wait:
            if os.path.exists(pause_file):
                try:
                    # Give a tiny buffer for file system synchronization
                    await asyncio.sleep(0.1)
                    with open(pause_file, "r") as f:
                        data = json.load(f)
                    
                    # Clean up file
                    try:
                        os.remove(pause_file)
                    except Exception:
                        pass
                    
                    if func_name == "ask_user_question":
                        return {"answers": data.get("answers") or [{"answer": data.get("answer", "")}]}
                    else: # request_plan_approval
                        return {"approved": data.get("approved", True)}
                except Exception as e:
                    logger.error("Failed to read pause file: %s", e)
                    
            await asyncio.sleep(1.0)
            elapsed += 1
            
        return {"error": "Timeout waiting for user response"}

    async def stream_run(self, payload_override: dict | None = None):
        """
        Yields raw Server-Sent Events from Prism. 
        If payload_override is provided, it passes those parameters to Prism 
        (useful for MCP tool execution where Prism loops internally).
        Otherwise it uses the local agent config.
        """
        import httpx
        import json
        
        url = f"{self.agent.llm_client.url}/agent"
        
        payload = {
            "provider": self.agent.provider,
            "model": self.agent.model,
            "messages": self.session.get_messages(),
            "maxTokens": 8192,
            "systemPrompt": self.agent.system_prompt,
            "agent": self.agent.name,
            "stream": True
        }
        
        if payload_override:
            payload.update(payload_override)
            
        async with httpx.AsyncClient(timeout=600.0) as client:
            async with client.stream(
                "POST", 
                url, 
                json=payload, 
                headers={"Accept": "text/event-stream"}
            ) as resp:
                if resp.status_code != 200:
                    error_body = ""
                    async for chunk in resp.aiter_text():
                        error_body += chunk
                    yield f'data: {json.dumps({"type": "error", "message": f"Prism error {resp.status_code}: {error_body[:500]}"})}\\n\\n'
                    return
                
                buffer = ""
                async for chunk in resp.aiter_text():
                    buffer += chunk
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        line = line.strip()
                        if line:
                            yield f"{line}\n\n"
