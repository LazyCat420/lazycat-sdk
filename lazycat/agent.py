import json
import logging
from typing import Any, Callable

from lazycat.llm import prism_client
from lazycat.tools import tool_executor
from lazycat.session import ConversationSession

logger = logging.getLogger(__name__)

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
        # (e.g. "BLOCKED: tool loop detected") and skip actual execution.
        self.on_tool_call = on_tool_call
        # Hook: called after each tool execution with (tool_name, arguments, result, was_blocked).
        # Use to record tool call outcomes for loop detection.
        self.on_tool_result = on_tool_result

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
                tools=self.agent.tools if self.agent.tools else None
            )
            
            resp_data = resp.json()
            
            # Handle both OpenAI format and Prism native format
            if "choices" in resp_data:
                message = resp_data.get("choices", [{}])[0].get("message", {})
                content = message.get("content", "")
                tool_calls = message.get("tool_calls", [])
            else:
                # Prism native format
                content = resp_data.get("text", "")
                tool_calls = resp_data.get("toolCalls", [])
            
            
            # 2. Add LLM response to history
            self.session.add_assistant_message(content, tool_calls)
            
            # 3. If no tool calls, we're done
            if not tool_calls:
                return content or ""
                
            # 4. Dispatch tool calls
            for tc in tool_calls:
                tc_id = tc.get("id", "")
                func = tc.get("function", {})
                func_name = func.get("name", "")
                try:
                    arguments = json.loads(func.get("arguments", "{}"))
                except json.JSONDecodeError:
                    arguments = {}
                
                logger.info(f"[{self.agent.name}] Executing tool: {func_name}")
                
                # Check hook before execution (e.g. ToolLoopDetector)
                override_result = None
                if self.on_tool_call is not None:
                    override_result = self.on_tool_call(func_name, arguments)
                
                if override_result is not None:
                    # Hook blocked this call — use override as result
                    logger.warning(f"[{self.agent.name}] Tool call blocked by hook: {func_name}")
                    result = {"blocked": True, "message": override_result}
                    was_blocked = True
                else:
                    # Execute via the tool service proxy
                    result = await tool_executor.execute_tool(func_name, arguments)
                    was_blocked = False
                
                # Notify post-call hook (e.g. ToolLoopDetector records outcome)
                if self.on_tool_result is not None:
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
