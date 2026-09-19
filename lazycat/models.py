from typing import Any, Literal
from pydantic import BaseModel, Field

class AgentProfile(BaseModel):
    name: str
    system_prompt: str
    model: str
    provider: str = "vllm"
    temperature: float = 0.0
    max_tokens: int = 8192
    min_p: float | None = None
    tools: list[dict] = Field(default_factory=list)

class RunRequest(BaseModel):
    idempotency_key: str | None = None
    profile: AgentProfile
    messages: list[dict]
    max_iterations: int = 15
    auto_approve: bool = True
    thinking_enabled: bool | None = None
    bench_task: str | None = None

class RunError(BaseModel):
    code: str
    message: str
    details: dict | None = None

class RunResult(BaseModel):
    status: Literal["completed", "failed", "cancelled"]
    final_output: str
    total_tokens: int
    completion_tokens: int
    prompt_tokens: int
    reasoning_tokens: int
    usage_requests: int
    error: RunError | None = None

class StreamEvent(BaseModel):
    seq: int
    type: str

class StreamChunk(StreamEvent):
    type: Literal["chunk"] = "chunk"
    content: str

class StreamToolCall(StreamEvent):
    type: Literal["tool_calls"] = "tool_calls"
    tool_calls: list[dict]

class StreamToolExecution(StreamEvent):
    type: Literal["tool_execution"] = "tool_execution"
    status: Literal["calling", "done", "error"]
    tool: dict
    elapsed_ms: int = 0

class StreamUsage(StreamEvent):
    type: Literal["usage_update"] = "usage_update"
    usage: dict

class StreamDone(StreamEvent):
    type: Literal["done"] = "done"
    model: str | None = None
    provider: str | None = None
    result: RunResult | None = None

class StreamError(StreamEvent):
    type: Literal["error"] = "error"
    error: RunError
