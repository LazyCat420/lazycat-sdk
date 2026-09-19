from __future__ import annotations

from enum import Enum
from typing import Any, List, Optional, Union
from pydantic import BaseModel, ConfigDict, Field, field_validator


class RunBudget(BaseModel):
    """Execution budget constraints matching docs/contracts/run-contract-v1.json."""
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    max_tokens: Optional[int] = Field(default=None, alias="maxTokens", ge=0, le=32768)
    max_tool_calls: Optional[int] = Field(default=None, alias="maxToolCalls", ge=0, le=50)
    max_retries: Optional[int] = Field(default=None, alias="maxRetries", ge=0, le=10)
    max_duration_ms: Optional[int] = Field(default=None, alias="maxDurationMs", ge=0, le=600000)


class Message(BaseModel):
    """Chat message object."""
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    role: str = Field(..., description="Role: system, user, assistant, or tool")
    content: str = Field(..., description="Message text content")


class CreateRunRequest(BaseModel):
    """Canonical v1 run creation request."""
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    profile_id: str = Field(..., alias="profileId")
    profile_version: Optional[str] = Field(default=None, alias="profileVersion")
    input: Union[str, List[Union[Message, dict[str, Any]]]] = Field(...)
    model: Optional[str] = None
    budget: Optional[RunBudget] = None
    tools: Optional[List[dict[str, Any]]] = None
    stream: bool = False
    idempotency_key: Optional[str] = Field(default=None, alias="idempotencyKey")


class RunEventType(str, Enum):
    """Canonical event types emitted during run streaming."""
    RUN_ADMITTED = "run.admitted"
    RUN_STARTED = "run.started"
    RUN_CREATED = "run.created"  # Compatibility alias
    MESSAGE_DELTA = "message.delta"
    MESSAGE_COMPLETED = "message.completed"
    TOOL_INVOKED = "tool.invoked"
    TOOL_CALLED = "tool.called"  # Compatibility alias
    TOOL_COMPLETED = "tool.completed"
    TOOL_RESULT = "tool.result"  # Compatibility alias
    TOOL_FAILED = "tool.failed"
    WORKER_DISPATCHED = "worker.dispatched"
    WORKER_COMPLETED = "worker.completed"
    RUN_COMPLETED = "run.completed"
    RUN_FAILED = "run.failed"
    RUN_CANCELLED = "run.cancelled"


class RunEvent(BaseModel):
    """Server-Sent Event envelope matching canonical v1 contract."""
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    id: str = Field(...)
    run_id: str = Field(..., alias="runId")
    type: str = Field(...)
    timestamp: str = Field(...)
    data: dict[str, Any] = Field(default_factory=dict)

    @field_validator("type")
    @classmethod
    def validate_type_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Event 'type' cannot be empty")
        return v


class StructuredError(BaseModel):
    """Structured error payload matching canonical v1 contract."""
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    code: str = Field(...)
    message: str = Field(...)
    retryable: bool = Field(...)
    category: Optional[str] = Field(default=None)
    details: Optional[dict[str, Any]] = None


class RunUsage(BaseModel):
    """Resource usage tracking."""
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    prompt_tokens: Optional[int] = Field(default=None, alias="promptTokens")
    completion_tokens: Optional[int] = Field(default=None, alias="completionTokens")
    total_tokens: Optional[int] = Field(default=None, alias="totalTokens")
    tool_calls_count: Optional[int] = Field(default=None, alias="toolCalls")
    retry_count: Optional[int] = Field(default=None, alias="retryCount")
    duration_ms: Optional[int] = Field(default=None, alias="durationMs")


class RunResult(BaseModel):
    """Authoritative final run result matching canonical v1 contract."""
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    run_id: str = Field(..., alias="id")
    status: str = Field(...)
    messages: List[dict[str, Any]] = Field(default_factory=list)
    usage: Optional[RunUsage] = None
    context_receipt: Optional[dict[str, Any]] = Field(default=None, alias="contextReceipt")
    evidence_records: Optional[List[dict[str, Any]]] = Field(default=None, alias="evidenceRecords")
    error: Optional[StructuredError] = None


class RunProfile(BaseModel):
    """Consumer profile mapping container."""
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    name: str
    system_prompt: str
    tools: List[str] = Field(default_factory=list)
    default_model: Optional[str] = None
    default_budget: Optional[RunBudget] = None
