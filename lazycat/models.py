from __future__ import annotations

from enum import Enum
from typing import Any, List, Literal, Optional, Union
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


class DecisionQuestion(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    type: Literal["choice"] = "choice"
    instructions: str = Field(..., min_length=1, max_length=2048)
    criteria: dict[str, str]
    required_abstain_option: Literal[True] = Field(default=True, alias="requiredAbstainOption")

    @field_validator("criteria")
    @classmethod
    def validate_criteria(cls, value: dict[str, str]) -> dict[str, str]:
        if not 2 <= len(value) <= 16 or "insufficient_evidence" not in value:
            raise ValueError("criteria must contain 2-16 choices including insufficient_evidence")
        return value


class DecisionConstraints(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    max_latency_ms: int = Field(..., alias="maxLatencyMs", ge=1, le=2000)
    shadow_only: Literal[True] = Field(default=True, alias="shadowOnly")
    no_side_effects: Literal[True] = Field(default=True, alias="noSideEffects")
    max_attempts: int = Field(default=1, alias="maxAttempts", ge=1, le=1)


class TypedDecisionRequest(BaseModel):
    """Decision-fabric request matching decision-provider.v1."""
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    request_id: str = Field(..., alias="requestId")
    run_id: str = Field(..., alias="runId")
    capability: Literal["semantic.choice.v1"] = "semantic.choice.v1"
    question_id: Literal["agent.next_readonly_action.v1", "agent.evidence_sufficiency.v1"] = Field(..., alias="questionId")
    policy_version: Literal["shadow.v1"] = Field(default="shadow.v1", alias="policyVersion")
    data_classification: Literal["public"] = Field(default="public", alias="dataClassification")
    state: str = Field(..., max_length=32768)
    questions: dict[str, DecisionQuestion]
    constraints: DecisionConstraints


class DecisionProviderInfo(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    id: Literal["tinymodels"] = "tinymodels"
    version: str
    model_id: str = Field(..., alias="modelId")
    artifact_id: Optional[str] = Field(default=None, alias="artifactId")
    deployment_id: str = Field(..., alias="deploymentId")
    deployment_state: Literal["candidate_shadow", "advisory_shadow"] = Field(..., alias="deploymentState")


class DecisionResultChoice(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    type: Literal["choice"] = "choice"
    value: str
    probabilities: Optional[dict[str, float]] = None
    raw_confidence: Optional[float] = Field(default=None, alias="rawConfidence", ge=0, le=1)
    calibrated_confidence: Optional[float] = Field(default=None, alias="calibratedConfidence", ge=0, le=1)
    calibration_state: Literal["uncalibrated", "calibrated", "not_available"] = Field(..., alias="calibrationState")
    abstained: bool


class DecisionEvidence(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    input_hash: str = Field(..., alias="inputHash", pattern=r"^sha256-[a-f0-9]{64}$")
    output_hash: str = Field(..., alias="outputHash", pattern=r"^sha256-[a-f0-9]{64}$")
    process_unloaded: bool = Field(..., alias="processUnloaded")
    latency_ms: float = Field(..., alias="latencyMs", ge=0)


class TypedDecisionResult(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    request_id: str = Field(..., alias="requestId")
    provider: DecisionProviderInfo
    results: dict[str, DecisionResultChoice]
    evidence: DecisionEvidence


class DecisionReceipt(BaseModel):
    """Authoritative decision receipt recorded against a run."""
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    receipt_id: str = Field(..., alias="receiptId")
    request_id: str = Field(..., alias="requestId")
    run_id: str = Field(..., alias="runId")
    agent_profile: str = Field(..., alias="agentProfile")
    policy_version: str = Field(..., alias="policyVersion")
    input_hash: str = Field(..., alias="inputHash")
    output_hash: Optional[str] = Field(default=None, alias="outputHash")
    created_at: str = Field(..., alias="createdAt")
    latency_ms: float = Field(..., alias="latencyMs", ge=0)
    policy_outcome: str = Field(default="shadow_only", alias="policyOutcome", pattern="^shadow_only$")
    authorizes_actions: bool = Field(default=False, alias="authorizesActions")
    fallback_reason: Optional[str] = Field(default=None, alias="fallbackReason")
    fallback: str = Field(..., pattern="^(primary_llm|none)$")
    signal: Optional[TypedDecisionResult] = None


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
    APPROVAL_REQUIRED = "approval.required"
    APPROVAL_RESOLVED = "approval.resolved"
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
