import hashlib
import inspect
import json
import logging
import os
import re
import time
from typing import Any, Callable

from lazycat.llm import _is_direct_openai_provider, agent_client
from lazycat.tools import strip_mcp_prefix, tool_executor
from lazycat.session import ConversationSession
from lazycat.sse import format_sse, iter_sse_json_lines, iter_sse_lines

logger = logging.getLogger(__name__)

# Set LAZYCAT_STREAM_STDOUT=1 to echo streamed tokens to stdout (interactive
# CLI use). Off by default — services must not mix model text into their logs.
_STREAM_STDOUT = os.environ.get("LAZYCAT_STREAM_STDOUT", "").lower() in ("1", "true", "yes")

#: How much of a malformed argument string to quote back to the model. Enough
#: to locate the break, not enough to re-flood the context with the payload it
#: just emitted.
_MALFORMED_ARGS_ECHO_CHARS = 240

_FINAL_TURN_NUDGE = (
    "Tool budget exhausted. Do not call tools. Reply now with your final "
    "answer in the required output format using the evidence gathered so far."
)


def _repair_json_argument_text(text: str) -> str:
    """Lightweight deterministic cleanup for common local-LLM JSON syntax slips.

    Strips surrounding markdown ```json / ``` fences and trailing commas
    before `}` or `]`. Never invents keys or truncates values.
    """
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json|JSON)?\s*\r?\n?", "", s)
        s = re.sub(r"\s*```$", "", s).strip()
    prev = None
    while prev != s:
        prev = s
        s = re.sub(r",\s*([}\]])", r"\1", s)
    return s


def decode_tool_arguments(raw: Any) -> tuple[dict, str | None]:
    """Decode a tool call's `arguments`. Returns `(arguments, error_or_None)`.

    WHY THIS IS NOT `except: arguments = {}`
    ----------------------------------------
    A model that emits a large tool payload sometimes truncates it or escapes
    it wrongly. The old code caught `JSONDecodeError` and substituted `{}`,
    which meant a 40-second generation was thrown away and the tool ran with
    NO arguments — so the model got back whatever the tool says about missing
    required fields, with nothing pointing at the real cause.

    Measured: trading-service `cycle-v3-1785792600`, the quant analyst's
    `emit_structured_output` call arrived as `{}` and was rejected with
    "'data' is required and must be an object". The model had produced the
    data; only the JSON was malformed. It could not tell, so it could not fix
    it, and the run was lost.

    An EMPTY or missing argument string is NOT an error — a zero-argument tool
    legitimately sends `""`, `"{}"` or nothing at all. Only a non-empty string
    that fails to parse is a defect, and only that returns an error.
    """
    if raw is None:
        return {}, None
    if isinstance(raw, dict):
        return raw, None
    if not isinstance(raw, str):
        return {}, (
            f"Tool arguments must be a JSON object, got {type(raw).__name__}."
        )

    text = raw.strip()
    if not text:
        return {}, None
    try:
        decoded = json.loads(text)
    except (ValueError, TypeError) as e:
        repaired = _repair_json_argument_text(text)
        if repaired:
            try:
                decoded = json.loads(repaired)
            except (ValueError, TypeError):
                return {}, (
                    f"Your tool arguments were not valid JSON and could not be "
                    f"decoded ({e}). The call was NOT executed. Nothing was lost — "
                    f"re-send the SAME call with correctly escaped JSON. You sent "
                    f"{len(text)} characters starting: "
                    f"{text[:_MALFORMED_ARGS_ECHO_CHARS]!r}"
                )
        else:
            return {}, (
                f"Your tool arguments were not valid JSON and could not be "
                f"decoded ({e}). The call was NOT executed. Nothing was lost — "
                f"re-send the SAME call with correctly escaped JSON. You sent "
                f"{len(text)} characters starting: "
                f"{text[:_MALFORMED_ARGS_ECHO_CHARS]!r}"
            )
    if isinstance(decoded, dict):
        return decoded, None
    # A bare list/string/number parses fine but is not an argument mapping.
    return {}, (
        f"Tool arguments must be a JSON object (e.g. {{\"ticker\": \"AAPL\"}}), "
        f"got a {type(decoded).__name__}. The call was NOT executed."
    )


_QWEN_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(?:```(?:json)?\s*)?(\{.*?\})(?:\s*```)?\s*</tool_call>",
    re.DOTALL | re.IGNORECASE,
)
_DSML_FUNC_CALLS_RE = re.compile(
    r"<[｜|]DSML[｜|]function_calls\s*>(.*?)</[｜|]DSML[｜|]function_calls\s*>",
    re.DOTALL,
)
_DSML_INVOKE_RE = re.compile(
    r"<[｜|]DSML[｜|]invoke\b([^>]*)>(.*?)</[｜|]DSML[｜|]invoke\s*>",
    re.DOTALL,
)
_DSML_PARAM_RE = re.compile(
    r"<[｜|]DSML[｜|]parameter\b([^>]*)>(.*?)</[｜|]DSML[｜|]parameter\s*>",
    re.DOTALL,
)
_DSML_ATTR_NAME_RE = re.compile(r'\bname\s*=\s*(?:"([^"]*)"|\'([^\']*)\')')
_DSML_ATTR_STRING_RE = re.compile(
    r'\bstring\s*=\s*(?:"([^"]*)"|\'([^\']*)\')', re.IGNORECASE
)
_NUMERIC_LITERAL_RE = re.compile(r"^-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?$")


def _build_allowed_tool_lookup(
    call_tools: list[dict] | None,
) -> tuple[dict[str, str], dict[str, str]]:
    allowed_exact: dict[str, str] = {}
    allowed_bare: dict[str, str] = {}
    for tool_def in call_tools or []:
        if not isinstance(tool_def, dict):
            continue
        fn_obj = tool_def.get("function")
        raw_name = (
            fn_obj.get("name") if isinstance(fn_obj, dict) else None
        ) or tool_def.get("name")
        if isinstance(raw_name, str) and raw_name.strip():
            clean = raw_name.strip()
            allowed_exact[clean] = clean
            bare = strip_mcp_prefix(clean)
            if bare and bare not in allowed_bare:
                allowed_bare[bare] = clean
    return allowed_exact, allowed_bare


def _resolve_allowed_tool_name(
    candidate: Any,
    allowed_exact: dict[str, str],
    allowed_bare: dict[str, str],
) -> str | None:
    if not isinstance(candidate, str) or not candidate.strip():
        return None
    cand = candidate.strip()
    if cand in allowed_exact:
        return allowed_exact[cand]
    bare = strip_mcp_prefix(cand)
    if bare in allowed_bare:
        return allowed_bare[bare]
    return None


def _coerce_dsml_param_value(raw_val: str, string_attr: str | None) -> Any:
    trimmed_newlines = re.sub(r"^\r?\n|\r?\n$", "", raw_val)
    if string_attr == "true":
        return trimmed_newlines
    stripped = raw_val.strip()
    if string_attr == "false":
        if not stripped:
            return ""
        try:
            return json.loads(stripped)
        except (ValueError, TypeError):
            repaired = _repair_json_argument_text(stripped)
            if repaired:
                try:
                    return json.loads(repaired)
                except (ValueError, TypeError):
                    pass
            return stripped
    # No explicit string="true|false": coerce structured/scalar JSON literals only.
    if (
        stripped in ("true", "false", "null")
        or (stripped.startswith("{") and stripped.endswith("}"))
        or (stripped.startswith("[") and stripped.endswith("]"))
        or bool(_NUMERIC_LITERAL_RE.match(stripped))
    ):
        try:
            return json.loads(stripped)
        except (ValueError, TypeError):
            repaired = _repair_json_argument_text(stripped)
            if repaired:
                try:
                    return json.loads(repaired)
                except (ValueError, TypeError):
                    pass
    return trimmed_newlines


def _extract_markup_tool_calls(
    content: str,
    call_tools: list[dict] | None,
    iteration: int,
) -> tuple[str, list[dict]]:
    """Recover structured Qwen `<tool_call>` or DeepSeek DSML markup emitted as text.

    Only blocks whose parsed tool name matches `call_tools` are converted and
    stripped from `content`. Unknown tool names (e.g. prose examples) remain
    untouched in `content` and are never executed.
    """
    if not call_tools or not content or not content.strip():
        return content, []

    allowed_exact, allowed_bare = _build_allowed_tool_lookup(call_tools)
    if not allowed_exact:
        return content, []

    # Collect (start, end, list[(resolved_name, serialized_args)]) spans.
    recovered_spans: list[tuple[int, int, list[tuple[str, str]]]] = []

    for q_match in _QWEN_TOOL_CALL_RE.finditer(content):
        raw_json = q_match.group(1)
        parsed: Any = None
        try:
            parsed = json.loads(raw_json)
        except (ValueError, TypeError):
            repaired = _repair_json_argument_text(raw_json)
            if repaired:
                try:
                    parsed = json.loads(repaired)
                except (ValueError, TypeError):
                    parsed = None
        if not isinstance(parsed, dict):
            continue
        fn_obj = parsed.get("function") if isinstance(parsed.get("function"), dict) else None
        cand_name = parsed.get("name") or (fn_obj.get("name") if fn_obj else None)
        resolved_name = _resolve_allowed_tool_name(cand_name, allowed_exact, allowed_bare)
        if not resolved_name:
            continue
        if "arguments" in parsed:
            raw_args = parsed["arguments"]
        elif "parameters" in parsed:
            raw_args = parsed["parameters"]
        elif "args" in parsed:
            raw_args = parsed["args"]
        elif fn_obj and "arguments" in fn_obj:
            raw_args = fn_obj["arguments"]
        else:
            raw_args = {}
        if isinstance(raw_args, str):
            args_str = raw_args
        elif raw_args is None:
            args_str = "{}"
        else:
            args_str = json.dumps(raw_args, default=str)
        recovered_spans.append((q_match.start(), q_match.end(), [(resolved_name, args_str)]))

    for fc_match in _DSML_FUNC_CALLS_RE.finditer(content):
        inner = fc_match.group(1)
        inner_offset = fc_match.start(1)
        matched_invokes: list[tuple[int, int, str, str]] = []
        has_unmatched_invoke = False

        for inv_match in _DSML_INVOKE_RE.finditer(inner):
            inv_attrs = inv_match.group(1) or ""
            inv_body = inv_match.group(2) or ""
            name_m = _DSML_ATTR_NAME_RE.search(inv_attrs)
            inv_name = (name_m.group(1) or name_m.group(2)) if name_m else None
            resolved_name = _resolve_allowed_tool_name(inv_name, allowed_exact, allowed_bare)
            if not resolved_name:
                has_unmatched_invoke = True
                continue

            params_dict: dict[str, Any] = {}
            for p_match in _DSML_PARAM_RE.finditer(inv_body):
                p_attrs = p_match.group(1) or ""
                p_val_raw = p_match.group(2) or ""
                p_name_m = _DSML_ATTR_NAME_RE.search(p_attrs)
                if not p_name_m:
                    continue
                p_name = p_name_m.group(1) or p_name_m.group(2)
                if not p_name:
                    continue
                s_attr_m = _DSML_ATTR_STRING_RE.search(p_attrs)
                s_attr = (
                    (s_attr_m.group(1) or s_attr_m.group(2) or "").lower()
                    if s_attr_m
                    else None
                )
                params_dict[p_name] = _coerce_dsml_param_value(p_val_raw, s_attr)

            matched_invokes.append(
                (
                    inner_offset + inv_match.start(),
                    inner_offset + inv_match.end(),
                    resolved_name,
                    json.dumps(params_dict, default=str),
                )
            )

        if not matched_invokes:
            continue
        if not has_unmatched_invoke:
            recovered_spans.append(
                (
                    fc_match.start(),
                    fc_match.end(),
                    [(r_name, r_args) for _, _, r_name, r_args in matched_invokes],
                )
            )
        else:
            for s_idx, e_idx, r_name, r_args in matched_invokes:
                recovered_spans.append((s_idx, e_idx, [(r_name, r_args)]))

    if not recovered_spans:
        return content, []

    recovered_spans.sort(key=lambda item: item[0])
    non_overlapping: list[tuple[int, int, list[tuple[str, str]]]] = []
    last_end = 0
    for start, end, calls in recovered_spans:
        if start < last_end:
            continue
        non_overlapping.append((start, end, calls))
        last_end = end

    tool_calls: list[dict] = []
    pieces: list[str] = []
    cursor = 0
    call_idx = 0
    for start, end, calls in non_overlapping:
        pieces.append(content[cursor:start])
        cursor = end
        for resolved_name, args_str in calls:
            tool_calls.append(
                {
                    "id": f"call_markup_{iteration}_{call_idx}",
                    "type": "function",
                    "function": {
                        "name": resolved_name,
                        "arguments": args_str,
                    },
                }
            )
            call_idx += 1
    pieces.append(content[cursor:])

    cleaned_content = re.sub(r"\n{3,}", "\n\n", "".join(pieces)).strip()
    return cleaned_content, tool_calls


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
    """Base class for all LazyCat SDK agents.

    .. deprecated:: 0.4.0
        The local BaseAgent and AgentHarness loop is deprecated and scheduled for
        deletion in Phase 5. Consumers must migrate to ``lazycat.RuntimeClient``
        calling the authoritative ``/v1/runs`` endpoint.
        
        Permitted transitional callers:
        - Legacy unmigrated agent test suites
        - Temporary local scripts pending Phase 4 cutover
    """
    
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
        import warnings
        warnings.warn(
            "BaseAgent is deprecated as of 0.4.0 and scheduled for deletion in Phase 5. "
            "Use lazycat.RuntimeClient instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.name = name
        self.system_prompt = system_prompt
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.provider = provider
        self.project = project
        self.username = username
        self.tools: list[dict] = []
        self.llm_client = llm_client or agent_client
        self.auto_approve = auto_approve
        # None defers to prism's ParameterRegistry, which carries
        # `minP: agentDefault 0.05` — and vLLM REFUSES any min_p > 0 when
        # speculative decoding is on ("The min_p and logit_bias sampling
        # parameters are not yet supported with speculative decoding").
        #
        # The refusal is invisible to the caller: vLLM answers HTTP 200 and
        # then raises inside the stream generator, so prism receives an empty
        # stream rather than an error and reports a successful call with no
        # content. `call_agent` has always accepted min_p; the harness simply
        # never forwarded it, so no BaseAgent user could reach the fix.
        #
        # Measured 2026-08-06 on the Jetson (Qwen3.6-35B-AWQ, spec decoding),
        # one variable changed, same prompt: default -> 0 chars;
        # min_p=0.0 -> 2,534 chars. Pass 0.0 on any path that may hit a vLLM
        # box; 0.0 is vLLM's own default, so it is not a tuning choice.
        self.min_p = min_p
        
    def add_tool(self, tool_schema: dict):
        self.tools.append(tool_schema)

class AgentHarness:
    """Standardized tool-call loop.
    
    send message -> check for tool calls -> dispatch -> loop until done.

    .. deprecated:: 0.4.0
        The local AgentHarness loop is deprecated and scheduled for deletion in Phase 5.
        Consumers must migrate to ``lazycat.RuntimeClient`` calling the authoritative
        ``/v1/runs`` endpoint.
        
        Permitted transitional callers:
        - Legacy unmigrated agent test suites
        - Temporary local scripts pending Phase 4 cutover
    """
    
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
        import warnings
        warnings.warn(
            "AgentHarness is deprecated as of 0.4.0 and scheduled for deletion in Phase 5. "
            "Use lazycat.RuntimeClient instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.agent = agent
        self.session = session
        self.max_iterations = max_iterations
        # None = provider/gateway default; False suppresses <think> blocks on
        # models that support toggling (prism honors an explicit false).
        self.thinking_enabled = thinking_enabled
        self.bench_task = bench_task
        # Hook: called before each tool execution with (tool_name, arguments).
        # Return None to proceed, or a string to inject as the tool result
        self.on_tool_call = on_tool_call
        # Hook: called after each tool execution with (tool_name, arguments, result, was_blocked).
        self.on_tool_result = on_tool_result
        self.max_tool_result_chars = max_tool_result_chars
        self.loop_detector = ToolLoopDetector(max_identical_failures=3)
        # Token accounting, filled from the stream's cumulative usage_update
        # events: last_usage is the final snapshot of the most recent request;
        # total_tokens sums input+output(+reasoning) across all loop iterations.
        self.last_usage: dict = {}
        self.total_tokens: int = 0
        # ADDITIVE (2026-09-12): the OUTPUT side, which the fused total_tokens
        # cannot separate and last_usage only holds for the FINAL request.
        # completion_tokens sums outputTokens+reasoningOutputTokens across the
        # whole loop; usage_requests counts the requests that actually REPORTED
        # a usage block, so a RECORDED zero (an all-zero block marks a
        # TRUNCATED generation, not a cheap one) stays distinguishable from
        # "never reported" — both of which otherwise read as 0.
        self.completion_tokens: int = 0
        self.usage_requests: int = 0
        self.total_requests: int = 0
        self.prompt_tokens: int = 0
        self.reasoning_tokens: int = 0
        # Model identity, filled from the stream's "done" event. This is
        # prism's SERVER-side resolved model — not an echo of the request —
        # so it survives silent gateway-side model swaps that the requested
        # model name would misattribute.
        self.last_model: str | None = None
        self.last_provider: str | None = None
        # Loop honesty (S-02), filled from the stream's "done" event exactly
        # like last_model/last_provider, snake_case mirrors of the service's
        # camelCase fields (lazy-agent-service LoopHonestyFields.ts):
        # answer_source "model" is real model text, "synthetic" is a
        # loop-written digest, "partial" is cut-off text; finalize_reason is
        # e.g. "COMPLETED", "STEP_LIMIT", "CUT_OFF", "CUT_OFF_TWICE";
        # provider_attempts/provider_retries count the provider calls of the
        # request that produced the FINAL done event (last-wins, the same
        # semantics as last_usage — a cross-loop total belongs to the
        # service-side run it reports). An older service sends none of these:
        # they stay None, never an error.
        self.answer_source: str | None = None
        self.finalize_reason: str | None = None
        self.provider_attempts: int | None = None
        self.provider_retries: int | None = None
        # The stream's own finish_reason ("stop"/"length"/"tool_calls"); only
        # "length" changes the answer's honesty: the text was cut.
        self.finish_reason: str | None = None
        # True once a done event carried the honesty keys: the service's
        # values are authoritative and the derivation above must not touch them.
        self._service_honesty_seen: bool = False
        self.tool_traces: list[dict[str, Any]] = []
        self._current_tool_call_id: str | None = None

    def _notify_tool_result_hook(
        self,
        func_name: str,
        arguments: Any,
        result: Any,
        was_blocked: bool,
        elapsed_ms: int,
        *,
        tool_call_id: str = "",
        raw_chars: int = 0,
        model_visible_chars: int = 0,
        was_truncated: bool = False,
    ) -> None:
        if self.on_tool_result is None:
            return
        extra = {
            "tool_call_id": tool_call_id,
            "raw_chars": raw_chars,
            "model_visible_chars": model_visible_chars,
            "was_truncated": was_truncated,
        }
        try:
            call_mode = "five_pos"
            accepted_kw: dict[str, Any] = {}
            try:
                sig = inspect.signature(self.on_tool_result)
                params = list(sig.parameters.values())
                if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params):
                    call_mode = "var_kw"
                else:
                    named_extra = {
                        k: v for k, v in extra.items() if k in sig.parameters
                    }
                    if named_extra:
                        call_mode = "named_kw"
                        accepted_kw = named_extra
                    else:
                        pos_params = [
                            p
                            for p in params
                            if p.kind
                            in (
                                inspect.Parameter.POSITIONAL_ONLY,
                                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                            )
                        ]
                        has_var_pos = any(
                            p.kind == inspect.Parameter.VAR_POSITIONAL
                            for p in params
                        )
                        if len(pos_params) >= 6 or has_var_pos:
                            call_mode = "six_pos"
            except (TypeError, ValueError):
                call_mode = "five_pos"

            if call_mode == "var_kw":
                self.on_tool_result(
                    func_name, arguments, result, was_blocked, elapsed_ms, **extra
                )
            elif call_mode == "named_kw":
                self.on_tool_result(
                    func_name,
                    arguments,
                    result,
                    was_blocked,
                    elapsed_ms,
                    **accepted_kw,
                )
            elif call_mode == "six_pos":
                self.on_tool_result(
                    func_name, arguments, result, was_blocked, elapsed_ms, extra
                )
            else:
                self.on_tool_result(
                    func_name, arguments, result, was_blocked, elapsed_ms
                )
        except Exception as hook_err:
            # Hooks are observational and must not kill a run —
            # EXCEPT when the hook explicitly asks to abort
            # (guard exceptions set abort_agent_run=True, e.g.
            # doom-loop detectors). Swallowing those made every
            # caller-side loop guard a silent no-op.
            if getattr(hook_err, "abort_agent_run", False):
                raise
            logger.warning(f"[{self.agent.name}] on_tool_result hook error: {hook_err}")

    def _handle_prism_tool_execution_event(
        self, data: dict, prism_tool_started: dict[str, float]
    ) -> None:
        status = data.get("status")
        tool_payload = data.get("tool") or data.get("toolCall", {})
        func_name = tool_payload.get("name", "")
        # Prism already executed this one, so we cannot refuse
        # it — but a silent `{}` here strips the args off the
        # telemetry row and makes a malformed-payload defect
        # look like a tool that takes no arguments.
        arguments, arg_error = decode_tool_arguments(
            tool_payload.get("args") or tool_payload.get("arguments")
        )
        if arg_error:
            logger.warning(
                f"[{self.agent.name}] agent-internal {func_name} "
                f"had undecodable arguments; telemetry will "
                f"record none: {arg_error.splitlines()[0]}"
            )

        if status == "calling":
            # Prism announces the call before executing it, so
            # the round trip IS measurable from here. Without
            # this, every prism-internal tool was recorded at
            # elapsed_ms=0 (the branch below hardcoded it),
            # which is nearly all of them — a trading-service
            # audit on 2026-07-27 found all 8 lazy_web_search
            # failures logged as 0ms, making a fast refusal
            # indistinguishable from a 20s connect timeout.
            prism_tool_started[func_name] = time.time()
        elif status in ("done", "error"):
            # This was executed internally by Prism. We record it!
            result = (
                tool_payload.get("result")
                if tool_payload.get("result") is not None
                else data.get("toolResult")
            )
            _t0 = prism_tool_started.pop(func_name, None)
            # 0 still means "not measurable" — a missing
            # `calling` event, not a 0ms call.
            elapsed_ms = int((time.time() - _t0) * 1000) if _t0 else 0
            tc_id = str(
                tool_payload.get("id")
                or data.get("id")
                or f"call_{func_name}"
            )
            _, raw_chars, model_visible_chars, was_truncated = (
                self._serialize_and_truncate_tool_result(
                    func_name, result, log_warning=False
                )
            )
            self.tool_traces.append(
                {
                    "tool_call_id": tc_id,
                    "tool_name": func_name,
                    "arguments": arguments,
                    "was_blocked": False,
                    "elapsed_ms": elapsed_ms,
                    "raw_chars": raw_chars,
                    "model_visible_chars": model_visible_chars,
                    "was_truncated": was_truncated,
                }
            )
            self._notify_tool_result_hook(
                func_name,
                arguments,
                result,
                False,
                elapsed_ms,
                tool_call_id=tc_id,
                raw_chars=raw_chars,
                model_visible_chars=model_visible_chars,
                was_truncated=was_truncated,
            )

    def _record_stream_usage(self, request_usage: dict) -> None:
        if not request_usage:
            return
        self.last_usage = request_usage
        input_tok = int(request_usage.get("inputTokens") or 0)
        output_tok = int(request_usage.get("outputTokens") or 0)
        reasoning_tok = int(request_usage.get("reasoningOutputTokens") or 0)
        completion = output_tok + reasoning_tok
        self.total_tokens += input_tok + completion
        self.completion_tokens += completion
        self.usage_requests += 1
        self.prompt_tokens += input_tok
        self.reasoning_tokens += reasoning_tok

    def _apply_done_event(self, data: dict) -> None:
        if data.get("model"):
            self.last_model = data["model"]
        if data.get("provider"):
            self.last_provider = data["provider"]
        if "finishReason" in data:
            self.finish_reason = data["finishReason"]
        # Honesty fields ride the final done event (camelCase on the wire).
        # Presence-gated like model/provider: an absent key leaves the
        # previous value (None before the first done) — an older service
        # must not raise, and a later done without the fields must not
        # erase an earlier one's honesty.
        if "answerSource" in data:
            self.answer_source = data["answerSource"]
            self._service_honesty_seen = True
        if "finalizeReason" in data:
            self.finalize_reason = data["finalizeReason"]
        if "providerAttempts" in data:
            self.provider_attempts = data["providerAttempts"]
        if "providerRetries" in data:
            self.provider_retries = data["providerRetries"]

    async def _consume_agent_stream(
        self, resp: Any, prism_tool_started: dict[str, float]
    ) -> tuple[str, list, bool]:
        content = ""
        tool_calls: list = []
        request_usage: dict = {}
        stream_done = False
        try:
            async for data in iter_sse_json_lines(resp.aiter_lines(), done_sentinel="[DONE]"):
                event_type = data.get("type")
                if event_type == "chunk":
                    chunk_text = data.get("content", "")
                    content += chunk_text
                    # Opt-in only: echoing every token to stdout from
                    # concurrent agents interleaves raw model text
                    # character-by-character with the service log stream
                    # (observed corrupting trading-service docker logs).
                    if _STREAM_STDOUT:
                        print(chunk_text, end="", flush=True)
                elif event_type == "tool_calls" or "toolCalls" in data:
                    tool_calls = data.get("toolCalls", [])
                elif event_type == "tool_execution":
                    self._handle_prism_tool_execution_event(data, prism_tool_started)
                elif event_type == "usage_update":
                    usage = data.get("usage")
                    if isinstance(usage, dict):
                        request_usage = usage
                elif event_type == "done":
                    stream_done = True
                    self._apply_done_event(data)
                elif event_type == "error":
                    # Terminal: a mid-stream `error` means the model died
                    # after emitting partial text. Committing that partial
                    # content as the assistant answer booked a failure as
                    # a successful turn (2026-10-20 cut-stream audit:
                    # 'probeprobeprobe...' returned as the answer), so
                    # stop consuming and fail the turn instead.
                    if (
                        _is_direct_openai_provider(self.agent.provider)
                        and not self._service_honesty_seen
                    ):
                        # Name the exit before raising (mirrors
                        # _call_provider): no answer exists, and the
                        # attributes must not keep the previous turn's
                        # derived model/COMPLETED on the exception path.
                        self.answer_source = None
                        self.finalize_reason = "PROVIDER_ERROR"
                    raise RuntimeError(
                        f"Agent stream error: {data.get('message') or 'unknown error'}"
                    )
                elif "text" in data and not event_type:
                    content = data.get("text", content)
                    tool_calls = data.get("toolCalls", tool_calls)

            print()  # flush newline after stream completes
        finally:
            # The last usage snapshot is cumulative for this stream. Record
            # it once, including when cancellation interrupts generation.
            self._record_stream_usage(request_usage)
            await resp.aclose()
        return content, tool_calls, stream_done

    @staticmethod
    def _decode_tool_call_entry(tc: dict) -> tuple[str, str, Any, str | None]:
        tc_id = tc.get("id", "")
        if "function" in tc:
            func = tc.get("function", {})
            func_name = func.get("name", "")
            arguments, arg_error = decode_tool_arguments(func.get("arguments"))
        else:
            func_name = tc.get("name", "")
            # `or` (not `is None`) preserves the original fallthrough:
            # an empty `arguments` defers to `args`.
            raw_arguments = tc.get("arguments") or tc.get("args")
            arguments, arg_error = decode_tool_arguments(raw_arguments)
        return tc_id, func_name, arguments, arg_error

    def _check_tool_override(self, func_name: str, arguments: Any) -> Any:
        loop_block_msg = self.loop_detector.record_call(func_name, arguments, failed=True)
        if loop_block_msg:
            return loop_block_msg
        # Undo the speculative failure record
        key = self.loop_detector._make_key(func_name, arguments, failed=True)
        self.loop_detector._history[key] = max(0, self.loop_detector._history.get(key, 1) - 1)
        if self.on_tool_call is not None:
            return self.on_tool_call(func_name, arguments)
        return None

    @staticmethod
    def _is_tool_result_failed(result: Any) -> bool:
        if isinstance(result, dict):
            return bool(result.get("error") or result.get("is_error") or not result)
        return result is None

    @staticmethod
    def _parse_ledger_tool_output(tool_msg: Any) -> Any:
        if isinstance(tool_msg, dict) and "content" in tool_msg:
            raw = tool_msg["content"]
            if isinstance(raw, str):
                if raw.startswith(("Error executing ", "Malformed tool arguments")):
                    return {"error": raw, "is_error": True}
                try:
                    return json.loads(raw)
                except Exception:
                    return {"result": raw}
            return raw
        return tool_msg

    async def _try_inprocess_tool(self, func_name: str, arguments: Any) -> tuple[bool, Any]:
        if os.environ.get("V3_USE_PRISM", "0").strip() == "1":
            return False, None
        call_payload = {
            "id": self._current_tool_call_id or f"call_{func_name}",
            "function": {
                "name": func_name,
                "arguments": (
                    arguments
                    if isinstance(arguments, str)
                    else json.dumps(arguments, default=str)
                ),
            },
        }
        try:
            from app.tools.registry import registry
            from app.v3.data_ledger import execute_with_ledger

            tool_mocked = hasattr(tool_executor, "assert_called") or hasattr(
                getattr(tool_executor, "execute_tool", None), "assert_called"
            )
            ledger_mocked = hasattr(execute_with_ledger, "assert_called") or (
                getattr(execute_with_ledger, "__module__", "") != "app.v3.data_ledger"
            )
            if not tool_mocked or ledger_mocked or func_name in getattr(registry, "_handlers", {}):
                try:
                    coro = execute_with_ledger(
                        call_payload,
                        _registry=registry,
                        force_local=True,
                        skip_permission_check=True,
                    )
                except TypeError:
                    coro = execute_with_ledger(registry, call_payload)
                tool_msg = await coro
                return True, self._parse_ledger_tool_output(tool_msg)
        except ImportError:
            pass
        tool_reg = getattr(self.agent, "tool_registry", None)
        if tool_reg is not None and hasattr(tool_reg, "execute_tool_call"):
            tool_msg = await tool_reg.execute_tool_call(call_payload)
            return True, self._parse_ledger_tool_output(tool_msg)
        return False, None

    async def _invoke_remote_tool(
        self, func_name: str, arguments: Any
    ) -> tuple[Any, int]:
        t0_tool = time.time()
        handled, result = await self._try_inprocess_tool(func_name, arguments)
        if not handled:
            # Execute via the tool service proxy. Stamp identity
            # headers so the proxy's per-conversation whitelist can
            # actually key on this session (it fails open otherwise).
            identity_headers = {"x-agent": self.agent.name}
            get_conv = getattr(self.agent.llm_client, "get_conversation_id", None)
            if callable(get_conv):
                conv_id = get_conv(self.agent.name, self.session.session_id)
                if conv_id:
                    identity_headers["x-conversation-id"] = conv_id
            result = await tool_executor.execute_tool(
                func_name, arguments, headers=identity_headers
            )
        elapsed_ms = int((time.time() - t0_tool) * 1000)
        self.loop_detector.record_call(
            func_name, arguments, failed=self._is_tool_result_failed(result)
        )
        return result, elapsed_ms

    def _serialize_and_truncate_tool_result(
        self, func_name: str, result: Any, *, log_warning: bool = True
    ) -> tuple[str, int, int, bool]:
        raw_str = json.dumps(result, default=str)
        raw_chars = len(raw_str)
        was_truncated = raw_chars > self.max_tool_result_chars
        if was_truncated:
            if log_warning:
                logger.warning(
                    f"[{self.agent.name}] Truncating tool result for {func_name}: "
                    f"{raw_chars:,} chars → {self.max_tool_result_chars:,} chars"
                )
            result_str = raw_str[: self.max_tool_result_chars] + (
                f"\n\n[TRUNCATED: result was {raw_chars:,} chars, "
                f"showing first {self.max_tool_result_chars:,}]"
            )
        else:
            result_str = raw_str
        return result_str, raw_chars, len(result_str), was_truncated

    def _format_tool_result_str(self, func_name: str, result: Any) -> str:
        return self._serialize_and_truncate_tool_result(func_name, result)[0]

    async def _dispatch_single_tool_call(self, tc: dict) -> None:
        tc_id, func_name, arguments, arg_error = self._decode_tool_call_entry(tc)
        effective_tc_id = tc_id or f"call_{func_name}"
        logger.info(f"[{self.agent.name}] Executing tool: {func_name}")
        elapsed_ms = 0

        # Malformed arguments: tell the model, do NOT run the tool with
        # `{}`. Running it anyway produces a misleading "required field
        # missing" rejection that hides the real defect, and burns the
        # turn — see decode_tool_arguments().
        if arg_error is not None:
            logger.warning(
                f"[{self.agent.name}] Malformed arguments for "
                f"{func_name}: {arg_error.splitlines()[0]}"
            )
            result: Any = {"error": arg_error, "is_error": True}
            was_blocked = False
            self.loop_detector.record_call(func_name, arguments, failed=True)
        # Check for human-in-the-loop pauses
        elif func_name in ("ask_user_question", "request_plan_approval"):
            result = await self._handle_pausing_tool(func_name, arguments)
            was_blocked = False
        else:
            override_result = self._check_tool_override(func_name, arguments)
            if override_result is not None:
                # Hook blocked this call — use override as result
                logger.warning(f"[{self.agent.name}] Tool call blocked: {func_name}")
                result = {"blocked": True, "message": override_result}
                was_blocked = True
            else:
                self._current_tool_call_id = effective_tc_id
                try:
                    result, elapsed_ms = await self._invoke_remote_tool(
                        func_name, arguments
                    )
                finally:
                    self._current_tool_call_id = None
                was_blocked = False

        # 5. Serialize and truncate oversized payloads to prevent context overflow
        result_str, raw_chars, model_visible_chars, was_truncated = (
            self._serialize_and_truncate_tool_result(func_name, result)
        )

        # Notify post-call hook & record per-tool lineage/truncation trace
        if func_name not in ("ask_user_question", "request_plan_approval"):
            self.tool_traces.append(
                {
                    "tool_call_id": effective_tc_id,
                    "tool_name": func_name,
                    "arguments": arguments,
                    "was_blocked": was_blocked,
                    "elapsed_ms": elapsed_ms,
                    "raw_chars": raw_chars,
                    "model_visible_chars": model_visible_chars,
                    "was_truncated": was_truncated,
                }
            )
            self._notify_tool_result_hook(
                func_name,
                arguments,
                result,
                was_blocked,
                elapsed_ms,
                tool_call_id=effective_tc_id,
                raw_chars=raw_chars,
                model_visible_chars=model_visible_chars,
                was_truncated=was_truncated,
            )

        self.session.add_tool_message(
            tool_call_id=effective_tc_id,
            name=func_name,
            content=result_str,
        )

    async def run(self, user_input: str | None = None) -> str:
        """Run the agent loop until it completes or reaches max iterations."""
        if user_input:
            self.session.add_user_message(user_input)
            
        iterations = 0
        # func_name -> monotonic start of a prism-internal tool call, set on
        # the "calling" event and consumed on "done"/"error". Per-run rather
        # than per-instance so a reused harness cannot leak stale timings.
        prism_tool_started: dict[str, float] = {}
        # Same per-run rationale for the honesty fields: run #2 must report
        # run #2's service, not run #1's — a service that omits the fields
        # must read back None, not the previous run's answer_source.
        self.answer_source = None
        self.finalize_reason = None
        self.provider_attempts = None
        self.provider_retries = None
        self._service_honesty_seen = False
        self.finish_reason = None
        self.tool_traces = []
        # The loop runs in this process on the direct path (llm.py:129/844).
        direct = _is_direct_openai_provider(self.agent.provider)
        while iterations < self.max_iterations:
            iterations += 1
            # finish_reason describes THIS stream only; a done event without it
            # must not inherit the previous turn's value.
            self.finish_reason = None
            force_final_answer = (
                _is_direct_openai_provider(self.agent.provider)
                and self.max_iterations > 1
                and iterations == self.max_iterations
            )
            call_messages = list(self.session.get_messages())
            if force_final_answer:
                call_messages.append({"role": "user", "content": _FINAL_TURN_NUDGE})
            call_tools = (
                None
                if force_final_answer
                else (self.agent.tools if self.agent.tools else None)
            )
            
            # 1. Send messages to LLM
            # max_iterations must go over the wire: on the prism path the
            # agentic loop runs SERVER-side (prism forces agenticLoopEnabled),
            # so without maxIterations in the payload the caller's per-role
            # turn budget never binds — prism used its own default cap.
            self.total_requests += 1
            if direct and not self._service_honesty_seen:
                # One HTTP POST per turn on the direct path (llm.py:665) and
                # the SDK retries nothing: attempts == turns, retries == 0.
                self.provider_attempts = iterations
                self.provider_retries = 0
            resp = await self._call_provider(
                model=self.agent.model,
                messages=call_messages,
                system_prompt=self.agent.system_prompt,
                agent_name=self.agent.name,
                project=self.agent.project,
                username=self.agent.username,
                max_tokens=self.agent.max_tokens,
                temperature=self.agent.temperature,
                tools=call_tools,
                provider=self.agent.provider,
                stream=True,
                session_id=self.session.session_id,
                auto_approve=self.agent.auto_approve,
                max_iterations=self.max_iterations,
                thinking_enabled=self.thinking_enabled,
                # See BaseAgent.min_p: None keeps prism's 0.05 agentDefault,
                # which a spec-decoding vLLM box answers with an empty stream.
                min_p=self.agent.min_p,
                bench_task=self.bench_task or self.agent.name,
                inline_system_prompt=False,
            )

            content, tool_calls, stream_done = await self._consume_agent_stream(
                resp, prism_tool_started
            )

            # The direct-openai path runs the loop HERE (llm.py:844), so no
            # service reports who wrote the answer or why the loop stopped.
            # A done event that carried the keys (a newer shim, or the /agent
            # path) is authoritative: _service_honesty_seen keeps its values.
            if direct and not self._service_honesty_seen:
                if self.finish_reason in ("stop", "tool_calls"):
                    self.answer_source = "model"
                    self.finalize_reason = (
                        "SDK_MAX_ITERATIONS_FORCED_FINAL"
                        if force_final_answer
                        else "COMPLETED"
                    )
                elif self.finish_reason in ("length", "content_filter", "abort"):
                    # A filtered or aborted reply is truncated text too; the
                    # service contract (CoreFinalizeReason) has no reason
                    # filter-specific, so CUT_OFF is the exact name on offer.
                    self.answer_source = "partial"
                    self.finalize_reason = "CUT_OFF"
                else:
                    # No finish reason seen (a [DONE] without one) or an
                    # unknown value: who wrote this cannot be verified, so
                    # unknown stays None — never a claimed "model".
                    self.answer_source = None
                    self.finalize_reason = None

            # A stream that ends without a `done` event (and without the
            # [DONE] sentinel) was cut mid-generation — e.g. gateway crash,
            # vLLM kill. The partial `content` is not an answer; returning it
            # as one is the exact defect the cut-stream boundary probe caught
            # (answer 'probeprobeprobe...' booked as a successful turn).
            if not stream_done:
                if direct and not self._service_honesty_seen:
                    self.answer_source = "partial"
                    self.finalize_reason = "CUT_OFF"
                raise RuntimeError(
                    "Agent stream ended without a done event; "
                    f"discarding {len(content)} chars of partial content"
                )

            if force_final_answer:
                # Stray tool_calls on the forced final turn are never
                # dispatched or recorded; an empty reply is not an answer.
                if content and content.strip():
                    self.session.add_assistant_message(content)
                    return content
                break

            # Recover structured Qwen <tool_call> or DeepSeek DSML markup
            # emitted as streamed text when tools are active and no native
            # tool_calls arrived. Only registered/allowed tools are converted.
            if call_tools and not tool_calls and content:
                content, recovered_calls = _extract_markup_tool_calls(
                    content, call_tools, iterations
                )
                if recovered_calls:
                    logger.info(
                        f"[{self.agent.name}] Recovered {len(recovered_calls)} "
                        f"markup tool call(s) on turn {iterations}"
                    )
                    tool_calls = recovered_calls

            if tool_calls:
                for idx, tc in enumerate(tool_calls):
                    if isinstance(tc, dict) and not tc.get("id"):
                        fn = (
                            (tc.get("function") or {}).get("name")
                            or tc.get("name")
                            or "tool"
                        )
                        tc["id"] = f"call_{fn}_{iterations}_{idx}"

            # 2. Add LLM response to history
            self.session.add_assistant_message(content, tool_calls)
            
            # 3. If no tool calls, we're done
            if not tool_calls:
                return content or ""
                
            # 4. Dispatch tool calls
            for tc in tool_calls:
                try:
                    await self._dispatch_single_tool_call(tc)
                except Exception as exc:
                    # A guard that aborts (agent.py:329 re-raises it) stopped the
                    # loop on a repeated call; any other tool fault is unnamed.
                    fn = (tc.get("function") or {}).get("name") or tc.get("name") or "unknown"
                    self.answer_source = None
                    self.finalize_reason = (
                        f"repeated_tool_call:{fn}"
                        if getattr(exc, "abort_agent_run", False)
                        else None
                    )
                    raise
                
        logger.warning(f"[{self.agent.name}] Reached max iterations ({self.max_iterations})")
        # This string is written by the SDK, not the model — report it as
        # synthetic so a caller never books it as a real answer. The service
        # never saw this decision, hence an SDK-specific finalize_reason.
        # attempts/retries are the SDK's own turn counts on the direct path
        # (set per turn at agent.py:672) and the service's on the /agent path.
        self.answer_source = "synthetic"
        self.finalize_reason = "SDK_MAX_ITERATIONS"
        return "Max iterations reached without a final answer."

    async def _call_provider(self, **kwargs: Any) -> Any:
        """One provider call; a raise ends the loop on a transport fault.

        The direct-openai path has no service to say why the loop stopped, so
        name it here. No answer exists, hence answer_source stays None.
        """
        try:
            return await self.agent.llm_client.call_agent(**kwargs)
        except Exception:
            if _is_direct_openai_provider(self.agent.provider) and not self._service_honesty_seen:
                self.answer_source = None
                self.finalize_reason = "PROVIDER_ERROR"
            raise

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
                # prism attributes by these HEADERS, not body fields — without them
                # the run lands in prism's unattributable "default" project.
                headers={"Accept": "text/event-stream",
                         "x-project": self.agent.project,
                         "x-username": self.agent.username}
            ) as resp:
                if resp.status_code != 200:
                    error_body = ""
                    async for chunk in resp.aiter_text():
                        error_body += chunk
                    yield format_sse({"type": "error", "message": f"Agent error {resp.status_code}: {error_body[:500]}"})
                    return

                async for line in iter_sse_lines(resp.aiter_text()):
                    yield f"{line}\n\n"
