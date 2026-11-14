---
part: Reference
status: reference
updated: 2026-11-13
---

# `AgentHarness` streaming, honesty metadata, and stream test fakes

`AgentHarness.run` (`lazycat/agent.py`) drives two streaming execution paths against
lazy-agent-service (`LAZY_AGENT_URL`, default `http://10.0.0.16:5591`):

1. **In-process direct OpenAI-compatible streaming (`b0757c8` + `a8ab691`):** when the resolved
   provider is a local vLLM endpoint (`vllm*`, `dgx-spark`, `gold-spark`, `jetson`, `qwen*`,
   `dell1`, `dell2`, `local*`) and `V3_USE_PRISM != "1"`, `AgentClient.call_agent(stream=True)`
   streams SSE directly over `/vllm-shim/<box>/v1/chat/completions` via `LLMStreamWrapper`,
   converting OpenAI chunk deltas (`content`, `reasoning` / `reasoning_content`, `tool_calls`,
   `finish_reason`) into canonical harness events and running the multi-turn tool loop inside
   `AgentHarness.run`.
2. **Gateway `/agent` SSE streaming:** for gateway-routed calls, `AgentHarness.run` consumes
   lazy-agent-service's `/agent` SSE stream (`delta` / `chunk`, `thinking`, `tool_execution`,
   `done`, `error`).

## Stream completion, honesty metadata & tool recovery invariants

- **Mandatory `done` frame:** a normal stream completion ends with a `{"type": "done"}` event
  before `[DONE]`. The `[DONE]` sentinel only ends SSE line iteration; a stream that ends without
  `done` stamps `answer_source = "partial"`, `finalize_reason = "CUT_OFF"`, and raises
  `RuntimeError("Agent stream ended without a done event; ...")` instead of returning partial text.
  Mid-stream `{"type": "error"}` events stamp `finalize_reason = "PROVIDER_ERROR"` and raise
  `RuntimeError`. Every fake stream in tests must emit a `done` event before `[DONE]`.
- **Loop honesty fields (`b0757c8`, `a8ab691`):** `AgentHarness` exposes `answer_source`
  (`"model"`, `"partial"`, `"synthetic"`, or `None`), `finalize_reason` (`"COMPLETED"`,
  `"SDK_MAX_ITERATIONS_FORCED_FINAL"`, `"SDK_MAX_ITERATIONS"`, `"CUT_OFF"`, `"PROVIDER_ERROR"`,
  `"repeated_tool_call:<fn>"`, or `None`), `provider_attempts`, and `provider_retries`. On the
  final turn (`iterations == max_iterations` with `max_iterations > 1`), direct execution appends
  `_FINAL_TURN_NUDGE` with `call_tools=None` and stamps `SDK_MAX_ITERATIONS_FORCED_FINAL` when the
  forced synthesis finishes cleanly. Missing or unrecognized `finish_reason` values leave
  `answer_source = None` and `finalize_reason = None` (never fabricated `"model"`).
- **Allowed-only markup tool-call recovery (`a8ab691`):** `_extract_markup_tool_calls` recovers
  Qwen `<tool_call>{...}</tool_call>` and DeepSeek DSML (`<｜DSML｜function_calls>` /
  `<｜DSML｜invoke>` / `<｜DSML｜parameter>`) markup from streamed text when tools are active on
  that turn and no native `tool_calls` arrived, **restricted strictly to tool names present in
  `call_tools`**.
- **Deterministic JSON argument repair & tool lineage (`b0757c8`, `a8ab691`):**
  `decode_tool_arguments` strips markdown ```` ```json ```` fences and trailing commas before `}`
  or `]` (`_repair_json_argument_text`) without inventing keys or substituting `{}` on broken JSON.
  Every dispatched or SSE-reported tool call records `tool_call_id`, `raw_chars`,
  `model_visible_chars`, and `was_truncated` in `harness.tool_traces` and forwards those fields to
  `on_tool_result` callbacks that accept them.
- **Zero hardcoded cloud model defaults (`a8ab691`):** `IntentRouter.classify` (`lazycat/router.py`)
  and `Validator.self_review_html` (`lazycat/validators.py`) default to `model=""` (with
  `provider="vllm"` on `IntentRouter`), removing legacy `"gpt-4o-mini"` defaults.

## Resolved tickets

| Ticket | Root cause | Fix |
|---|---|---|
| 8 failing tests: `test_min_p_passthrough` (3), `test_agent_tool_timing` (3), `test_malformed_tool_arguments` (1), `test_stream_and_budget_fixes` (1) | `fb6edc0` (2026-10-20) made a stream without a `done` event raise; these fakes ended without one | `f802263` adds the `done` event to each fake; assertions unchanged. Cut-stream raising is covered in `tests/test_agent.py`. |
| Loop honesty & forced-final synthesis (`S-01` / `T6`) | Exhausted tool loops or unknown `finish_reason` values could misreport completion source, and tool truncation lineage lacked `tool_call_id` / char counts | `b0757c8` + `a8ab691` add `answer_source`, `finalize_reason` (`SDK_MAX_ITERATIONS_FORCED_FINAL`), `provider_attempts`, `provider_retries`, markup recovery gated to allowed tools, lightweight JSON repair, per-tool truncation lineage, and zero cloud model defaults (`tests/test_agent_honesty_and_markup.py`). |
