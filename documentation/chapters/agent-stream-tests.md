---
part: Reference
status: reference
updated: 2026-10-23
---

# Agent stream test fakes

`AgentHarness.run` (`lazycat/agent.py`) consumes prism's SSE stream. A normal
completion ends with a `{"type": "done"}` event. The `[DONE]` sentinel only ends
iteration; it does not mark the stream complete. A stream that ends without a
`done` event is treated as cut mid-generation and `run()` raises
`RuntimeError("Agent stream ended without a done event; ...")` instead of
returning the partial content. Every fake stream in the tests must therefore
send a `done` event before `[DONE]`.

## Resolved tickets

| Ticket | Root cause | Fix |
|---|---|---|
| 8 failing tests: `test_min_p_passthrough` (3), `test_agent_tool_timing` (3), `test_malformed_tool_arguments` (1), `test_stream_and_budget_fixes` (1) | `fb6edc0` (2026-10-20) made a stream without a `done` event raise; these fakes ended without one | `f802263` adds the `done` event to each fake; assertions unchanged. No test expected a cut stream to return content, so none flipped to "raises" (that case is covered in `tests/test_agent.py`). |
