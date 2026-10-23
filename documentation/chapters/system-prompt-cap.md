# The `systemPrompt` cap: cut loudly, never silently

**Status 2026-10-23.**
- `AgentClient.call_agent` and `AgentClient.get_stream_payload_and_url` still cap the `systemPrompt` field at 15,000 characters by default.
- A cut is no longer silent:
  - The SDK logs a WARNING once per agent name per process, with the prompt's length, the cap and the number of characters dropped.
  - Every cut adds a `lazycat.system_prompt_truncated` event to the caller's current OpenTelemetry span, when the caller runs OpenTelemetry. The SDK does not depend on it.
- `LAZYCAT_SYSTEM_PROMPT_MAX_CHARS` overrides the cap for a process; a value of 0 or less disables it.

## Why

- The cap (`system_prompt[:15000]`) has been in the SDK since its first commit. lazy-agent-service accepts a `systemPrompt` of any length (`z.string()`), so nothing downstream could see a cut.
- On 2026-10-22 the trading Board's system prompt was 19,598 characters. The cut dropped its last 4,598 characters on every call, and those characters were every output example in the Board's decision rules. 12 of 13 first answers failed the schema.
- The trading tests asserted on the prompt handed to the model call. That prompt was whole: the cut happened afterwards, inside this SDK.
- Findings and plan: trading-client `documentation/chapters/47-week-audit-2026-10-22.md`, plan item 1.1.

## What to do when you see the warning

- **Preferred:** shorten the system prompt, or move rules that must be followed to the letter into the user message. The trading Board's fix moved its output rules into the user prompt.
- **Otherwise:** raise the cap for that service with `LAZYCAT_SYSTEM_PROMPT_MAX_CHARS`. The cap protects nothing downstream today. It is kept as the default so services that never set it keep their current behaviour.

## Tests

`tests/test_system_prompt_cap.py` covers:
- the payload actually posted (`httpx.AsyncClient.post` patched): whole at the cap, cut above it, warned once per agent;
- the same cap on the streaming payload builder;
- the disable switch (0 or less);
- the span event, which is skipped where `opentelemetry` is not installed.
