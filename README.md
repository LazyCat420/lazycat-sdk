# lazycat-sdk

## Agent client & direct `/vllm-shim` harness streaming (updated 2026-11-13)

`lazycat.agent_client` (an `AgentClient`) talks only to lazy-agent-service (`LAZY_AGENT_URL`,
default `http://10.0.0.16:5591`; set `agent_client.url = ...` to override per process). The old
`prism_client` / `PrismClient` names, `PRISM_URL`, `PRISM_ENABLED`, the `:7777` default, and the
`/prism-proxy` path were removed; model-to-provider resolution reads `{LAZY_AGENT_URL}/config`.

`AgentHarness` (`lazycat/agent.py`) and `AgentClient` (`lazycat/llm.py`) support **in-process
direct OpenAI-compatible streaming** over lazy-agent-service's
`/vllm-shim/<box>/v1/chat/completions` (`b0757c8` + `a8ab691`, active for local `vllm*`,
`dgx-spark`, `gold-spark`, `jetson`, `qwen*`, `dell1`, `dell2`, and `local*` providers when
`V3_USE_PRISM != "1"`) alongside the gateway `/agent` path:

- **Loop honesty metadata (`b0757c8`, `a8ab691`):** every `AgentHarness.run()` tracks
  `answer_source` (`"model"`, `"partial"`, `"synthetic"`, or `None` when `finish_reason` is
  absent/unknown — never fabricated `"model"`), `finalize_reason` (`"COMPLETED"`,
  `"SDK_MAX_ITERATIONS_FORCED_FINAL"` on the final iteration when tools are suppressed and
  `_FINAL_TURN_NUDGE` forces synthesis, `"SDK_MAX_ITERATIONS"`, `"CUT_OFF"`, `"PROVIDER_ERROR"`,
  `"repeated_tool_call:<fn>"`, or `None`), `provider_attempts`, and `provider_retries`.
- **Streamed markup tool-call recovery (`a8ab691`):** `_extract_markup_tool_calls` recovers Qwen
  `<tool_call>{...}</tool_call>` and DeepSeek DSML (`<｜DSML｜function_calls>` /
  `<｜DSML｜invoke>` / `<｜DSML｜parameter>`) blocks emitted in streamed text when tools are
  active and no native `tool_calls` arrived, **strictly restricted to allowed tools** registered
  in `call_tools` (unknown tool names in prose are never executed).
- **Lightweight JSON tool-argument repair (`a8ab691`):** `decode_tool_arguments` applies
  deterministic `_repair_json_argument_text` cleanup (stripping surrounding ```` ```json ````
  fences and trailing commas before `}` or `]`) without inventing keys or substituting `{}` on
  malformed payloads.
- **Per-tool `tool_call_id` & truncation lineage (`b0757c8`, `a8ab691`):** `harness.tool_traces`
  and `on_tool_result` record `tool_call_id`, `raw_chars`, `model_visible_chars`, and
  `was_truncated` across both local tool dispatch and server-side SSE tool events.
- **Zero hardcoded cloud model defaults (`a8ab691`):** `IntentRouter.classify` (`lazycat/router.py`)
  and `Validator.self_review_html` (`lazycat/validators.py`) default to `model=""` /
  `provider="vllm"`, with no hardcoded cloud model names (`gpt-4o-mini` removed).

## grounded_research: where results come from (2026-10-06)

`lazycat.grounded_research` gathers sources from three places:

- **News:** lazy-agent-service `news_search` (keyed news APIs), `POST {LAZY_AGENT_SERVICE_URL}/execute/news_search`.
- **Web:** lazy-agent-service `web_search` (Exa's keyless index), `POST {LAZY_AGENT_SERVICE_URL}/execute/web_search`. It has one cache and one rate limit for every project on the network.
- **Finnhub,** when a ticker is given.

`LAZY_AGENT_SERVICE_URL` defaults to `http://10.0.0.16:5591`.

It used to call the `ddgs` library for news and web results. `ddgs` scrapes DuckDuckGo, and on a blocked IP tries about nine engines for every call. The free search engines bot-block this network's one public IP (lazy-agent-service `documentation/chapters/04-shared-web-search.md`), so every one of those calls made the block worse for every project. `ddgs` is no longer a dependency.

A search that is refused (`rate_limited`, `busy`) or cannot be reached contributes no articles; it never raises. `tests/test_grounded_research_sources.py` covers:

- both request URLs and bodies;
- the mapping to `Article`, including RFC 2822 news dates;
- refused and failed searches;
- a check on the module's imports that it does not import `ddgs`.


## `/execute` API key (2026-10-11)

lazy-agent-service now enforces `EXECUTE_API_KEY` on `POST /execute/*`. This repo sends it as
`x-api-key` when the env var is set (it comes from the deployed `.env`); without it the shared
search answers 401. Rollout record, live checks and how to rotate the key:
lazy-agent-service `documentation/chapters/11-security-hardening-plan-2026-10-11.md` (Item 3).
