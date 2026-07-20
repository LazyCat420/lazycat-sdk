# HANDOFF — lazycat-sdk v0.3.0 (2026-07-20)

## What shipped

Five pieces of generic infrastructure that other services had each
reimplemented now live in the SDK, plus a consolidation of SSE parsing that
was duplicated four times *inside this package*.

### New modules

| Module | Public surface | Extracted from |
|---|---|---|
| `lazycat/llm_json.py` | `parse_json_response`, `parse_json_strict`, `parse_json_list_response`, `extract_json_str`, `strip_think_tags` | trading-service `app/utils/text_utils.py` + music-player `_strip_markdown_fences` |
| `lazycat/resilience.py` | `aresilient_call`, `resilient_call`, `classify_exception`, `FailureType`, `AttemptRecord`, `ResilientCallError`, `set_failure_emitter`, `NON_RETRYABLE_EXCEPTION_NAMES` | trading-service `app/utils/resilience.py` |
| `lazycat/cache.py` | `timed_cache`, `invalidate_cache`, `get_cache_stats`, `clear_cache` | trading-service `app/cache.py` |
| `lazycat/ratelimit.py` | `KeyedRateLimiter`, `KeyedSemaphore` | trading-service scraper `rate_limiter.py` + `api_rate_limiter.py` |
| `lazycat/sse.py` | `iter_sse_lines`, `iter_sse_json`, `iter_sse_json_lines`, `format_sse` | the four copy-pasted loops in this package |

`llm_json` helpers are exported at package level. The cross-cutting four stay
behind explicit submodule imports (`from lazycat.resilience import ...`).

### Design rule applied throughout

**No domain knowledge in the SDK.** Where the original code had trading
specifics baked in, the SDK takes a hook instead:

- `parse_json_response(text, *, reject=..., fallback=...)` — the caller
  supplies placeholder-ticker rejection and its prose fallback.
- `resilience` — `NON_RETRYABLE_EXCEPTION_NAMES` (by class *name*, so the SDK
  never imports app exceptions) and `set_failure_emitter()`.
- `ratelimit` — rate/limit tables stay with the application.

### Bugs fixed on the way through

1. **`html_auditor.audit_functional_html` vouched for HTML it never
   inspected.** A broad `except Exception` swallowed a missing-bs4 ImportError
   and returned `is_valid=True` having run none of the dead-button/dead-link
   checks. This is what the two long-failing tests were reporting.
2. **`AgentHarness.stream_run` emitted a malformed SSE error frame.** The
   non-200 branch used `'...\\n\\n'` inside a single-quoted f-string, so it
   wrote a literal backslash-n twice instead of a frame terminator. Any
   consumer splitting on blank lines saw one unterminated frame and stalled.
3. **`parse_json_strict` choked on prose appended after the JSON.** Caught by
   diffing against HTML-Notes' extractor before migrating it.

### Deliberate behaviour change (one)

On the OpenAI-compatible stream path, `data: [DONE]` now ends iteration.
Previously the `break` exited only the inner buffer-drain loop and reading
continued. `[DONE]` is terminal in that protocol, so nothing is dropped and
the response is released promptly.

## Testing

105 → 107 tests, all green (`.venv/bin/python -m pytest tests/ -q`).
Was 21 passed / 2 failed before this work.

The SSE consolidation touches live streaming paths in two services, so
`tests/test_sse.py` includes a **characterization test** pinning the exact
frames `stream_prism_events` emits. It was written and made green *against the
old hand-rolled loop* before the refactor landed, then re-run after.

## Consumers — IMPORTANT

Three repos consume this SDK, in two different ways:

- **trading-service** and **HTML-Notes** mount the sibling checkout
  (`../lazycat-sdk:/app/lazycat-sdk:ro`, `PYTHONPATH=/app/lazycat-sdk`).
  **Both deploy scripts tar this working tree to the NAS**, so deploying
  either one updates the SDK directory the other mounts. Running containers
  keep their old code until they restart — so **deploy both consumers in the
  same session** after changing the SDK.
- **lazy-agent-service** bundles a *copy* at `python/lazycat/`. It must be
  synced by hand after SDK changes or the twins drift:
  `cp lazycat-sdk/lazycat/*.py lazy-agent-service/python/lazycat/`

## Known gaps / next

- `sse-starlette` is declared in pyproject but imported nowhere — dead dep.
- No README beyond the one-liner; no `py.typed` marker.
- `tool_registry.py` and `logging.py` still carry trading vocabulary
  (`ticker`, `cycle_id`) in supposedly generic code.
- Untested modules remain: `grounded_research`, `router`, `validators`,
  `html_skills`, `health`, `tools`.
- The bigger extraction candidates found in the audit but **not** taken:
  web-search provider chain, geocoding, SQLite helper layer, structured
  logging setup, FastAPI rate-limit/security middleware.
- music-player, SmartGardenDashBoard and RedditPurgeScraper each reimplement
  the prism client + JSON parsing and don't import `lazycat` at all. They are
  the obvious next migrations.
