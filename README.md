# lazycat-sdk (Retired 2026-11-21)

`platform/lazycat-sdk` has been **retired** and replaced with an `ImportError` tombstone (`lazycat/__init__.py`).

All runtime callers across the workspace now bundle lightweight, repo-local `lazy-agent-service` clients and drop `/app/lazycat-sdk` volume mounts and `PYTHONPATH` injection:

- **trading-service (`finance/trading/trading-service`):** `app/lib/*` (`las_client.py`, `llm_json.py`, `config.py`, `resilience.py`, `sse.py`, `tool_registry.py`, `tool_telemetry.py`, `tools.py`, etc.)
- **trading-client (`finance/trading/trading-client`):** `app/lib/las_client.py`
- **html-notes (`apps/html-notes`):** `app/lib/*` (`las_client.py`, `llm_json.py`, `grounded_research.py`, `logging.py`, `research.py`, `resilience.py`, `sse.py`, `tool_telemetry.py`)
- **music-player (`apps/media/music-player`):** `apps/api/app/agent/las_client.py`
- **scraper-service (`agents/scraper-service`):** stages `app/lib/llm_json.py` from `trading-service`

Importing `lazycat` raises `ImportError` by design (`tests/test_retired_tombstone.py`).
