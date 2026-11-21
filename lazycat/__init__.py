"""Retired shared SDK tombstone.

`platform/lazycat-sdk` was retired on 2026-11-21 when every LazyCat420 Python
service internalized its runtime client and shared utilities directly:

- `finance/trading/trading-service`: `app/lib/*` (`las_client.py`, `llm_json.py`, ...)
- `finance/trading/trading-client`: `app/lib/las_client.py`
- `apps/html-notes`: `app/lib/*` (`las_client.py`, `llm_json.py`, ...)
- `apps/media/music-player`: `apps/api/app/agent/las_client.py`
- `agents/scraper-service`: stages `app/lib/llm_json.py` from `trading-service`
"""

raise ImportError(
    "lazycat-sdk was retired on 2026-11-21; use each repo's internalized app/lib/* client instead."
)
