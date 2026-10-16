"""One `tool_execution` span per tool execution, to the telemetry-service.

WHY. The per-tool log line (`Tool Execution: <tool> ...`) lives in docker
stdout, which rotates within ~1.5 h — a week of per-tool stats was impossible
(documentation/chapters/08-tool-metrics.md, takeaway 5). The durable source is
the telemetry-service SQLite `spans` table, which lazy-agent-service fills by
POSTing `SpanData` batches to the collector (`TraceExporter.ts`). This module
gives the SDK's Python consumers the same mechanism: the same ingest URL, the
same span shape (`kind: "tool_execution"`, `name: "tool_exec:<tool>"`), so
`WHERE kind='tool_execution'` returns rows from every service.

Attributes follow the brief: tool, agent, cycle_id, ticker, status,
duration_ms, error (truncated to 500 chars — the DB stores raw_json and an
unbounded exception string would bloat every row).

FIRE AND FORGET, ALWAYS. Emission goes through a bounded queue drained by a
daemon thread; every failure path (thread dead, URL bad, socket refused)
swallows its exception and drops the span. A telemetry outage must never fail
a tool call — the same contract as trading_adapter.TelemetryAsyncExporter and
lazy-agent-service's TraceExporter (both suppress unhandled rejections).

Env:
- TELEMETRY_COLLECTOR_URL (default http://10.0.0.16:5595/v1/telemetry)
- TELEMETRY_SERVICE_SOURCE (default "trading-service")
"""
from __future__ import annotations

import collections
import datetime
import hashlib
import json
import logging
import os
import threading
import uuid
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)

MAX_ERROR_CHARS = 500
DEFAULT_COLLECTOR_URL = "http://10.0.0.16:5595/v1/telemetry"

_queue: collections.deque = collections.deque(maxlen=2000)
_lock = threading.Lock()
_thread: threading.Thread | None = None
_thread_started = threading.Event()


def _collector_url() -> str:
    return os.environ.get(
        "TELEMETRY_COLLECTOR_URL", DEFAULT_COLLECTOR_URL
    ).rstrip("/")


def _service_source() -> str:
    return os.environ.get("TELEMETRY_SERVICE_SOURCE", "trading-service")


def _post_batch(spans: list[dict[str, Any]]) -> None:
    payload = {
        "schema_version": "1.0",
        "service_source": _service_source(),
        "exported_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "spans": spans,
        "runs": [],
    }
    req = urllib.request.Request(
        _collector_url(),
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=2.5):
        pass


def _worker_loop() -> None:
    while True:
        batch: list[dict[str, Any]] = []
        with _lock:
            while _queue and len(batch) < 50:
                batch.append(_queue.popleft())
        if not batch:
            # Nothing to send; wait before re-checking.
            if threading.Event().wait(1.0):
                return
            continue
        try:
            _post_batch(batch)
        except Exception as e:  # noqa: BLE001 — telemetry must never surface
            logger.debug("[tool_telemetry] export failed: %s", e)


def _ensure_worker() -> None:
    if _thread_started.is_set():
        return
    with _lock:
        if _thread_started.is_set():
            return
        t = threading.Thread(target=_worker_loop, daemon=True, name="lazycat-tool-telemetry")
        t.start()
        _thread_started.set()


def build_tool_execution_span(
    tool: str,
    agent: str | None,
    *,
    success: bool,
    duration_ms: int,
    error: str | None = None,
    ticker: str | None = None,
    cycle_id: str | None = None,
) -> dict[str, Any]:
    """The SpanData dict, shaped like lazy-agent-service's tool_execution spans."""
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    cycle = cycle_id or "tool"
    trace_id = hashlib.sha256(cycle.encode("utf-8")).hexdigest()[:32]
    return {
        "trace_id": trace_id,
        "span_id": uuid.uuid4().hex[:16],
        "parent_span_id": None,
        "run_id": cycle,
        "name": f"tool_exec:{tool}",
        "kind": "tool_execution",
        "status": "OK" if success else "ERROR",
        "status_message": (error or "")[:200] if not success else None,
        "start_time": now,
        "end_time": now,
        "duration_ms": int(duration_ms),
        "attributes": {
            "tool": tool,
            "agent": agent or "unknown",
            "cycle_id": cycle_id,
            "ticker": (ticker or "").upper() or None,
            "status": "OK" if success else "ERROR",
            "duration_ms": int(duration_ms),
            "error": (error or "")[:MAX_ERROR_CHARS],
        },
        "events": [],
        "links": [],
    }


def emit_tool_execution_span(
    tool: str,
    agent: str | None,
    *,
    success: bool,
    duration_ms: int,
    error: str | None = None,
    ticker: str | None = None,
    cycle_id: str | None = None,
) -> None:
    """Queue one `tool_execution` span. Never raises, never blocks."""
    try:
        span = build_tool_execution_span(
            tool, agent, success=success, duration_ms=duration_ms,
            error=error, ticker=ticker, cycle_id=cycle_id,
        )
        with _lock:
            _queue.append(span)
        _ensure_worker()
    except Exception as e:  # noqa: BLE001
        logger.debug("[tool_telemetry] emit failed: %s", e)


def flush() -> None:
    """Drain the queue synchronously (tests, shutdown)."""
    batch: list[dict[str, Any]] = []
    with _lock:
        while _queue:
            batch.append(_queue.popleft())
    if batch:
        _post_batch(batch)
