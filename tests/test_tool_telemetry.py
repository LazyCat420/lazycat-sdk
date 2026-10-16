"""One `tool_execution` span per tool execution, captured at the HTTP boundary.

The span must reach the telemetry-service collector as JSON POSTed to
TELEMETRY_COLLECTOR_URL — the same ingest lazy-agent-service's TraceExporter
uses — so week-scale per-tool stats survive docker-log rotation
(documentation/chapters/08-tool-metrics.md, takeaway 5). And a telemetry
outage must never fail (or even touch) the tool call.
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from lazycat import tool_telemetry
from lazycat.tool_registry import ToolRegistry


class _Collector(BaseHTTPRequestHandler):
    received: list = []

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        _Collector.received.append(body)
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"{}")

    def log_message(self, *args):
        pass


@pytest.fixture
def collector(monkeypatch):
    server = HTTPServer(("127.0.0.1", 0), _Collector)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    _Collector.received.clear()
    monkeypatch.setenv("TELEMETRY_COLLECTOR_URL",
                       f"http://127.0.0.1:{server.server_port}/v1/telemetry")
    yield _Collector.received
    server.shutdown()


@pytest.mark.asyncio
async def test_tool_execution_emits_one_span_with_brief_attributes(collector):
    registry = ToolRegistry()

    @registry.register(name="echo_ticker")
    async def echo_ticker(ticker: str) -> dict:
        return {"ticker": ticker}

    with tool_telemetry._lock:
        tool_telemetry._queue.clear()

    tool_call = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "echo_ticker",
                     "arguments": json.dumps({"ticker": "msft"})},
    }
    res = await registry.execute_tool_call(
        tool_call=tool_call, agent_name="research_analyst",
        ticker="MSFT", cycle_id="cycle-42",
    )
    assert res["role"] == "tool"  # the call itself succeeded

    tool_telemetry.flush()

    spans = [s for body in collector for s in body["spans"]]
    assert len(spans) == 1
    span = spans[0]
    assert span["kind"] == "tool_execution"
    assert span["name"] == "tool_exec:echo_ticker"
    assert span["status"] == "OK"
    assert isinstance(span["duration_ms"], int)
    attrs = span["attributes"]
    assert attrs["tool"] == "echo_ticker"
    assert attrs["agent"] == "research_analyst"
    assert attrs["cycle_id"] == "cycle-42"
    assert attrs["ticker"] == "MSFT"
    assert attrs["status"] == "OK"
    assert attrs["duration_ms"] == span["duration_ms"]
    assert attrs["error"] == ""


@pytest.mark.asyncio
async def test_failed_tool_span_carries_truncated_error(collector):
    registry = ToolRegistry()

    @registry.register(name="boom_tool")
    async def boom_tool() -> dict:
        raise ValueError("x" * 800)

    tool_call = {
        "id": "call_2",
        "type": "function",
        "function": {"name": "boom_tool", "arguments": "{}"},
    }
    res = await registry.execute_tool_call(
        tool_call=tool_call, agent_name="analyst",
        ticker="aapl", cycle_id="cycle-7",
    )
    assert "error" in json.loads(res["content"])

    tool_telemetry.flush()

    spans = [s for body in collector for s in body["spans"]]
    span = spans[0]
    assert span["status"] == "ERROR"
    assert len(span["attributes"]["error"]) == 500
    assert span["attributes"]["ticker"] == "AAPL"


@pytest.mark.asyncio
async def test_telemetry_outage_does_not_raise_or_delay_the_tool(monkeypatch):
    # Port 1 on localhost is a refusal; the span must be dropped silently.
    monkeypatch.setenv("TELEMETRY_COLLECTOR_URL", "http://127.0.0.1:1/v1/telemetry")
    with tool_telemetry._lock:
        tool_telemetry._queue.clear()

    registry = ToolRegistry()

    @registry.register(name="echo_ok")
    async def echo_ok(ticker: str) -> dict:
        return {"ticker": ticker}

    tool_call = {
        "id": "call_3",
        "type": "function",
        "function": {"name": "echo_ok",
                     "arguments": json.dumps({"ticker": "NVDA"})},
    }
    res = await registry.execute_tool_call(
        tool_call=tool_call, agent_name="analyst",
        ticker="NVDA", cycle_id="cycle-9",
    )
    assert res["role"] == "tool"
    assert json.loads(res["content"])["ticker"] == "NVDA"

    with pytest.raises(Exception):
        tool_telemetry.flush()  # the POST itself fails — but nothing above did
    with tool_telemetry._lock:
        tool_telemetry._queue.clear()
