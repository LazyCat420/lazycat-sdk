"""The `systemPrompt` cap must cut loudly, never silently.

WHY THIS EXISTS. `call_agent` and `get_stream_payload_and_url` sent
`system_prompt[:15000]` since the SDK's first commit, and lazy-agent-service
accepts any length, so nothing downstream could notice a cut. On 2026-10-22 the
trading Board's 19,598-char system prompt was found losing its last 4,598
characters on every call: the output rules' every example. 12 of 13 first
answers failed the schema (trading-client ch.47, plan item 1.1). The trading
tests asserted on the prompt handed to the model call, which was whole; the cut
happened after it, in here.

These tests assert on the payload the SDK actually posts.
"""
import logging
from unittest.mock import MagicMock, patch

import pytest

import lazycat.llm as llm
from lazycat.llm import AgentClient


@pytest.fixture(autouse=True)
def _fresh_warning_memory(monkeypatch):
    monkeypatch.setattr(llm, "_WARNED_SYSTEM_PROMPT_CUTS", set())


async def _posted_payload(system_prompt: str, agent_name: str = "board") -> dict:
    client = AgentClient()
    client.url = "http://agent"
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"choices": []}
    with patch("httpx.AsyncClient.post", return_value=resp) as post:
        await client.call_agent(
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            system_prompt=system_prompt,
            agent_name=agent_name,
            inline_system_prompt=False,
        )
    return post.call_args.kwargs["json"]


def _cut_warnings(caplog) -> list[logging.LogRecord]:
    return [r for r in caplog.records
            if r.levelno == logging.WARNING and "systemPrompt for agent" in r.getMessage()]


@pytest.mark.asyncio
async def test_a_prompt_at_the_cap_is_sent_whole_and_quietly(caplog):
    prompt = "x" * llm.SYSTEM_PROMPT_MAX_CHARS
    payload = await _posted_payload(prompt)
    assert payload["systemPrompt"] == prompt
    assert not _cut_warnings(caplog)


@pytest.mark.asyncio
async def test_a_cut_is_logged_with_its_size_once_per_agent(caplog):
    cap = llm.SYSTEM_PROMPT_MAX_CHARS
    prompt = "r" * cap + "HOLD Example"
    caplog.set_level(logging.WARNING, logger="lazycat.llm")

    first = await _posted_payload(prompt, "board")
    await _posted_payload(prompt, "board")
    await _posted_payload(prompt, "fundamental")

    # Still cut (the default is unchanged), so the tail never leaves the SDK...
    assert first["systemPrompt"] == prompt[:cap]
    assert "HOLD Example" not in first["systemPrompt"]
    # ...but now it says so: once per agent, with the sizes.
    warnings = _cut_warnings(caplog)
    assert [w.args[0] for w in warnings] == ["board", "fundamental"]
    assert "12 chars" in warnings[0].getMessage() and f"{cap}-char cap" in warnings[0].getMessage()


def test_the_stream_payload_builder_applies_the_same_cap(caplog, monkeypatch):
    monkeypatch.setattr(llm, "SYSTEM_PROMPT_MAX_CHARS", 10)
    payload, _, _ = AgentClient().get_stream_payload_and_url(
        model="m", messages=[{"role": "user", "content": "hi"}], max_tokens=10,
        temperature=0.1, system_prompt="0123456789TAIL", agent_name="board",
        conversation_id="c", session_id="s", project="p", username="u",
        is_new=True, enable_thinking=None,
    )
    assert payload["systemPrompt"] == "0123456789"
    assert len(_cut_warnings(caplog)) == 1


@pytest.mark.parametrize("cap", [0, -1])
def test_a_cap_of_zero_or_less_disables_the_cut(monkeypatch, cap, caplog):
    monkeypatch.setattr(llm, "SYSTEM_PROMPT_MAX_CHARS", cap)
    assert llm._cap_system_prompt("x" * 40_000, "board") == "x" * 40_000
    assert not _cut_warnings(caplog)


def test_every_cut_is_an_event_on_the_callers_span(monkeypatch):
    sdk_trace = pytest.importorskip("opentelemetry.sdk.trace")
    exporter_mod = pytest.importorskip("opentelemetry.sdk.trace.export.in_memory_span_exporter")
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    exporter = exporter_mod.InMemorySpanExporter()
    provider = sdk_trace.TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(llm, "SYSTEM_PROMPT_MAX_CHARS", 10)

    with provider.get_tracer("t").start_as_current_span("agent-run"):
        llm._cap_system_prompt("0123456789TAIL", "board")
        llm._cap_system_prompt("0123456789TAIL", "board")  # warned already; still an event

    (span,) = exporter.get_finished_spans()
    events = [e for e in span.events if e.name == "lazycat.system_prompt_truncated"]
    assert len(events) == 2
    assert events[0].attributes["lazycat.agent"] == "board"
    assert events[0].attributes["lazycat.system_prompt.dropped_chars"] == 4
