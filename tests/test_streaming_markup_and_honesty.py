"""Tests for W3.1 + W2.2 in lazycat-sdk:
- Streaming Qwen `<tool_call>` and DeepSeek DSML markup recovery in `AgentHarness.run()`
- Unknown-tool prose protection (never execute unregistered tools from markup)
- Lightweight JSON syntax repair in `decode_tool_arguments`
- Forced-final-turn honesty (`SDK_MAX_ITERATIONS_FORCED_FINAL` vs `COMPLETED` / `CUT_OFF` / `SDK_MAX_ITERATIONS`)
- Per-tool lineage & truncation metadata (`tool_call_id`, `raw_chars`, `model_visible_chars`, `was_truncated`)
- Removal of hardcoded cloud model defaults (`gpt-4o`) in `router.py` and `validators.py`
"""
import inspect
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from lazycat.agent import AgentHarness, BaseAgent, decode_tool_arguments
from lazycat.router import IntentDefinition, IntentRouter
from lazycat.session import ConversationSession
from lazycat.validators import Validator


def _build_direct_harness(*turn_events, tools=None, **harness_kwargs):
    agent = BaseAgent(
        model="local-qwen3-coder",
        name="v3_junior_analyst",
        system_prompt="You are a local financial analyst.",
        provider="vllm",
    )
    for t in tools or []:
        agent.add_tool(t)

    agent.llm_client = MagicMock()
    queued_turns = list(turn_events)
    sent_payloads = []

    async def _call_agent(**kwargs):
        sent_payloads.append(kwargs)
        events = queued_turns.pop(0)

        async def _aiter_lines():
            for ev in events:
                yield "data: " + json.dumps(ev)
            yield "data: [DONE]"

        resp = MagicMock()
        resp.aiter_lines = _aiter_lines
        resp.aclose = AsyncMock()
        return resp

    agent.llm_client.call_agent = _call_agent
    harness = AgentHarness(
        agent=agent,
        session=ConversationSession(session_id="sess-markup"),
        **harness_kwargs,
    )
    return harness, sent_payloads


def _attach_inprocess_registry(harness, handler):
    captured_payloads = []

    async def _execute_tool_call(call_payload):
        captured_payloads.append(call_payload)
        fn = call_payload["function"]["name"]
        args = json.loads(call_payload["function"]["arguments"])
        return {"content": json.dumps(handler(fn, args))}

    harness.agent.tool_registry = MagicMock()
    harness.agent.tool_registry.execute_tool_call = _execute_tool_call
    return captured_payloads


def _stream_text_turn(text: str, finish_reason: str = "stop"):
    return [
        {"type": "chunk", "content": text},
        {
            "type": "done",
            "model": "local-qwen3-coder",
            "provider": "vllm",
            "finishReason": finish_reason,
        },
    ]


def _stream_native_tool_turn(
    name: str = "get_market_data",
    args: str = '{"ticker": "NVDA"}',
    call_id: str = "call_native_1",
    finish_reason: str = "tool_calls",
):
    return [
        {
            "type": "tool_calls",
            "toolCalls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": args},
                }
            ],
        },
        {
            "type": "done",
            "model": "local-qwen3-coder",
            "provider": "vllm",
            "finishReason": finish_reason,
        },
    ]


# ============================================================================
# 1. Streaming Qwen `<tool_call>` & DeepSeek DSML Markup Recovery
# ============================================================================


@pytest.mark.asyncio
async def test_qwen_single_and_multi_tool_call_markup_recovery():
    tools = [
        {"type": "function", "function": {"name": "get_market_data"}},
        {"type": "function", "function": {"name": "get_earnings_data"}},
    ]
    turn1_text = (
        "Let me check NVDA price and earnings first.\n\n"
        '<tool_call>\n{"name": "get_market_data", "arguments": {"ticker": "NVDA"}}\n</tool_call>\n'
        '<tool_call>{"name": "mcp__lazy-agent-service__get_earnings_data", "arguments": {"ticker": "NVDA", "quarters": 4,}}</tool_call>'
    )
    harness, _ = _build_direct_harness(
        _stream_text_turn(turn1_text, finish_reason="stop"),
        _stream_text_turn('{"verdict": "bullish", "price": 138.5}', finish_reason="stop"),
        tools=tools,
        max_iterations=4,
    )
    dispatched = _attach_inprocess_registry(
        harness,
        lambda fn, args: {"tool": fn, "echo": args, "price": 138.5},
    )

    final_answer = await harness.run("Analyze NVDA")

    assert final_answer == '{"verdict": "bullish", "price": 138.5}'
    assert (harness.answer_source, harness.finalize_reason) == ("model", "COMPLETED")
    assert len(dispatched) == 2
    assert dispatched[0]["id"] == "call_markup_1_0"
    assert dispatched[0]["function"]["name"] == "get_market_data"
    assert json.loads(dispatched[0]["function"]["arguments"]) == {"ticker": "NVDA"}

    assert dispatched[1]["id"] == "call_markup_1_1"
    assert dispatched[1]["function"]["name"] == "get_earnings_data"
    assert json.loads(dispatched[1]["function"]["arguments"]) == {
        "ticker": "NVDA",
        "quarters": 4,
    }

    assistant_msgs = [m for m in harness.session.messages if m.get("role") == "assistant"]
    assert assistant_msgs[0]["content"] == "Let me check NVDA price and earnings first."
    assert "<tool_call>" not in assistant_msgs[0]["content"]


@pytest.mark.asyncio
async def test_deepseek_dsml_fullwidth_and_ascii_markup_recovery():
    tools = [
        {"type": "function", "function": {"name": "get_market_data"}},
        {"type": "function", "function": {"name": "screener_query"}},
    ]
    turn1_dsml = (
        "Gathering semiconductor comparables.\n"
        "<｜DSML｜function_calls>\n"
        '<｜DSML｜invoke name="get_market_data">\n'
        '<｜DSML｜parameter name="ticker" string="true">AMD</｜DSML｜parameter>\n'
        "</｜DSML｜invoke>\n"
        "</｜DSML｜function_calls>"
    )
    turn2_dsml_ascii = (
        "<|DSML|function_calls>"
        '<|DSML|invoke name="screener_query">'
        '<|DSML|parameter name="sector" string="true">Semiconductors</|DSML|parameter>'
        '<|DSML|parameter name="limit" string="false">5</|DSML|parameter>'
        '<|DSML|parameter name="tickers" string="false">["AMD", "NVDA",]</|DSML|parameter>'
        "</|DSML|invoke>"
        "</|DSML|function_calls>"
    )
    harness, _ = _build_direct_harness(
        _stream_text_turn(turn1_dsml),
        _stream_text_turn(turn2_dsml_ascii),
        _stream_text_turn("Comparison complete.", finish_reason="stop"),
        tools=tools,
        max_iterations=5,
    )
    dispatched = _attach_inprocess_registry(
        harness,
        lambda fn, args: {"ok": True, "fn": fn, "args": args},
    )

    answer = await harness.run("Compare AMD")
    assert answer == "Comparison complete."
    assert len(dispatched) == 2
    assert dispatched[0]["id"] == "call_markup_1_0"
    assert dispatched[0]["function"]["name"] == "get_market_data"
    assert json.loads(dispatched[0]["function"]["arguments"]) == {"ticker": "AMD"}

    assert dispatched[1]["id"] == "call_markup_2_0"
    assert dispatched[1]["function"]["name"] == "screener_query"
    assert json.loads(dispatched[1]["function"]["arguments"]) == {
        "sector": "Semiconductors",
        "limit": 5,
        "tickers": ["AMD", "NVDA"],
    }

    assistant_msgs = [m for m in harness.session.messages if m.get("role") == "assistant"]
    assert assistant_msgs[0]["content"] == "Gathering semiconductor comparables."
    assert "DSML" not in assistant_msgs[0]["content"]
    assert assistant_msgs[1].get("content", "") == ""


@pytest.mark.asyncio
async def test_unknown_tool_in_prose_markup_is_never_executed():
    tools = [{"type": "function", "function": {"name": "get_market_data"}}]
    prose_with_example = (
        "For example, an external caller might write "
        '<tool_call>{"name": "delete_production_db", "arguments": {"force": true}}</tool_call> '
        "or <｜DSML｜function_calls><｜DSML｜invoke name=\"rm_rf\">"
        '<｜DSML｜parameter name="path" string="true">/</｜DSML｜parameter>'
        "</｜DSML｜invoke></｜DSML｜function_calls>, which is forbidden."
    )
    harness, _ = _build_direct_harness(
        _stream_text_turn(prose_with_example, finish_reason="stop"),
        tools=tools,
        max_iterations=3,
    )
    dispatched = _attach_inprocess_registry(harness, lambda fn, args: {"ran": fn})

    answer = await harness.run("Explain forbidden calls")
    assert dispatched == []
    assert answer == prose_with_example
    assert (harness.answer_source, harness.finalize_reason) == ("model", "COMPLETED")


# ============================================================================
# 2. Lightweight JSON Arg Syntax Repair
# ============================================================================


def test_decode_tool_arguments_repairs_fences_and_trailing_commas():
    fenced = '```json\n{\n  "ticker": "MSFT",\n  "metrics": ["pe", "ev_ebitda", ],\n}\n```'
    decoded, err = decode_tool_arguments(fenced)
    assert err is None
    assert decoded == {"ticker": "MSFT", "metrics": ["pe", "ev_ebitda"]}


def test_decode_tool_arguments_still_rejects_unrepairable_or_non_dict():
    decoded_bad, err_bad = decode_tool_arguments('```json\n{"ticker": "MSFT"\n```')
    assert decoded_bad == {}
    assert err_bad is not None
    assert "NOT executed" in err_bad

    decoded_list, err_list = decode_tool_arguments('```json\n["MSFT", "AAPL",]\n```')
    assert decoded_list == {}
    assert err_list is not None
    assert "got a list" in err_list


# ============================================================================
# 3. Forced-Final Honesty & Tool Lineage / Truncation Metadata
# ============================================================================


@pytest.mark.asyncio
async def test_forced_final_turn_honesty_distinguishes_clean_truncated_and_empty():
    # Case 1: Clean forced final -> ("model", "SDK_MAX_ITERATIONS_FORCED_FINAL")
    h_clean, _ = _build_direct_harness(
        _stream_native_tool_turn(),
        _stream_text_turn("Synthesized artifact", finish_reason="stop"),
        tools=[{"name": "get_market_data"}],
        max_iterations=2,
    )
    _attach_inprocess_registry(h_clean, lambda fn, args: {"price": 200})
    ans_clean = await h_clean.run("go")
    assert ans_clean == "Synthesized artifact"
    assert (h_clean.answer_source, h_clean.finalize_reason) == (
        "model",
        "SDK_MAX_ITERATIONS_FORCED_FINAL",
    )

    # Case 2: Truncated forced final -> ("partial", "CUT_OFF")
    h_cut, _ = _build_direct_harness(
        _stream_native_tool_turn(),
        _stream_text_turn("Half artifact...", finish_reason="length"),
        tools=[{"name": "get_market_data"}],
        max_iterations=2,
    )
    _attach_inprocess_registry(h_cut, lambda fn, args: {"price": 200})
    ans_cut = await h_cut.run("go")
    assert ans_cut == "Half artifact..."
    assert (h_cut.answer_source, h_cut.finalize_reason) == ("partial", "CUT_OFF")

    # Case 3: Empty forced final -> ("synthetic", "SDK_MAX_ITERATIONS")
    h_empty, _ = _build_direct_harness(
        _stream_native_tool_turn(),
        _stream_text_turn("   ", finish_reason="stop"),
        tools=[{"name": "get_market_data"}],
        max_iterations=2,
    )
    _attach_inprocess_registry(h_empty, lambda fn, args: {"price": 200})
    ans_empty = await h_empty.run("go")
    assert ans_empty == "Max iterations reached without a final answer."
    assert (h_empty.answer_source, h_empty.finalize_reason) == (
        "synthetic",
        "SDK_MAX_ITERATIONS",
    )


@pytest.mark.asyncio
async def test_tool_lineage_and_truncation_metadata_on_kwargs_and_legacy_hooks():
    callback_events = []

    def _modern_hook(tool_name, arguments, result, was_blocked, elapsed_ms, **meta):
        callback_events.append(
            {
                "tool_name": tool_name,
                "was_blocked": was_blocked,
                "elapsed_ms": elapsed_ms,
                **meta,
            }
        )

    oversized_blob = "X" * 60_000
    harness, _ = _build_direct_harness(
        _stream_native_tool_turn(
            name="get_sec_filings",
            args='{"ticker": "AAPL"}',
            call_id="call_lineage_42",
        ),
        _stream_text_turn("Filed summary.", finish_reason="stop"),
        tools=[{"name": "get_sec_filings"}],
        on_tool_result=_modern_hook,
        max_iterations=3,
    )
    inproc_payloads = _attach_inprocess_registry(
        harness,
        lambda fn, args: {"filing_text": oversized_blob},
    )

    await harness.run("Check AAPL 10-K")

    assert inproc_payloads[0]["id"] == "call_lineage_42"
    assert len(callback_events) == 1
    ev = callback_events[0]
    assert ev["tool_call_id"] == "call_lineage_42"
    assert ev["was_truncated"] is True
    assert ev["raw_chars"] > 60_000
    assert 50_000 < ev["model_visible_chars"] < ev["raw_chars"]

    tool_msgs = [m for m in harness.session.messages if m.get("role") == "tool"]
    assert tool_msgs[0]["tool_call_id"] == "call_lineage_42"
    assert len(tool_msgs[0]["content"]) == ev["model_visible_chars"]

    assert len(harness.tool_traces) == 1
    assert harness.tool_traces[0]["tool_call_id"] == "call_lineage_42"
    assert harness.tool_traces[0]["was_truncated"] is True
    assert harness.tool_traces[0]["raw_chars"] == ev["raw_chars"]
    assert harness.tool_traces[0]["model_visible_chars"] == ev["model_visible_chars"]

    # Legacy 5-positional-arg hook (no **kwargs) must continue to work cleanly.
    legacy_calls = []

    def _legacy_five_arg_hook(tool_name, arguments, result, was_blocked, elapsed_ms):
        legacy_calls.append((tool_name, was_blocked))

    h_legacy, _ = _build_direct_harness(
        _stream_native_tool_turn(
            name="get_market_data",
            args='{"ticker": "AAPL"}',
            call_id="call_small_1",
        ),
        _stream_text_turn("Done.", finish_reason="stop"),
        tools=[{"name": "get_market_data"}],
        on_tool_result=_legacy_five_arg_hook,
        max_iterations=3,
    )
    _attach_inprocess_registry(h_legacy, lambda fn, args: {"price": 225.0})
    await h_legacy.run("Quote AAPL")

    assert legacy_calls == [("get_market_data", False)]
    assert len(h_legacy.tool_traces) == 1
    assert h_legacy.tool_traces[0]["tool_call_id"] == "call_small_1"
    assert h_legacy.tool_traces[0]["was_truncated"] is False
    assert (
        h_legacy.tool_traces[0]["raw_chars"]
        == h_legacy.tool_traces[0]["model_visible_chars"]
    )


# ============================================================================
# 4. No Cloud Model Defaults in IntentRouter & Validator
# ============================================================================


@pytest.mark.asyncio
async def test_router_and_validator_have_no_cloud_model_defaults():
    classify_sig = inspect.signature(IntentRouter.classify)
    review_sig = inspect.signature(Validator.self_review_html)

    assert classify_sig.parameters["model"].default == ""
    assert review_sig.parameters["model"].default == ""

    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = '{"intent": "lookup", "query": "NVDA", "reasoning": "local", "extracted_entities": {}}'
    mock_client.call_agent = AsyncMock(return_value=mock_resp)

    router = IntentRouter(
        intents=[IntentDefinition(name="lookup", description="Ticker lookup")],
        llm_client=mock_client,
    )
    await router.classify("Check NVDA", model="Qwen3-30B-A3B")
    _, router_kwargs = mock_client.call_agent.call_args
    assert router_kwargs["model"] == "Qwen3-30B-A3B"

    mock_resp.text = "<div>ok</div>"
    await Validator.self_review_html("<div>ok</div>", mock_client, model="Qwen3-30B-A3B")
    _, val_kwargs = mock_client.call_agent.call_args
    assert val_kwargs["model"] == "Qwen3-30B-A3B"
