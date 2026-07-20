"""
Tests for lazycat.llm_json — robust JSON extraction from LLM output.
"""
import json

import pytest

from lazycat.llm_json import (
    extract_json_str,
    parse_json_list_response,
    parse_json_response,
    parse_json_strict,
    strip_think_tags,
)


# ── strip_think_tags ────────────────────────────────────────────────────


def test_strips_closed_think_block():
    assert strip_think_tags('<think>reasoning</think>{"a": 1}') == '{"a": 1}'


def test_unclosed_think_tag_keeps_the_payload():
    # The model ran out of budget mid-thought. Dropping everything after an
    # unclosed <think> would throw away the JSON that follows it.
    assert strip_think_tags('<think>{"a": 1}') == '{"a": 1}'


def test_returns_think_content_when_requested():
    cleaned, think = strip_think_tags("<think>why</think>answer", return_think_content=True)
    assert cleaned == "answer"
    assert think == "why"


# ── extract_json_str ────────────────────────────────────────────────────


def test_extract_strips_fences():
    assert extract_json_str('```json\n{"a": 1}\n```') == '{"a": 1}'


def test_extract_skips_conversational_prefix():
    assert extract_json_str('Sure! Here you go: {"a": 1}') == '{"a": 1}'


def test_extract_is_string_aware_for_braces_inside_strings():
    # A '}' inside a string literal must not close the object early.
    src = '{"note": "a } brace", "b": 2}'
    assert json.loads(extract_json_str(f"prefix {src}")) == {
        "note": "a } brace",
        "b": 2,
    }


def test_extract_handles_escaped_quotes():
    src = '{"q": "he said \\"hi\\"", "n": 1}'
    assert json.loads(extract_json_str(f"x {src}")) == {"q": 'he said "hi"', "n": 1}


def test_extract_tries_next_opener_when_first_never_balances():
    assert extract_json_str('{unbalanced ... then {"good": 1}') .endswith('{"good": 1}')


def test_extract_returns_input_unchanged_when_nothing_found():
    assert extract_json_str("no json here") == "no json here"


def test_text_already_starting_with_brace_is_returned_verbatim():
    # Early return before the balanced scan: a response that is already bare
    # JSON is handed back untouched, truncated or not. allow_truncated cannot
    # repair such input (there is no closer to clip back to), so both modes
    # agree here and the caller's own json.loads decides.
    truncated = '{"a": 1, "b": [1, 2'
    assert extract_json_str(truncated) == truncated
    assert extract_json_str(truncated, allow_truncated=True) == truncated


def test_extract_truncated_salvages_only_behind_a_prefix():
    # Salvage applies when a conversational prefix pushed the JSON off the
    # front and the tail was cut mid-structure.
    truncated = 'Sure, here: {"a": 1, "b": [1, 2'
    assert extract_json_str(truncated) == truncated  # default: unchanged
    assert extract_json_str(truncated, allow_truncated=True) == '{"a": 1, "b": [1, 2'


def test_extract_truncated_recovers_closable_span():
    src = 'here: {"a": 1} trailing garbage'
    assert extract_json_str(src, allow_truncated=True) == '{"a": 1}'


# ── parse_json_response ─────────────────────────────────────────────────


def test_parses_fenced_json():
    assert parse_json_response('```json\n{"action": "BUY"}\n```') == {"action": "BUY"}


def test_parses_json_after_prose():
    assert parse_json_response('Here is my answer:\n{"action": "HOLD"}') == {"action": "HOLD"}


def test_returns_outer_object_not_inner_fragment():
    # Regression: the brace scan used to walk into a parsed object and return
    # its last nested sub-dict, silently dropping the real payload.
    text = '{"action": "BUY", "meta": {"score": 3}}'
    assert parse_json_response(text) == {"action": "BUY", "meta": {"score": 3}}


def test_empty_after_think_strip_raises():
    with pytest.raises(ValueError):
        parse_json_response("<think>only reasoning, no answer</think>")


def test_returns_empty_dict_when_nothing_parses():
    assert parse_json_response("I could not comply.") == {}


def test_think_streaming_markers_are_stripped():
    text = '__THINK__ working...\n{"a": 1}'
    assert parse_json_response(text) == {"a": 1}


def test_reject_hook_prefers_a_non_rejected_candidate():
    text = '```json\n{"t": "TICKER1"}\n```\n```json\n{"t": "NVDA"}\n```'
    reject = lambda d: any("TICKER1" in str(v) for v in d.values())
    assert parse_json_response(text, reject=reject) == {"t": "NVDA"}


def test_reject_hook_falls_back_when_all_rejected():
    text = '```json\n{"t": "TICKER1"}\n```'
    reject = lambda d: any("TICKER1" in str(v) for v in d.values())
    assert parse_json_response(text, reject=reject) == {"t": "TICKER1"}


def test_fallback_hook_used_only_when_no_json_parses():
    called = []

    def fallback(cleaned):
        called.append(cleaned)
        return {"action": "HOLD", "source": "prose"}

    assert parse_json_response("Recommendation: HOLD", fallback=fallback)["source"] == "prose"
    assert called

    called.clear()
    assert parse_json_response('{"a": 1}', fallback=fallback) == {"a": 1}
    assert not called  # real JSON present — fallback must not run


def test_fallback_exception_is_contained():
    def boom(cleaned):
        raise RuntimeError("bad parser")

    assert parse_json_response("no json", fallback=boom) == {}


# ── parse_json_list_response ────────────────────────────────────────────


def test_parses_fenced_list():
    assert parse_json_list_response('```json\n[1, 2, 3]\n```') == [1, 2, 3]


def test_parses_bare_list_after_prose():
    assert parse_json_list_response('Results: ["a", "b"]') == ["a", "b"]


def test_list_returns_empty_when_absent():
    assert parse_json_list_response("nothing here") == []


def test_list_empty_input_returns_empty():
    assert parse_json_list_response("<think>hmm</think>") == []


# ── parse_json_strict ───────────────────────────────────────────────────


def test_strict_parses_fenced_json():
    assert parse_json_strict('```json\n{"a": 1}\n```') == {"a": 1}


def test_strict_raises_on_garbage():
    with pytest.raises(json.JSONDecodeError):
        parse_json_strict("absolutely not json")


def test_strict_handles_think_block_then_json():
    assert parse_json_strict('<think>x</think>```json\n{"ok": true}\n```') == {"ok": True}
