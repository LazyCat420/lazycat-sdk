"""
Robust JSON extraction from LLM responses.

Language models rarely emit clean JSON. They wrap it in markdown fences, prefix
it with conversational filler, leak chain-of-thought markers, and truncate it
when they hit a token ceiling. This module consolidates the extraction/repair
logic that was independently reimplemented across several services.

Usage:
    from lazycat.llm_json import parse_json_response, parse_json_strict

    data = parse_json_response(llm_text)        # -> dict, {} when nothing parses
    data = parse_json_strict(llm_text)          # -> Any, RAISES on bad input

Domain-specific behavior is injected rather than baked in: `parse_json_response`
takes a `reject` predicate (skip candidates that are obviously placeholder
output) and a `fallback` parser (last-ditch structured extraction from prose).
"""

import json
import logging
import re
from typing import Any, Callable

logger = logging.getLogger(__name__)

__all__ = [
    "strip_think_tags",
    "extract_json_str",
    "parse_json_response",
    "parse_json_list_response",
    "parse_json_strict",
]


def strip_think_tags(text: str, return_think_content: bool = False):
    """Remove <think>...</think> blocks from LLM responses.

    Reasoning models (Qwen3 et al.) emit <think> blocks for chain-of-thought.
    These must be stripped before parsing the actual response content.
    If return_think_content is True, returns (cleaned_text, think_block_content).
    """
    think_content = ""
    if return_think_content:
        match = re.search(r"<think>(.*?)(?:</think>|$)", text, flags=re.DOTALL)
        if match:
            think_content = match.group(1).strip()

    if "</think>" in text:
        cleaned = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()
    else:
        # If unclosed, just remove the <think> tag itself so we don't delete the JSON!
        cleaned = text.replace("<think>", "").strip()

    if return_think_content:
        return cleaned, think_content
    return cleaned


def _strip_think_markers(cleaned: str, *, warn: bool = False) -> str:
    """Drop __THINK__ streaming status markers that leaked into a response.

    These come from streaming mode and should never appear in a non-streaming
    completion, but when they do they kill the JSON parser.
    """
    if "__THINK__" not in cleaned:
        return cleaned
    if warn:
        logger.warning(
            "[LLM_JSON] __THINK__ marker found in response — stripping before JSON parse. "
            "This indicates a streaming marker leaked into the pipeline. Preview: %s",
            cleaned[:200],
        )
    lines = cleaned.split("\n")
    return "\n".join(l for l in lines if not l.strip().startswith("__THINK__")).strip()


def extract_json_str(text: str, allow_truncated: bool = False) -> str:
    """Best-effort extraction of the first JSON object/array as a STRING.

    For callers that need the JSON text itself rather than a parsed object.
    Strips markdown fences, then returns the earliest balanced {...} or [...]
    block (string-aware, so braces inside string literals don't confuse the
    depth count; tries the next opener if one never balances).

    Returns the input unchanged when nothing better is found, unless
    `allow_truncated` is set — then a best-effort span is returned for output
    that was cut off mid-structure by a token limit.
    """
    if not text:
        return text
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    text = text.strip()
    if text.startswith("{") or text.startswith("["):
        return text

    span = _scan_balanced(text)
    if span is not None:
        return span

    if allow_truncated:
        return _salvage_truncated(text)
    return text


def _scan_balanced(text: str) -> str | None:
    """Return the earliest balanced {...} or [...] span, or None.

    String-aware, so braces and brackets inside string literals don't affect
    the depth count. Tries the next opener when one never balances.
    """
    pairs = {"{": "}", "[": "]"}
    starts = [i for i, ch in enumerate(text) if ch in pairs][:10]
    for start in starts:
        opener = text[start]
        closer = pairs[opener]
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            ch = text[i]
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    return None


def _object_starts(text: str, *, string_aware: bool) -> list[int]:
    """Indices of every '{' that opens at brace depth 0.

    A depth-0 opener is the only honest definition of "top-level", and unlike
    a skip-on-success cursor it does not depend on the object parsing. That
    distinction is the whole point: when the outer object is malformed, a
    success-gated cursor never advances, so the scan walks into the interior
    and the caller ends up holding a nested fragment.

    `string_aware` tracks quotes so braces inside string literals don't move
    the depth. It is the correct reading and is tried first; the naive count
    is kept as a second pass for prose whose quotes are unbalanced *before*
    the JSON (`He said "hi. {"a": 1}`), where quote tracking would swallow the
    real opener.
    """
    starts: list[int] = []
    depth = 0
    in_str = False
    esc = False
    for i, ch in enumerate(text):
        if string_aware:
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
        if ch == "{":
            if depth == 0:
                starts.append(i)
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
    return starts


def _balanced_end(text: str, start: int, *, string_aware: bool) -> int | None:
    """Index of the '}' closing the object opened at `start`, or None."""
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if string_aware:
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
    return None


def _salvage_truncated(text: str) -> str:
    """Recover a usable JSON span from output that was cut off mid-structure.

    Walks from the first opener to the LAST matching closer. When that span
    still doesn't parse and the discarded tail contains further JSON tokens,
    the whole suffix is returned instead so the caller can attempt its own
    repair rather than silently losing data.
    """
    first_brace = text.find("{")
    first_bracket = text.find("[")

    if first_brace != -1 and (first_bracket == -1 or first_brace < first_bracket):
        start_idx, end_char = first_brace, "}"
    elif first_bracket != -1:
        start_idx, end_char = first_bracket, "]"
    else:
        return text

    suffix = text[start_idx:]
    end_idx = suffix.rfind(end_char)
    if end_idx == -1:
        return suffix

    candidate = suffix[: end_idx + 1]
    try:
        json.loads(candidate)
        return candidate
    except json.JSONDecodeError:
        if len(suffix) > len(candidate) and "," in suffix[end_idx:]:
            return suffix
        return candidate


def parse_json_response(
    text: str,
    *,
    reject: Callable[[dict], bool] | None = None,
    fallback: Callable[[str], dict | None] | None = None,
) -> dict:
    """Extract a JSON object from an LLM response.

    Tries, in order:
        1. Markdown JSON code blocks (```json ... ```)
        2. Balanced brace-counting over top-level objects
        3. The raw cleaned text
        4. The caller-supplied `fallback` parser

    Args:
        text: Raw LLM response (may contain <think> blocks, markdown, prose).
        reject: Optional predicate marking a parsed dict as unusable (e.g.
            obvious template placeholders). Rejected candidates are only
            returned if every candidate was rejected.
        fallback: Optional last-resort parser invoked with the cleaned text
            when no JSON parses. Its truthy return value is returned as-is.

    Returns:
        Parsed dict, or {} if nothing usable was found.

    Raises:
        ValueError: if the response is empty after stripping reasoning blocks
            (i.e. the model produced no answer at all).
    """
    cleaned = strip_think_tags(text)
    cleaned = _strip_think_markers(cleaned, warn=True)

    if not cleaned:
        raise ValueError(
            "LLM response is empty after stripping <think> tags (model failed to output JSON)."
        )

    def _pick(candidates: list[dict]) -> dict:
        if reject is None:
            return candidates[-1]
        accepted = [c for c in candidates if not reject(c)]
        return accepted[-1] if accepted else candidates[-1]

    # 1. Markdown code blocks (non-greedy so we don't span multiple blocks)
    markdown_candidates = []
    for match in re.finditer(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL):
        try:
            parsed = json.loads(match.group(1))
            if isinstance(parsed, dict):
                markdown_candidates.append(parsed)
        except json.JSONDecodeError:
            pass
    if markdown_candidates:
        return _pick(markdown_candidates)

    # 2. Balanced brace scan over TOP-LEVEL objects only.
    #
    # "Top-level" is decided by BRACE DEPTH, never by whether a span parsed.
    # The previous cursor only skipped an object's interior once that object
    # parsed, so a malformed outer object left the cursor parked and the scan
    # walked straight into it — collecting every inner sub-object, with
    # `_pick`'s [-1] then handing back the LAST nested fragment as if it were
    # the payload. Measured in trading-service on 2026-08-04: a fundamental
    # report whose outer object did not parse came back as its `metrics`
    # block alone (17 keys, none of them required), and two quant reports
    # came back as a single entry from their trailing `overlays` array. All
    # three read downstream as "the model produced a useless artifact" rather
    # than "we failed to parse it", so the caller's repair pass never ran.
    #
    # A nested fragment is never a better answer than no answer: returning {}
    # lets the caller repair, retry, or fail honestly.
    brace_candidates: list[dict] = []
    for string_aware in (True, False):
        for start_idx in _object_starts(cleaned, string_aware=string_aware):
            end_idx = _balanced_end(cleaned, start_idx, string_aware=string_aware)
            if end_idx is None:
                continue
            try:
                parsed = json.loads(cleaned[start_idx : end_idx + 1])
            except json.JSONDecodeError:
                continue  # This opening brace didn't work, try next
            if isinstance(parsed, dict):
                brace_candidates.append(parsed)
        # The passes are tried in order and never mixed: a naive depth count
        # can mistake a nested object for a top-level one when the enclosing
        # object holds a '}' inside a string, which is the exact failure the
        # string-aware pass exists to prevent.
        if brace_candidates:
            break
    if brace_candidates:
        return _pick(brace_candidates)

    # 3. The entire cleaned text. Deliberately unfiltered by type: a bare JSON
    # array here is returned as-is, matching long-standing caller expectations.
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        pass

    # 4. Caller-supplied prose parser
    if fallback is not None:
        try:
            fallback_data = fallback(cleaned)
            if fallback_data:
                logger.info(
                    "[LLM_JSON] fallback parser recovered fields: %s",
                    list(fallback_data.keys()),
                )
                return fallback_data
        except Exception as e:
            logger.debug("[LLM_JSON] fallback parser failed: %s", e)

    return {}


def parse_json_list_response(text: str) -> list:
    """Extract a JSON list from an LLM response.

    Same strategy as parse_json_response, for array-valued output.
    Returns [] if no valid JSON list is found.
    """
    cleaned = strip_think_tags(text)
    cleaned = _strip_think_markers(cleaned)

    if not cleaned:
        return []

    for match in re.finditer(r"```(?:json)?\s*(\[.*?\])\s*```", cleaned, re.DOTALL):
        try:
            parsed = json.loads(match.group(1))
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass

    for start_idx in range(len(cleaned)):
        if cleaned[start_idx] != "[":
            continue
        depth = 0
        for end_idx in range(start_idx, len(cleaned)):
            if cleaned[end_idx] == "[":
                depth += 1
            elif cleaned[end_idx] == "]":
                depth -= 1
            if depth == 0:
                candidate = cleaned[start_idx : end_idx + 1]
                try:
                    parsed = json.loads(candidate)
                    if isinstance(parsed, list):
                        return parsed
                except json.JSONDecodeError:
                    break  # This opening bracket didn't work, try next

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            return parsed
    except (json.JSONDecodeError, TypeError):
        pass

    return []


def parse_json_strict(text: str) -> Any:
    """Extract and parse JSON, raising if the response contains none.

    The strict counterpart to parse_json_response: use this where an
    unparseable response is a real error the caller must handle, rather than
    something to paper over with an empty dict.

    Raises:
        json.JSONDecodeError: if no valid JSON could be extracted.
    """
    cleaned = strip_think_tags(text)
    cleaned = _strip_think_markers(cleaned)
    candidate = extract_json_str(cleaned, allow_truncated=True)

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # extract_json_str hands back input that already looks like bare JSON
    # verbatim, which fails to parse when the model appended prose after the
    # object ('{"a": 1} hope that helps!'). Fall back to a balanced scan.
    span = _scan_balanced(candidate)
    if span is not None:
        return json.loads(span)

    # Nothing salvageable — re-raise against the best candidate we had so the
    # error message points at real content.
    return json.loads(candidate)
