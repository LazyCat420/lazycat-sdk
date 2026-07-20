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

    if allow_truncated:
        return _salvage_truncated(text)
    return text


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

    # 2. Balanced brace scan. Only TOP-LEVEL objects may become candidates:
    # once a span parses, every opening brace inside it is skipped. Without
    # this, a valid nested response ends up returning its last inner sub-dict
    # (the outer object parses first, but the scan keeps walking into it and
    # [-1] picks the innermost fragment) — silently dropping the real payload.
    brace_candidates = []
    skip_until = -1
    for start_idx in range(len(cleaned)):
        if cleaned[start_idx] != "{" or start_idx < skip_until:
            continue
        depth = 0
        for end_idx in range(start_idx, len(cleaned)):
            if cleaned[end_idx] == "{":
                depth += 1
            elif cleaned[end_idx] == "}":
                depth -= 1
            if depth == 0:
                candidate = cleaned[start_idx : end_idx + 1]
                try:
                    parsed = json.loads(candidate)
                    if isinstance(parsed, dict):
                        brace_candidates.append(parsed)
                        skip_until = end_idx + 1
                except json.JSONDecodeError:
                    pass  # This opening brace didn't work, try next
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
    return json.loads(extract_json_str(cleaned, allow_truncated=True))
