"""Models send list/object tool arguments JSON-encoded as strings.

2026-10-16 trading cycle: screener_query received
`filters='["ticker:eq:AMZN"]'`, `columns='["rsi_14", ...]'`, `limit='1'` and
failed validation with 2 errors, which degraded AMZN's research and forced a
FULL triage. Pydantic's lax mode coerces '1' -> 1 but not a JSON string into a
list, so the registry decodes those before giving up.
"""

import pytest
from pydantic import BaseModel, ValidationError

from lazycat.tool_registry import ToolRegistry


class _ScreenerInput(BaseModel):
    filters: list[str]
    columns: list[str] = []
    limit: int = 10


def _registry() -> ToolRegistry:
    registry = ToolRegistry()

    async def screener_query(filters, columns=(), limit=10):
        return {"filters": filters, "columns": columns, "limit": limit}

    registry.register(screener_query, input_model=_ScreenerInput)
    return registry


def test_json_encoded_lists_are_decoded():
    out = _registry()._validate_input(
        "screener_query",
        {"filters": '["ticker:eq:AMZN"]', "columns": '["rsi_14", "earnings_date"]', "limit": "1"},
    )
    assert out == {"filters": ["ticker:eq:AMZN"], "columns": ["rsi_14", "earnings_date"], "limit": 1}


def test_a_string_field_that_looks_like_json_is_left_alone():
    class _Note(BaseModel):
        text: str

    registry = ToolRegistry()

    async def note(text):
        return text

    registry.register(note, input_model=_Note)
    assert registry._validate_input("note", {"text": "[draft] hello"}) == {"text": "[draft] hello"}
    assert registry._validate_input("note", {"text": '["a"]'}) == {"text": '["a"]'}


def test_still_invalid_after_decoding_raises():
    with pytest.raises(ValidationError):
        _registry()._validate_input("screener_query", {"filters": '{"not": "a list"}'})


def test_python_repr_lists_are_decoded():
    # 2026-10-10 junior: filters="['ticker:eq:MP']" — single quotes, not JSON.
    out = _registry()._validate_input(
        "screener_query",
        {"filters": "['ticker:eq:MP']", "columns": "['rsi_14', 'earnings_date']"},
    )
    assert out == {"filters": ["ticker:eq:MP"], "columns": ["rsi_14", "earnings_date"], "limit": 10}


def test_python_repr_dict_is_decoded():
    class _Opts(BaseModel):
        options: dict[str, int]

    registry = ToolRegistry()

    async def opts(options):
        return options

    registry.register(opts, input_model=_Opts)
    assert registry._validate_input("opts", {"options": "{'a': 1, 'b': True}"}) == {
        "options": {"a": 1, "b": 1}
    }


def test_repr_decode_never_evaluates_code_or_rescues_garbage():
    # literal_eval refuses calls; a set literal is not a list; a malformed
    # mixed string stays malformed — all still fail validation.
    for bad in ("[__import__('os').getcwd()]", "{'ticker:eq:MP'}", "['ticker:eq:MP', columns=['x']]"):
        with pytest.raises(ValidationError):
            _registry()._validate_input("screener_query", {"filters": bad})
