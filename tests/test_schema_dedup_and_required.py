"""One schema per tool name, and a missing argument that reads as a schema error.

Two defects measured in trading-service on 2026-07-31.

1. The registry has two schema sources — the compiled catalog (`load_from_json`)
   and the `@register` decorator — and both appended blindly. 54 of 56 tools were
   defined by both, so every agent's tool array was exactly 2x its whitelist on
   every LLM call. Three pairs also disagreed about their contract, which is how
   `get_sec_filings` was advertised as both `required: ["ticker"]` and
   `required: []` with a `symbol` alias.

2. The missing-required-field check was nested under `if _dropped_keys`, so it
   only fired when the model ALSO sent an undeclared key. A plain omission fell
   through to `func(**kwargs)` and died as a raw TypeError the model cannot act
   on.
"""

import json

import pytest

from lazycat.tool_registry import ToolRegistry

CATALOG = [
    {
        "name": "get_sec_filings",
        "description": "Catalog version — takes ONLY a ticker.",
        "parameters": {
            "type": "object",
            "properties": {"ticker": {"type": "string"}},
            "required": ["ticker"],
        },
    }
]


def _write_catalog(tmp_path, entries=CATALOG):
    p = tmp_path / "tool_schemas.json"
    p.write_text(json.dumps(entries), encoding="utf-8")
    return str(p)


def _call(name, arguments):
    return {
        "id": "call_1",
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def _names(reg):
    return [s["function"]["name"] for s in reg.schemas]


# ── 1. Deduplication ──────────────────────────────────────────────────────────


def test_catalog_then_decorator_yields_one_schema(tmp_path):
    """The load order trading-service actually uses: catalog first, then imports."""
    reg = ToolRegistry()
    reg.load_from_json(_write_catalog(tmp_path))

    @reg.register(
        name="get_sec_filings",
        description="Decorator version — permissive.",
        parameters={
            "type": "object",
            "properties": {"ticker": {"type": "string"}, "symbol": {"type": "string"}},
            "required": [],
        },
    )
    async def get_sec_filings(ticker: str = "", **_extra):
        return "ok"

    assert _names(reg).count("get_sec_filings") == 1
    fn = reg.get_schemas_by_names(["get_sec_filings"])
    assert len(fn) == 1, "the whitelist path is what gets sent to the model"
    # The catalog wins, so what the model is SHOWN matches what _schema_params
    # ENFORCES — previously the model saw both and the executor used the catalog.
    assert fn[0]["function"]["parameters"]["required"] == ["ticker"]
    assert "symbol" not in fn[0]["function"]["parameters"]["properties"]


def test_decorator_then_catalog_also_yields_one_catalog_schema(tmp_path):
    """Reverse order (other consumers) resolves the same way — the catalog wins."""
    reg = ToolRegistry()

    @reg.register(
        name="get_sec_filings",
        description="Decorator version.",
        parameters={
            "type": "object",
            "properties": {"symbol": {"type": "string"}},
            "required": [],
        },
    )
    async def get_sec_filings(symbol: str = "", **_extra):
        return "ok"

    reg.load_from_json(_write_catalog(tmp_path))

    assert _names(reg).count("get_sec_filings") == 1
    assert reg.schemas[0]["function"]["parameters"]["required"] == ["ticker"]


def test_decorator_only_tool_keeps_its_schema(tmp_path):
    """The decorator is the fallback for anything the catalog does not carry."""
    reg = ToolRegistry()
    reg.load_from_json(_write_catalog(tmp_path))

    @reg.register(
        name="local_only",
        description="Not in the catalog.",
        parameters={
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "required": ["x"],
        },
    )
    async def local_only(x: str):
        return x

    assert _names(reg).count("local_only") == 1
    assert reg.get_schemas_by_names(["local_only"])[0]["function"]["description"] == (
        "Not in the catalog."
    )


def test_reloading_the_catalog_does_not_grow_the_list(tmp_path):
    """Boot paths that load twice must not double the tools array."""
    reg = ToolRegistry()
    path = _write_catalog(tmp_path)
    reg.load_from_json(path)
    reg.load_from_json(path)
    assert len(reg.schemas) == 1


def test_mismatched_contract_is_logged_not_silently_merged(tmp_path, caplog):
    reg = ToolRegistry()
    reg.load_from_json(_write_catalog(tmp_path))
    with caplog.at_level("WARNING"):

        @reg.register(
            name="get_sec_filings",
            description="Decorator version.",
            parameters={
                "type": "object",
                "properties": {"symbol": {"type": "string"}},
                "required": [],
            },
        )
        async def get_sec_filings(symbol: str = "", **_extra):
            return "ok"

    assert "DIFFERENT" in caplog.text and "get_sec_filings" in caplog.text


# ── 2. Required-field check without junk keys ─────────────────────────────────


@pytest.fixture
def reg():
    registry = ToolRegistry()

    @registry.register(
        name="get_market_data",
        description="Market data for a ticker.",
        parameters={
            "type": "object",
            "properties": {"ticker": {"type": "string"}},
            "required": ["ticker"],
        },
    )
    async def get_market_data(ticker: str):
        return f"data for {ticker}"

    # The shape that must NOT newly break: the schema calls `section` required,
    # but the function defaults it, so omitting it is a working call today.
    @registry.register(
        name="whiteboard_read",
        description="Read the desk's scratchpad.",
        parameters={
            "type": "object",
            "properties": {"ticker": {"type": "string"}, "section": {"type": "string"}},
            "required": ["ticker", "section"],
        },
    )
    async def whiteboard_read(ticker: str, section: str = "", **_extra):
        return f"board for {ticker}, section={section!r}"

    return registry


@pytest.mark.asyncio
async def test_plain_missing_field_returns_schema_error_not_typeerror(reg):
    """No junk keys at all — the case that used to reach func() and TypeError."""
    out = await reg.execute_tool_call(_call("get_market_data", json.dumps({})))
    body = json.loads(out["content"])
    assert "ticker" in body["error"]
    assert body["required_arguments"] == ["ticker"]
    assert body["expected_arguments"] == ["ticker"]
    # The old message blamed JSON escaping, which is wrong when nothing was dropped.
    assert "escaped" not in body["error"]


@pytest.mark.asyncio
async def test_missing_field_alongside_junk_still_explains_the_junk(reg):
    args = json.dumps({'"oops"], "r': ["junk"]})
    out = await reg.execute_tool_call(_call("get_market_data", args))
    body = json.loads(out["content"])
    assert "ticker" in body["error"] and "escaped" in body["error"]


@pytest.mark.asyncio
async def test_defaulted_field_is_not_blocked(reg):
    """Schema says `section` is required; the function defaults it. Must execute.

    This is the regression guard for rejecting on the schema's `required` list
    instead of on whether the call can actually bind.
    """
    out = await reg.execute_tool_call(
        _call("whiteboard_read", json.dumps({"ticker": "NVDA"}))
    )
    assert "board for NVDA" in out["content"]
    assert "Malformed" not in out["content"]


@pytest.mark.asyncio
async def test_satisfied_call_still_executes(reg):
    out = await reg.execute_tool_call(
        _call("get_market_data", json.dumps({"ticker": "NVDA"}))
    )
    assert "data for NVDA" in out["content"]
