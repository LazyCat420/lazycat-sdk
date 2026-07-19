"""Schema-based tool-argument filtering.

Regression cover for a production failure where a model emitted malformed JSON
and a FRAGMENT of an array value arrived as a keyword argument:

    whiteboard_write() got an unexpected keyword argument
    '"regulatory clearance for ai features in china"], "r'

That TypeError killed the write, so an analyst's red-flag never reached the
whiteboard and the board decided without it.
"""

import json

import pytest

from lazycat.tool_registry import ToolRegistry


@pytest.fixture
def reg():
    registry = ToolRegistry()

    @registry.register(
        name="whiteboard_write",
        description="Write a section to the whiteboard.",
        parameters={
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "section": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["ticker", "section"],
        },
    )
    async def whiteboard_write(ticker: str, section: str, content: str = ""):
        return f"wrote {section} for {ticker}: {content}"

    return registry


def _call(name, arguments):
    return {
        "id": "call_1",
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


@pytest.mark.asyncio
async def test_undeclared_argument_is_dropped_not_splatted(reg):
    """The exact production payload shape must not raise TypeError."""
    args = json.dumps(
        {
            "ticker": "NVDA",
            "section": "red_flags",
            "content": "regulatory risk",
            '"regulatory clearance for ai features in china"], "r': ["junk"],
        }
    )
    result = await reg.execute_tool_call(_call("whiteboard_write", args), skip_permission_check=True, force_local=True)
    assert "unexpected keyword argument" not in json.dumps(result)
    assert "wrote red_flags for NVDA" in result["content"]


@pytest.mark.asyncio
async def test_missing_required_after_filtering_returns_schema_not_typeerror(reg):
    """When the noise removal leaves a required field unset, teach the model."""
    args = json.dumps({"ticker": "NVDA", '"broken"], "section': "red_flags"})
    result = await reg.execute_tool_call(_call("whiteboard_write", args), skip_permission_check=True, force_local=True)
    payload = json.loads(result["content"])

    assert "section" in payload["required_arguments"]
    assert set(payload["expected_arguments"]) == {"ticker", "section", "content"}
    # Must be actionable: a bare error just gets retried verbatim.
    assert "escaped" in payload["error"]


@pytest.mark.asyncio
async def test_wellformed_call_is_untouched(reg):
    args = json.dumps({"ticker": "AMD", "section": "thesis", "content": "solid"})
    result = await reg.execute_tool_call(_call("whiteboard_write", args), skip_permission_check=True, force_local=True)
    assert result["content"] == "wrote thesis for AMD: solid"


@pytest.mark.asyncio
async def test_uppercase_keys_still_normalise(reg):
    """Lowercasing must run before filtering, or valid args get dropped."""
    args = json.dumps({"Ticker": "AMD", "SECTION": "thesis"})
    result = await reg.execute_tool_call(_call("whiteboard_write", args), skip_permission_check=True, force_local=True)
    assert "wrote thesis for AMD" in result["content"]


@pytest.mark.asyncio
async def test_tool_without_declared_properties_passes_through():
    """No schema means no contract to filter against — must not drop everything."""
    registry = ToolRegistry()

    @registry.register(name="freeform", description="anything", parameters={"type": "object"})
    async def freeform(**kwargs):
        return f"got {sorted(kwargs)}"

    args = json.dumps({"anything": 1, "goes": 2})
    result = await registry.execute_tool_call(_call("freeform", args), skip_permission_check=True, force_local=True)
    assert "anything" in result["content"] and "goes" in result["content"]
