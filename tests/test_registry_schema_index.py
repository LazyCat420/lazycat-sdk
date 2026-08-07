"""`_schema_index` is a second copy of what `schemas` already knows.

Any caller that edits `schemas` directly desynchronises it. That is how a
trading-service test fixture had been erroring at setup since before
2026-08-06: its teardown removed a tool with

    registry.schemas = [s for s in registry.schemas if ...]

leaving `_schema_index["test_dummy_tool"]` pointing past the end, so the next
registration of that name raised IndexError inside `_put_schema`.

The IndexError is the MILD half. Removing an entry also shifts every later
schema down one position, so a stale index that still happens to be in range
resolves to the WRONG tool — and `_put_schema` would overwrite an unrelated
tool's schema with no error at all. These tests cover both, plus the
`unregister()` that means callers no longer have to touch `schemas` by hand.
"""

import pytest

from lazycat.tool_registry import ToolRegistry


def _schema(name: str, prop: str = "val") -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"{name} description",
            "parameters": {
                "type": "object",
                "properties": {prop: {"type": "integer"}},
                "required": [prop],
            },
        },
    }


@pytest.fixture
def registry() -> ToolRegistry:
    r = ToolRegistry()
    for n in ("alpha", "beta", "gamma"):
        r._put_schema(n, _schema(n), source="decorator")
    return r


class TestUnregisterKeepsTheThreeStructuresTogether:
    def test_it_removes_the_schema_and_reindexes(self, registry):
        assert registry.unregister("alpha") is True

        names = [s["function"]["name"] for s in registry.schemas]
        assert names == ["beta", "gamma"]
        assert registry._schema_index == {"beta": 0, "gamma": 1}

    def test_removing_an_unknown_tool_is_false_not_an_error(self, registry):
        assert registry.unregister("never_registered") is False
        assert len(registry.schemas) == 3

    def test_a_reregistered_tool_lands_once(self, registry):
        registry.unregister("beta")
        registry._put_schema("beta", _schema("beta"), source="decorator")

        names = [s["function"]["name"] for s in registry.schemas]
        assert names.count("beta") == 1

    def test_it_clears_the_callable_and_the_meta(self):
        r = ToolRegistry()

        @r.register(name="tmp_tool", description="d", parameters={
            "type": "object", "properties": {}, "required": []})
        async def tmp_tool():
            return "x"

        assert "tmp_tool" in r.tools
        assert r.unregister("tmp_tool") is True
        assert "tmp_tool" not in r.tools
        assert "tmp_tool" not in r._meta
        assert all(s["function"]["name"] != "tmp_tool" for s in r.schemas)


class TestAStaleIndexIsRepairedNotTrusted:
    """External mutation of `schemas` is the case that was crashing."""

    def test_an_out_of_range_index_does_not_raise(self, registry):
        # Exactly what the trading-service fixture's teardown did.
        registry.schemas = [
            s for s in registry.schemas if s["function"]["name"] != "gamma"
        ]

        registry._put_schema("gamma", _schema("gamma"), source="decorator")

        names = [s["function"]["name"] for s in registry.schemas]
        assert names == ["alpha", "beta", "gamma"]

    def test_a_shifted_index_does_not_overwrite_a_different_tool(self, registry):
        """The silent half: after removing the FIRST entry, `_schema_index`
        still says gamma is at 2 — which now holds nothing, while beta and
        gamma have slid to 0 and 1. An unverified index would clobber the
        wrong schema."""
        registry.schemas = [
            s for s in registry.schemas if s["function"]["name"] != "alpha"
        ]

        registry._put_schema(
            "gamma", _schema("gamma", prop="replaced"), source="catalog"
        )

        by_name = {s["function"]["name"]: s for s in registry.schemas}
        assert set(by_name) == {"beta", "gamma"}
        # beta must be untouched...
        assert list(by_name["beta"]["function"]["parameters"]["properties"]) == ["val"]
        # ...and gamma must be the one that changed.
        assert list(by_name["gamma"]["function"]["parameters"]["properties"]) == ["replaced"]

    def test_the_repaired_index_agrees_with_the_list(self, registry):
        registry.schemas = [
            s for s in registry.schemas if s["function"]["name"] != "alpha"
        ]
        registry._put_schema("gamma", _schema("gamma"), source="decorator")

        for name, at in registry._schema_index.items():
            assert registry.schemas[at]["function"]["name"] == name


class TestOrderIsStillStable:
    """Position is load-bearing: schema order is what keeps prompt caching
    effective across boots, so a replacement must not move a tool to the end."""

    def test_replacing_a_schema_keeps_its_position(self, registry):
        registry._put_schema("alpha", _schema("alpha", prop="v2"), source="catalog")

        names = [s["function"]["name"] for s in registry.schemas]
        assert names == ["alpha", "beta", "gamma"]
        assert list(
            registry.schemas[0]["function"]["parameters"]["properties"]
        ) == ["v2"]
