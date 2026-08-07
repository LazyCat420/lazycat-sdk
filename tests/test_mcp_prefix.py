"""`ToolExecutor` must strip either MCP namespace before routing.

The prefix is minted by prism from its MCP server registration name, not by
this SDK. When the backing service was renamed `lazy-tool-service` ->
`lazy-agent-service` the two spellings went live in different prism scopes at
different times, so a client that knows only one of them forwards the other
verbatim to `/execute/<name>` — which answers "Unknown tool". That surfaces to
the model as a missing capability, not as a routing bug, so nothing upstream
reports a failure worth chasing.
"""

import pytest

from lazycat.tools import strip_mcp_prefix


@pytest.mark.parametrize(
    "namespaced",
    [
        "mcp__lazy-agent-service__get_stock_price",
        "mcp__lazy-tool-service__get_stock_price",
    ],
)
def test_both_service_names_strip_to_the_same_bare_name(namespaced):
    assert strip_mcp_prefix(namespaced) == "get_stock_price"


def test_bare_and_empty_names_pass_through():
    assert strip_mcp_prefix("get_stock_price") == "get_stock_price"
    assert strip_mcp_prefix("") == ""
    assert strip_mcp_prefix(None) == ""


def test_an_unrelated_mcp_namespace_is_left_alone():
    """Only OUR namespaces are stripped. Another server's tool must keep its
    prefix, or it would be routed to our /execute as if it were ours."""
    other = "mcp__some-other-server__get_stock_price"
    assert strip_mcp_prefix(other) == other
