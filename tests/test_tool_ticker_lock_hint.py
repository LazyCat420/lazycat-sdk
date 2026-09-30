import json
import pytest
from lazycat.tool_registry import ToolRegistry


@pytest.mark.asyncio
async def test_tool_ticker_lock_hint_recommends_screener_query():
    """Verify that ticker-lock guardrail blocks cross-ticker access with screener_query guidance."""
    registry = ToolRegistry()
    tool_call = {
        "id": "call_test_1",
        "type": "function",
        "function": {
            "name": "get_institutional_holdings",
            "arguments": json.dumps({"ticker": "V"}),
        },
    }

    res = await registry.execute_tool_call(
        tool_call=tool_call,
        ticker="MA",
    )

    assert res["role"] == "tool"
    assert res["name"] == "get_institutional_holdings"
    content = json.loads(res["content"])
    assert "Unauthorized ticker access" in content["error"]
    assert "MA" in content["error"] and "V" in content["error"]
    assert "screener_query" in content["hint"]
