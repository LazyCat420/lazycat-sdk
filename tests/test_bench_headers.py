import pytest
from unittest.mock import MagicMock, patch
from lazycat.llm import (
    AgentClient,
    set_bench_context,
    get_bench_context,
    clear_bench_context,
)

@pytest.mark.asyncio
async def test_call_agent_with_bench_headers_explicit():
    client = AgentClient()
    client.url = "http://agent"

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"choices": []}

    with patch("httpx.AsyncClient.post", return_value=mock_resp) as mock_post:
        await client.call_agent(
            model="deepseek-v4-flash",
            messages=[{"role": "user", "content": "test message"}],
            system_prompt="sys prompt",
            agent_name="analyst_AAPL",
            bench_harness="trading-cycle",
            bench_run_id="tc-2026-09-03T21:14:05Z",
            bench_task="analyze:AAPL",
        )

        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args

        headers = kwargs.get("headers", {})
        assert headers.get("X-Bench-Harness") == "trading-cycle"
        assert headers.get("X-Bench-Run") == "tc-2026-09-03T21:14:05Z"
        assert headers.get("X-Bench-Task") == "analyze:AAPL"

        payload = kwargs.get("json", {})
        # OpenAI user field fallback
        assert payload.get("user") == "tc-2026-09-03T21:14:05Z"


@pytest.mark.asyncio
async def test_call_agent_with_bench_context_propagation():
    clear_bench_context()
    set_bench_context(
        run_id="tc-2026-09-03T22:00:00Z",
        harness="trading-cycle",
        task="screen",
    )

    client = AgentClient()
    client.url = "http://agent"

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"choices": []}

    with patch("httpx.AsyncClient.post", return_value=mock_resp) as mock_post:
        await client.call_agent(
            model="deepseek-v4-flash",
            messages=[{"role": "user", "content": "screening tickers"}],
            system_prompt="screener prompt",
            agent_name="gatekeeper",
        )

        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args

        headers = kwargs.get("headers", {})
        assert headers.get("X-Bench-Harness") == "trading-cycle"
        assert headers.get("X-Bench-Run") == "tc-2026-09-03T22:00:00Z"
        assert headers.get("X-Bench-Task") == "screen"

        payload = kwargs.get("json", {})
        assert payload.get("user") == "tc-2026-09-03T22:00:00Z"

    clear_bench_context()
