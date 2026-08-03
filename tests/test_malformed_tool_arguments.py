"""Malformed tool arguments must reach the model, not become `{}`.

The harness used to catch `json.JSONDecodeError` on a tool call's `arguments`
and substitute `{}`. The tool then ran with no arguments and rejected the call
for a missing required field — a message that says nothing about the real
defect, so the model could not correct it.

Measured: trading-service `cycle-v3-1785792600` (2026-08-03), the quant
analyst's `emit_structured_output` call arrived as `{}` after ~40s of
generation and was rejected with "'data' is required and must be an object".
The model HAD produced the data; only its JSON escaping was wrong. The run was
lost and the agent re-run from scratch.
"""
import json

import pytest

from lazycat.agent import decode_tool_arguments


# ── An empty payload is not an error: zero-argument tools are normal ────────

@pytest.mark.parametrize("raw", [None, "", "   ", "{}", {}])
def test_empty_arguments_are_not_an_error(raw):
    arguments, error = decode_tool_arguments(raw)

    assert arguments == {}
    assert error is None, "a zero-argument tool call must not be reported broken"


def test_a_valid_object_decodes():
    arguments, error = decode_tool_arguments(json.dumps({"ticker": "PLTR"}))

    assert arguments == {"ticker": "PLTR"}
    assert error is None


def test_a_dict_passes_through_untouched():
    arguments, error = decode_tool_arguments({"ticker": "PLTR"})

    assert arguments == {"ticker": "PLTR"}
    assert error is None


# ── A malformed payload must be reported, not swallowed ────────────────────

def test_truncated_json_is_reported():
    """The observed shape: a large payload cut off mid-string."""
    truncated = '{"data": {"summary": "PLTR beat and rais'

    arguments, error = decode_tool_arguments(truncated)

    assert arguments == {}
    assert error is not None, "a truncated payload must not silently become {}"


def test_the_error_tells_the_model_what_to_do():
    arguments, error = decode_tool_arguments('{"data": {"summary": "cut off')

    assert "NOT executed" in error, "the model must know the tool did not run"
    assert "re-send" in error.lower(), "the model needs an instruction, not just a diagnosis"
    assert "cut off" in error, "quoting the payload back locates the break"


def test_the_echo_is_bounded():
    """Quoting the payload back must not re-flood the context."""
    huge = '{"data": "' + "x" * 100_000

    _, error = decode_tool_arguments(huge)

    assert error is not None
    assert len(error) < 1000, "the echo must be truncated, not the whole payload"


@pytest.mark.parametrize("raw", ['["a", "b"]', '"a string"', "42", "true"])
def test_valid_json_that_is_not_an_object_is_reported(raw):
    """`json.loads` succeeds but the result cannot be used as kwargs."""
    arguments, error = decode_tool_arguments(raw)

    assert arguments == {}
    assert error is not None


def test_a_non_string_non_dict_is_reported():
    arguments, error = decode_tool_arguments(12345)

    assert arguments == {}
    assert error is not None


# ── The load-bearing behaviour: the tool must NOT run ───────────────────────

@pytest.mark.asyncio
async def test_a_malformed_call_never_reaches_the_executor():
    """Running the tool with `{}` is what produced the misleading rejection."""
    from unittest.mock import AsyncMock, MagicMock, patch

    from lazycat.agent import AgentHarness, BaseAgent
    from lazycat.session import ConversationSession

    agent = BaseAgent(name="test_agent", system_prompt="test")
    agent.add_tool({"name": "emit_structured_output", "description": "dummy"})
    harness = AgentHarness(agent=agent, session=ConversationSession(session_id="t1"))

    truncated = '{\\"data\\": {\\"summary\\": \\"cut off mid-str'

    first = MagicMock()
    first.aclose = AsyncMock()

    async def _first():
        yield 'data: {"type": "chunk", "content": ""}'
        yield (
            'data: {"toolCalls": [{"id": "c1", "name": "emit_structured_output",'
            f' "arguments": "{truncated}"}}]}}'
        )
        yield "data: [DONE]"
    first.aiter_lines = _first

    second = MagicMock()
    second.aclose = AsyncMock()

    async def _second():
        yield 'data: {"type": "chunk", "content": "Recovered."}'
        yield "data: [DONE]"
    second.aiter_lines = _second

    with patch("lazycat.agent.prism_client.call_agent", new_callable=AsyncMock) as mock_call, \
         patch("lazycat.agent.tool_executor.execute_tool", new_callable=AsyncMock) as mock_exec:
        mock_call.side_effect = [first, second]
        mock_exec.return_value = {"status": "ok"}

        result = await harness.run("go")

    assert not mock_exec.called, (
        "a call with undecodable arguments must not execute with {}"
    )
    assert result == "Recovered.", "the loop must continue so the model can retry"

    # And the model must actually be told why.
    tool_messages = [
        m for m in harness.session.messages if m.get("role") == "tool"
    ]
    assert tool_messages, "the model needs a tool result to react to"
    assert "NOT executed" in tool_messages[-1]["content"]
