import pytest
from unittest.mock import MagicMock

from lazycat.agent import AgentHarness, BaseAgent, decode_tool_arguments, ToolLoopDetector
from lazycat.session import ConversationSession


def test_base_agent_deprecation_warning():
    with pytest.deprecated_call():
        _ = BaseAgent(
            name="legacy_agent",
            system_prompt="You are legacy",
            model="mock-model",
        )


def test_agent_harness_deprecation_warning():
    with pytest.deprecated_call():
        agent = BaseAgent(
            name="legacy_agent",
            system_prompt="You are legacy",
            model="mock-model",
        )
    session = MagicMock(spec=ConversationSession)
    with pytest.deprecated_call():
        _ = AgentHarness(
            agent=agent,
            session=session,
        )


def test_shared_tool_utilities_not_deprecated():
    # decode_tool_arguments must NOT emit deprecation warnings
    args, err = decode_tool_arguments('{"ticker": "AAPL"}')
    assert args == {"ticker": "AAPL"}
    assert err is None

    # ToolLoopDetector must NOT emit deprecation warnings
    detector = ToolLoopDetector()
    detector.record_call("get_quote", {"ticker": "AAPL"}, failed=False)
