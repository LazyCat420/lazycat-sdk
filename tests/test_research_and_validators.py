import time
from unittest.mock import AsyncMock, MagicMock
import pytest
from pydantic import BaseModel, Field, ValidationError

from lazycat.health import (
    configure_health,
    health_check,
    router as health_router,
    _START_TIME,
)
import lazycat.health as health_module
from lazycat.validators import Validator
from lazycat.router import IntentDefinition, IntentRouter
from lazycat.tool_registry import ToolRegistry


# ============================================================================
# lazycat/validators.py Tests
# ============================================================================

class SampleModel(BaseModel):
    name: str
    age: int
    tags: list[str] = Field(default_factory=list)


def test_validator_validate_schema_success():
    data = {"name": "Alice", "age": 30, "tags": ["admin", "user"]}
    result = Validator.validate_schema(SampleModel, data)
    assert result == {"name": "Alice", "age": 30, "tags": ["admin", "user"]}


def test_validator_validate_schema_with_defaults():
    data = {"name": "Bob", "age": 25}
    result = Validator.validate_schema(SampleModel, data)
    assert result == {"name": "Bob", "age": 25, "tags": []}


def test_validator_validate_schema_missing_required_field():
    with pytest.raises(ValueError, match="Validation failed:"):
        Validator.validate_schema(SampleModel, {"age": 25})


def test_validator_validate_schema_type_mismatch():
    with pytest.raises(ValueError, match="Validation failed:"):
        Validator.validate_schema(SampleModel, {"name": "Charlie", "age": "not-a-number"})


@pytest.mark.asyncio
async def test_validator_self_review_html_strips_html_fence():
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = "```html\n<div id=\"unique-comp\">Reviewed</div>\n```"
    mock_client.call_agent = AsyncMock(return_value=mock_resp)

    raw_html = "<div id=\"comp\">Original</div>"
    result = await Validator.self_review_html(raw_html, mock_client, project="my-project")

    assert result == "<div id=\"unique-comp\">Reviewed</div>"
    mock_client.call_agent.assert_awaited_once()
    _, kwargs = mock_client.call_agent.call_args
    assert kwargs["model"] == "gpt-4o"
    assert kwargs["agent_name"] == "html_reviewer"
    assert kwargs["project"] == "my-project"
    assert kwargs["temperature"] == 0.1
    assert kwargs["max_tokens"] == 4096


@pytest.mark.asyncio
async def test_validator_self_review_html_strips_generic_fence():
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = "```\n<p>No language tag</p>\n```"
    mock_client.call_agent = AsyncMock(return_value=mock_resp)

    raw_html = "<p>Original</p>"
    result = await Validator.self_review_html(raw_html, mock_client)
    assert result == "<p>No language tag</p>"


@pytest.mark.asyncio
async def test_validator_self_review_html_plain_content():
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = "  <span>Clean output</span>  "
    mock_client.call_agent = AsyncMock(return_value=mock_resp)

    result = await Validator.self_review_html("<span>Raw</span>", mock_client)
    assert result == "<span>Clean output</span>"


@pytest.mark.asyncio
async def test_validator_self_review_html_exception_fallback():
    mock_client = MagicMock()
    mock_client.call_agent = AsyncMock(side_effect=RuntimeError("LLM call failed"))

    original = "<button onclick=\"return false\">Click</button>"
    result = await Validator.self_review_html(original, mock_client)
    # On exception, falls back to original html_content
    assert result == original


# ============================================================================
# lazycat/health.py Tests
# ============================================================================

def test_health_configure():
    orig_name = health_module.SERVICE_NAME
    orig_version = health_module.SERVICE_VERSION
    try:
        configure_health("my_test_service", "2.1.0")
        assert health_module.SERVICE_NAME == "my_test_service"
        assert health_module.SERVICE_VERSION == "2.1.0"
    finally:
        configure_health(orig_name, orig_version)


@pytest.mark.asyncio
async def test_health_check_output():
    orig_name = health_module.SERVICE_NAME
    orig_version = health_module.SERVICE_VERSION
    try:
        configure_health("test_runner", "1.0.0")
        res = await health_check()
        assert res["service"] == "test_runner"
        assert res["version"] == "1.0.0"
        assert res["status"] == "healthy"
        assert isinstance(res["uptime_seconds"], int)
        assert res["uptime_seconds"] >= 0
        assert "h" in res["uptime_display"]
        assert "m" in res["uptime_display"]
        assert "s" in res["uptime_display"]
    finally:
        configure_health(orig_name, orig_version)


@pytest.mark.asyncio
async def test_health_check_uptime_calculation(monkeypatch):
    start = 1_000_000.0
    now = start + 3665.0  # 1 hour, 1 minute, 5 seconds
    monkeypatch.setattr(health_module, "_START_TIME", start)
    monkeypatch.setattr(time, "time", lambda: now)

    res = await health_check()
    assert res["uptime_seconds"] == 3665
    assert res["uptime_display"] == "1h 1m 5s"


def test_health_router_endpoints():
    route_paths = [route.path for route in health_router.routes]
    assert "/health" in route_paths


# ============================================================================
# lazycat/router.py Tests
# ============================================================================

def test_intent_definition():
    intent = IntentDefinition(name="search", description="Search for items")
    assert intent.name == "search"
    assert intent.description == "Search for items"


def test_intent_router_build_system_prompt():
    intents = [
        IntentDefinition(name="chat", description="Casual conversation"),
        IntentDefinition(name="trade", description="Execute a trade"),
    ]
    router = IntentRouter(intents=intents)
    prompt = router._build_system_prompt()
    assert "1. chat: Casual conversation" in prompt
    assert "2. trade: Execute a trade" in prompt
    assert '"chat" | "trade"' in prompt
    assert "extracted_entities" in prompt


@pytest.mark.asyncio
async def test_intent_router_classify_valid_json():
    intents = [
        IntentDefinition(name="chat", description="Casual conversation"),
        IntentDefinition(name="order", description="Place an order"),
    ]
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = '{"intent": "order", "query": "buy TSLA", "reasoning": "wants shares", "extracted_entities": {"ticker": "TSLA"}}'
    mock_client.call_agent = AsyncMock(return_value=mock_resp)

    router = IntentRouter(intents=intents, llm_client=mock_client)
    res = await router.classify("buy TSLA", model="test-model", provider="test-provider")

    assert res["intent"] == "order"
    assert res["query"] == "buy TSLA"
    assert res["extracted_entities"] == {"ticker": "TSLA"}
    mock_client.call_agent.assert_awaited_once()
    _, kwargs = mock_client.call_agent.call_args
    assert kwargs["model"] == "test-model"
    assert kwargs["provider"] == "test-provider"


@pytest.mark.asyncio
async def test_intent_router_classify_with_json_codeblock():
    intents = [IntentDefinition(name="info", description="Get info")]
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = '```json\n{"intent": "info", "query": "status", "reasoning": "check", "extracted_entities": {}}\n```'
    mock_client.call_agent = AsyncMock(return_value=mock_resp)

    router = IntentRouter(intents=intents, llm_client=mock_client)
    res = await router.classify("status")
    assert res["intent"] == "info"
    assert res["query"] == "status"


@pytest.mark.asyncio
async def test_intent_router_classify_with_generic_codeblock():
    intents = [IntentDefinition(name="info", description="Get info")]
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = '```\n{"intent": "info", "query": "status", "reasoning": "check", "extracted_entities": {}}\n```'
    mock_client.call_agent = AsyncMock(return_value=mock_resp)

    router = IntentRouter(intents=intents, llm_client=mock_client)
    res = await router.classify("status")
    assert res["intent"] == "info"


@pytest.mark.asyncio
async def test_intent_router_classify_conversation_history_trimmed():
    intents = [IntentDefinition(name="chat", description="Chat")]
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = '{"intent": "chat", "query": "hello", "reasoning": "greeting", "extracted_entities": {}}'
    mock_client.call_agent = AsyncMock(return_value=mock_resp)

    router = IntentRouter(intents=intents, llm_client=mock_client)
    history = [
        {"role": "user", "content": "1"},
        {"role": "assistant", "content": "2"},
        {"role": "user", "content": "3"},
        {"role": "assistant", "content": "4"},
        {"role": "user", "content": "5"},
        {"role": "assistant", "content": "6"},
    ]
    await router.classify("current input", conversation_history=history)

    _, kwargs = mock_client.call_agent.call_args
    sent_messages = kwargs["messages"]
    # system message (index 0) + last 4 history messages + current user message = 6 messages
    assert len(sent_messages) == 6
    assert sent_messages[0]["role"] == "system"
    assert sent_messages[1]["content"] == "3"
    assert sent_messages[2]["content"] == "4"
    assert sent_messages[3]["content"] == "5"
    assert sent_messages[4]["content"] == "6"
    assert sent_messages[5]["content"] == "current input"


@pytest.mark.asyncio
async def test_intent_router_classify_exception_fallback():
    intents = [
        IntentDefinition(name="query", description="Query"),
        IntentDefinition(name="fallback_intent", description="Default intent"),
    ]
    mock_client = MagicMock()
    mock_client.call_agent = AsyncMock(side_effect=Exception("API failure"))

    router = IntentRouter(intents=intents, llm_client=mock_client)
    res = await router.classify("test query")

    assert res["intent"] == "fallback_intent"
    assert res["query"] == "test query"
    assert "Fallback due to failure: API failure" in res["reasoning"]
    assert res["extracted_entities"] == {}


@pytest.mark.asyncio
async def test_intent_router_classify_exception_fallback_empty_intents():
    mock_client = MagicMock()
    mock_client.call_agent = AsyncMock(side_effect=Exception("Network error"))

    router = IntentRouter(intents=[], llm_client=mock_client)
    res = await router.classify("test query")

    assert res["intent"] == "UNKNOWN"
    assert res["query"] == "test query"
    assert "Fallback due to failure: Network error" in res["reasoning"]


# ============================================================================
# lazycat/tool_registry.py _validate_input Tests
# ============================================================================

class FilterModel(BaseModel):
    filters: list[str]
    options: dict[str, int] = Field(default_factory=dict)
    limit: int = 10


def test_tool_registry_validate_input_no_model():
    registry = ToolRegistry()

    async def unmodeled_tool(a, b=1):
        return a + b

    registry.register(unmodeled_tool)
    raw = {"a": 10, "b": 20, "extra": "ignored"}
    assert registry._validate_input("unmodeled_tool", raw) == raw


def test_tool_registry_validate_input_unregistered_tool():
    registry = ToolRegistry()
    raw = {"x": 1}
    assert registry._validate_input("nonexistent_tool", raw) == raw


def test_tool_registry_validate_input_valid():
    registry = ToolRegistry()

    async def filter_tool(**kwargs):
        pass

    registry.register(filter_tool, input_model=FilterModel)
    out = registry._validate_input(
        "filter_tool",
        {"filters": ["ticker:eq:AAPL"], "options": {"retries": 3}, "limit": 5},
    )
    assert out == {
        "filters": ["ticker:eq:AAPL"],
        "options": {"retries": 3},
        "limit": 5,
    }


def test_tool_registry_validate_input_coerces_stringified_json_array_and_object():
    registry = ToolRegistry()

    async def filter_tool(**kwargs):
        pass

    registry.register(filter_tool, input_model=FilterModel)
    out = registry._validate_input(
        "filter_tool",
        {
            "filters": '["sector:eq:TECH", "cap:gt:10B"]',
            "options": '{"timeout": 30, "priority": 1}',
            "limit": 20,
        },
    )
    assert out == {
        "filters": ["sector:eq:TECH", "cap:gt:10B"],
        "options": {"timeout": 30, "priority": 1},
        "limit": 20,
    }


def test_tool_registry_validate_input_coerces_python_repr():
    registry = ToolRegistry()

    async def filter_tool(**kwargs):
        pass

    registry.register(filter_tool, input_model=FilterModel)
    out = registry._validate_input(
        "filter_tool",
        {
            "filters": "['pe:lt:25', 'yield:gt:2']",
            "options": "{'depth': 2}",
        },
    )
    assert out == {
        "filters": ["pe:lt:25", "yield:gt:2"],
        "options": {"depth": 2},
        "limit": 10,
    }


def test_tool_registry_validate_input_missing_required_field_raises():
    registry = ToolRegistry()

    async def filter_tool(**kwargs):
        pass

    registry.register(filter_tool, input_model=FilterModel)
    # 'filters' is required
    with pytest.raises(ValidationError):
        registry._validate_input("filter_tool", {"options": {"a": 1}})


def test_tool_registry_validate_input_type_mismatch_raises():
    registry = ToolRegistry()

    async def filter_tool(**kwargs):
        pass

    registry.register(filter_tool, input_model=FilterModel)
    # 'filters' should be a list, passing integer
    with pytest.raises(ValidationError):
        registry._validate_input("filter_tool", {"filters": 12345})


def test_tool_registry_validate_input_unparseable_stringified_json_raises():
    registry = ToolRegistry()

    async def filter_tool(**kwargs):
        pass

    registry.register(filter_tool, input_model=FilterModel)
    # Invalid JSON that cannot be parsed as a list
    with pytest.raises(ValidationError):
        registry._validate_input("filter_tool", {"filters": "[broken json {("})
