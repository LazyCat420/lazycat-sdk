"""A registered custom agent can carry DENY tool policies.

`availableTools` is NOT a restriction on a prism custom agent. The persona
registry copies availableTools out of the Mongo document but not
`coreToolsLocked`, so the resolver's `persona?.coreToolsLocked ?? true`
defaults a custom agent to LOCKED and force-adds every CORE_AGENTIC / system
tool on top of the list we registered.

That is how trading-service's v3 analysts reached `execute_command`,
`write_file` and `query_datastore` while carrying an explicit whitelist that
named none of them.

A DENY policy is the one restriction that holds: AutoApprovalEngine evaluates
policies before the tier system AND before full-auto, and treats DENY as a
terminal rejection. These agents run with auto_approve=True, so nothing else
can stop a call.
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lazycat.llm import PrismClient


def _mock_client(captured: dict):
    """An httpx-ish client that captures the registration payload."""
    client = MagicMock()

    list_response = MagicMock()
    list_response.raise_for_status = MagicMock()
    list_response.json = MagicMock(return_value=[])
    client.get = AsyncMock(return_value=list_response)

    write_response = MagicMock()
    write_response.raise_for_status = MagicMock()

    async def _post(url, json=None, headers=None, timeout=None):
        captured["url"] = url
        captured["payload"] = json
        return write_response

    client.post = AsyncMock(side_effect=_post)
    client.put = AsyncMock(side_effect=_post)
    return client


async def _register(**kwargs) -> dict:
    captured: dict = {}
    prism = PrismClient()
    with patch.object(
        PrismClient, "_get_client", new=AsyncMock(return_value=_mock_client(captured))
    ):
        await prism.register_or_update_custom_agent(
            name="v3_quant_analyst",
            identity="test identity",
            **kwargs,
        )
    return captured["payload"]


@pytest.mark.asyncio
async def test_policies_are_sent_in_the_registration_payload():
    deny_rules = [
        {"tool": "execute_command", "decision": "DENY", "name": "deny(execute_command)"},
        {"tool": "write_file", "decision": "DENY", "name": "deny(write_file)"},
    ]

    payload = await _register(enabled_tools=["a"], policies=deny_rules)

    assert payload["policies"] == deny_rules


@pytest.mark.asyncio
async def test_policies_are_omitted_when_not_supplied():
    """Absent policies must not overwrite anything already configured."""
    payload = await _register(enabled_tools=["a"])

    assert "policies" not in payload


@pytest.mark.asyncio
async def test_an_empty_policy_list_is_also_omitted():
    payload = await _register(enabled_tools=["a"], policies=[])

    assert "policies" not in payload


@pytest.mark.asyncio
async def test_core_tools_locked_still_defaults_to_false():
    """Regression guard: locking ADDS the core set, it does not remove it."""
    payload = await _register(enabled_tools=["a"])

    assert payload["coreToolsLocked"] is False
