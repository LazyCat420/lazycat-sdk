"""Universal deep-research client.

One place, reused by every repo, for "research this topic deeply and hand me back
a structured brief." It drives the server-side DEEP_RESEARCH agent (registered in
lazy-tool-service, see personas/clients/DeepResearchPersona.ts) over the gateway's
/agent endpoint: the agent decomposes the topic, fans out parallel sub-researchers,
reads sources, and finishes by calling emit_structured_output. This helper supplies
the task + output contract, drains the SSE stream, and returns the structured data.

Before this, each consumer re-implemented the whole loop — music-player's
apps/api/app/services/llm.py (RESEARCH_TOOLS + a bespoke SSE drain) and
HTML-Notes' build_stock_report_config (an in-process fan-out that bypassed the
agent). Callers now just:

    from lazycat.research import research
    brief = await research(
        "what moved the US stock market today and why",
        schema={
            "title": "short headline for the brief",
            "overview": "one-sentence bottom line",
            "answer": "the full brief in GitHub-flavored Markdown",
            "sources": [{"title": "...", "url": "...", "publisher": "..."}],
        },
        domain="finance",
    )
    # brief == the dict the agent emitted, shaped like `schema`

and render `brief` however they like. The output *shape* is enforced by prompt
(the agent has no JSON-schema validator), so `schema` is a plain dict whose values
DESCRIBE each field — precise, human-readable hints, not JSON-Schema.
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional

import httpx

from .logging import get_logger

logger = get_logger(__name__)

# The gateway (lazy-tool-service) is where the DEEP_RESEARCH persona lives — NOT
# real prism (:7777), which has no such catalog registered. Host :5591 maps to the
# container's :7778. Overridable for other hosts / the in-container port.
DEFAULT_GATEWAY_URL = os.getenv(
    "LAZY_AGENT_URL", os.getenv("LAZY_TOOL_SERVICE_URL", "http://10.0.0.16:5591")
)

# Mirrors DeepResearchPersona.availableTools. Sent as enabledTools so the tools are
# active even on a gateway/prism instance that hasn't loaded the persona (the run
# then behaves like a generic agent with exactly this scope instead of erroring).
RESEARCH_TOOLS = [
    "emit_structured_output",
    "search_web",
    "read_url",
    "read_web_page",
    "search_news",
    "create_subagents",
    "get_subagent_output",
]

# Fallback model/provider pair when /config-local discovery fails. Gold Spark
# (10.0.0.141) is registered as "vllm-2" on the gateway; the Jetson pair is "vllm".
_FALLBACK_PROVIDER = "vllm"
_FALLBACK_MODEL = "cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit"


async def _discover_model(client: httpx.AsyncClient, base: str) -> tuple[str, str]:
    """Pick a tool-calling conversation model from the gateway's local catalog so
    the provider instance and model always match (vllm vs vllm-2 serve different
    models; a mismatched pair 404s with "model does not exist")."""
    try:
        r = await client.get(f"{base}/config-local", timeout=12.0)
        if r.status_code == 200:
            for provider_id, models in (r.json().get("models") or {}).items():
                for m in models:
                    if m.get("modelType") == "conversation" and "Tool Calling" in (
                        m.get("tools") or []
                    ):
                        return provider_id, m.get("name")
    except Exception as e:
        logger.warning("research: model discovery failed (%s); using fallback", e)
    return _FALLBACK_PROVIDER, _FALLBACK_MODEL


def _build_system_prompt(topic: str, schema: Optional[dict], domain: Optional[str]) -> str:
    """The task + output contract. The DEEP_RESEARCH persona already carries the
    research method (decompose → fan out → read → synthesize); this only states
    WHAT to research and the EXACT JSON to emit."""
    parts = [f"RESEARCH TASK: {topic.strip()}"]
    if domain:
        parts.append(f"DOMAIN: {domain}. Prefer authoritative {domain} sources.")
    if schema:
        contract = json.dumps(schema, indent=2, ensure_ascii=False)
        parts.append(
            "OUTPUT CONTRACT — finish by calling emit_structured_output with `data` "
            "as a JSON object with EXACTLY these keys. Each value below DESCRIBES "
            "what that key must contain; replace it with the real value:\n" + contract
        )
    else:
        parts.append(
            "Finish by calling emit_structured_output with `data` as a JSON object "
            "holding: title (short headline), overview (one-sentence bottom line), "
            "answer (the full findings in GitHub-flavored Markdown), and sources "
            "(a list of {title, url, publisher} you actually used)."
        )
    parts.append(
        "Base every claim on sources you actually read — never invent figures, "
        "dates, quotes, or events. If a source is missing, say so briefly rather "
        "than padding."
    )
    return "\n\n".join(parts)


def _extract_json(text: str) -> Optional[dict]:
    """Best-effort: pull the first balanced JSON object out of free text (the
    fallback when the agent narrated the answer instead of calling the tool)."""
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            c = text[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


async def research(
    topic: str,
    schema: Optional[dict] = None,
    *,
    domain: Optional[str] = None,
    gateway_url: Optional[str] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    agent: str = "DEEP_RESEARCH",
    max_iterations: int = 14,
    max_tokens: int = 3000,
    temperature: float = 0.2,
    # Default OFF: on Qwen-class models the <think> stream is most of the latency,
    # and a research run is many iterations — leaving it on (the gateway's agent
    # default) pushed even a scoped run past 280s. Pass True to trade speed for
    # deliberation; the DEEP_RESEARCH persona also defaults thinking off.
    thinking_enabled: Optional[bool] = False,
    timeout: float = 240.0,
    project: str = "lazycat-sdk-research",
    username: str = "lazycat",
) -> Optional[dict[str, Any]]:
    """Run a deep-research pass and return the agent's structured `data` object.

    Args:
        topic: what to research, in plain language.
        schema: a dict whose keys are the required output fields and whose values
            DESCRIBE each field (human-readable hints, not JSON-Schema). Omit for a
            sensible default brief (title/overview/answer/sources).
        domain: optional subject hint ("finance", "medicine", ...) to steer sourcing.
        gateway_url: override the DEEP_RESEARCH gateway (defaults to LAZY_AGENT_URL).
        provider/model: pin the backend; auto-discovered from /config-local if unset.
        agent: gateway agent id; DEEP_RESEARCH by default. Falls to a generic agent
            with the same tool scope if the persona isn't registered on that host.
        thinking_enabled: None leaves the persona default (off). True buys quality at
            the cost of a long <think> stream on Qwen-class models.

    Returns:
        The emitted `data` dict (shaped like `schema`), or a best-effort dict parsed
        from the agent's final text, or None if the run produced nothing usable.
    """
    base = (gateway_url or DEFAULT_GATEWAY_URL).rstrip("/")
    system_prompt = _build_system_prompt(topic, schema, domain)

    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=8.0)) as client:
        if not (provider and model):
            provider, model = await _discover_model(client, base)

        payload: dict[str, Any] = {
            "provider": provider,
            "model": model,
            "agent": agent,
            "enabledTools": RESEARCH_TOOLS,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": topic.strip()},
            ],
            "maxTokens": max_tokens,
            "temperature": temperature,
            "maxIterations": max_iterations,
            "autoApprove": True,
            "skipConversation": True,
            "memoryEnabled": False,
            "project": project,
            "username": username,
        }
        if thinking_enabled is not None:
            payload["thinkingEnabled"] = thinking_enabled

        emitted: Optional[dict] = None
        final_text = ""
        tool_names: list[str] = []
        try:
            async with client.stream(
                "POST",
                f"{base}/agent",
                json=payload,
                headers={"Accept": "text/event-stream"},
            ) as resp:
                if resp.status_code != 200:
                    body = (await resp.aread()).decode("utf-8", "replace")[:500]
                    logger.error("research: gateway %s: %s", resp.status_code, body)
                    return None
                buf = ""
                async for chunk in resp.aiter_text():
                    buf += chunk
                    while "\n" in buf:
                        line, buf = buf.split("\n", 1)
                        line = line.strip()
                        if not line.startswith("data: "):
                            continue
                        try:
                            ev = json.loads(line[6:])
                        except json.JSONDecodeError:
                            continue
                        etype = ev.get("type")
                        if etype == "chunk":
                            final_text += ev.get("content", "")
                        elif etype == "tool_execution":
                            tool = ev.get("tool") or ev.get("toolCall") or {}
                            name = tool.get("name", "")
                            if name and name not in tool_names:
                                tool_names.append(name)
                            if name == "emit_structured_output":
                                args = tool.get("args") or tool.get("arguments") or {}
                                if isinstance(args, str):
                                    try:
                                        args = json.loads(args)
                                    except json.JSONDecodeError:
                                        args = {}
                                data = args.get("data", args)
                                if isinstance(data, str):
                                    try:
                                        data = json.loads(data)
                                    except json.JSONDecodeError:
                                        pass
                                if isinstance(data, dict) and data:
                                    emitted = data  # last emit wins
                        elif etype == "error":
                            logger.warning("research: agent error: %s", ev.get("message"))
        except Exception as e:
            logger.error("research(%r) stream failed: %s", topic, e)
            # fall through — a partial final_text may still parse

    if emitted:
        logger.info("research(%r): emitted %d keys (tools: %s)",
                    topic[:60], len(emitted), ",".join(tool_names))
        return emitted
    parsed = _extract_json(final_text)
    if parsed:
        logger.info("research(%r): no emit_structured_output; parsed JSON from text", topic[:60])
        return parsed
    logger.warning("research(%r): no structured result (tools used: %s)", topic[:60], tool_names)
    return None
