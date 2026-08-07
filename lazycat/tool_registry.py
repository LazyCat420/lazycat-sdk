"""
Tool Registry — Central registry for all agent tools.

Inspired by Claude Code's buildTool() pattern:
  - Every tool has typed metadata (tier, source, permission, size limits)
  - Input validation via Pydantic models (optional per tool)
  - Result truncation prevents context overflow
  - Permission levels gate destructive operations
"""

import inspect
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)


# ── Permission Levels (inspired by Claude Code's isReadOnly/isDestructive) ──
class PermissionLevel(str, Enum):
    """Permission level for a tool. Higher levels require more oversight.

    READ_ONLY:   Safe to call freely — no side effects (e.g., get_market_data)
    WRITE:       Creates/modifies data — logged but auto-approved (e.g., write_memory_note)
    DESTRUCTIVE: Irreversible operations — requires human approval (e.g., run_local_command, deploy_fix)
    """

    READ_ONLY = "read_only"
    WRITE = "write"
    DESTRUCTIVE = "destructive"


@dataclass
class ToolMeta:
    """Metadata for a registered tool.

    Modeled after Claude Code's Tool type, which includes isEnabled(),
    isReadOnly(), isDestructive(), isConcurrencySafe(), and maxResultSizeChars.
    """

    tier: int = 0  # 0=collect, 1=analyze, 2=validate
    source: str = ""  # e.g. "yfinance", "hermes", "reddit"
    fallback_only: bool = False  # if True, only invoked when primary tools return empty
    permission: PermissionLevel = PermissionLevel.READ_ONLY
    max_result_chars: int = (
        50_000  # Truncate results beyond this to prevent context overflow
    )
    input_model: type[BaseModel] | None = (
        None  # Optional Pydantic model for input validation
    )
    concurrency_safe: bool = True  # If False, tool should not be called in parallel
    tags: list[str] = field(
        default_factory=list
    )  # Search keywords for future ToolSearch
    domain: str | None = None
    labels: list[str] = field(
        default_factory=list
    )


def get_default_domain_and_labels(tool_name: str) -> tuple[str, list[str]]:
    # Market Data tools
    if tool_name in [
        "get_market_data",
        "get_finnhub_news",
        "get_technical_indicators",
        "get_sec_filings",
        "get_options_flow",
        "get_insider_trades",
        "get_congress_trades",
        "get_earnings_data",
        "get_finviz_fundamentals",
        "get_polygon_price_history",
    ]:
        return "Market Data", ["market-data", "stock-analysis"]

    # Trading & Execution tools
    if tool_name in [
        "buy_stock",
        "sell_stock",
        "add_to_watchlist",
        "remove_from_watchlist",
        "get_portfolio_state",
        "get_position_pnl",
        "get_performance_metrics",
        "set_price_trigger",
        "list_active_triggers",
        "cancel_price_trigger",
    ]:
        return "Trading & Execution", ["trading", "execution", "portfolio"]

    # Research & Intelligence tools
    if tool_name in [
        "search_web",
        "web_search",
        "scrape_url",
        "query_hermes",
        "hermes_web_research",
        "search_internal_database",
        "browser_navigate",
        "run_playwright_script",
        "youtube_test_channel",
        "update_youtube_channel_handle",
        "youtube_search",
    ]:
        return "Research & Intelligence", ["research", "scraping", "web-search"]

    # Quant & Analytics tools
    if tool_name in [
        "execute_momentum_strategy",
        "execute_value_strategy",
        "calculate_position_size",
        "calculate_stop_loss",
        "calculate_risk_reward",
        "calculate_portfolio_allocation",
        "execute_quant_script",
        "run_python_script",
    ]:
        return "Quant & Analytics", ["quant", "analytics", "calculations"]

    # Memory & Knowledge tools
    if tool_name in [
        "write_memory_note",
        "read_memory_note",
        "search_wiki",
        "read_profile",
        "update_preference",
        "add_agent_note",
        "search_trading_skills",
    ]:
        return "Memory & Knowledge", ["memory", "knowledge", "wiki"]

    # Agent Coordination tools
    if tool_name in [
        "post_finding",
        "read_team_findings",
        "request_investigation",
        "check_open_investigations",
        "get_cycle_context",
        "get_cycle_context_all",
    ]:
        return "Agent Coordination", ["coordination", "teamwork", "cycle-context"]

    # System & Autonomy tools
    if tool_name in [
        "run_local_command",
        "audit_decision_quality",
        "check_hallucination",
        "propose_constitution_amendment",
        "create_or_update_schedule",
        "list_active_schedules",
    ]:
        return "System & Autonomy", ["system", "autonomy", "audit"]

    return "General", ["tool"]


class ToolRegistry:
    """Registry of tool implementations and the schemas agents are shown.

    ONE SCHEMA PER NAME. `schemas` is a list because every consumer sends it
    straight to an LLM as the `tools` array, but a name may appear at most
    once. Two sources feed it — the compiled catalog (`load_from_json`) and the
    `@register` decorator — and both used to append blindly, so any tool
    present in both was sent to the model TWICE.

    Measured in trading-service on 2026-07-31: 54 of 56 tools doubled, so every
    agent's tool array was exactly 2x its whitelist (user_chat 29 -> 56 schemas,
    v3_worker_fundamental 4 -> 8) on every single LLM call. Three of the pairs
    disagreed about their contract, which is how `get_sec_filings` was shown as
    both `required: ["ticker"]` and `required: []` with a `symbol` alias, and
    failed ~20% of calls.

    THE CATALOG WINS. When both sources define a name, the compiled catalog's
    schema is the one kept, because that is already the one enforced: the arg
    filter and required-field check read `_schema_params`, which returns the
    first match, and the catalog loads first. Keeping the catalog makes what
    the model is SHOWN identical to what the executor ENFORCES. The decorator's
    schema is the fallback for tools the catalog does not carry — which is also
    why `description=` on the decorator appears inert for catalogued tools.
    """

    def __init__(self):
        self.tools: dict[str, Callable] = {}
        self.schemas: list[dict] = []
        self._meta: dict[str, ToolMeta] = {}
        # name -> index into self.schemas, so replacing a schema keeps its
        # position (order is stable across boots, which keeps prompt caching
        # effective) instead of moving it to the end.
        self._schema_index: dict[str, int] = {}

    def _reindex(self) -> None:
        """Rebuild `_schema_index` from `schemas`, which is the authority."""
        self._schema_index = {}
        for i, s in enumerate(self.schemas):
            n = (s.get("function") or {}).get("name")
            if n and n not in self._schema_index:
                self._schema_index[n] = i

    def _index_of(self, name: str) -> int | None:
        """The position of `name`'s schema, verified against `schemas`.

        `_schema_index` is a second copy of information `schemas` already
        carries, so any code that edits `schemas` directly desynchronises it.
        That is not hypothetical: a caller removing a tool with

            registry.schemas = [s for s in registry.schemas if ...]

        leaves a stale entry pointing past the end (IndexError on the next
        registration of that name) and shifts every later tool's position by
        one — which is the WORSE half, because a stale-but-in-range index
        silently overwrites an UNRELATED tool's schema. Verify the slot before
        trusting it and rebuild when it does not match.
        """
        at = self._schema_index.get(name)
        if at is None:
            return None
        if at < len(self.schemas):
            found = (self.schemas[at].get("function") or {}).get("name")
            if found == name:
                return at
        self._reindex()
        return self._schema_index.get(name)

    def unregister(self, name: str) -> bool:
        """Remove a tool, its schema and its metadata. Returns whether it existed.

        Exists so callers never have to reach into `schemas`/`tools`/`_meta`
        by hand — the index cannot be kept correct from outside, and the three
        structures have to move together. Mainly for test teardown, where a
        tool registered by one test must not leak into the next.
        """
        existed = False
        if name in self.tools:
            del self.tools[name]
            existed = True
        if name in self._meta:
            del self._meta[name]
            existed = True
        before = len(self.schemas)
        self.schemas = [
            s for s in self.schemas
            if (s.get("function") or {}).get("name") != name
        ]
        if len(self.schemas) != before:
            existed = True
        self._reindex()
        return existed

    def _put_schema(self, name: str, schema: dict, *, source: str) -> None:
        """Insert or replace the single schema for `name`.

        `source` is "catalog" or "decorator". A catalog entry overwrites a
        decorator entry; a decorator entry never overwrites a catalog entry.
        Contract disagreements are logged rather than silently resolved — a
        shadowed schema is drift between the Python signature and the compiled
        catalog, and it should be fixed at the source folder, not here.
        """
        if not name:
            return
        existing_at = self._index_of(name)
        if existing_at is None:
            self._schema_index[name] = len(self.schemas)
            self.schemas.append(schema)
            return

        kept, incoming = self.schemas[existing_at], schema
        if source == "catalog":
            self.schemas[existing_at] = incoming
            kept, incoming = incoming, kept

        def _contract(s: dict) -> tuple:
            p = (s.get("function", {}).get("parameters") or {})
            return (
                tuple(sorted((p.get("properties") or {}).keys())),
                tuple(sorted(p.get("required") or [])),
            )

        if _contract(kept) != _contract(incoming):
            logger.warning(
                "[ToolRegistry] %s: catalog and decorator declare DIFFERENT "
                "parameters — keeping the catalog's %s, ignoring %s. Fix the "
                "mismatch in tool_schemas/ so the model sees what the executor "
                "enforces.",
                name, _contract(kept), _contract(incoming),
            )

    def load_from_json(self, filepath: str):
        """Load schemas from a pre-compiled JSON file (e.g. tool_schemas.json)."""
        import os
        if not os.path.exists(filepath):
            logger.warning(f"[ToolRegistry] Schema file not found: {filepath}")
            return
            
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            
            raw_list = []
            if isinstance(data, list):
                raw_list = data
            elif isinstance(data, dict) and "schemas" in data:
                raw_list = data["schemas"]

            normalized_schemas = []
            for s in raw_list:
                if "function" in s:
                    normalized = s
                    func_info = s["function"]
                    name = func_info.get("name")
                else:
                    name = s.get("name")
                    func_info = {
                        "name": name,
                        "description": s.get("description", ""),
                        "parameters": s.get("parameters", {})
                    }
                    normalized = {
                        "type": "function",
                        "function": func_info
                    }
                
                normalized_schemas.append(normalized)
                
                if name:
                    if name not in self.tools:
                        self.tools[name] = None
                    
                    perm_str = s.get("permission", "read_only")
                    try:
                        perm = PermissionLevel(perm_str)
                    except ValueError:
                        perm = PermissionLevel.READ_ONLY
                        
                    self._meta[name] = ToolMeta(
                        tier=s.get("tier", 0),
                        source=s.get("source", ""),
                        fallback_only=s.get("fallback_only", False),
                        permission=perm,
                        max_result_chars=s.get("max_result_chars", 50000),
                        concurrency_safe=s.get("concurrency_safe", True),
                        tags=s.get("tags", []),
                        domain=s.get("domain"),
                        labels=s.get("labels", []),
                    )
            
            for norm in normalized_schemas:
                self._put_schema(
                    norm.get("function", {}).get("name", ""), norm, source="catalog"
                )
        logger.info(
            "[ToolRegistry] Loaded %d schemas from %s (%d unique tools registered)",
            len(normalized_schemas), filepath, len(self.schemas),
        )

    def register(
        self,
        func: Callable | None = None,
        name: str | None = None,
        description: str | None = None,
        parameters: dict | None = None,
        *,
        tier: int = 0,
        source: str = "",
        fallback_only: bool = False,
        permission: PermissionLevel = PermissionLevel.READ_ONLY,
        max_result_chars: int = 50_000,
        input_model: type[BaseModel] | None = None,
        concurrency_safe: bool = True,
        tags: list[str] | None = None,
        domain: str | None = None,
        labels: list[str] | None = None,
    ):
        """Register an async function as a tool. Can be used as a decorator with or without arguments.

        Args:
            tier: Processing tier (0=collect, 1=analyze, 2=validate).
            source: Data source label (e.g. "yfinance", "hermes").
            fallback_only: If True, only invoke when primary tools return empty results.
            permission: Permission level (read_only, write, destructive).
            max_result_chars: Max chars for tool output before truncation.
            input_model: Optional Pydantic BaseModel for input validation.
            concurrency_safe: If False, this tool should not be called in parallel.
            tags: Search keywords for future ToolSearch feature.
        """

        def decorator(f: Callable):
            tool_name = name or f.__name__
            self.tools[tool_name] = f

            resolved_domain = domain
            resolved_labels = labels
            if resolved_domain is None or resolved_labels is None:
                def_domain, def_labels = get_default_domain_and_labels(tool_name)
                if resolved_domain is None:
                    resolved_domain = def_domain
                if resolved_labels is None:
                    resolved_labels = def_labels

            self._meta[tool_name] = ToolMeta(
                tier=tier,
                source=source,
                fallback_only=fallback_only,
                permission=permission,
                max_result_chars=max_result_chars,
                input_model=input_model,
                concurrency_safe=concurrency_safe,
                tags=tags or [],
                domain=resolved_domain,
                labels=resolved_labels,
            )

            self._put_schema(
                tool_name,
                {
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "description": description
                        or f.__doc__
                        or "No description provided.",
                        "parameters": parameters
                        or {"type": "object", "properties": {}, "required": []},
                    },
                },
                source="decorator",
            )
            return f

        if func is None:
            return decorator
        else:
            return decorator(func)

    def is_fallback(self, name: str) -> bool:
        """Check if a tool is marked as fallback-only."""
        meta = self._meta.get(name)
        return meta.fallback_only if meta else False

    def get_tool_meta(self, name: str) -> ToolMeta | None:
        """Get metadata for a registered tool."""
        return self._meta.get(name)

    def get_primary_schemas(self) -> list[dict]:
        """Get schemas for non-fallback tools only (Layer 1 structured APIs)."""
        return [s for s in self.schemas if not self.is_fallback(s["function"]["name"])]

    def get_fallback_schemas(self) -> list[dict]:
        """Get schemas for fallback-only tools (Layer 2 Hermes)."""
        return [s for s in self.schemas if self.is_fallback(s["function"]["name"])]

    def get_schemas_by_names(self, names: list[str]) -> list[dict]:
        """Get schemas for specific tools by name (whitelist filtering)."""
        return [s for s in self.schemas if s["function"]["name"] in names]

    def get_schemas_by_tier(self, tier: int) -> list[dict]:
        """Get schemas for tools at a specific processing tier."""
        return [
            s
            for s in self.schemas
            if self._meta.get(s["function"]["name"], ToolMeta()).tier == tier
        ]

    def get_schemas_by_permission(self, permission: PermissionLevel) -> list[dict]:
        """Get schemas for tools at a specific permission level."""
        return [
            s
            for s in self.schemas
            if self._meta.get(s["function"]["name"], ToolMeta()).permission
            == permission
        ]

    def _validate_input(self, func_name: str, kwargs: dict) -> dict:
        """Validate tool input against Pydantic model if one is registered.

        Returns validated kwargs (potentially with defaults filled in).
        Raises ValidationError if input is invalid.
        """
        meta = self._meta.get(func_name)
        if not meta or not meta.input_model:
            return kwargs

        try:
            validated = meta.input_model(**kwargs)
            return validated.model_dump()
        except ValidationError as e:
            logger.warning(
                "[ToolRegistry] Input validation FAILED for %s: %s | raw kwargs: %s",
                func_name,
                e.error_count(),
                kwargs,
            )
            raise

    def _schema_params(self, func_name: str) -> tuple[set[str], set[str]]:
        """Declared (properties, required) parameter names for a tool.

        Returns empty sets when the tool declares no schema, which callers must
        treat as "pass everything through" rather than "accept nothing".
        """
        for schema in self.schemas:
            fn = schema.get("function", {})
            if fn.get("name") != func_name:
                continue
            params = fn.get("parameters") or {}
            props = params.get("properties") or {}
            required = params.get("required") or []
            return set(props.keys()), set(required)
        return set(), set()

    def _unbindable_params(self, func_name: str, kwargs: dict) -> list[str] | None:
        """Parameters that make `func(**kwargs)` raise TypeError, or None.

        This is the oracle for "the call cannot execute", and it is deliberately
        the FUNCTION's signature rather than the schema's `required` list. The
        two disagree: 4 of the 40 tools with required fields declare one the
        Python function happily defaults (`get_sec_filings(ticker='')`,
        `whiteboard_read(section='')`, `schedule_research(reason='')`,
        `save_trading_chart(overlays=None)`). Rejecting on the schema's list
        would newly block those calls, and `whiteboard_read` omitting `section`
        is a legitimate read of the desk's own scratchpad.

        Binding instead means we intercept exactly the calls that were going to
        die anyway, and change nothing else.

        Returns None when the question does not apply — no local implementation
        (remote/schema-only tools cannot raise a local TypeError) or a callable
        that cannot be introspected.
        """
        func = self.tools.get(func_name)
        if func is None:
            return None
        try:
            sig = inspect.signature(func)
        except (TypeError, ValueError):
            return None
        try:
            sig.bind(**kwargs)
            return None
        except TypeError:
            pass
        return sorted(
            name
            for name, p in sig.parameters.items()
            if p.default is inspect.Parameter.empty
            and p.kind
            not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
            and name not in kwargs
        )

    def _filter_kwargs_to_schema(
        self, func_name: str, kwargs: dict
    ) -> tuple[dict, list[str]]:
        """Drop arguments the tool never declared.

        Model-emitted arguments are splatted into the Python function, so a
        malformed key becomes a TypeError that kills the call. Observed in
        production: whiteboard_write received a key that was a FRAGMENT of a
        JSON array value —

            whiteboard_write() got an unexpected keyword argument
            '"regulatory clearance for ai features in china"], "r'

        …which failed the write, so a red-flag the analyst had already found
        never reached the whiteboard and the board decided without it. The
        schema is the contract the model was handed; anything outside it is
        noise and must not be able to break the call.
        """
        props, _ = self._schema_params(func_name)
        if not props:
            return kwargs, []  # no declared schema — nothing to filter against
        clean = {k: v for k, v in kwargs.items() if k in props}
        dropped = [k for k in kwargs if k not in props]
        return clean, dropped

    def _truncate_result(self, func_name: str, result: str) -> str:
        """Truncate tool result if it exceeds max_result_chars.

        Prevents context window overflow when tools return huge payloads
        (e.g., scraping a full webpage, dumping entire price histories).
        """
        meta = self._meta.get(func_name)
        max_chars = meta.max_result_chars if meta else 50_000

        if len(result) <= max_chars:
            return result

        # Keep first 80% + last 10%, with a truncation notice in the middle
        head_chars = int(max_chars * 0.8)
        tail_chars = int(max_chars * 0.1)
        truncated_count = len(result) - head_chars - tail_chars
        notice = (
            f"\n\n[TRUNCATED: {truncated_count:,} characters removed. "
            f"Original size: {len(result):,} chars.]\n\n"
        )
        truncated = result[:head_chars] + notice + result[-tail_chars:]

        logger.info(
            "[ToolRegistry] Truncated %s output: %d → %d chars (-%d)",
            func_name,
            len(result),
            len(truncated),
            truncated_count,
        )
        return truncated

    def check_permission(self, func_name: str) -> tuple[bool, str]:
        """Check if a tool is allowed to execute based on its permission level.

        Returns (allowed: bool, reason: str).
        DESTRUCTIVE tools are blocked by default — caller must handle approval.
        """
        meta = self._meta.get(func_name)
        if not meta:
            return True, "no metadata (legacy tool, allowed)"

        if meta.permission == PermissionLevel.DESTRUCTIVE:
            return False, (
                f"Tool '{func_name}' is DESTRUCTIVE (permission={meta.permission.value}). "
                f"Requires human approval before execution."
            )

        return True, f"permission={meta.permission.value}"

    async def execute_tool_call(
        self,
        tool_call: dict,
        *,
        skip_permission_check: bool = False,
        agent_name: str = "",
        ticker: str = "",
        cycle_id: str = "",
        tool_cache: dict | None = None,
        enforce_ticker: bool = False,
        force_local: bool = False,
    ) -> dict:
        """Execute a single tool call from the LLM and return the formatted result.

        Args:
            tool_call: The tool call dict from the LLM response.
            skip_permission_check: If True, bypass permission checks (for pre-approved calls).
            agent_name: The agent requesting this tool (for usage tracking).
            ticker: Current ticker context (for usage tracking).
            cycle_id: Current cycle context (for usage tracking).
            tool_cache: Pre-fetched tool results to intercept redundant executions.
            enforce_ticker: If True, block tool calls where the 'ticker' argument
                            doesn't match the context ticker. Used during debates
                            to prevent cross-ticker data contamination.
            force_local: If True, ignore USE_LAZY_TOOL_SERVICE and run the local
                         registration function. Used by HTTP execute endpoints that
                         lazy-tool-service itself calls — without it the call would
                         bounce back to lazy-tool-service in an infinite loop.
        """
        tool_call_id = tool_call.get("id")
        function_info = tool_call.get("function", {})
        func_name = function_info.get("name")
        arguments_json = function_info.get("arguments", "{}")

        # ── Robust Normalization of Tool Name & Arguments ──
        if func_name:
            # Strip trailing tags like </Function
            if "</" in func_name:
                func_name = func_name.split("</")[0].strip()
            
            # Map capitalized names with spaces/hyphens to registered lowercase snake_case names
            def clean_str(s: str) -> str:
                return s.lower().replace(" ", "").replace("_", "").replace("-", "")
            
            target_cleaned = clean_str(func_name)
            matched_name = None
            for registered_name in self.tools:
                reg_cleaned = clean_str(registered_name)
                # Check for exact normalized match
                if reg_cleaned == target_cleaned:
                    matched_name = registered_name
                    break
                # Check with get_ prefix variation (e.g. "Technical Indicators" -> "get_technical_indicators")
                if clean_str("get_" + registered_name) == target_cleaned or reg_cleaned == clean_str("get_" + target_cleaned):
                    matched_name = registered_name
                    break

            if matched_name:
                if matched_name != func_name:
                    logger.info("[ToolRegistry] Normalized tool call name: '%s' -> '%s'", func_name, matched_name)
                func_name = matched_name
                # Update function info representation
                function_info["name"] = func_name

        # ── Parse and Normalize JSON arguments ──
        try:
            # Clean up trailing tags from arguments string if present
            if arguments_json and "</" in arguments_json:
                arguments_json = arguments_json.split("</")[0].strip()
                
            kwargs = json.loads(arguments_json)
            
            # Convert all argument keys to lowercase (e.g. {"Ticker": "IP"} -> {"ticker": "IP"})
            kwargs = {k.lower(): v for k, v in kwargs.items()}

            # Drop anything the tool never declared, so a malformed key cannot
            # reach func(**kwargs) and raise TypeError.
            kwargs, _dropped_keys = self._filter_kwargs_to_schema(func_name, kwargs)
            if _dropped_keys:
                logger.warning(
                    "[ToolRegistry] %s: dropped %d undeclared argument(s): %s",
                    func_name,
                    len(_dropped_keys),
                    [k[:60] for k in _dropped_keys],
                )

            # Required-field check, for EVERY call — not just the ones that also
            # carried junk. This block used to be nested under `if _dropped_keys`,
            # so it only ever fired when the model sent an undeclared key as well.
            # A plain omission skipped the check entirely and died inside
            # `func(**kwargs)` as a raw `TypeError: ... missing 1 required
            # positional argument`, which the model cannot act on and simply
            # retries verbatim.
            #
            # Two triggers, deliberately different. `_unbindable_params` is the
            # strict one: the call physically cannot execute, so intercepting it
            # costs nothing and replaces a guaranteed TypeError. The schema's
            # `required` list is used only where it already was — alongside
            # dropped keys — because it over-declares (see _unbindable_params)
            # and promoting it to a standalone gate would block working calls.
            _props, _required = self._schema_params(func_name)
            _unbindable = self._unbindable_params(func_name, kwargs)
            _missing = _unbindable or (
                sorted(_required - set(kwargs)) if _dropped_keys else []
            )
            if _missing:
                # Hand the model the schema instead of a TypeError it cannot
                # interpret — a bare error just gets retried unchanged.
                self._log_usage(
                    func_name or "unknown", agent_name, ticker, cycle_id,
                    False, 0,
                    f"Malformed arguments: missing {_missing}",
                )
                if _dropped_keys:
                    _cause = (
                        f", and {len(_dropped_keys)} argument(s) were not part of "
                        f"this tool's schema. This usually means the JSON was not "
                        f"escaped correctly — check that quotes inside string values "
                        f"are escaped and that arrays are closed"
                    )
                else:
                    _cause = " and must be supplied"
                return {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": func_name,
                    "content": json.dumps(
                        {
                            "error": (
                                f"Malformed tool arguments. Required field(s) "
                                f"{_missing} were missing{_cause}."
                            ),
                            "expected_arguments": sorted(_props),
                            "required_arguments": sorted(_required),
                            "hint": "Re-issue the call with valid JSON matching the schema above.",
                        }
                    ),
                }

            # Update the parsed arguments representation
            arguments_json = json.dumps(kwargs)
            function_info["arguments"] = arguments_json
            
            normalized_args = json.dumps(kwargs, sort_keys=True, separators=(',', ':'))
            cache_key = f"{func_name}:{normalized_args}"
        except Exception as e:
            logger.error(
                "[ToolRegistry] Failed to parse JSON arguments for %s: %s | raw: %s",
                func_name,
                e,
                arguments_json[:200],
            )
            self._log_usage(
                func_name or "unknown",
                agent_name,
                ticker,
                cycle_id,
                False,
                0,
                f"JSON parse error: {e}",
            )
            return {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": func_name,
                "content": json.dumps(
                    {
                        "error": f"Invalid JSON arguments: {e}",
                        "hint": "Please provide valid JSON arguments.",
                    }
                ),
            }

        # ── Ticker-Lock Guardrail ──
        if ticker and "ticker" in kwargs:
            tool_ticker = str(kwargs["ticker"]).upper().strip()
            context_ticker = str(ticker).upper().strip()
            if tool_ticker and tool_ticker != context_ticker:
                logger.warning(
                    "[ToolRegistry] CROSS-CONTAMINATION BLOCKED: %s requested ticker %s but context is %s",
                    func_name,
                    tool_ticker,
                    context_ticker,
                )
                self._log_usage(
                    func_name, agent_name, False, 0, f"Cross-contamination blocked: {tool_ticker} != {context_ticker}"
                )
                return {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": func_name,
                    "content": json.dumps({
                        "error": f"Unauthorized ticker access. You are analyzing {context_ticker}, but you requested {tool_ticker}.",
                        "hint": f"Only query data for the assigned ticker ({context_ticker})."
                    })
                }

        # ── Tool existence check ──
        if func_name not in self.tools:
            self._log_usage(
                func_name or "unknown",
                agent_name,
                ticker,
                cycle_id,
                False,
                0,
                f"Tool '{func_name}' not found",
            )
            return {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": func_name,
                "content": json.dumps({"error": f"Tool '{func_name}' not found."}),
            }

        # ── Cache interception check ──
        if tool_cache:
            cached_content = None
            hit_key = None
            if cache_key in tool_cache:
                cached_content = tool_cache[cache_key]
                hit_key = cache_key
            elif func_name in tool_cache:
                cached_content = tool_cache[func_name]
                hit_key = func_name

            if cached_content is not None:
                logger.info(
                    "[ToolRegistry] Intercepted %s via tool_cache (%s) to prevent redundant execution.",
                    func_name,
                    hit_key,
                )
                # Log cache hit usage (0ms execution)
                self._log_usage(
                    func_name, agent_name, True, 0, "Cache Hit"
                )
                return {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": func_name,
                    "content": cached_content,
                }

        # ── Permission check ──
        if not skip_permission_check:
            allowed, reason = self.check_permission(func_name)
            if not allowed:
                logger.warning(
                    "[ToolRegistry] PERMISSION DENIED: %s — %s", func_name, reason
                )
                self._log_usage(
                    func_name,
                    agent_name,
                    ticker,
                    cycle_id,
                    False,
                    0,
                    f"Permission denied: {reason}",
                )
                return {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": func_name,
                    "content": json.dumps(
                        {
                            "error": reason,
                            "requires_approval": True,
                            "pending_command": arguments_json,
                            "action_required": "This tool requires human approval. The request has been logged for review.",
                        }
                    ),
                }

        # ── Input validation (Pydantic) ──
        try:
            kwargs = self._validate_input(func_name, kwargs)
        except ValidationError as e:
            error_details = e.errors()
            self._log_usage(
                func_name,
                agent_name,
                ticker,
                cycle_id,
                False,
                0,
                f"Validation failed: {len(error_details)} errors",
            )
            return {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": func_name,
                "content": json.dumps(
                    {
                        "error": f"Input validation failed: {len(error_details)} error(s)",
                        "details": [
                            {"field": err.get("loc", []), "message": err.get("msg", "")}
                            for err in error_details[:5]
                        ],
                        "hint": "Fix the arguments and try again.",
                    }
                ),
            }

        logger.info(
            "[ToolRegistry] Executing tool: %s with args: %s", func_name, kwargs
        )

        # ── Execute ──
        t0 = time.monotonic()
        try:
            import os
            use_lazy = (
                not force_local
                and os.getenv("USE_LAZY_TOOL_SERVICE", "false").lower() == "true"
            )
            # Bypass lazy-tool-service for save_trading_chart to execute it directly inside trading-service
            if use_lazy and func_name != "save_trading_chart":
                from lazycat.tools import tool_executor
                resp_json = await tool_executor.execute_tool(func_name, kwargs)
                if "error" in resp_json:
                    raise RuntimeError(resp_json["error"])
                result = resp_json.get("content", "")
                service_source = "lazy-tool-service"
            else:
                service_source = "trading-service"
                func = self.tools[func_name]
                if func is None:
                    raise RuntimeError(f"Tool '{func_name}' has no local registration function in this service context.")
                if inspect.iscoroutinefunction(func):
                    result = await func(**kwargs)
                else:
                    result = func(**kwargs)

            if not isinstance(result, str):
                result = json.dumps(result)

            # ── Result truncation ──
            result = self._truncate_result(func_name, result)

            elapsed_ms = int((time.monotonic() - t0) * 1000)
            self._log_usage(func_name, agent_name, True, elapsed_ms)

            return {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": func_name,
                "content": result,
                "service_source": service_source,
            }
        except Exception as e:
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            logger.exception("[ToolRegistry] Tool execution failed for %s", func_name)
            service_source = "lazy-tool-service"
            self._log_usage(func_name, agent_name, False, elapsed_ms, str(e))
            return {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": func_name,
                "content": json.dumps({"error": str(e)}),
                "service_source": service_source,
            }

    def _log_usage(self, *args, **kwargs) -> None:
        """Log a tool usage event.
        
        Supports both:
          - 5-arg style: (tool_name, agent_name, success, execution_ms, error_message)
          - 7-arg style: (tool_name, agent_name, ticker, cycle_id, success, execution_ms, error_message)
        """
        tool_name = "unknown"
        agent_name = None
        ticker = None
        cycle_id = None
        success = True
        execution_ms = 0
        error_message = None

        if len(args) >= 1:
            tool_name = args[0]
        if len(args) >= 2:
            agent_name = args[1]

        if len(args) >= 3:
            if isinstance(args[2], bool):
                # 5-argument style: (tool_name, agent_name, success, execution_ms, error_message)
                success = args[2]
                if len(args) >= 4:
                    execution_ms = args[3] if isinstance(args[3], int) else 0
                if len(args) >= 5:
                    error_message = args[4]
            else:
                # 7-argument style: (tool_name, agent_name, ticker, cycle_id, success, execution_ms, error_message)
                ticker = args[2]
                if len(args) >= 4:
                    cycle_id = args[3]
                if len(args) >= 5:
                    success = args[4] if isinstance(args[4], bool) else True
                if len(args) >= 6:
                    execution_ms = args[5] if isinstance(args[5], int) else 0
                if len(args) >= 7:
                    error_message = args[6]

        # Apply kwargs overrides if present
        tool_name = kwargs.get("tool_name", tool_name)
        agent_name = kwargs.get("agent_name", agent_name)
        ticker = kwargs.get("ticker", ticker)
        cycle_id = kwargs.get("cycle_id", cycle_id)
        success = kwargs.get("success", success)
        execution_ms = kwargs.get("execution_ms", execution_ms)
        error_message = kwargs.get("error_message", error_message)

        import os
        if getattr(self, "_telemetry_callback", None):
            try:
                self._telemetry_callback(tool_name, agent_name, success, execution_ms, error_message)
            except Exception as e:
                logger.debug(f"[ToolRegistry] Telemetry callback failed: {e}")
                
        if os.environ.get("SKIP_TOOL_USAGE_LOG", "false").lower() == "true":
            return

        status = "SUCCESS" if success else "FAILED"
        msg = f"Tool Execution: {tool_name} by {agent_name or 'unknown'} - {status} ({execution_ms}ms)"
        if error_message:
            msg += f" Error: {error_message}"
            logger.error(msg)
        else:
            logger.info(msg)

    def get_registry_snapshot(self) -> list[dict]:
        """Return a snapshot of all registered tools with their metadata.

        Used by the /tools API endpoint for frontend introspection.
        """
        snapshot = []
        for schema in self.schemas:
            func_info = schema.get("function", {})
            name = func_info.get("name", "")
            meta = self._meta.get(name, ToolMeta())
            snapshot.append(
                {
                    "name": name,
                    "description": func_info.get("description", ""),
                    "parameters": func_info.get("parameters", {}),
                    "tier": meta.tier,
                    "source": meta.source,
                    "permission": meta.permission.value,
                    "fallback_only": meta.fallback_only,
                    "concurrency_safe": meta.concurrency_safe,
                    "max_result_chars": meta.max_result_chars,
                    "tags": meta.tags,
                    "domain": meta.domain,
                    "labels": meta.labels,
                }
            )
        return snapshot

    def set_telemetry_callback(self, callback: Callable):
        """Set a callback function for tool telemetry (e.g. database logging)."""
        self._telemetry_callback = callback


registry = ToolRegistry()
