from .llm import prism_client, PrismClient, LLMStreamWrapper, LLMResponseWrapper
from .research import research
from .grounded_research import grounded_research
from .llm_json import (
    extract_json_str,
    parse_json_list_response,
    parse_json_response,
    parse_json_strict,
    strip_think_tags,
)

# The cross-cutting utilities (cache, ratelimit, resilience, sse) stay behind
# explicit submodule imports — e.g. `from lazycat.resilience import ...` — so
# importing the package stays cheap and the top-level surface stays readable.

# Keep in sync with pyproject.toml [project] version
__version__ = "0.3.1"

__all__ = [
    "prism_client",
    "PrismClient",
    "LLMStreamWrapper",
    "LLMResponseWrapper",
    "research",
    "grounded_research",
    "parse_json_response",
    "parse_json_list_response",
    "parse_json_strict",
    "extract_json_str",
    "strip_think_tags",
]
