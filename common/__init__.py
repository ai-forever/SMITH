"""Shared level-0 contracts used by every harness family.

Family packages (`smit_harness`, `search_r1_harness`) own loop-specific
config/agents/tools. Import shared types from here instead of reaching into
a sibling family package.
"""

from .agent import AgentLike
from .format import search_hit_to_ranked_dict
from .protocols import (
    DocumentFetcher,
    FetchedDocument,
    LLMClient,
    LLMResponse,
    Message,
    SearchClient,
    SearchHit,
    ToolCall,
    ToolSchema,
)
from .progress import RetrieveProgressSink
from .result import RetrievalResult
from .environment_messages import (
    DEFAULT_TOOL_CALLS_EXCEEDED_WARNING,
    GENERIC_TOOL_CALLS_EXCEEDED_WARNING,
    ToolCallsExceededWarning,
    render_generic_tool_calls_exceeded_warning,
    render_tool_calls_exceeded_warning,
    resolve_tool_calls_exceeded_warning,
)
from .turns import DEFAULT_MAX_TURNS, estimate_max_turns, tool_policy_offers_pick

__all__ = [
    "AgentLike",
    "DEFAULT_MAX_TURNS",
    "DEFAULT_TOOL_CALLS_EXCEEDED_WARNING",
    "DocumentFetcher",
    "FetchedDocument",
    "GENERIC_TOOL_CALLS_EXCEEDED_WARNING",
    "LLMClient",
    "LLMResponse",
    "Message",
    "RetrievalResult",
    "RetrieveProgressSink",
    "SearchClient",
    "SearchHit",
    "ToolCall",
    "ToolCallsExceededWarning",
    "ToolSchema",
    "estimate_max_turns",
    "render_generic_tool_calls_exceeded_warning",
    "render_tool_calls_exceeded_warning",
    "resolve_tool_calls_exceeded_warning",
    "search_hit_to_ranked_dict",
    "tool_policy_offers_pick",
]
