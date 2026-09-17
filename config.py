"""Resolved behavior settings for the service-agnostic retrieval agent.

``HarnessConfig`` is a *resolved* level-0 behavior slice: it carries prompt
version, final-response strategy, and retrieval/tool/loop limits. It never
carries a dataset id, qrels, OBS path, service URL, or provider credential --
those stay in the host's own settings and get translated into a
``HarnessConfig`` by the host.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from common import DEFAULT_MAX_TURNS, estimate_max_turns
from common.environment_messages import (
    DEFAULT_TOOL_CALLS_EXCEEDED_WARNING,
    ToolCallsExceededWarning,
)

ContextManagementStrategy = Literal["hide_shown_retrieved_docs"]
ToolResultFormatVersion = Literal["v0", "v1", "v2"]
SearchLimitPolicy = Literal["warn_search_limit", "truncate_search_limit"]
SEARCH_ONLY_STRATEGY = "search_only"

#: Final-response strategies the harness tool policy understands directly.
#: Hosts may resolve other provider-facing strategy names (e.g. "provider_strategy",
#: "json_text") down to one of these before building a ``HarnessConfig`` -- the
#: harness itself only needs to know which final tool schema to bind and how to
#: interpret its arguments.
FULL_ANSWER_STRATEGIES = frozenset({"tool_strategy", "custom_giga_tool_strategy"})
CORPUS_IDS_ONLY_STRATEGIES = frozenset(
    {"custom_tool_strategy_corpus_ids_only", "native_structured_tool_corpus_ids_only"}
)
NATIVE_ANSWER_STRATEGY = "native_tool_without_supporting_ids"
OPENRESEARCHER_STRATEGY = "openresearcher"

__all__ = [
    "CORPUS_IDS_ONLY_STRATEGIES",
    "DEFAULT_MAX_TURNS",
    "FULL_ANSWER_STRATEGIES",
    "HarnessConfig",
    "NATIVE_ANSWER_STRATEGY",
    "OPENRESEARCHER_STRATEGY",
    "SEARCH_ONLY_STRATEGY",
    "SearchLimitPolicy",
    "estimate_max_turns",
]


_DEFAULT_SYSTEM_PROMPT_VERSION = "v4.3"
_DEFAULT_FINAL_RESPONSE_STRATEGY = "custom_giga_tool_strategy"
_DEFAULT_MAX_SEARCH_CALLS = 6
_DEFAULT_MAX_SEARCH_QUERY_CHARS = 256


@dataclass(frozen=True)
class HarnessConfig:
    """SMITH-Exp operating point. Override per field; see ``AGENTS.md``."""

    system_prompt_version: str = _DEFAULT_SYSTEM_PROMPT_VERSION
    final_response_strategy: str = _DEFAULT_FINAL_RESPONSE_STRATEGY
    retriever_k: int = 10
    agent_max_k: Optional[int] = 10
    agent_fixed_k: Optional[int] = None
    max_search_calls: int = _DEFAULT_MAX_SEARCH_CALLS
    search_limit_policy: SearchLimitPolicy = "warn_search_limit"
    max_turns: int = estimate_max_turns(
        _DEFAULT_MAX_SEARCH_CALLS,
        final_response_strategy=_DEFAULT_FINAL_RESPONSE_STRATEGY,
        system_prompt_version=_DEFAULT_SYSTEM_PROMPT_VERSION,
    )
    retriever_only: bool = False
    context_management_strategy: Optional[ContextManagementStrategy] = None
    #: Shape of ``search_index`` tool observations shown to the model.
    #: ``v2`` is ``{corpus_id,title,text}`` only (gigatool / GigaChat SFT).
    tool_result_format_version: ToolResultFormatVersion = "v2"
    #: Named observation when ``search_index`` is called past ``max_search_calls``.
    #: ``generic`` → LangChain-style "Tool call limit exceeded…".
    tool_calls_exceeded_warning: ToolCallsExceededWarning = (
        DEFAULT_TOOL_CALLS_EXCEEDED_WARNING
    )
    #: Constrain answer ``supporting_corpus_ids`` via xgrammar enum of retrieved
    #: ids. The OpenAI tool schema stays ``string[]`` so GigaChat prompts match SFT.
    constrain_supporting_ids_to_retrieved: bool = True
    #: GBNF length of answer ``supporting_corpus_ids`` when guided decoding is on.
    guided_supporting_ids_min_items: int = 2
    guided_supporting_ids_max_items: int = 10
    #: Word-boundary cap on ``search_index`` / ``browser.search`` queries.
    #: Longer queries are truncated by ``truncate_search_query`` at dispatch.
    max_search_query_chars: int = _DEFAULT_MAX_SEARCH_QUERY_CHARS
    #: Sampling for the injected ``LLMClient``. The harness loop does not send
    #: these itself; adapters should copy them onto each model request.
    temperature: float = 0.0
    top_p: float = 1.0
    tool_choice: str = "required"

    @property
    def uses_fixed_agent_k(self) -> bool:
        return self.agent_fixed_k is not None

    def agent_prompt_k(self) -> int:
        if self.agent_fixed_k is not None:
            return int(self.agent_fixed_k)
        return int(self.agent_max_k or 1)

    def uses_pick_supporting_documents(self) -> bool:
        return self.system_prompt_version in {"v5", "v5.1", "v5.2"}

    def uses_native_answer_tool(self) -> bool:
        return self.final_response_strategy == NATIVE_ANSWER_STRATEGY

    def uses_corpus_ids_only_tool(self) -> bool:
        return self.final_response_strategy in CORPUS_IDS_ONLY_STRATEGIES

    def uses_full_answer_tool(self) -> bool:
        return self.final_response_strategy in FULL_ANSWER_STRATEGIES

    def uses_openresearcher_browser_tools(self) -> bool:
        return self.final_response_strategy == OPENRESEARCHER_STRATEGY

    def uses_search_only_strategy(self) -> bool:
        return self.final_response_strategy == SEARCH_ONLY_STRATEGY

    def agent_uses_corpus_ids_only_output(self) -> bool:
        return self.uses_corpus_ids_only_tool()
