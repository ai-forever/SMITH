"""Tool visibility policy for ``RetrieverAgent``."""

from __future__ import annotations

from config import HarnessConfig
from common import ToolSchema
from state import SessionState
from .schemas import (
    PICK_SUPPORTING_DOCUMENTS_TOOL,
    final_answer_tool,
    openresearcher_browser_tools,
    search_index_tool,
    with_guided_supporting_corpus_ids,
)


class ToolPolicy:
    """Select which tool schemas are offered to the model for this turn.

    Once ``max_search_calls`` successful searches have been executed,
    ``search_index`` / ``browser.search`` is omitted so the model can only
    finish with pick/answer (or browse open/find) tools.
    """

    def schemas(self, state: SessionState, config: HarnessConfig) -> list[ToolSchema]:
        tools: list[ToolSchema] = []
        if config.uses_openresearcher_browser_tools():
            browser_search, browser_open, browser_find = openresearcher_browser_tools()
            if state.tool_calls_executed < config.max_search_calls:
                tools.append(browser_search)
            tools.append(browser_open)
            tools.append(browser_find)
            tools.append(self._answer_tool(state, config))
            return tools

        if state.tool_calls_executed < config.max_search_calls:
            tools.append(search_index_tool(config))
        if config.uses_search_only_strategy():
            return tools
        if config.uses_pick_supporting_documents():
            tools.append(PICK_SUPPORTING_DOCUMENTS_TOOL)
        tools.append(self._answer_tool(state, config))
        return tools

    @staticmethod
    def _answer_tool(state: SessionState, config: HarnessConfig) -> ToolSchema:
        schema = final_answer_tool(config)
        if not config.constrain_supporting_ids_to_retrieved:
            return schema
        return with_guided_supporting_corpus_ids(schema, state.seen_corpus_ids)

    def final_tool_name(self, config: HarnessConfig) -> str:
        return final_answer_tool(config).name
