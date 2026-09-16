"""The synchronous, service-agnostic SMIT retrieval-agent workflow.

This is the real SMIT loop (search -> optionally pick supporting documents ->
final answer, or OpenResearcher browser tools). Hosts inject ``LLMClient`` /
``SearchClient`` backends (and optional ``DocumentFetcher``); this module owns
only the service-agnostic tool-calling loop. ``RetrieverAgent`` never touches an
HTTP session, a provider SDK object, OBS path, or other concrete backend; state
for one call lives entirely in a fresh ``SessionState``.
"""

from __future__ import annotations

from typing import Optional, Sequence

from harness.common import DocumentFetcher, LLMClient, Message, RetrievalResult, SearchClient
from harness.common.progress import RetrieveProgressSink

from .config import HarnessConfig
from .prompts import PromptPolicy
from .state import SessionState
from .tools import ToolDispatcher, ToolPolicy


class RetrieverAgent:
    """Borrow injected clients and create fresh state for every retrieval call.

    ``RetrieverAgent`` never owns client lifecycle -- hosts provision ``llm``/``search``
    protocol clients and the agent only ever borrows them. The
    agent instance itself is stateless between calls and may be reused
    concurrently by a host runner, since all mutable state lives in the
    ``SessionState`` created fresh inside ``retrieve``.
    """

    def __init__(
        self,
        config: HarnessConfig,
        llm: LLMClient,
        search: SearchClient,
        document_fetcher: Optional[DocumentFetcher] = None,
    ) -> None:
        self.config = config
        self.llm = llm
        self.search = search
        self.document_fetcher = document_fetcher
        self._prompts = PromptPolicy()
        self._tools = ToolPolicy()
        self._dispatcher = ToolDispatcher(search, document_fetcher=document_fetcher)

    def retrieve(
        self,
        query: str,
        *,
        progress: Optional[RetrieveProgressSink] = None,
    ) -> RetrievalResult:
        state = SessionState(query=str(query or "").strip())

        if self.config.retriever_only:
            return self._result(state)

        for _ in range(self.config.max_turns):
            messages = self._prompts.render(state, self.config)
            schemas = self._tools.schemas(state, self.config)
            try:
                response = self.llm.call(messages, schemas)
            except Exception as exc:
                state.errors.append(f"{type(exc).__name__}: {exc}")
                break
            state.raw_responses.append(response.raw)
            tool_calls = tuple(response.tool_calls or ())
            content = str(response.content or "")
            state.messages.append(
                Message(role="assistant", content=content, tool_calls=tool_calls)
            )

            if not tool_calls:
                if self.config.uses_search_only_strategy():
                    break
                state.errors.append("Agent ended without calling the final answer tool.")
                if not state.answer:
                    state.answer = response.content
                break

            stop = False
            history_before = len(state.search_history)
            for call in tool_calls:
                try:
                    outcome = self._dispatcher.execute(call, state, self.config)
                except Exception as exc:
                    state.errors.append(f"{type(exc).__name__}: {exc}")
                    stop = True
                    break
                state.messages.append(
                    Message(role="tool", content=outcome.content, tool_call_id=call.id, name=call.name)
                )
                if len(state.search_history) > history_before:
                    step_idx = len(state.search_history) - 1
                    subquery, hits = state.search_history[-1]
                    self._emit_search(
                        progress,
                        step_idx=step_idx,
                        subquery=subquery,
                        hits=hits,
                    )
                    history_before = len(state.search_history)
                if outcome.done:
                    stop = True
                if (
                    self.config.uses_search_only_strategy()
                    and self.config.search_limit_policy == "truncate_search_limit"
                    and state.tool_calls_executed >= self.config.max_search_calls
                ):
                    stop = True
            if stop:
                break
        else:
            state.errors.append("Agent turn limit reached.")

        self._emit_answer(progress, state.answer, state.supporting_corpus_ids)

        return self._result(state)

    @staticmethod
    def _emit_search(
        progress: Optional[RetrieveProgressSink],
        *,
        step_idx: int,
        subquery: str,
        hits: Sequence,
    ) -> None:
        if progress is None:
            return
        progress.on_search_step(step_idx, subquery, hits)

    @staticmethod
    def _emit_answer(
        progress: Optional[RetrieveProgressSink],
        answer: str,
        supporting_corpus_ids: Sequence[str],
    ) -> None:
        if progress is None:
            return
        text = str(answer or "").strip()
        if not text:
            return
        progress.on_answer(text, list(supporting_corpus_ids))

    def _result(self, state: SessionState) -> RetrievalResult:
        return RetrievalResult(
            query=state.query,
            answer=state.answer,
            supporting_corpus_ids=list(state.supporting_corpus_ids),
            retrieved=(),
            search_history=list(state.search_history),
            tool_calls_attempted=state.tool_calls_attempted,
            tool_calls_executed=state.tool_calls_executed,
            errors=list(state.errors),
            messages=list(state.messages),
            raw_responses=list(state.raw_responses),
            browser_cursor_to_corpus_id=dict(state.browser.cursor_to_corpus_id),
            answer_extras=dict(state.answer_extras),
        )
