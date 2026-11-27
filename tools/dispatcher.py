"""Tool dispatch for ``RetrieverAgent``.

Executes one validated ``ToolCall`` against injected state and the borrowed
``SearchClient`` (and optional ``DocumentFetcher`` for OpenResearcher open).
Nothing here is keyed by query id or thread; everything reads and writes the
``SessionState`` passed in by the caller.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

from common import DocumentFetcher, FetchedDocument, SearchClient, SearchHit, ToolCall
from common.environment_messages import render_tool_calls_exceeded_warning

from config import HarnessConfig, ToolResultFormatVersion
from state import SessionState
from browser import (
    build_document_page,
    build_find_page,
    build_serp_page,
    coerce_int,
    render_viewport,
    supporting_corpus_ids_from_explanation,
)
from .schemas import (
    ANSWER_TOOL_NAME,
    BROWSER_FIND_TOOL_NAME,
    BROWSER_OPEN_TOOL_NAME,
    BROWSER_SEARCH_TOOL_NAME,
    PICK_SUPPORTING_DOCUMENTS_TOOL,
    SEARCH_INDEX_TOOL,
    TOOL_STRATEGY_FINAL_TOOL_NAME,
    final_answer_tool,
)


def truncate_search_query(query: str, *, max_chars: int) -> str:
    """Cap a search query at ``max_chars``, preferring a word boundary.

    This is the harness overflow control for ``search_index`` /
    ``browser.search`` queries. Longer model-generated queries are shortened
    here rather than reconstructed from truncated tool-call text.
    """
    text = str(query or "").strip()
    limit = max(1, int(max_chars))
    if len(text) <= limit:
        return text
    cut = text[:limit]
    space = cut.rfind(" ")
    if space >= limit // 2:
        cut = cut[:space]
    return cut.strip()


@dataclass(frozen=True)
class ToolOutcome:
    """The ``role="tool"`` message content plus whether the loop should stop."""

    content: str
    done: bool = False


def _hide_shown_hits(
    hits: Sequence[SearchHit],
    state: SessionState,
    *,
    search_number: int,
) -> List[SearchHit]:
    modified: List[SearchHit] = []
    for hit in hits:
        corpus_id = str(hit.corpus_id or "").strip()
        if corpus_id and corpus_id in state.seen_corpus_id_to_search_number:
            modified.append(
                SearchHit(
                    corpus_id=corpus_id,
                    text=(
                        "retrieved earlier at search "
                        f"{state.seen_corpus_id_to_search_number[corpus_id]}"
                    ),
                    score=hit.score,
                    title=hit.title,
                    metadata=hit.metadata,
                )
            )
        else:
            if corpus_id:
                state.seen_corpus_id_to_search_number[corpus_id] = search_number
            modified.append(hit)
    return modified


def _format_search_results(
    hits: Sequence[SearchHit],
    state: SessionState,
    *,
    tool_result_format_version: ToolResultFormatVersion = "v0",
) -> str:
    if not hits:
        return json.dumps({"results": []}, ensure_ascii=False, separators=(",", ":"))

    if tool_result_format_version == "v2":
        results = [
            {
                "corpus_id": str(hit.corpus_id),
                "title": str(hit.title or ""),
                "text": str(hit.text or "").replace("\n", " ").strip(),
            }
            for hit in hits
        ]
        return json.dumps({"results": results}, ensure_ascii=False, separators=(",", ":"))

    if tool_result_format_version == "v1":
        results = [
            {
                "rank": rank,
                "corpus_id": str(hit.corpus_id),
                "title": str(hit.title or ""),
                "text": str(hit.text or "").replace("\n", " ").strip(),
                "score": float(hit.score) if hit.score is not None else None,
            }
            for rank, hit in enumerate(hits, start=1)
        ]
        return json.dumps({"results": results}, ensure_ascii=False, separators=(",", ":"))

    # v0: rich canonical JSON (rank + doc_index + score).
    results = []
    for rank, hit in enumerate(hits, start=1):
        results.append(
            {
                "rank": rank,
                "doc_index": state.doc_ref_for(hit.corpus_id),
                "corpus_id": str(hit.corpus_id),
                "title": str(hit.title or ""),
                "text": str(hit.text or "").replace("\n", " ").strip(),
                "score": float(hit.score) if hit.score is not None else None,
            }
        )
    return json.dumps({"results": results}, ensure_ascii=False, separators=(",", ":"))


class ToolDispatcher:
    def __init__(
        self,
        search: SearchClient,
        document_fetcher: Optional[DocumentFetcher] = None,
    ) -> None:
        self.search = search
        self.document_fetcher = document_fetcher

    def execute(self, call: ToolCall, state: SessionState, config: HarnessConfig) -> ToolOutcome:
        if config.uses_openresearcher_browser_tools():
            if call.name == BROWSER_SEARCH_TOOL_NAME:
                return self._execute_browser_search(call, state, config)
            if call.name == BROWSER_OPEN_TOOL_NAME:
                return self._execute_browser_open(call, state)
            if call.name == BROWSER_FIND_TOOL_NAME:
                return self._execute_browser_find(call, state)
            if call.name == ANSWER_TOOL_NAME:
                return self._execute_openresearcher_answer(call, state)
            state.errors.append(f"Rejected unsolicited tool call: {call.name}")
            return ToolOutcome(
                content=json.dumps(
                    {"error": f"Unknown or unsolicited tool: {call.name}"},
                    ensure_ascii=False,
                )
            )

        if call.name == SEARCH_INDEX_TOOL.name:
            return self._execute_search(call, state, config)
        if call.name == PICK_SUPPORTING_DOCUMENTS_TOOL.name and config.uses_pick_supporting_documents():
            return self._execute_pick_supporting_documents(call, state)
        if call.name in {
            final_answer_tool(config).name,
            ANSWER_TOOL_NAME,
            TOOL_STRATEGY_FINAL_TOOL_NAME,
        }:
            return self._execute_answer(call, state, config)
        state.errors.append(f"Rejected unsolicited tool call: {call.name}")
        return ToolOutcome(
            content=json.dumps(
                {"error": f"Unknown or unsolicited tool: {call.name}"},
                ensure_ascii=False,
            )
        )

    def _execute_search(
        self, call: ToolCall, state: SessionState, config: HarnessConfig
    ) -> ToolOutcome:
        state.tool_calls_attempted += 1
        if state.tool_calls_executed >= config.max_search_calls:
            return ToolOutcome(
                content=json.dumps(
                    {
                        "error": render_tool_calls_exceeded_warning(
                            config.tool_calls_exceeded_warning,
                            tool_name=SEARCH_INDEX_TOOL.name,
                            max_search_calls=config.max_search_calls,
                        )
                    },
                    ensure_ascii=False,
                )
            )

        query = truncate_search_query(
            str(call.arguments.get("query") or state.query).strip() or state.query,
            max_chars=config.max_search_query_chars,
        )
        k = self._resolve_k(call.arguments.get("k"), config)
        hits = list(self.search.search(query, k))
        state.register_hits(hits)
        state.tool_calls_executed += 1
        search_number = state.tool_calls_executed
        if config.context_management_strategy == "hide_shown_retrieved_docs":
            hits = _hide_shown_hits(hits, state, search_number=search_number)
        state.search_history.append((query, hits))
        return ToolOutcome(
            content=_format_search_results(
                hits,
                state,
                tool_result_format_version=config.tool_result_format_version,
            )
        )

    def _execute_browser_search(
        self, call: ToolCall, state: SessionState, config: HarnessConfig
    ) -> ToolOutcome:
        state.tool_calls_attempted += 1
        if state.tool_calls_executed >= config.max_search_calls:
            return ToolOutcome(
                content=json.dumps(
                    {
                        "error": render_tool_calls_exceeded_warning(
                            config.tool_calls_exceeded_warning,
                            tool_name=BROWSER_SEARCH_TOOL_NAME,
                            max_search_calls=config.max_search_calls,
                        )
                    },
                    ensure_ascii=False,
                )
            )

        query = truncate_search_query(
            str(call.arguments.get("query") or state.query).strip() or state.query,
            max_chars=config.max_search_query_chars,
        )
        topn = call.arguments.get("topn")
        if topn is None:
            k = self._resolve_k(None, config)
        else:
            k = self._resolve_k(topn, config)
        hits = list(self.search.search(query, k))
        state.register_hits(hits)
        state.tool_calls_executed += 1
        search_number = state.tool_calls_executed
        if config.context_management_strategy == "hide_shown_retrieved_docs":
            hits = _hide_shown_hits(hits, state, search_number=search_number)
        state.search_history.append((query, hits))
        _, viewport = build_serp_page(state.browser, query=query, hits=hits)
        return ToolOutcome(content=viewport)

    def _execute_browser_open(self, call: ToolCall, state: SessionState) -> ToolOutcome:
        cursor = state.browser.resolve_cursor(call.arguments.get("cursor", -1))
        loc = coerce_int(call.arguments.get("loc", -1), default=-1)
        num_lines = coerce_int(call.arguments.get("num_lines", -1), default=-1)
        raw_id = call.arguments.get("id", -1)

        # Scroll current/specified page when id is absent / -1.
        scroll_only = raw_id is None or raw_id == "" or raw_id == -1 or str(raw_id).strip() == "-1"
        if scroll_only:
            page = state.browser.get(cursor)
            if page is None:
                return ToolOutcome(
                    content=json.dumps(
                        {"error": f"Unknown page cursor: {cursor}"},
                        ensure_ascii=False,
                    )
                )
            state.browser.current_cursor = page.cursor
            return ToolOutcome(content=render_viewport(page, loc=loc, num_lines=num_lines))

        # Open by URL string.
        if isinstance(raw_id, str) and not str(raw_id).strip().lstrip("-").isdigit():
            url = str(raw_id).strip()
            doc = self._fetch_document(url=url, corpus_id=None, state=state)
            if doc is None:
                return ToolOutcome(
                    content=json.dumps(
                        {"error": f"Could not open URL: {url}"},
                        ensure_ascii=False,
                    )
                )
            _, viewport = build_document_page(
                state.browser,
                corpus_id=doc.corpus_id,
                text=doc.text,
                url=doc.url or url,
                title=doc.title,
            )
            return ToolOutcome(content=viewport)

        # Open link id from the SERP/page at cursor.
        link_id = coerce_int(raw_id, default=-1)
        page = state.browser.get(cursor)
        if page is None:
            return ToolOutcome(
                content=json.dumps(
                    {"error": f"Unknown page cursor: {cursor}"},
                    ensure_ascii=False,
                )
            )
        corpus_id = str(page.link_id_to_corpus_id.get(link_id) or "").strip()
        url = str(page.link_id_to_url.get(link_id) or "").strip()
        if not corpus_id and not url:
            return ToolOutcome(
                content=json.dumps(
                    {"error": f"Unknown link id {link_id} on cursor {cursor}"},
                    ensure_ascii=False,
                )
            )
        doc = self._fetch_document(url=url or None, corpus_id=corpus_id or None, state=state)
        if doc is None:
            # Fall back to search-hit snippet text already seen this session.
            fallback_text = self._snippet_text_for_corpus_id(corpus_id, state)
            if not fallback_text:
                return ToolOutcome(
                    content=json.dumps(
                        {
                            "error": (
                                f"Could not fetch document for link {link_id} "
                                f"(corpus_id={corpus_id!r}, url={url!r})"
                            )
                        },
                        ensure_ascii=False,
                    )
                )
            doc = FetchedDocument(
                corpus_id=corpus_id or url,
                text=fallback_text,
                title="",
                url=url,
            )
        _, viewport = build_document_page(
            state.browser,
            corpus_id=doc.corpus_id,
            text=doc.text,
            url=doc.url or url,
            title=doc.title,
        )
        return ToolOutcome(content=viewport)

    def _execute_browser_find(self, call: ToolCall, state: SessionState) -> ToolOutcome:
        pattern = str(call.arguments.get("pattern") or "")
        cursor = state.browser.resolve_cursor(call.arguments.get("cursor", -1))
        page = state.browser.get(cursor)
        if page is None:
            return ToolOutcome(
                content=json.dumps(
                    {"error": f"Unknown page cursor: {cursor}"},
                    ensure_ascii=False,
                )
            )
        _, viewport = build_find_page(state.browser, source=page, pattern=pattern)
        return ToolOutcome(content=viewport)

    def _fetch_document(
        self,
        *,
        url: Optional[str],
        corpus_id: Optional[str],
        state: SessionState,
    ) -> Optional[FetchedDocument]:
        fetcher = self.document_fetcher
        if fetcher is None:
            return None
        if corpus_id:
            doc = fetcher.fetch_by_corpus_id(str(corpus_id))
            if doc is not None:
                return doc
        if url:
            # Prefer session link maps (URL → corpus_id) before fetcher URL lookup.
            for page in state.browser.pages.values():
                for link_id, link_url in page.link_id_to_url.items():
                    if link_url == url:
                        mapped = page.link_id_to_corpus_id.get(link_id)
                        if mapped:
                            doc = fetcher.fetch_by_corpus_id(str(mapped))
                            if doc is not None:
                                return doc
            return fetcher.fetch_by_url(str(url))
        return None

    @staticmethod
    def _snippet_text_for_corpus_id(corpus_id: str, state: SessionState) -> str:
        target = str(corpus_id or "").strip()
        if not target:
            return ""
        for _, hits in reversed(state.search_history):
            for hit in hits:
                if str(hit.corpus_id) == target and hit.text:
                    return str(hit.text)
        return ""

    @staticmethod
    def _resolve_k(requested: Any, config: HarnessConfig) -> int:
        if config.agent_fixed_k is not None:
            return int(config.agent_fixed_k)
        agent_max_k = int(config.agent_max_k or 1)
        if requested is None:
            return agent_max_k
        try:
            requested_k = int(requested)
        except (TypeError, ValueError):
            return agent_max_k
        return max(1, min(requested_k, agent_max_k))

    def _execute_pick_supporting_documents(
        self, call: ToolCall, state: SessionState
    ) -> ToolOutcome:
        raw_ids = call.arguments.get("document_ids") or []
        retrieved = state.retrieved_doc_refs or state.latest_doc_refs
        valid: list[int] = []
        invalid: list[int] = []
        seen: set[int] = set()
        for raw_id in raw_ids:
            try:
                doc_ref = int(raw_id)
            except (TypeError, ValueError):
                continue
            if doc_ref in seen:
                continue
            seen.add(doc_ref)
            if doc_ref in retrieved:
                valid.append(doc_ref)
            else:
                invalid.append(doc_ref)

        for doc_ref in valid:
            if doc_ref not in state.picked_document_refs:
                state.picked_document_refs.append(doc_ref)

        payload: dict[str, Any] = {
            "picked_document_ids": valid,
            "status": "ok" if not invalid else "partial",
        }
        if invalid:
            payload["invalid_document_ids"] = invalid
        return ToolOutcome(content=json.dumps(payload, ensure_ascii=False, separators=(",", ":")))

    def _execute_answer(
        self, call: ToolCall, state: SessionState, config: HarnessConfig
    ) -> ToolOutcome:
        if config.uses_native_answer_tool():
            answer_text = str(call.arguments.get("answer") or "").strip()
            supporting_corpus_ids = self._supporting_ids_from_picked(state)
        elif config.uses_corpus_ids_only_tool():
            answer_text = ""
            supporting_corpus_ids = self._filter_seen_corpus_ids(
                call.arguments.get("supporting_corpus_ids"), state
            )
        else:
            answer_text = str(call.arguments.get("answer") or "").strip()
            supporting_corpus_ids = self._filter_seen_corpus_ids(
                call.arguments.get("supporting_corpus_ids"), state
            )

        state.answer = answer_text
        state.supporting_corpus_ids = supporting_corpus_ids
        state.done = True
        payload = {"answer": answer_text, "supporting_corpus_ids": supporting_corpus_ids}
        return ToolOutcome(
            content=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            done=True,
        )

    def _execute_openresearcher_answer(
        self, call: ToolCall, state: SessionState
    ) -> ToolOutcome:
        explanation = str(call.arguments.get("explanation") or "").strip()
        exact_answer = str(call.arguments.get("exact_answer") or "").strip()
        confidence_raw = call.arguments.get("confidence")
        try:
            confidence = float(confidence_raw) if confidence_raw is not None else None
        except (TypeError, ValueError):
            confidence = None

        supporting = supporting_corpus_ids_from_explanation(
            explanation,
            state.browser.cursor_to_corpus_id,
        )
        state.answer = exact_answer
        state.supporting_corpus_ids = supporting
        state.answer_extras = {
            "explanation": explanation,
            "exact_answer": exact_answer,
            "confidence": confidence,
        }
        state.done = True
        payload = {
            "explanation": explanation,
            "exact_answer": exact_answer,
            "confidence": confidence,
            "answer": exact_answer,
            "supporting_corpus_ids": supporting,
        }
        return ToolOutcome(
            content=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            done=True,
        )

    @staticmethod
    def _supporting_ids_from_picked(state: SessionState) -> list[str]:
        supporting: list[str] = []
        seen: set[str] = set()
        for doc_ref in state.picked_document_refs:
            corpus_id = str(state.corpus_id_by_doc_ref.get(doc_ref) or "").strip()
            if not corpus_id or corpus_id in seen:
                continue
            seen.add(corpus_id)
            supporting.append(corpus_id)
        return supporting

    @staticmethod
    def _filter_seen_corpus_ids(raw_ids: Any, state: SessionState) -> list[str]:
        if not isinstance(raw_ids, (list, tuple)):
            return []
        supporting: list[str] = []
        seen: set[str] = set()
        for raw_id in raw_ids:
            corpus_id = str(raw_id or "").strip()
            if not corpus_id or corpus_id in seen or corpus_id not in state.seen_corpus_ids:
                continue
            seen.add(corpus_id)
            supporting.append(corpus_id)
        return supporting
