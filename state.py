"""Mutable state belonging to exactly one ``RetrieverAgent.retrieve`` call.

Everything the SMIT agent loop used to keep in module globals or thread-local
registries lives here instead, scoped to a single call. There is no shared or
thread-local registry anywhere in ``harness/``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Set, Tuple

from common import Message, SearchHit

from browser import BrowserPageStore


@dataclass
class SessionState:
    query: str
    messages: List[Message] = field(default_factory=list)

    #: One entry per ``search_index`` / ``browser.search`` call (successful or
    #: budget-rejected), in issue order: ``(subquery, hits)``.
    search_history: List[Tuple[str, List[SearchHit]]] = field(default_factory=list)

    #: Search-specific tool-call counters (mirrors the old
    #: ``ToolCallLimitMiddleware(tool_name="search_index")`` accounting).
    tool_calls_attempted: int = 0
    tool_calls_executed: int = 0

    errors: List[str] = field(default_factory=list)

    #: corpus_id -> stable small integer handle, assigned on first sight.
    #: Replaces legacy backend row indices with a session-scoped, backend-agnostic
    #: equivalent for tool-result formatting.
    doc_ref_by_corpus_id: Dict[str, int] = field(default_factory=dict)
    corpus_id_by_doc_ref: Dict[int, str] = field(default_factory=dict)

    #: All corpus ids ever returned by ``search_index`` this session, used to
    #: reject hallucinated ids passed directly to a final-answer tool.
    seen_corpus_ids: Set[str] = field(default_factory=set)

    #: doc refs returned by the most recent search / ever retrieved this session.
    latest_doc_refs: Set[int] = field(default_factory=set)
    retrieved_doc_refs: Set[int] = field(default_factory=set)

    #: Ordered, de-duplicated doc refs accepted by ``pick_supporting_documents``.
    picked_document_refs: List[int] = field(default_factory=list)

    #: Raw provider responses for every LLM turn, for host-side token-usage
    #: extraction. Never inspected by the harness itself.
    raw_responses: List[object] = field(default_factory=list)

    #: corpus_id -> 1-based search step that first returned it; used by
    #: ``hide_shown_retrieved_docs`` to replace duplicate passage text.
    seen_corpus_id_to_search_number: Dict[str, int] = field(default_factory=dict)

    #: OpenResearcher browser page cursor state (SERP / document / find).
    browser: BrowserPageStore = field(default_factory=BrowserPageStore)

    answer: str = ""
    supporting_corpus_ids: List[str] = field(default_factory=list)
    #: OpenResearcher answer-tool extras retained for postprocessing / traces.
    answer_extras: Dict[str, Any] = field(default_factory=dict)
    done: bool = False

    def doc_ref_for(self, corpus_id: str) -> int:
        ref = self.doc_ref_by_corpus_id.get(corpus_id)
        if ref is None:
            ref = len(self.doc_ref_by_corpus_id)
            self.doc_ref_by_corpus_id[corpus_id] = ref
            self.corpus_id_by_doc_ref[ref] = corpus_id
        return ref

    def register_hits(self, hits: Sequence[SearchHit]) -> None:
        latest: Set[int] = set()
        for hit in hits:
            ref = self.doc_ref_for(hit.corpus_id)
            latest.add(ref)
            self.retrieved_doc_refs.add(ref)
            self.seen_corpus_ids.add(hit.corpus_id)
        self.latest_doc_refs = latest
