"""Optional progress sink for mid-retrieve streaming hosts (e.g. demo SSE)."""

from __future__ import annotations

from typing import Protocol, Sequence

from .protocols import SearchHit


class RetrieveProgressSink(Protocol):
    """Called from the agent thread during ``RetrieverAgent.retrieve``."""

    def on_search_step(
        self,
        step_idx: int,
        subquery: str,
        hits: Sequence[SearchHit],
    ) -> None:
        """One retrieval step (baseline or ``search_index`` tool)."""

    def on_answer(
        self,
        answer: str,
        supporting_corpus_ids: Sequence[str],
    ) -> None:
        """Final answer is available (may still finish the turn loop)."""
