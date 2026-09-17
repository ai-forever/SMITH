"""Host-facing formatting helpers for ``SearchHit`` values.

These are pure functions over harness value types. Host bridges use them to
translate a ``RetrievalResult`` into a host-specific trace/record schema. The
agent loop itself never calls these; only host-side bridges do.
"""

from __future__ import annotations

from typing import Any, Dict

from .protocols import SearchHit


def search_hit_to_ranked_dict(hit: SearchHit, rank: int) -> Dict[str, Any]:
    """Render one ``SearchHit`` as the ``{rank, doc_index, corpus_id, ...}`` shape
    hosts historically stored per search result.

    ``doc_index`` prefers a backend-assigned index carried in ``hit.metadata``
    when present and otherwise falls back to the hit's rank so every hit
    still gets a stable, unique integer within one ranked list.
    """
    metadata = hit.metadata or {}
    doc_index = metadata.get("doc_index")
    if doc_index is None:
        doc_index = rank - 1
    return {
        "rank": int(rank),
        "doc_index": int(doc_index),
        "corpus_id": str(hit.corpus_id),
        "title": str(hit.title or ""),
        "text": str(hit.text or ""),
        "score": float(hit.score) if hit.score is not None else None,
    }
