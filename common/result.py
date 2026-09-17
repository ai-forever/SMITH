"""Value returned by the level-0 retrieval workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence, Tuple

from .protocols import Message, SearchHit


@dataclass(frozen=True)
class RetrievalResult:
    query: str
    answer: str = ""
    supporting_corpus_ids: Sequence[str] = ()
    retrieved: Sequence[SearchHit] = ()
    search_history: Sequence[Tuple[str, Sequence[SearchHit]]] = ()
    tool_calls_attempted: int = 0
    tool_calls_executed: int = 0
    errors: Sequence[str] = ()
    messages: Sequence[Message] = field(default_factory=tuple)
    raw_responses: Sequence[Any] = ()
    #: OpenResearcher browser: opened document page cursor → corpus_id.
    browser_cursor_to_corpus_id: Mapping[int, str] = field(default_factory=dict)
    #: Optional answer-tool extras (explanation / confidence / exact_answer).
    answer_extras: Mapping[str, Any] = field(default_factory=dict)
