"""Backend-neutral client contracts used by the retrieval agent.

Every type here is a small, serializable value object. None of them may carry an
HTTP session, provider SDK object, OBS path, or process handle -- those live one
level up, behind concrete ``SearchClient`` / ``LLMClient`` adapters owned by the
host runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence


@dataclass(frozen=True)
class SearchHit:
    """A normalized search result independent of an index implementation."""

    corpus_id: str
    text: str
    score: float | None = None
    title: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolCall:
    """One tool invocation requested by the model."""

    name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    id: str = ""


@dataclass(frozen=True)
class Message:
    """One turn of the conversation sent to or received from the model.

    ``tool_calls`` is populated on assistant turns that requested tool calls;
    ``tool_call_id``/``name`` identify which call a ``role="tool"`` turn answers.
    """

    role: str
    content: str = ""
    tool_calls: Sequence[ToolCall] = ()
    tool_call_id: str = ""
    name: str = ""


@dataclass(frozen=True)
class ToolSchema:
    name: str
    description: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    #: Field-name → allowed string values for xgrammar only. Never copied into
    #: the OpenAI ``tools`` blob (GigaChat chat templates would inline enums).
    guided_string_enums: Mapping[str, Sequence[str]] = field(default_factory=dict)


@dataclass(frozen=True)
class LLMResponse:
    content: str = ""
    tool_calls: Sequence[ToolCall] = ()
    raw: Any = None


class SearchClient(Protocol):
    def search(self, query: str, top_k: int) -> Sequence[SearchHit]: ...


@dataclass(frozen=True)
class FetchedDocument:
    """Full document payload for ``browser.open`` (host-hydrated)."""

    corpus_id: str
    text: str
    title: str = ""
    url: str = ""


class DocumentFetcher(Protocol):
    """Optional host-injected corpus lookup used by OpenResearcher browser tools."""

    def fetch_by_corpus_id(self, corpus_id: str) -> FetchedDocument | None: ...

    def fetch_by_url(self, url: str) -> FetchedDocument | None: ...


class LLMClient(Protocol):
    def call(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSchema],
    ) -> LLMResponse: ...
