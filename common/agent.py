"""Shared agent protocol used by host runners across harness families."""

from __future__ import annotations

from typing import Protocol

from .result import RetrievalResult


class AgentLike(Protocol):
    def retrieve(self, query: str) -> RetrievalResult: ...
