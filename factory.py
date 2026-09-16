"""Construct the SMIT ``RetrieverAgent`` for a resolved harness config."""

from __future__ import annotations

from typing import Optional

from harness.common import AgentLike, DocumentFetcher, LLMClient, SearchClient

from .agent import RetrieverAgent
from .config import HarnessConfig

__all__ = [
    "AgentLike",
    "RetrieverAgent",
    "build_retriever_agent",
]


def build_retriever_agent(
    config: HarnessConfig,
    llm: LLMClient,
    search: SearchClient,
    document_fetcher: Optional[DocumentFetcher] = None,
) -> RetrieverAgent:
    return RetrieverAgent(config, llm, search, document_fetcher=document_fetcher)
