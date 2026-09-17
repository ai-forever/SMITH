"""Tool schemas for the SMIT retrieval-agent tool namespace.

There is exactly one search tool family and exactly one final-answer tool per
turn, but the shapes depend on ``HarnessConfig.final_response_strategy``:
a full free-text answer with self-cited corpus ids, a corpus-ids-only
submission, (paired with the v5.x prompts) a bare answer whose supporting
ids are gathered from prior ``pick_supporting_documents`` calls, or the
OpenResearcher browser toolset (``browser.*`` + ``answer(explanation, …)``).
"""

from __future__ import annotations

from typing import Sequence

from config import HarnessConfig
from common import ToolSchema

#: Runtime + metadata name for ``custom_giga_tool_strategy`` / native / corpus-ids-only.
ANSWER_TOOL_NAME = "answer"
#: Historical LangChain ToolStrategy name; runtime + Trace metadata for ``tool_strategy``.
TOOL_STRATEGY_FINAL_TOOL_NAME = "AgentStructuredResponse"
BROWSER_SEARCH_TOOL_NAME = "browser.search"
BROWSER_OPEN_TOOL_NAME = "browser.open"
BROWSER_FIND_TOOL_NAME = "browser.find"

_SEARCH_INDEX_DESCRIPTION = (
    "Search the local semantic index and return relevant text snippets. "
    "Use the returned text as the only evidence source."
)

#: Variable-k schema (optional ``k``). Prefer :func:`search_index_tool` at runtime.
SEARCH_INDEX_TOOL = ToolSchema(
    name="search_index",
    description=_SEARCH_INDEX_DESCRIPTION,
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query string."},
            "k": {
                "type": "integer",
                "description": "Optional number of top results to return.",
            },
        },
        "required": ["query"],
    },
)

#: Fixed-k schema (query only) — matches BrowseComp-Plus / author GPT-OSS client.
SEARCH_INDEX_TOOL_FIXED_K = ToolSchema(
    name="search_index",
    description=_SEARCH_INDEX_DESCRIPTION,
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query string."},
        },
        "required": ["query"],
    },
)


def search_index_tool(config: HarnessConfig) -> ToolSchema:
    """Return the search tool schema offered to the model for this config.

    With ``agent_fixed_k`` set, omit ``k`` so the model cannot request a different
    top-k (dispatcher already ignores model ``k`` in that mode).
    """
    if config.uses_fixed_agent_k:
        return SEARCH_INDEX_TOOL_FIXED_K
    return SEARCH_INDEX_TOOL


PICK_SUPPORTING_DOCUMENTS_TOOL = ToolSchema(
    name="pick_supporting_documents",
    description=(
        "Record integer doc_index ids from retrieved search results that contain "
        "direct evidence."
    ),
    parameters={
        "type": "object",
        "properties": {
            "document_ids": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "doc_index values from previously retrieved passages.",
            },
        },
        "required": ["document_ids"],
    },
)

ANSWER_WITH_CORPUS_IDS_TOOL = ToolSchema(
    name=ANSWER_TOOL_NAME,
    description="Submit the final answer after retrieval is complete.",
    parameters={
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "description": "Final answer grounded only in retrieved passages.",
            },
            "supporting_corpus_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Supporting corpus ids copied exactly from retrieved passages, "
                    "ordered from most to least important."
                ),
            },
        },
        "required": ["answer", "supporting_corpus_ids"],
    },
)

ANSWER_CORPUS_IDS_ONLY_TOOL = ToolSchema(
    name=ANSWER_TOOL_NAME,
    description="Submit supporting corpus ids after retrieval is complete.",
    parameters={
        "type": "object",
        "properties": {
            "supporting_corpus_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Supporting corpus ids copied exactly from retrieved passages, "
                    "ordered from most to least important."
                ),
            },
        },
        "required": ["supporting_corpus_ids"],
    },
)

ANSWER_NATIVE_TOOL = ToolSchema(
    name=ANSWER_TOOL_NAME,
    description=(
        "Submit the final answer; supporting corpus ids are gathered from documents "
        "previously picked with pick_supporting_documents."
    ),
    parameters={
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "description": "Final answer grounded only in picked passages.",
            },
        },
        "required": ["answer"],
    },
)

# GPT-OSS / OpenResearcher browser tool schemas (from sft.openresearcher_harmony).
BROWSER_SEARCH_TOOL = ToolSchema(
    name=BROWSER_SEARCH_TOOL_NAME,
    description="Searches for information related to `query` and displays `topn` results.",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "topn": {"type": "number", "default": 10},
            "source": {"type": "string"},
        },
        "required": ["query"],
    },
)

BROWSER_OPEN_TOOL = ToolSchema(
    name=BROWSER_OPEN_TOOL_NAME,
    description=(
        "Opens the link `id` from the page indicated by `cursor` starting at "
        "line number `loc`, showing `num_lines` lines."
    ),
    parameters={
        "type": "object",
        "properties": {
            "cursor": {"type": "number", "default": -1},
            "id": {"type": ["number", "string"], "default": -1},
            "loc": {"type": "number", "default": -1},
            "num_lines": {"type": "number", "default": -1},
            "source": {"type": "string"},
            "view_source": {"type": "boolean", "default": False},
        },
    },
)

BROWSER_FIND_TOOL = ToolSchema(
    name=BROWSER_FIND_TOOL_NAME,
    description=(
        "Finds exact matches of `pattern` in the current page, or the page given by "
        "`cursor`."
    ),
    parameters={
        "type": "object",
        "properties": {
            "pattern": {"type": "string"},
            "cursor": {"type": "number", "default": -1},
        },
        "required": ["pattern"],
    },
)

ANSWER_OPENRESEARCHER_TOOL = ToolSchema(
    name=ANSWER_TOOL_NAME,
    description=(
        "Submit the final answer after browsing. Put a cited explanation in "
        "`explanation`, the short answer in `exact_answer`, and a 0-100 confidence."
    ),
    parameters={
        "type": "object",
        "properties": {
            "explanation": {
                "type": "string",
                "description": (
                    "Cited explanation using 【cursor†L…】 markers from opened pages."
                ),
            },
            "exact_answer": {
                "type": "string",
                "description": "Short final answer graded by eval.",
            },
            "confidence": {
                "type": "number",
                "description": "Confidence in the exact answer, typically 0-100.",
            },
        },
        "required": ["explanation", "exact_answer", "confidence"],
    },
)


def openresearcher_browser_tools() -> list[ToolSchema]:
    return [BROWSER_SEARCH_TOOL, BROWSER_OPEN_TOOL, BROWSER_FIND_TOOL]


def with_guided_supporting_corpus_ids(
    schema: ToolSchema,
    corpus_ids: Sequence[str],
) -> ToolSchema:
    """Attach retrieved-id enums without changing ``parameters`` (prompt schema)."""
    properties = schema.parameters.get("properties") if isinstance(schema.parameters, dict) else None
    if not isinstance(properties, dict) or "supporting_corpus_ids" not in properties:
        return schema
    values = sorted({str(item).strip() for item in corpus_ids if str(item or "").strip()})
    if not values:
        return schema
    return ToolSchema(
        name=schema.name,
        description=schema.description,
        parameters=dict(schema.parameters),
        guided_string_enums={"supporting_corpus_ids": values},
    )


def final_answer_tool(config: HarnessConfig) -> ToolSchema:
    if config.uses_openresearcher_browser_tools():
        return ANSWER_OPENRESEARCHER_TOOL
    if config.uses_native_answer_tool():
        return ANSWER_NATIVE_TOOL
    if config.uses_corpus_ids_only_tool():
        return ANSWER_CORPUS_IDS_ONLY_TOOL
    if config.final_response_strategy == "tool_strategy":
        # Keep the historical ToolStrategy function name in sync with Trace.tools /
        # SFT converters (not the giga ``answer`` alias).
        return ToolSchema(
            name=TOOL_STRATEGY_FINAL_TOOL_NAME,
            description=ANSWER_WITH_CORPUS_IDS_TOOL.description,
            parameters=dict(ANSWER_WITH_CORPUS_IDS_TOOL.parameters),
        )
    return ANSWER_WITH_CORPUS_IDS_TOOL
