"""Compatibility shim — shared contracts live in ``common``."""

from common.protocols import (  # noqa: F401
    LLMClient,
    LLMResponse,
    Message,
    SearchClient,
    SearchHit,
    ToolCall,
    ToolSchema,
)

__all__ = [
    "LLMClient",
    "LLMResponse",
    "Message",
    "SearchClient",
    "SearchHit",
    "ToolCall",
    "ToolSchema",
]
