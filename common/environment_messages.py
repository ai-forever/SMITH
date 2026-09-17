"""Named environment / tool-observation messages shared by harness families."""

from __future__ import annotations

from typing import Literal, Optional

#: Named options for the observation returned when ``search_index`` is called
#: after ``max_search_calls`` has already been exhausted.
ToolCallsExceededWarning = Literal["generic", "search_r1"]

DEFAULT_TOOL_CALLS_EXCEEDED_WARNING: ToolCallsExceededWarning = "generic"

GENERIC_TOOL_CALLS_EXCEEDED_WARNING = (
    "Tool call limit exceeded. Do not call 'search_index' again."
)


def render_generic_tool_calls_exceeded_warning(tool_name: str = "search_index") -> str:
    """LangChain-style tool-limit observation (``generic`` option)."""

    return f"Tool call limit exceeded. Do not call '{tool_name}' again."


def render_search_r1_tool_calls_exceeded_warning(max_search_calls: int) -> str:
    """Search-R1 style note when another ``<search>`` is attempted past the budget."""

    return (
        f"Search call limit ({int(max_search_calls)}) reached; "
        "model attempted another <search> action."
    )


def resolve_tool_calls_exceeded_warning(
    option: Optional[str],
    *,
    default: ToolCallsExceededWarning = DEFAULT_TOOL_CALLS_EXCEEDED_WARNING,
) -> ToolCallsExceededWarning:
    """Resolve a configured option; omitted / empty → ``generic``."""

    text = str(option or "").strip()
    if text in {"generic", "search_r1"}:
        return text  # type: ignore[return-value]
    return default


def render_tool_calls_exceeded_warning(
    option: Optional[str],
    *,
    tool_name: str = "search_index",
    max_search_calls: int = 6,
) -> str:
    """Render the tool-limit observation for a named option (default ``generic``)."""

    resolved = resolve_tool_calls_exceeded_warning(option)
    if resolved == "search_r1":
        return render_search_r1_tool_calls_exceeded_warning(max_search_calls)
    return render_generic_tool_calls_exceeded_warning(tool_name=tool_name)
