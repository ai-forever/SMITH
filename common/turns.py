"""Shared turn-budget helpers for harness family configs."""

from __future__ import annotations

DEFAULT_MAX_TURNS = 12

#: Prompt versions that expose ``pick_supporting_documents`` in ``ToolPolicy``.
_PICK_PROMPT_VERSIONS = frozenset({"v5", "v5.1", "v5.2"})

#: Final-response strategies whose intended loop requires pick tools
#: (answer does not carry supporting ids; picks accumulate evidence).
_PICK_REQUIRED_STRATEGIES = frozenset({"native_tool_without_supporting_ids"})

#: OpenResearcher browsing: search + open/find extras per search budget.
_OPENRESEARCHER_STRATEGY = "openresearcher"


def tool_policy_offers_pick(
    *,
    final_response_strategy: str,
    system_prompt_version: str,
) -> bool:
    """Whether the SMIT tool policy can offer ``pick_supporting_documents``.

    Mirrors ``HarnessConfig.uses_pick_supporting_documents`` (prompt gate) and
    also treats pick-required strategies as needing pick budget even if the
    prompt version is misconfigured.
    """
    if str(system_prompt_version or "") in _PICK_PROMPT_VERSIONS:
        return True
    return str(final_response_strategy or "") in _PICK_REQUIRED_STRATEGIES


def estimate_max_turns(
    max_search_calls: int,
    *,
    final_response_strategy: str = "tool_strategy",
    system_prompt_version: str = "v4.3",
) -> int:
    """Per-query turn budget for exhausting ``max_search_calls`` searches.

    One turn is one LLM call (typically one tool dispatch). The budget covers
    a full search budget plus the other tools that strategy / prompt require,
    and **one extra turn after the search limit** so the model can call the
    final answer tool after receiving ``Search call limit reached…``:

    * Always: ``max_search_calls`` search turns + **1** final-answer turn
      + **1** post-limit turn (answer after a rejected over-budget search).
    * When pick is in the toolset (v5* prompts, or
      ``native_tool_without_supporting_ids``): **+ ``max_search_calls``** pick
      turns — v5 prompts instruct pick-after-search, so at most one pick per
      search.
    * ``openresearcher``: ``+ 2 * max_search_calls`` for open/find turns
      (search budget still counts ``browser.search`` only).

    Without the post-limit turn, a model that burns all ``S`` searches and
    tries one more search on turn ``S+1`` gets the limit warning on that
    turn and has no remaining turn to answer (empirically 0% answer-after-
    warn on BrowseComp ms30).

    Examples (``S = max_search_calls``):

    ======== ====================================== ==========
    S        Strategy / prompt                      max_turns
    ======== ====================================== ==========
    30       tool_strategy / v4.3                   32
    30       corpus_ids_only / v4.3                 32
    30       native_tool_without_supporting_ids/v5  62
    6        tool_strategy / v4.3                   8
    6        openresearcher / v6                    20
    ======== ====================================== ==========
    """
    searches = max(1, int(max_search_calls))
    answer_turns = 1
    # One turn after search budget: answer once the limit warning is returned.
    post_search_limit_turns = 1
    pick_turns = 0
    if tool_policy_offers_pick(
        final_response_strategy=final_response_strategy,
        system_prompt_version=system_prompt_version,
    ):
        pick_turns = searches
    browse_turns = 0
    if str(final_response_strategy or "") == _OPENRESEARCHER_STRATEGY:
        browse_turns = 2 * searches
    return searches + pick_turns + browse_turns + answer_turns + post_search_limit_turns
