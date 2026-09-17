"""Focused unit tests for OpenResearcher browser tools + citation postprocessing."""

from __future__ import annotations

import os
import sys
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from common import FetchedDocument, LLMClient, LLMResponse, SearchClient, SearchHit, ToolCall
from agent import RetrieverAgent
from config import HarnessConfig
from browser import (
    parse_citation_cursors,
    supporting_corpus_ids_from_explanation,
)
from prompts.system_prompt import render_system_prompt
from tools.policy import ToolPolicy
from state import SessionState


class _FakeSearch(SearchClient):
    def __init__(self, hits: list[SearchHit] | None = None) -> None:
        self.hits = hits or [
            SearchHit(
                corpus_id="42",
                text="Paris is the capital of France.",
                title="Paris",
                metadata={"url": "https://en.wikipedia.org/wiki/Paris"},
            ),
            SearchHit(
                corpus_id="99",
                text="Lyon is a city in France.",
                title="Lyon",
                metadata={"url": "https://en.wikipedia.org/wiki/Lyon"},
            ),
        ]
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, top_k: int):
        self.calls.append((query, top_k))
        return list(self.hits[:top_k])


class _FakeFetcher:
    def __init__(self, docs: dict[str, FetchedDocument]) -> None:
        self.docs = docs
        self.fetch_by_corpus_id_calls: list[str] = []

    def fetch_by_corpus_id(self, corpus_id: str):
        self.fetch_by_corpus_id_calls.append(str(corpus_id))
        return self.docs.get(str(corpus_id))

    def fetch_by_url(self, url: str):
        for doc in self.docs.values():
            if doc.url == url:
                return doc
        return None


class _ScriptedLLM(LLMClient):
    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
        self.seen_tools: list[list[str]] = []
        self.call_count = 0

    def call(self, messages, tools):
        self.call_count += 1
        self.seen_tools.append([tool.name for tool in tools])
        if not self._responses:
            return LLMResponse(content="")
        return self._responses.pop(0)


def _openresearcher_config(**overrides) -> HarnessConfig:
    base = dict(
        final_response_strategy="openresearcher",
        system_prompt_version="v6",
        retriever_k=2,
        agent_max_k=5,
        max_search_calls=6,
        max_turns=12,
    )
    base.update(overrides)
    return HarnessConfig(**base)


class OpenResearcherBrowserTests(unittest.TestCase):
    def test_tool_policy_lists_four_tools(self) -> None:
        config = _openresearcher_config()
        state = SessionState(query="q")
        names = [tool.name for tool in ToolPolicy().schemas(state, config)]
        self.assertEqual(
            names,
            ["browser.search", "browser.open", "browser.find", "answer"],
        )
        answer = ToolPolicy().schemas(state, config)[-1]
        self.assertEqual(answer.parameters["required"], ["explanation", "exact_answer", "confidence"])
        self.assertNotIn("supporting_corpus_ids", answer.parameters["properties"])

    def test_tool_policy_omits_browser_search_after_budget(self) -> None:
        config = _openresearcher_config(max_search_calls=1)
        state = SessionState(query="q")
        state.tool_calls_executed = 1
        names = [tool.name for tool in ToolPolicy().schemas(state, config)]
        self.assertEqual(names, ["browser.open", "browser.find", "answer"])


    def test_v6_system_prompt_omits_hardcoded_tool_names(self) -> None:
        for fixed_k in (False, True):
            with self.subTest(fixed_k=fixed_k):
                config = _openresearcher_config(
                    agent_max_k=None if fixed_k else 5,
                    agent_fixed_k=3 if fixed_k else None,
                    max_search_calls=4,
                )
                prompt = render_system_prompt(config)
                self.assertIn("browsing research agent", prompt.lower())
                self.assertIn("exact_answer", prompt)
                self.assertIn("confidence", prompt)
                self.assertIn("4", prompt)  # max_search_calls budget
                for banned in (
                    "browser.search",
                    "browser.open",
                    "browser.find",
                    "search_index",
                    "pick_supporting_documents",
                ):
                    self.assertNotIn(banned, prompt)

    def test_search_open_find_answer_roundtrip(self) -> None:
        search = _FakeSearch()
        fetcher = _FakeFetcher(
            {
                "42": FetchedDocument(
                    corpus_id="42",
                    text="Paris is the capital of France.\nIt is on the Seine.",
                    title="Paris",
                    url="https://en.wikipedia.org/wiki/Paris",
                )
            }
        )
        # Trailing unused response proves answer terminates the loop.
        unused = LLMResponse(content="should-not-run")
        llm = _ScriptedLLM(
            [
                LLMResponse(
                    tool_calls=(
                        ToolCall(
                            name="browser.search",
                            arguments={"query": "capital of France", "topn": 2},
                            id="s1",
                        ),
                    )
                ),
                LLMResponse(
                    tool_calls=(
                        ToolCall(
                            name="browser.open",
                            arguments={"cursor": 0, "id": 0},
                            id="o1",
                        ),
                    )
                ),
                LLMResponse(
                    tool_calls=(
                        ToolCall(
                            name="browser.find",
                            arguments={"pattern": "Seine", "cursor": 1},
                            id="f1",
                        ),
                    )
                ),
                LLMResponse(
                    tool_calls=(
                        ToolCall(
                            name="answer",
                            arguments={
                                "explanation": "The capital is Paris【1†L2-L3】.",
                                "exact_answer": "Paris",
                                "confidence": 90,
                            },
                            id="a1",
                        ),
                    )
                ),
                unused,
            ]
        )
        agent = RetrieverAgent(
            _openresearcher_config(),
            llm,
            search,
            document_fetcher=fetcher,
        )
        result = agent.retrieve("What is the capital of France?")
        self.assertEqual(result.answer, "Paris")
        self.assertEqual(list(result.supporting_corpus_ids), ["42"])
        self.assertEqual(result.answer_extras.get("confidence"), 90.0)
        self.assertEqual(result.errors, [])
        self.assertEqual(llm.call_count, 4)
        self.assertEqual(llm._responses, [unused])
        self.assertIn(1, result.browser_cursor_to_corpus_id)
        self.assertEqual(result.browser_cursor_to_corpus_id[1], "42")
        # SERP cursor 0 is not a document page → absent from citation map.
        self.assertNotIn(0, result.browser_cursor_to_corpus_id)
        # SERP observation is teacher-style text, not JSON results[].
        search_tool_msgs = [m for m in result.messages if m.name == "browser.search"]
        self.assertTrue(search_tool_msgs)
        self.assertIn("【0†Doc 42†en.wikipedia.org】", search_tool_msgs[0].content)
        self.assertNotIn('"results"', search_tool_msgs[0].content)
        # Open returns a document viewport with full text.
        open_tool_msgs = [m for m in result.messages if m.name == "browser.open"]
        self.assertTrue(open_tool_msgs)
        self.assertIn("Paris is the capital of France.", open_tool_msgs[0].content)
        self.assertTrue(fetcher.fetch_by_corpus_id_calls)
        # Find stays in-memory (no extra search / FAISS).
        self.assertEqual(len(search.calls), 1)  # browser.search only
        find_tool_msgs = [m for m in result.messages if m.name == "browser.find"]
        self.assertTrue(find_tool_msgs)
        self.assertIn("Seine", find_tool_msgs[0].content)
        self.assertIn("Find results", find_tool_msgs[0].content)

    def test_open_scroll_without_id_reuses_page(self) -> None:
        search = _FakeSearch()
        fetcher = _FakeFetcher(
            {
                "42": FetchedDocument(
                    corpus_id="42",
                    text="\n".join(f"line {i}" for i in range(20)),
                    title="Paris",
                    url="https://en.wikipedia.org/wiki/Paris",
                )
            }
        )
        llm = _ScriptedLLM(
            [
                LLMResponse(
                    tool_calls=(
                        ToolCall(
                            name="browser.search",
                            arguments={"query": "paris", "topn": 1},
                            id="s1",
                        ),
                    )
                ),
                LLMResponse(
                    tool_calls=(
                        ToolCall(
                            name="browser.open",
                            arguments={"cursor": 0, "id": 0},
                            id="o1",
                        ),
                    )
                ),
                LLMResponse(
                    tool_calls=(
                        ToolCall(
                            name="browser.open",
                            arguments={"cursor": 1, "loc": 5, "num_lines": 3},
                            id="o2",
                        ),
                    )
                ),
                LLMResponse(
                    tool_calls=(
                        ToolCall(
                            name="answer",
                            arguments={
                                "explanation": "scrolled【1†L5】",
                                "exact_answer": "ok",
                                "confidence": 50,
                            },
                            id="a1",
                        ),
                    )
                ),
            ]
        )
        result = RetrieverAgent(
            _openresearcher_config(),
            llm,
            search,
            document_fetcher=fetcher,
        ).retrieve("q")
        open_msgs = [m for m in result.messages if m.name == "browser.open"]
        self.assertEqual(len(open_msgs), 2)
        self.assertIn("line 5", open_msgs[1].content)
        # Scroll does not allocate a new document cursor / refetch.
        self.assertEqual(result.browser_cursor_to_corpus_id, {1: "42"})
        self.assertEqual(fetcher.fetch_by_corpus_id_calls.count("42"), 1)

    def test_citation_parser(self) -> None:
        self.assertEqual(parse_citation_cursors("x【71†L0-L6】 y【3†L2】 z【71†L9】"), [71, 3])
        mapped = supporting_corpus_ids_from_explanation(
            "see【1†L0-L2】 and【9†L1】 and【1†L5】",
            {1: "doc-a", 2: "doc-b"},
        )
        self.assertEqual(mapped, ["doc-a"])  # 9 dropped (unknown / SERP-only)



if __name__ == "__main__":
    unittest.main()
