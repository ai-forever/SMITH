from __future__ import annotations

import os
import sys
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from harness.common import LLMResponse, ToolCall
from harness.smit_harness import HarnessConfig, RetrieverAgent
from harness.smit_harness.tools import ANSWER_TOOL_NAME, truncate_search_query
from harness.smit_harness.tests.test_agent import FakeLLM, FakeSearch


class TruncateSearchQueryTests(unittest.TestCase):
    def test_keeps_short_query(self) -> None:
        self.assertEqual(
            truncate_search_query("Ламздорф Варшава", max_chars=256),
            "Ламздорф Варшава",
        )

    def test_cuts_at_word_boundary(self) -> None:
        query = "Ламздорф " + " ".join(str(year) for year in range(1800, 2000))
        out = truncate_search_query(query, max_chars=40)
        self.assertLessEqual(len(out), 40)
        self.assertTrue(out.startswith("Ламздорф"))
        self.assertFalse(out.endswith(" "))

    def test_default_config_window_is_256(self) -> None:
        self.assertEqual(HarnessConfig().max_search_query_chars, 256)


class SearchQueryDispatchTests(unittest.TestCase):
    def test_long_search_query_is_truncated(self) -> None:
        years = " ".join(str(year) for year in range(1800, 2300))
        query = "Ламздорф " + years
        search = FakeSearch()
        llm = FakeLLM(
            [
                LLMResponse(
                    tool_calls=(
                        ToolCall(
                            name="search_index",
                            arguments={"query": query},
                            id="c1",
                        ),
                    )
                ),
                LLMResponse(
                    tool_calls=(
                        ToolCall(
                            name=ANSWER_TOOL_NAME,
                            arguments={
                                "answer": "1860",
                                "supporting_corpus_ids": ["Ламздорф_0"],
                            },
                            id="a1",
                        ),
                    )
                ),
            ]
        )
        agent = RetrieverAgent(
            HarnessConfig(max_search_query_chars=40),
            llm,
            search,
        )
        result = agent.retrieve("в каком году ламздорф?")
        self.assertEqual(len(search.calls), 1)
        self.assertTrue(search.calls[0][0].startswith("Ламздорф"))
        self.assertLessEqual(len(search.calls[0][0]), 40)
        self.assertFalse(result.errors)
        self.assertEqual(result.answer, "1860")


if __name__ == "__main__":
    unittest.main()
