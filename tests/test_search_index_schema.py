"""Fixed-k search_index schema omits optional ``k`` (BrowseComp-Plus parity)."""

from __future__ import annotations

import unittest

from config import HarnessConfig
from state import SessionState
from tools.policy import ToolPolicy
from tools.schemas import (
    SEARCH_INDEX_TOOL,
    SEARCH_INDEX_TOOL_FIXED_K,
    search_index_tool,
)


class SearchIndexSchemaTests(unittest.TestCase):
    def test_fixed_k_schema_omits_k(self) -> None:
        schema = search_index_tool(HarnessConfig(agent_fixed_k=5, agent_max_k=None))
        self.assertIs(schema, SEARCH_INDEX_TOOL_FIXED_K)
        self.assertNotIn("k", schema.parameters["properties"])
        self.assertEqual(schema.parameters["required"], ["query"])

    def test_variable_k_schema_keeps_k(self) -> None:
        schema = search_index_tool(HarnessConfig(agent_fixed_k=None, agent_max_k=5))
        self.assertIs(schema, SEARCH_INDEX_TOOL)
        self.assertIn("k", schema.parameters["properties"])

    def test_tool_policy_offers_fixed_k_schema(self) -> None:
        config = HarnessConfig(
            agent_fixed_k=5,
            agent_max_k=None,
            final_response_strategy="custom_tool_strategy_corpus_ids_only",
        )
        tools = ToolPolicy().schemas(SessionState(query="q"), config)
        search = next(t for t in tools if t.name == "search_index")
        self.assertNotIn("k", search.parameters["properties"])


if __name__ == "__main__":
    unittest.main()
