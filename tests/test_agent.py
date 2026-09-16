"""Fake-client unit tests for the service-agnostic SMIT retrieval agent."""

from __future__ import annotations

import os
import sys
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from harness.smit_harness import HarnessConfig, Message, RetrievalResult, RetrieverAgent, SearchHit, ToolCall
from harness.common import LLMClient, LLMResponse, SearchClient
from harness.smit_harness.tools import ANSWER_TOOL_NAME, TOOL_STRATEGY_FINAL_TOOL_NAME


class FakeSearch(SearchClient):
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, top_k: int):
        self.calls.append((query, top_k))
        return [
            SearchHit(
                corpus_id=f"{query.split()[0] if query.split() else 'doc'}_{index}",
                text=f"text for {query} #{index}",
                score=1.0 - index * 0.1,
            )
            for index in range(top_k)
        ]


class FakeLLM(LLMClient):
    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
        self.calls = 0
        self.seen_tools: list[list[str]] = []

    def call(self, messages, tools):
        self.calls += 1
        self.seen_tools.append([tool.name for tool in tools])
        if not self._responses:
            return LLMResponse(content="")
        return self._responses.pop(0)


class RetrieverAgentTests(unittest.TestCase):
    def test_retriever_only_short_circuit(self) -> None:
        search = FakeSearch()
        llm = FakeLLM([])
        agent = RetrieverAgent(
            HarnessConfig(retriever_only=True, retriever_k=3),
            llm,
            search,
        )
        result = agent.retrieve("what is SMIT?")
        self.assertIsInstance(result, RetrievalResult)
        self.assertEqual(len(result.retrieved), 0)
        self.assertEqual(search.calls, [])
        self.assertEqual(llm.calls, 0)
        self.assertEqual(result.tool_calls_executed, 0)

    def test_tool_strategy_full_answer_with_self_cited_ids(self) -> None:
        search = FakeSearch()
        llm = FakeLLM(
            [
                LLMResponse(
                    tool_calls=(
                        ToolCall(
                            name="search_index",
                            arguments={"query": "SMIT harness", "k": 2},
                            id="call_1",
                        ),
                    )
                ),
                LLMResponse(
                    tool_calls=(
                        ToolCall(
                            name=TOOL_STRATEGY_FINAL_TOOL_NAME,
                            arguments={
                                "answer": "SMIT uses a harness.",
                                # Includes one hallucinated id that was never returned;
                                # the dispatcher must drop it rather than trust the model.
                                "supporting_corpus_ids": ["SMIT_0", "never_seen"],
                            },
                            id="call_2",
                        ),
                    )
                ),
            ]
        )
        agent = RetrieverAgent(
            HarnessConfig(
                system_prompt_version="v4.3",
                final_response_strategy="tool_strategy",
                retriever_k=5,
                agent_max_k=2,
                max_search_calls=3,
            ),
            llm,
            search,
        )
        result = agent.retrieve("what is SMIT?")
        self.assertEqual(result.tool_calls_executed, 1)
        self.assertEqual(result.answer, "SMIT uses a harness.")
        self.assertEqual(list(result.supporting_corpus_ids), ["SMIT_0"])
        self.assertEqual(len(search.calls), 1)
        self.assertEqual(search.calls[0], ("SMIT harness", 2))
        # search_index stays visible while budget remains.
        self.assertTrue(all("search_index" in turn_tools for turn_tools in llm.seen_tools))

    def test_corpus_ids_only_strategy_has_no_answer_field(self) -> None:
        search = FakeSearch()
        llm = FakeLLM(
            [
                LLMResponse(
                    tool_calls=(
                        ToolCall(name="search_index", arguments={"query": "q"}, id="c1"),
                    )
                ),
                LLMResponse(
                    tool_calls=(
                        ToolCall(
                            name=ANSWER_TOOL_NAME,
                            arguments={"supporting_corpus_ids": ["q_0", "q_1"]},
                            id="c2",
                        ),
                    )
                ),
            ]
        )
        agent = RetrieverAgent(
            HarnessConfig(
                system_prompt_version="v3",
                final_response_strategy="custom_tool_strategy_corpus_ids_only",
                agent_max_k=2,
            ),
            llm,
            search,
        )
        result = agent.retrieve("question")
        self.assertEqual(result.answer, "")
        self.assertEqual(list(result.supporting_corpus_ids), ["q_0", "q_1"])

    def test_retrieved_enum_stays_off_openai_schema(self) -> None:
        class RecordingLLM(FakeLLM):
            def __init__(self, responses):
                super().__init__(responses)
                self.seen_schemas = []

            def call(self, messages, tools):
                self.seen_schemas.append(list(tools))
                return super().call(messages, tools)

        search = FakeSearch()
        llm = RecordingLLM(
            [
                LLMResponse(
                    tool_calls=(
                        ToolCall(name="search_index", arguments={"query": "q", "k": 2}, id="c1"),
                    )
                ),
                LLMResponse(
                    tool_calls=(
                        ToolCall(
                            name=ANSWER_TOOL_NAME,
                            arguments={"supporting_corpus_ids": ["q_0"]},
                            id="c2",
                        ),
                    )
                ),
            ]
        )
        agent = RetrieverAgent(
            HarnessConfig(
                system_prompt_version="v4.3",
                final_response_strategy="custom_tool_strategy_corpus_ids_only",
                agent_max_k=2,
                constrain_supporting_ids_to_retrieved=True,
            ),
            llm,
            search,
        )
        result = agent.retrieve("question")
        self.assertEqual(list(result.supporting_corpus_ids), ["q_0"])
        first_answer = next(tool for tool in llm.seen_schemas[0] if tool.name == ANSWER_TOOL_NAME)
        second_answer = next(tool for tool in llm.seen_schemas[1] if tool.name == ANSWER_TOOL_NAME)
        self.assertEqual(first_answer.guided_string_enums, {})
        self.assertEqual(
            list(second_answer.guided_string_enums["supporting_corpus_ids"]),
            ["q_0", "q_1"],
        )
        items = second_answer.parameters["properties"]["supporting_corpus_ids"]["items"]
        self.assertEqual(items, {"type": "string"})
        self.assertNotIn("enum", items)

    def test_native_answer_tool_derives_ids_from_pick_supporting_documents(self) -> None:
        search = FakeSearch()
        llm = FakeLLM(
            [
                LLMResponse(
                    tool_calls=(
                        ToolCall(name="search_index", arguments={"query": "bridge"}, id="c1"),
                    )
                ),
                LLMResponse(
                    tool_calls=(
                        ToolCall(
                            name="pick_supporting_documents",
                            arguments={"document_ids": [0, 1]},
                            id="c2",
                        ),
                    )
                ),
                LLMResponse(
                    tool_calls=(
                        ToolCall(
                            name=ANSWER_TOOL_NAME,
                            arguments={"answer": "Brazil"},
                            id="c3",
                        ),
                    )
                ),
            ]
        )
        agent = RetrieverAgent(
            HarnessConfig(
                system_prompt_version="v5",
                final_response_strategy="native_tool_without_supporting_ids",
                retriever_k=0,
                agent_max_k=2,
            ),
            llm,
            search,
        )
        result = agent.retrieve("where is Rio?")
        self.assertEqual(result.answer, "Brazil")
        self.assertEqual(list(result.supporting_corpus_ids), ["bridge_0", "bridge_1"])
        self.assertEqual(llm.seen_tools[0], ["search_index", "pick_supporting_documents", "answer"])

    def test_pick_supporting_documents_rejects_ids_never_retrieved(self) -> None:
        search = FakeSearch()
        llm = FakeLLM(
            [
                LLMResponse(
                    tool_calls=(
                        ToolCall(name="search_index", arguments={"query": "q"}, id="c1"),
                    )
                ),
                LLMResponse(
                    tool_calls=(
                        ToolCall(
                            name="pick_supporting_documents",
                            arguments={"document_ids": [0, 99]},
                            id="c2",
                        ),
                    )
                ),
                LLMResponse(
                    tool_calls=(ToolCall(name=ANSWER_TOOL_NAME, arguments={"answer": "ok"}, id="c3"),)
                ),
            ]
        )
        agent = RetrieverAgent(
            HarnessConfig(
                system_prompt_version="v5",
                final_response_strategy="native_tool_without_supporting_ids",
                agent_max_k=1,
            ),
            llm,
            search,
        )
        result = agent.retrieve("q")
        self.assertEqual(list(result.supporting_corpus_ids), ["q_0"])

    def test_search_budget_exhausted_hides_search_tool(self) -> None:
        search = FakeSearch()
        llm = FakeLLM(
            [
                LLMResponse(
                    tool_calls=(ToolCall(name="search_index", arguments={"query": f"q{i}"}, id=f"s{i}"),)
                )
                for i in range(2)
            ]
            + [
                LLMResponse(
                    tool_calls=(
                        ToolCall(
                            name=TOOL_STRATEGY_FINAL_TOOL_NAME,
                            arguments={"answer": "done", "supporting_corpus_ids": []},
                            id="a",
                        ),
                    )
                )
            ]
        )
        agent = RetrieverAgent(
            HarnessConfig(final_response_strategy="tool_strategy", max_search_calls=2, max_turns=10),
            llm,
            search,
        )
        result = agent.retrieve("q")
        self.assertEqual(result.tool_calls_executed, 2)
        self.assertEqual(result.tool_calls_attempted, 2)
        self.assertEqual(result.answer, "done")
        # After two successful searches, the third LLM turn must not offer search_index.
        self.assertEqual(llm.seen_tools[-1], [TOOL_STRATEGY_FINAL_TOOL_NAME])

    def test_tool_result_format_v2_omits_rank_and_score(self) -> None:
        search = FakeSearch()
        llm = FakeLLM(
            [
                LLMResponse(
                    tool_calls=(ToolCall(name="search_index", arguments={"query": "q"}, id="c1"),)
                ),
                LLMResponse(
                    tool_calls=(
                        ToolCall(
                            name=ANSWER_TOOL_NAME,
                            arguments={"answer": "ok", "supporting_corpus_ids": ["q_0"]},
                            id="c2",
                        ),
                    )
                ),
            ]
        )
        agent = RetrieverAgent(
            HarnessConfig(
                final_response_strategy="custom_giga_tool_strategy",
                tool_result_format_version="v2",
                agent_max_k=1,
            ),
            llm,
            search,
        )
        result = agent.retrieve("q")
        tool_msg = next(m for m in result.messages if m.role == "tool" and m.name == "search_index")
        payload = __import__("json").loads(tool_msg.content)
        self.assertEqual(set(payload["results"][0]), {"corpus_id", "title", "text"})

    def test_hide_shown_retrieved_docs_replaces_duplicate_text(self) -> None:
        class StickySearch(SearchClient):
            def search(self, query: str, top_k: int):
                return [
                    SearchHit(corpus_id="shared", text=f"full text for {query}", score=0.9),
                ]

        search = StickySearch()
        llm = FakeLLM(
            [
                LLMResponse(
                    tool_calls=(ToolCall(name="search_index", arguments={"query": "first"}, id="s1"),)
                ),
                LLMResponse(
                    tool_calls=(ToolCall(name="search_index", arguments={"query": "second"}, id="s2"),)
                ),
                LLMResponse(
                    tool_calls=(
                        ToolCall(
                            name=TOOL_STRATEGY_FINAL_TOOL_NAME,
                            arguments={"answer": "ok", "supporting_corpus_ids": ["shared"]},
                            id="a",
                        ),
                    )
                ),
            ]
        )
        agent = RetrieverAgent(
            HarnessConfig(
                final_response_strategy="tool_strategy",
                context_management_strategy="hide_shown_retrieved_docs",
                max_search_calls=3,
                agent_max_k=1,
                retriever_k=0,
            ),
            llm,
            search,
        )
        result = agent.retrieve("q")
        tool_msgs = [m for m in result.messages if m.role == "tool" and m.name == "search_index"]
        self.assertEqual(len(tool_msgs), 2)
        first = __import__("json").loads(tool_msgs[0].content)
        second = __import__("json").loads(tool_msgs[1].content)
        self.assertIn("full text", first["results"][0]["text"])
        self.assertEqual(second["results"][0]["text"], "retrieved earlier at search 1")

    def test_rejected_unsolicited_tool(self) -> None:
        search = FakeSearch()
        llm = FakeLLM(
            [
                LLMResponse(
                    tool_calls=(ToolCall(name="unknown_tool", arguments={}, id="call_bad"),)
                ),
                LLMResponse(content="fallback answer"),
            ]
        )
        agent = RetrieverAgent(HarnessConfig(final_response_strategy="tool_strategy"), llm, search)
        result = agent.retrieve("question")
        self.assertTrue(any("Rejected unsolicited tool call" in error for error in result.errors))
        self.assertEqual(result.answer, "fallback answer")

    def test_turn_limit(self) -> None:
        search = FakeSearch()
        llm = FakeLLM(
            [
                LLMResponse(
                    tool_calls=(ToolCall(name="search_index", arguments={"query": "loop"}, id=f"call_{index}"),)
                )
                for index in range(40)
            ]
        )
        agent = RetrieverAgent(
            HarnessConfig(max_turns=3, max_search_calls=10),
            llm,
            search,
        )
        result = agent.retrieve("loop forever")
        self.assertTrue(any("turn limit" in error.lower() for error in result.errors))

    def test_conversation_messages_are_well_formed(self) -> None:
        search = FakeSearch()
        llm = FakeLLM(
            [
                LLMResponse(
                    tool_calls=(ToolCall(name="search_index", arguments={"query": "q"}, id="c1"),)
                ),
                LLMResponse(
                    tool_calls=(
                        ToolCall(
                            name=TOOL_STRATEGY_FINAL_TOOL_NAME,
                            arguments={"answer": "ok", "supporting_corpus_ids": []},
                            id="c2",
                        ),
                    )
                ),
            ]
        )
        agent = RetrieverAgent(HarnessConfig(final_response_strategy="tool_strategy"), llm, search)
        result = agent.retrieve("q")
        roles = [message.role for message in result.messages]
        self.assertEqual(roles[0], "system")
        self.assertEqual(roles[1], "user")
        # Every assistant tool_calls turn is immediately followed by matching tool turns.
        for index, message in enumerate(result.messages):
            if message.role == "assistant" and message.tool_calls:
                for offset, call in enumerate(message.tool_calls, start=1):
                    follow_up = result.messages[index + offset]
                    self.assertEqual(follow_up.role, "tool")
                    self.assertEqual(follow_up.tool_call_id, call.id)

    def test_llm_failure_preserves_partial_messages_and_search_counts(self) -> None:
        class BoomLLM(LLMClient):
            def __init__(self) -> None:
                self.calls = 0

            def call(self, messages, tools):
                self.calls += 1
                if self.calls == 1:
                    return LLMResponse(
                        tool_calls=(
                            ToolCall(
                                name="search_index",
                                arguments={"query": "partial", "k": 1},
                                id="c1",
                            ),
                        )
                    )
                raise RuntimeError("context length exceeded")

        search = FakeSearch()
        agent = RetrieverAgent(
            HarnessConfig(final_response_strategy="tool_strategy", max_turns=5),
            BoomLLM(),
            search,
        )
        result = agent.retrieve("q")
        self.assertGreaterEqual(result.tool_calls_executed, 1)
        self.assertTrue(any(message.role == "tool" for message in result.messages))
        self.assertTrue(
            any("RuntimeError: context length exceeded" in error for error in result.errors)
        )




    def test_search_only_eos_stop_is_success(self) -> None:
        search = FakeSearch()
        llm = FakeLLM(
            [
                LLMResponse(
                    tool_calls=(
                        ToolCall(name="search_index", arguments={"query": "q"}, id="c1"),
                    )
                ),
                LLMResponse(content=""),
            ]
        )
        agent = RetrieverAgent(
            HarnessConfig(
                system_prompt_version="v7",
                final_response_strategy="search_only",
                max_search_calls=3,
            ),
            llm,
            search,
        )
        result = agent.retrieve("question?")
        self.assertEqual(result.tool_calls_executed, 1)
        self.assertEqual(result.errors, [])
        self.assertEqual(llm.calls, 2)
        # Budget remains, so search_index is still offered; model stops with EOS.
        self.assertEqual(llm.seen_tools[-1], ["search_index"])

    def test_search_only_warn_hides_search_after_budget(self) -> None:
        search = FakeSearch()
        llm = FakeLLM(
            [
                LLMResponse(
                    tool_calls=(
                        ToolCall(name="search_index", arguments={"query": "q1"}, id="s1"),
                    )
                ),
                LLMResponse(content=""),
            ]
        )
        agent = RetrieverAgent(
            HarnessConfig(
                system_prompt_version="v7",
                final_response_strategy="search_only",
                max_search_calls=1,
                search_limit_policy="warn_search_limit",
            ),
            llm,
            search,
        )
        result = agent.retrieve("q")
        self.assertEqual(result.tool_calls_executed, 1)
        self.assertEqual(result.errors, [])
        self.assertEqual(llm.seen_tools[0], ["search_index"])
        self.assertEqual(llm.seen_tools[1], [])

    def test_search_only_truncate_stops_after_budget(self) -> None:
        search = FakeSearch()
        llm = FakeLLM(
            [
                LLMResponse(
                    tool_calls=(
                        ToolCall(name="search_index", arguments={"query": "q1"}, id="s1"),
                    )
                ),
                LLMResponse(
                    tool_calls=(
                        ToolCall(name="search_index", arguments={"query": "should_not_run"}, id="s2"),
                    )
                ),
            ]
        )
        agent = RetrieverAgent(
            HarnessConfig(
                system_prompt_version="v7",
                final_response_strategy="search_only",
                max_search_calls=1,
                search_limit_policy="truncate_search_limit",
            ),
            llm,
            search,
        )
        result = agent.retrieve("q")
        self.assertEqual(result.tool_calls_executed, 1)
        self.assertEqual(llm.calls, 1)
        self.assertEqual(result.errors, [])

if __name__ == "__main__":
    unittest.main()
