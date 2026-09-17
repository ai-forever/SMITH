# Using SMIT

The SMIT repository provides the tool-calling loop for the SMITH search agent.
It is backend-neutral: applications supply a search component and an
agent-model component, while the harness owns prompts, tool schemas, search
budgets, conversation state, and final result normalization.

The three main application components are:

- an **embedder model** that converts a text query into an index-compatible
  vector;
- an **index and corpus** that retrieve passages and expose stable corpus IDs;
- an **agent model** that chooses search queries and submits the final answer
  through tool calls.

Create these components once and reuse them across calls to
`RetrieverAgent.retrieve`.

## Pinned agent defaults

`HarnessConfig()` is the SMITH-Exp operating point. Read the pin from the
config rather than restating it in application code:

```python
config = HarnessConfig()
system_prompt_version = config.system_prompt_version
final_response_strategy = config.final_response_strategy
```

Use this pair for SMITH-Exp. The pinned prompt is
[`prompts/system/default.md`](prompts/system/default.md), a symlink to the
variable-k v4.3 template at
[`prompts/system/v4.3/agent_max_k.md`](prompts/system/v4.3/agent_max_k.md).
`config.agent_max_k` is substituted into that template.

`custom_giga_tool_strategy` exposes the GigaChat-compatible `answer` tool with
both `answer` and `supporting_corpus_ids`. Do not replace it with
`tool_strategy` for this model: that strategy uses the historical
`AgentStructuredResponse` tool name intended for other tool-calling clients.

## Index and embedder

Expose retrieval through the `SearchClient` protocol. The harness calls
`search(query, top_k)` whenever the agent invokes `search_index`.

```python
from common import SearchHit


class DenseSearch:
    def __init__(self, embedder, index, corpus):
        self.embedder = embedder
        self.index = index
        self.corpus = corpus

    def search(self, query: str, top_k: int):
        query_vector = self.embedder.encode_query(query)
        scores, rows = self.index.search(query_vector, top_k)

        hits = []
        for rank, (score, row) in enumerate(zip(scores, rows), start=1):
            if int(row) < 0:
                continue
            document = self.corpus[int(row)]
            hits.append(
                SearchHit(
                    corpus_id=str(document["corpus_id"]),
                    title=str(document.get("title", "")),
                    text=str(document["text"]),
                    score=float(score),
                    metadata={"doc_index": int(row), "rank": rank},
                )
            )
        return hits
```

The example intentionally leaves model and index APIs abstract. For FAISS,
convert the query vector to the dtype and batch shape expected by the index.
The query embedder must use the same model, dimensions, pooling,
normalization, and query instruction used to build the document index.

Every hit must contain:

- `corpus_id`: a stable ID copied from the corpus, never a rank or list index;
- `text`: the bounded evidence passage shown to the agent;
- optional `title`, `score`, and application metadata.

The harness serializes these hits into the tool result, remembers all retrieved
corpus IDs, and exposes each search in `RetrievalResult.search_history`.

## Agent model

Expose the model through the `LLMClient` protocol. Its `call` method receives
the complete conversation and the tools currently allowed by the harness.

```python
import json

from common import LLMResponse, ToolCall


class AgentModel:
    def __init__(self, client, model, config):
        self.client = client
        self.model = model
        self.config = config

    def call(self, messages, tools):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=to_openai_messages(messages),
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": dict(tool.parameters),
                    },
                }
                for tool in tools
            ],
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            tool_choice=self.config.tool_choice,
        )
        message = response.choices[0].message
        calls = [
            ToolCall(
                name=call.function.name,
                arguments=json.loads(call.function.arguments or "{}"),
                id=call.id,
            )
            for call in (message.tool_calls or [])
        ]
        return LLMResponse(
            content=message.content or "",
            tool_calls=calls,
            raw=response,
        )
```

`to_openai_messages` is a small application adapter. Preserve system/user
content, assistant tool calls, and the `tool_call_id` on tool-result messages.
Keep tool-call IDs unchanged between the assistant request and corresponding
tool result.

The recommended experimental agent model is
[ai-forever/SMITH-Exp](https://huggingface.co/ai-forever/SMITH-Exp).
Follow its model card for the chat template, tool parser, and serving command.

## Run the agent loop

With `HarnessConfig` and `RetrieverAgent` imported from the package:

```python
config = HarnessConfig()

agent = RetrieverAgent(
    config=config,
    llm=AgentModel(openai_client, "ai-forever/SMITH-Exp", config),
    search=DenseSearch(embedder, index, corpus),
)

result = agent.retrieve("What evidence answers the question?")

print(result.answer)
print(result.supporting_corpus_ids)
for search_query, hits in result.search_history:
    print(search_query, [hit.corpus_id for hit in hits])
```

One `retrieve` call creates one independent conversation. The model may call
the search tool repeatedly until it submits the configured final-answer tool
or reaches the search/turn budget. Inspect `result.errors` when a run ends
without a valid final answer.

## Guided decoding

> **Strongly recommended for answer strategies that return
> `supporting_corpus_ids`.**

Enabling guided decoding removes document ID hallucination by constraining
model output to seen corpus IDs only.

The prompt already asks the model to copy IDs from retrieved passages.
Guided decoding enforces that same constraint at generation time, so every
`supporting_corpus_ids` item is one of `SessionState.seen_corpus_ids`. The
harness then filters and de-duplicates the final list against those IDs, which
keeps the result aligned with retrieved evidence.

Retrieved-ID guidance follows `config.constrain_supporting_ids_to_retrieved`
and is **on by default**:

```python
config = HarnessConfig()
guided_min = config.guided_supporting_ids_min_items
guided_max = config.guided_supporting_ids_max_items
```

Keep guidance enabled when the server loads a compatible plugin. Set
`config.constrain_supporting_ids_to_retrieved` to `False` only for backends
that do not accept the extra payload below.

When this option is enabled, the harness rebuilds the final-answer
`ToolSchema` on every turn. After at least one search, it attaches:

```python
tool.guided_string_enums == {
    "supporting_corpus_ids": ["retrieved-id-1", "retrieved-id-2", "..."]
}
```

The values are the unique corpus IDs retrieved so far. This metadata is kept
separate from `tool.parameters` so the ordinary OpenAI tool schema — and the
prompt rendered from it — stays compact. A large retrieved-ID list belongs in
the guided-decoding payload, not in the chat template.

The `LLMClient` adapter should collect this metadata and pass it to a
compatible serving backend in the request `extra_body`. For the SMITH vLLM
plugin, the payload has this form:

```python
guided = {
    tool.name: {
        field: list(values)
        for field, values in tool.guided_string_enums.items()
    }
    for tool in tools
    if tool.guided_string_enums
}

extra_body = {
    "smit_guided_string_enums": guided,
    "smit_guided_array_min_items": config.guided_supporting_ids_min_items,
    "smit_guided_array_max_items": config.guided_supporting_ids_max_items,
}
```

The model-side guided-decoding plugin consumes that payload and patches the
grammar for the current final-answer tool. During token generation:

- each generated `supporting_corpus_ids` item is one of the retrieved IDs;
- the generated array length stays within the configured minimum and maximum;
- the search query, answer text, document ranking, and evidence selection
  remain decisions of the model and retriever.

The grammar bounds the number of generated array items. The harness
de-duplicates the final list, so the normalized result reports unique IDs.

Guided decoding therefore makes citation membership match the retrieved set.
Answer wording and which retrieved passages to cite stay with the model; the
harness still de-duplicates and validates the submitted IDs.

Guidance is attached to answer schemas that contain `supporting_corpus_ids`,
after IDs have been retrieved. With a compatible plugin, send these custom
`extra_body` keys so generation stays on retrieved IDs. Without a plugin, omit
them; the harness still keeps only retrieved IDs in the result.

## Hyperparameters

You are **not expected to change these** — the defaults hardcoded on
`HarnessConfig` are the SMITH-Exp operating point used for the model card
evaluation. Copy values from a `HarnessConfig` instance in application code.
`HarnessConfig()` is a valid operating point.

| field | default | what it controls |
|-------|---------|------------------|
| `system_prompt_version` | `"v4.3"` | system prompt template family |
| `final_response_strategy` | `"custom_giga_tool_strategy"` | final-answer tool name and arguments (`answer` + `supporting_corpus_ids`) |
| `agent_max_k` | `10` | cap on `search_index` `k`; omitted `k` also uses this value |
| `agent_fixed_k` | `None` | if set, omit `k` from the search schema and ignore a model-requested `k` |
| `max_search_calls` | `6` | successful `search_index` calls before the search tool is hidden |
| `max_turns` | `8` | LLM-call ceiling (`estimate_max_turns(6)` for this prompt/strategy pair) |
| `tool_result_format_version` | `"v2"` | `search_index` observations as `{corpus_id,title,text}` |
| `constrain_supporting_ids_to_retrieved` | `True` | attach retrieved-ID guidance metadata for `supporting_corpus_ids` |
| `guided_supporting_ids_min_items` | `2` | GBNF minimum array length when guidance is on |
| `guided_supporting_ids_max_items` | `10` | GBNF maximum array length when guidance is on |
| `max_search_query_chars` | `256` | word-boundary cap on `search_index` / `browser.search` queries |
| `temperature` / `top_p` | `0.0` / `1.0` | sampling on each LLM turn (copy onto the `LLMClient` request) |
| `tool_choice` | `"required"` | force a tool call each turn (copy onto the `LLMClient` request) |

The prompts under [`prompts/system/`](prompts/system/) and the tool schemas in
[`tools/schemas.py`](tools/schemas.py) *are* the harness. Changing them changes
the model's behavior.

The dispatcher uses `agent_max_k` (or `agent_fixed_k`) as the search `top_k`.
A host-side `retriever_k` used for a separate baseline retrieval is not an
effective harness search parameter.

Search-query overflow is controlled only by `config.max_search_query_chars`.
`truncate_search_query` applies that window at dispatch, so a long
`search_index` query still runs as a shorter search.
