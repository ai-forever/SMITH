# SMITH search agent

This repository contains the harness for search agent
[ai-forever/SMITH-v1](https://huggingface.co/ai-forever/SMITH-v1)
and other models from the
[SMITH collection](https://huggingface.co/collections/ai-forever/smith).

This is a lightweight, backend-neutral harness which owns the search agent's
prompt, tool policy, turn limits, and normalized trace result. The LLM agent,
embedder and index components are to be implemented on top of the harness by
downstream applications.

Use `HarnessConfig.smith_v1()` for the v4.3 retrieval-only profile. It returns ranked corpus IDs and shows the remaining search budget each turn.

Implementation and integration details are described in
[`AGENTS.md`](AGENTS.md).

## License

Licensed under the [Apache License 2.0](LICENSE).
