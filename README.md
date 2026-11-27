# SMITH search agent

This repository contains the harness for search agent
[ai-forever/SMITH-Exp](https://huggingface.co/ai-forever/SMITH-Exp)
and other models from the
[SMITH collection](https://huggingface.co/collections/ai-forever/smith).

This is a lightweight, backend-neutral harness which owns the search agent's
prompt, tool policy, turn limits, and normalized trace result. The LLM agent,
embedder and index components are to be implemented on top of the harness by
downstream applications.

Implementation and integration details are described in
[`AGENTS.md`](AGENTS.md).

## License

Licensed under the [Apache License 2.0](LICENSE).
