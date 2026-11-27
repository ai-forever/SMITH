# Route retrieval with FRIDA-Decisions

Before retrieval, ask FRIDA-Decisions whether the question requires identifying an entity from clues. Send the original question to SMITH when the “yes” probability is at most 0.75 and to your main agent otherwise. The [SMITH-v1 evaluation](https://huggingface.co/ai-forever/SMITH-Exp/blob/v1/README.md#metrics) used DeepSeek V4.1 Flash as the main agent.

The evaluation used FRIDA-Decisions V2 28B with in-process scoring. The client below sends the same question and criteria to a FRIDA-Decisions `/pooling` endpoint. To use the evaluated model, supply an endpoint serving that checkpoint. The [public FRIDA-Decisions model and server instructions](https://huggingface.co/ai-forever/FRIDA-Decisions#vllm-server) provide an 823M alternative whose routing decisions can differ.

Run from the harness repository with Python 3.9 or later. Set `--model` to the name advertised by your server.

```bash
python3 decisions_router/router.py --url http://localhost:8400 --model m \
  'В каком городе родился автор романа «Обломов»?'
```

In Python, call `route(question, url=..., model=...)` from `decisions_router.router`. It returns the selected retrieval agent (`"smith"` or `"main"`) and the “yes” probability. Pass the question to that agent using your existing retrieval setup.
