"""Route a question using a FRIDA-Decisions pooling endpoint."""

import argparse
import json
from urllib.request import Request, urlopen


def select_agent(p_identify_first: float) -> str:
    if not 0 <= p_identify_first <= 1:
        raise ValueError("Expected a probability between 0 and 1")
    # The evaluation stored probabilities to five decimal places before dispatch.
    return "smith" if round(p_identify_first, 5) <= 0.75 else "main"


def route(question: str, *, url: str, model: str) -> tuple[str, float]:
    """Return the retrieval agent and the router's probability for this question."""
    payload = {
        "model": model,
        "data": {
            "state": f"Вопрос: {question}",
            "questions": {
                "q": {
                    "type": "noul",
                    "instructions": "Нужно ли сначала установить, о ком или о чём идёт речь, и только потом искать сам ответ?",
                    "criteria": {
                        "true": "сначала нужно установить, о ком или о чём речь",
                        "false": "о ком или о чём речь, ясно из вопроса",
                    },
                }
            },
        },
    }
    request = Request(
        url.rstrip("/") + "/pooling",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=60) as response:
        probability = json.load(response)["data"]["answers"]["q"]["noul"]
    return select_agent(probability), probability


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--url", default="http://localhost:8400")
    parser.add_argument("--model", default="m", help="Model name configured on the server")
    args = parser.parse_args()
    agent, probability = route(args.question, url=args.url, model=args.model)
    print(json.dumps({"agent": agent, "p_identify_first": probability}))
