"""Optional DeepSeek answer from retrieved, numbered triples."""

import os
import re

import requests


def answer_question(question: str, triples: list[dict]) -> tuple[str, list[str]]:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    model = os.getenv("DEEPSEEK_MODEL")
    if not api_key or not model:
        raise RuntimeError("Set DEEPSEEK_API_KEY and DEEPSEEK_MODEL to enable answers")
    evidence = "\n".join(
        f"[{t['id']}] {t['head']} | {t['relation']} | {t['tail']}"
        for t in triples[:30]
    )
    response = requests.post(
        os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
        + "/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": (
                    "Answer using only the provided knowledge graph triples. "
                    "Cite every factual claim with its triple ID, such as [T2]. "
                    "If the triples do not support an answer, say so."
                )},
                {"role": "user", "content": f"Question: {question}\n\nTriples:\n{evidence}"},
            ],
            "temperature": 0,
        },
        timeout=45,
    )
    response.raise_for_status()
    answer = response.json()["choices"][0]["message"]["content"]
    allowed = {triple["id"] for triple in triples[:30]}
    citations = set(re.findall(r"\[(T\d+)\]", answer))
    warnings = [f"Unknown citation [{item}]" for item in sorted(citations - allowed)]
    if not citations:
        warnings.append("No triple citations in the answer")
    return answer, warnings
