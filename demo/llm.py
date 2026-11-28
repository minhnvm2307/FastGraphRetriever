"""DeepSeek P2h answers over ranked RBF evidence."""

from __future__ import annotations

import os
from pathlib import Path

from .p2h_inference import p2h_inference

DEFAULT_MODEL = "deepseek-flash"


def load_local_env() -> None:
    """Load only DeepSeek settings from the ignored project-local .env."""
    path = Path(__file__).resolve().parents[1] / ".env"
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        name, separator, value = line.partition("=")
        name = name.strip()
        if separator and name in {"DEEPSEEK_API_KEY", "DEEPSEEK_MODEL"}:
            os.environ.setdefault(name, value.strip().strip('"\''))


load_local_env()


def deepseek_ready() -> bool:
    return bool(os.getenv("DEEPSEEK_API_KEY"))


def answer_with_details(
    question: str, triples: list[dict], topic_entities: list[str],
    history: list[dict[str, str]] | None = None,
) -> dict:
    """Run the supplied P2h format and preserve its timing and usage details."""
    if not deepseek_ready():
        raise RuntimeError("Set DEEPSEEK_API_KEY to enable DeepSeek answers")
    triplets = [(item["head"], item["relation"], item["tail"]) for item in triples]
    return p2h_inference(
        triplets=triplets, question=question, topic_entities=topic_entities,
        model=os.getenv("DEEPSEEK_MODEL", DEFAULT_MODEL),
        k=min(len(triplets), 100), max_tokens=1024,
        history=history,
    )


def answer_question(
    question: str, triples: list[dict], topic_entities: list[str],
) -> tuple[str, list[str]]:
    result = answer_with_details(question, triples, topic_entities)
    warnings = [f"Đã chạy lượt sửa đáp án ({result['repaired']})."] if result["repaired"] else []
    return result["response"], warnings
