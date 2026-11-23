"""On-disk, question-scoped GTSQA graphs and their manifest."""

from __future__ import annotations

import gzip
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .features import clean_name

DEFAULT_DATA_DIR = Path(os.getenv("DEMO_DATA_DIR", "data/gtsqa_demo"))


@dataclass(frozen=True)
class GraphRecord:
    id: str
    question: str
    names: list[str]
    relations: list[str]
    h: np.ndarray
    r: np.ndarray
    t: np.ndarray
    seeds: list[int]
    answers: list[int]
    gold: np.ndarray
    n_hops: int
    test_type: list[str]

    @property
    def n_entities(self) -> int:
        return len(self.names)

    @property
    def n_triples(self) -> int:
        return len(self.h)

    def triple(self, index: int) -> tuple[str, str, str]:
        return (
            self.names[int(self.h[index])],
            self.relations[int(self.r[index])],
            self.names[int(self.t[index])],
        )


def graph_from_row(row: dict) -> GraphRecord:
    """Preserve the notebook's entity/relation index ordering and gold labels."""
    names: list[str] = []
    relations: list[str] = []
    entity_ids: dict[str, int] = {}
    relation_ids: dict[str, int] = {}
    heads: list[int] = []
    rels: list[int] = []
    tails: list[int] = []
    gold_set = {
        (str(h), clean_name(str(r)), str(t))
        for h, r, t in row["answer_subgraph"]
    }
    labels: list[bool] = []
    for h, r, t in row["graph"]:
        h, r, t = str(h), clean_name(str(r)), str(t)
        if h not in entity_ids:
            entity_ids[h] = len(names)
            names.append(h)
        if t not in entity_ids:
            entity_ids[t] = len(names)
            names.append(t)
        if r not in relation_ids:
            relation_ids[r] = len(relations)
            relations.append(r)
        heads.append(entity_ids[h])
        rels.append(relation_ids[r])
        tails.append(entity_ids[t])
        labels.append((h, r, t) in gold_set)
    seeds = sorted({entity_ids[str(s)] for s in row["seed_entities"] if str(s) in entity_ids})
    answer_name = str(row["answer_node"])
    answers = [entity_ids[answer_name]] if answer_name in entity_ids else []
    if not heads or not seeds or not answers or not any(labels):
        raise ValueError("question graph has no triples, seed, answer, or annotated gold triple")
    return GraphRecord(
        id=str(row["id"]), question=str(row["question"]), names=names,
        relations=relations, h=np.asarray(heads, dtype=np.int32),
        r=np.asarray(rels, dtype=np.int32), t=np.asarray(tails, dtype=np.int32),
        seeds=seeds, answers=answers, gold=np.asarray(labels, dtype=bool),
        n_hops=int(row["n_hops"]), test_type=list(row["test_type"]),
    )


def write_graph(record: GraphRecord, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "id": record.id, "question": record.question, "names": record.names,
        "relations": record.relations, "h": record.h.tolist(), "r": record.r.tolist(),
        "t": record.t.tolist(), "seeds": record.seeds, "answers": record.answers,
        "gold": record.gold.astype(int).tolist(), "n_hops": record.n_hops,
        "test_type": record.test_type,
    }
    with gzip.open(directory / f"{record.id}.json.gz", "wt", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, separators=(",", ":"))


def read_graph(graph_id: str, directory: Path = DEFAULT_DATA_DIR) -> GraphRecord:
    if not graph_id.isdecimal():
        raise ValueError("graph id must be numeric")
    with gzip.open(directory / "graphs" / f"{graph_id}.json.gz", "rt", encoding="utf-8") as stream:
        value = json.load(stream)
    return GraphRecord(
        id=value["id"], question=value["question"], names=value["names"],
        relations=value["relations"], h=np.asarray(value["h"], dtype=np.int32),
        r=np.asarray(value["r"], dtype=np.int32), t=np.asarray(value["t"], dtype=np.int32),
        seeds=value["seeds"], answers=value["answers"],
        gold=np.asarray(value["gold"], dtype=bool), n_hops=value["n_hops"],
        test_type=value["test_type"],
    )


def read_manifest(directory: Path = DEFAULT_DATA_DIR) -> dict:
    return json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
