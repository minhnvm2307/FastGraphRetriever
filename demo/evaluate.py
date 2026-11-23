"""Evaluate the prepared local GTSQA sample at 50/100/200 triples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .data import DEFAULT_DATA_DIR, GraphRecord, read_graph, read_manifest
from .encoders import FeatureStore
from .retriever import Retriever

BUDGETS = (50, 100, 200)


def evaluate_records(records: list[GraphRecord], features, retriever: Retriever) -> dict:
    per_budget = {k: {"answer_node_recall": [], "gold_triple_recall": []} for k in BUDGETS}
    times = []
    for rec in records:
        result = retriever.retrieve(rec, rec.question, rec.seeds, 200, features)
        times.append(result["timings_ms"]["rbf_score"])
        for k in BUDGETS:
            indices = [item["index"] for item in result["triples"][:k]]
            found = set(rec.h[indices].tolist()) | set(rec.t[indices].tolist())
            per_budget[k]["answer_node_recall"].append(
                len(found.intersection(rec.answers)) / max(len(rec.answers), 1)
            )
            per_budget[k]["gold_triple_recall"].append(
                float(rec.gold[indices].sum() / max(int(rec.gold.sum()), 1))
            )
    return {
        "questions": len(records), "split": "sampled GTSQA test",
        "target": "answer_node and annotated answer_subgraph",
        "budgets": {
            str(k): {
                metric: float(np.mean(values)) if values else 0.0
                for metric, values in scores.items()
            }
            for k, scores in per_budget.items()
        },
        "rbf_score_ms": {
            "median": float(np.median(times)) if times else 0.0,
            "p95": float(np.percentile(times, 95)) if times else 0.0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--out", help="optional output JSON path")
    args = parser.parse_args()
    directory = Path(args.data_dir)
    manifest = read_manifest(directory)
    if not manifest.get("features_ready"):
        parser.error("run python -m demo.prepare before evaluating")
    records = [read_graph(item["id"], directory) for item in manifest["questions"]]
    results = evaluate_records(records, FeatureStore(directory), Retriever(threads=args.threads))
    results["dataset_revision"] = manifest["revision"]
    results["source_sha256"] = manifest["source_sha256"]
    output = json.dumps(results, indent=2)
    if args.out:
        Path(args.out).write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
