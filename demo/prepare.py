"""Prepare a small, paper-aligned local GTSQA test set.

Run: python -m demo.prepare --count 100
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict, deque
from pathlib import Path

from .data import DEFAULT_DATA_DIR, graph_from_row, write_graph
from .encoders import (
    BGE_MODEL, BGE_REVISION, DEFAULT_ADAPTER, GTE_CODE_REVISION, GTE_MODEL, GTE_REVISION,
    ensure_adapter, prepare_embeddings,
)

DATASET = "Graphcore/GTSQA"
REVISION = "21dfffc"
PARQUET = "gtsqa-with-graphs/test-00000-of-00001.parquet"


def sample_order(metadata: list[dict]) -> list[int]:
    """Round robin across question shape, with stable hash order within each group."""
    groups: dict[tuple, list[int]] = defaultdict(list)
    for index, row in enumerate(metadata):
        kind = tuple(sorted(str(value) for value in row["test_type"]))
        key = (kind, min(int(row["n_hops"]), 4), len(row["seed_entities"]) > 1)
        groups[key].append(index)
    buckets = {}
    for key, indices in groups.items():
        indices.sort(key=lambda i: hashlib.blake2b(
            f"42:{metadata[i]['id']}".encode(), digest_size=8
        ).digest())
        buckets[key] = deque(indices)
    result = []
    while buckets:
        for key in sorted(tuple(buckets)):
            result.append(buckets[key].popleft())
            if not buckets[key]:
                del buckets[key]
    return result


def prepare(args: argparse.Namespace) -> dict:
    import pyarrow.parquet as pq

    adapter_dir = Path(args.adapter)
    adapter_info = ensure_adapter(
        adapter_dir, download_if_missing=adapter_dir.resolve() == DEFAULT_ADAPTER.resolve()
    )
    if args.source_parquet:
        source = Path(args.source_parquet)
        revision = "local-file"
    else:
        from huggingface_hub import hf_hub_download

        source = Path(hf_hub_download(
            repo_id=DATASET, repo_type="dataset", filename=PARQUET, revision=REVISION,
        ))
        revision = REVISION
    if not source.is_file():
        raise FileNotFoundError(source)
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    parquet = pq.ParquetFile(source)
    metadata = parquet.read(columns=[
        "id", "n_hops", "test_type", "seed_entities",
    ]).to_pylist()
    order = sample_order(metadata)
    candidate_indices = set(order[:args.candidate_buffer])
    candidates = {}
    row_index = 0
    columns = [
        "id", "question", "seed_entities", "answer_node", "answer_subgraph",
        "graph", "n_hops", "test_type",
    ]
    for batch in parquet.iter_batches(batch_size=8, columns=columns):
        rows = batch.to_pylist()
        for row in rows:
            if row_index in candidate_indices:
                try:
                    candidates[row_index] = graph_from_row(row)
                except (ValueError, TypeError, KeyError):
                    pass
            row_index += 1
    records = [candidates[i] for i in order if i in candidates][:args.count]
    if len(records) < args.count:
        raise RuntimeError(
            f"Only {len(records)} usable graphs among {args.candidate_buffer} candidates; "
            "increase --candidate-buffer"
        )
    directory = Path(args.data_dir)
    graph_dir = directory / "graphs"
    graph_dir.mkdir(parents=True, exist_ok=True)
    for record in records:
        write_graph(record, graph_dir)
    manifest = {
        "dataset": DATASET, "subset": "gtsqa-with-graphs", "split": "test",
        "revision": revision, "source": str(source),
        "source_sha256": digest.hexdigest(), "adapter": adapter_info,
        "encoders": {
            "gte": {"model": GTE_MODEL, "revision": GTE_REVISION,
                    "code_revision": GTE_CODE_REVISION},
            "bge": {"model": BGE_MODEL, "revision": BGE_REVISION},
        },
        "count": len(records),
        "features_ready": False,
        "questions": [
            {
                "id": rec.id, "question": rec.question, "n_hops": rec.n_hops,
                "n_seeds": len(rec.seeds), "test_type": rec.test_type,
                "n_triples": rec.n_triples,
            }
            for rec in records
        ],
    }
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    prepare_embeddings(records, directory, adapter_dir, args.device)
    manifest["features_ready"] = True
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--candidate-buffer", type=int, default=200)
    parser.add_argument("--source-parquet", help="use an existing official test parquet")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--adapter", default=os.getenv("BGE_ADAPTER_DIR", str(DEFAULT_ADAPTER)))
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    if args.count <= 0 or args.candidate_buffer < args.count:
        parser.error("candidate-buffer must be at least count, and count must be positive")
    result = prepare(args)
    print(f"Prepared {result['count']} GTSQA test graphs in {args.data_dir}")


if __name__ == "__main__":
    main()
