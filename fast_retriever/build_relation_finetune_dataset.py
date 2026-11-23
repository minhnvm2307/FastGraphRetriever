#!/usr/bin/env python3
"""Build the training data of the BGE-M3 relation encoder (question -> relations of its labelled paths).

Example (from the repository root)::

    python retrieve/build_relation_finetune_dataset.py \
      --data-root retrieve/data_files \
      --output-dir retrieve/data_files/bge_ft/webqsp_entity_disjoint \
      --dataset webqsp --source-split train --relation-format split --seed 42

The source data root is read-only from this script's perspective.  All output
files are written atomically, and ``manifest.json`` is committed last.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import torch

try:
    from retrieve.bge_ft.data import (
        SCHEMA_VERSION,
        BuilderConfig,
        atomic_write_json,
        atomic_write_jsonl,
        build_records,
        builder_config_dict,
        collect_relation_vocabularies,
        file_sha256,
        object_sha256,
        summarize_records,
        vocabularies_from_rich_records,
        vocabulary_overlap,
    )
except ModuleNotFoundError:  # Direct ``python retrieve/script.py`` execution.
    from bge_ft.data import (  # type: ignore[no-redef]
        SCHEMA_VERSION,
        BuilderConfig,
        atomic_write_json,
        atomic_write_jsonl,
        build_records,
        builder_config_dict,
        collect_relation_vocabularies,
        file_sha256,
        object_sha256,
        summarize_records,
        vocabularies_from_rich_records,
        vocabulary_overlap,
    )


DEFAULT_COMPARISONS = ("webqsp:test", "cwq:test")
OUTPUT_NAMES = {
    "rich": "rich_train.jsonl",
    "fit": "flat_fit.jsonl",
    "dev": "flat_dev.jsonl",
    "stats": "stats.json",
    "manifest": "manifest.json",
}


def _load_pickle(path: Path) -> list[dict[str, Any]]:
    with path.open("rb") as handle:
        value = pickle.load(handle)
    if not isinstance(value, list):
        raise ValueError(f"processed source must contain a list: {path}")
    return value


def _load_labels(path: Path) -> dict[str, dict[str, Any]]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # torch 2.1 compatibility.
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, dict):
        raise ValueError(f"label source must contain a dictionary: {path}")
    return {str(key): entry for key, entry in value.items()}


def _source_paths(data_root: Path, dataset: str, split: str) -> tuple[Path, Path]:
    return (
        data_root / dataset / "processed" / f"{split}.pkl",
        data_root / dataset / "triple_scores" / f"{split}.pth",
    )


def _require_source_paths(data_root: Path, dataset: str, split: str) -> tuple[Path, Path]:
    processed, labels = _source_paths(data_root, dataset, split)
    missing = [str(path) for path in (processed, labels) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing BGE-M3 relation-encoder source file(s): {missing}")
    return processed, labels


def _parse_comparison(value: str) -> tuple[str, str]:
    try:
        dataset, split = value.split(":", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("comparison must use DATASET:SPLIT") from error
    if not dataset or not split:
        raise argparse.ArgumentTypeError("comparison must use non-empty DATASET:SPLIT")
    return dataset, split


def _ensure_output_policy(output_dir: Path, overwrite: bool) -> None:
    existing = [output_dir / name for name in OUTPUT_NAMES.values() if (output_dir / name).exists()]
    if existing and not overwrite:
        rendered = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"BGE-M3 relation-encoder output already exists ({rendered}); pass --overwrite "
            "to atomically replace it"
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def build_dataset_files(args: argparse.Namespace) -> dict[str, Any]:
    data_root = Path(args.data_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    _ensure_output_policy(output_dir, args.overwrite)

    config = BuilderConfig(
        relation_format=args.relation_format,
        seed=args.seed,
        dev_ratio=args.dev_ratio,
        split_strategy=args.split_strategy,
        max_negatives=args.max_negatives,
        lexical_threshold=args.lexical_threshold,
        graph_hard_max_distance=args.graph_hard_max_distance,
        l1_ratio=args.l1_ratio,
        l2_ratio=args.l2_ratio,
        l3_ratio=args.l3_ratio,
        l4_ratio=args.l4_ratio,
    )
    config.validate()

    processed_path, label_path = _require_source_paths(
        data_root, args.dataset, args.source_split
    )
    samples = _load_pickle(processed_path)
    labels = _load_labels(label_path)
    rich_records, fit_records, dev_records = build_records(
        samples,
        labels,
        dataset=args.dataset,
        source_split=args.source_split,
        config=config,
    )
    # Reuse the authoritative rich records instead of traversing every train
    # graph a second time.  Comparison splits are still traversed once below.
    train_vocabularies = vocabularies_from_rich_records(rich_records)

    input_files: dict[str, dict[str, Any]] = {
        f"{args.dataset}:{args.source_split}:processed": {
            "path": str(processed_path),
            "sha256": file_sha256(processed_path),
            "size_bytes": processed_path.stat().st_size,
        },
        f"{args.dataset}:{args.source_split}:labels": {
            "path": str(label_path),
            "sha256": file_sha256(label_path),
            "size_bytes": label_path.stat().st_size,
        },
    }
    overlap: dict[str, Any] = {}
    for dataset, split in args.comparison:
        comparison_processed, comparison_labels = _require_source_paths(
            data_root, dataset, split
        )
        comparison_samples = _load_pickle(comparison_processed)
        comparison_label_values = _load_labels(comparison_labels)
        comparison_vocabularies = collect_relation_vocabularies(
            comparison_samples, comparison_label_values
        )
        comparison_key = f"{dataset}:{split}"
        overlap[comparison_key] = vocabulary_overlap(
            train_vocabularies, comparison_vocabularies
        )
        input_files[f"{comparison_key}:processed"] = {
            "path": str(comparison_processed),
            "sha256": file_sha256(comparison_processed),
            "size_bytes": comparison_processed.stat().st_size,
        }
        input_files[f"{comparison_key}:labels"] = {
            "path": str(comparison_labels),
            "sha256": file_sha256(comparison_labels),
            "size_bytes": comparison_labels.stat().st_size,
        }
        del comparison_samples, comparison_label_values, comparison_vocabularies

    artifact_paths = {
        key: output_dir / OUTPUT_NAMES[key] for key in ("rich", "fit", "dev", "stats")
    }
    atomic_write_jsonl(artifact_paths["rich"], rich_records)
    atomic_write_jsonl(artifact_paths["fit"], fit_records)
    atomic_write_jsonl(artifact_paths["dev"], dev_records)

    stats = summarize_records(rich_records, fit_records, dev_records)
    stats.update(
        {
            "dataset": args.dataset,
            "source_split": args.source_split,
            "builder_config": builder_config_dict(config),
            "source_vocab_sizes": {
                name: len(values) for name, values in sorted(train_vocabularies.items())
            },
            "vocabulary_overlap": overlap,
            "artifact_sha256": {
                key: file_sha256(path)
                for key, path in artifact_paths.items()
                if key != "stats"
            },
        }
    )
    atomic_write_json(artifact_paths["stats"], stats)

    output_files = {
        key: {
            "path": str(path),
            "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for key, path in artifact_paths.items()
    }
    effective_config = builder_config_dict(config)
    builder_path = Path(__file__).resolve()
    implementation_paths = {
        "builder": builder_path,
        "data": builder_path.parent / "bge_ft" / "data.py",
        "relation_text": builder_path.parent / "bge_ft" / "relation_text.py",
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "builder": "retrieve/build_relation_finetune_dataset.py",
        "data_root": str(data_root),
        "dataset": args.dataset,
        "source_split": args.source_split,
        "config": effective_config,
        "config_sha256": object_sha256(effective_config),
        "comparisons": [f"{dataset}:{split}" for dataset, split in args.comparison],
        "inputs": dict(sorted(input_files.items())),
        "outputs": output_files,
        "implementation": {
            name: {"path": str(path), "sha256": file_sha256(path)}
            for name, path in implementation_paths.items()
        },
        "record_counts": stats["counts"],
        "manifest_commit_protocol": "manifest_written_atomically_after_all_other_artifacts",
    }
    manifest_path = output_dir / OUTPUT_NAMES["manifest"]
    atomic_write_json(manifest_path, manifest)
    return {
        "output_dir": str(output_dir),
        "manifest": str(manifest_path),
        "record_counts": stats["counts"],
        "verification": stats["verification"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build edge-complete, auditable relation fine-tuning data for the BGE-M3 relation encoder"
    )
    parser.add_argument(
        "--data-root",
        default="retrieve/data_files",
        help="read-only root containing <dataset>/processed and triple_scores",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset", default="webqsp")
    parser.add_argument("--source-split", default="train")
    parser.add_argument(
        "--comparison",
        action="append",
        type=_parse_comparison,
        help="comparison vocabulary split as DATASET:SPLIT; repeatable",
    )
    parser.add_argument("--relation-format", choices=("raw", "split", "natural"), default="split")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dev-ratio", type=float, default=0.1)
    parser.add_argument(
        "--split-strategy",
        choices=("record_hash", "entity_component"),
        default="record_hash",
        help="fit/dev split by query hash or topic+answer entity components",
    )
    parser.add_argument("--max-negatives", type=int, default=32)
    parser.add_argument("--lexical-threshold", type=float, default=0.5)
    parser.add_argument("--graph-hard-max-distance", type=int, default=1)
    parser.add_argument("--l1-ratio", type=float, default=0.15)
    parser.add_argument("--l2-ratio", type=float, default=0.20)
    parser.add_argument("--l3-ratio", type=float, default=0.30)
    parser.add_argument("--l4-ratio", type=float, default=0.25)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.comparison is None:
        args.comparison = [_parse_comparison(value) for value in DEFAULT_COMPARISONS]
    # Duplicate comparisons waste several GB of source reads and can hide a typo.
    if len(set(args.comparison)) != len(args.comparison):
        parser.error("--comparison values must be unique")
    result = build_dataset_files(args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
