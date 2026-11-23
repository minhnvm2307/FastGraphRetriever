"""Streaming intrinsic evaluation utilities for the BGE-M3 relation encoder.

The core evaluator deliberately knows nothing about answer entities.  A scorer
receives only the question text and formatted candidate-relation texts, which
makes the no-answer-at-inference contract straightforward to test.

``BGEM3RelationScorer`` is imported lazily and is therefore optional for unit
tests and data validation.  It supports a base/merged BGE-M3 model directory as
well as a PEFT adapter merged into the in-memory backbone.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Protocol, Sequence

from .relation_text import FORMATTERS, format_relation


DEFAULT_K_VALUES = (1, 3, 5, 10)
SUPPORTED_SCORE_MODES = frozenset(
    {"dense", "sparse", "dense_sparse", "colbert", "ensemble"}
)


class RelationScorer(Protocol):
    """Minimal scorer boundary used by evaluation and dynamic mining."""

    supported_modes: frozenset[str]

    def score(
        self,
        query: str,
        passages: Sequence[str],
        score_modes: Sequence[str],
    ) -> Mapping[str, Sequence[float]]:
        """Return one score per passage for every requested mode."""


def sha256_file(path: os.PathLike[str] | str, chunk_size: int = 1024 * 1024) -> str:
    """Hash a file without loading it into memory."""

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_reference(reference: os.PathLike[str] | str) -> dict[str, str]:
    """Return a deterministic fingerprint for a local checkpoint or model tag.

    Local directories are hashed as a canonical stream of relative path names,
    symlink targets, and file bytes.  A remote model identifier cannot be
    content-addressed without downloading it, so its literal identifier is
    hashed and the weaker fingerprint kind is reported explicitly.
    """

    value = str(reference)
    path = Path(value)
    if path.is_file():
        return {
            "reference": value,
            "fingerprint_kind": "file_content_sha256",
            "sha256": sha256_file(path),
        }
    if path.is_dir():
        digest = hashlib.sha256()
        for child in sorted(path.rglob("*"), key=lambda item: item.as_posix()):
            relative = child.relative_to(path).as_posix().encode("utf-8")
            if child.is_symlink():
                digest.update(b"L\0" + relative + b"\0")
                digest.update(os.readlink(child).encode("utf-8"))
            elif child.is_file():
                digest.update(b"F\0" + relative + b"\0")
                with child.open("rb") as handle:
                    while True:
                        chunk = handle.read(1024 * 1024)
                        if not chunk:
                            break
                        digest.update(chunk)
        return {
            "reference": value,
            "fingerprint_kind": "directory_content_sha256",
            "sha256": digest.hexdigest(),
        }

    return {
        "reference": value,
        "fingerprint_kind": "identifier_sha256",
        "sha256": hashlib.sha256(value.encode("utf-8")).hexdigest(),
    }


def canonical_json_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def resolve_peft_artifacts(
    adapter_reference: os.PathLike[str] | str | None,
    m3_heads_reference: os.PathLike[str] | str | None = None,
) -> tuple[str | None, str | None]:
    """Resolve either a direct PEFT adapter or a Phase-11 checkpoint root.

    Training checkpoints use ``checkpoint-*/adapter`` and, for unified mode,
    store ``m3_heads.pt`` beside that directory.  Accepting the checkpoint root
    prevents the easy-to-miss error of evaluating tuned backbone weights with
    stale base sparse/ColBERT heads.
    """

    if adapter_reference is None:
        if m3_heads_reference is None:
            return None, None
        heads_path = Path(m3_heads_reference)
        if not heads_path.is_file():
            raise FileNotFoundError(f"M3 heads artifact not found: {heads_path}")
        return None, str(heads_path)

    raw_adapter = Path(adapter_reference)
    resolved_adapter = raw_adapter
    if raw_adapter.is_dir() and not (raw_adapter / "adapter_config.json").is_file():
        nested_adapter = raw_adapter / "adapter"
        if nested_adapter.is_dir():
            resolved_adapter = nested_adapter

    if m3_heads_reference is not None:
        resolved_heads: Path | None = Path(m3_heads_reference)
        if not resolved_heads.is_file():
            raise FileNotFoundError(f"M3 heads artifact not found: {resolved_heads}")
    else:
        resolved_heads = None
        if raw_adapter.is_dir():
            candidates = [
                raw_adapter / "m3_heads.pt",
                resolved_adapter.parent / "m3_heads.pt",
            ]
            resolved_heads = next((path for path in candidates if path.is_file()), None)

    return str(resolved_adapter), str(resolved_heads) if resolved_heads else None


def load_m3_heads(wrapper: Any, heads_path: os.PathLike[str] | str) -> None:
    """Load the combined unified-training sparse and ColBERT head artifact."""

    try:
        import torch
    except ImportError as error:  # pragma: no cover - BGE environment concern
        raise RuntimeError("Loading M3 heads requires torch") from error
    path = Path(heads_path)
    if not path.is_file():
        raise FileNotFoundError(f"M3 heads artifact not found: {path}")
    heads = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(heads, Mapping):
        raise ValueError(f"Malformed M3 heads artifact: {path}")
    for name in ("colbert_linear", "sparse_linear"):
        state = heads.get(name)
        module = getattr(wrapper, name, None)
        if not isinstance(state, Mapping) or module is None:
            raise ValueError(f"M3 heads artifact lacks compatible {name}: {path}")
        module.load_state_dict(state, strict=True)
        module.eval()


def write_json_report(path: os.PathLike[str] | str, report: Mapping[str, Any]) -> dict[str, Any]:
    """Atomically write a report with a self-describing canonical JSON hash."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    materialized = dict(report)
    materialized.pop("report_sha256", None)
    materialized["report_sha256"] = canonical_json_sha256(materialized)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(materialized, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()

    sidecar = output.with_name(f"{output.name}.sha256")
    sidecar_tmp = sidecar.with_name(f".{sidecar.name}.tmp-{os.getpid()}")
    try:
        sidecar_tmp.write_text(f"{sha256_file(output)}  {output.name}\n", encoding="utf-8")
        os.replace(sidecar_tmp, sidecar)
    finally:
        if sidecar_tmp.exists():
            sidecar_tmp.unlink()
    return materialized


def iter_jsonl(path: os.PathLike[str] | str) -> Iterator[dict[str, Any]]:
    """Yield JSON objects one at a time with useful source-line errors."""

    with open(path, "r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {error}") from error
            if not isinstance(record, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            yield record


def stable_unique_strings(values: Iterable[Any], field_name: str) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value:
            raise ValueError(f"{field_name} must contain non-empty strings")
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def get_record_id(record: Mapping[str, Any]) -> str:
    value = record.get("query_id", record.get("id"))
    if value is None:
        raise ValueError("Rich record lacks query_id/id")
    return str(value)


def get_query(record: Mapping[str, Any]) -> str:
    value = record.get("query", record.get("question"))
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Record {get_record_id(record)} lacks a non-empty query")
    return value


def get_positive_relations(record: Mapping[str, Any]) -> list[str]:
    values = record.get("positive_relations", record.get("positives", []))
    if values is None:
        values = []
    if not isinstance(values, list):
        raise ValueError(
            f"Record {get_record_id(record)} positive_relations must be a list"
        )
    return stable_unique_strings(values, "positive_relations")


def _embedded_candidates(record: Mapping[str, Any]) -> tuple[list[str] | None, str | None]:
    value = record.get("candidate_relations")
    if value is None:
        return None, None
    source: str | None = None
    if isinstance(value, dict):
        source_value = value.get("source")
        source = str(source_value) if source_value is not None else None
        value = value.get("all", value.get("relations"))
    if value is None:
        return None, source
    if not isinstance(value, list):
        raise ValueError(
            f"Record {get_record_id(record)} candidate_relations must be a list or object"
        )
    return stable_unique_strings(value, "candidate_relations"), source


class ProcessedCandidateProvider:
    """Optional fallback index for legacy processed ``.pkl`` candidates.

    The canonical rich record embeds its candidate pool, so this index
    is loaded only when explicitly requested for backwards compatibility.  A
    pickle list cannot be streamed safely by Python's standard unpickler; the
    evaluator therefore reports use of this fallback and otherwise remains
    record-streaming.
    """

    def __init__(
        self,
        candidates_by_id: Mapping[str, Sequence[str]],
        *,
        source_path: os.PathLike[str] | str | None = None,
        source_sha256: str | None = None,
    ):
        self._candidates = {
            str(key): stable_unique_strings(value, "processed relation_list")
            for key, value in candidates_by_id.items()
        }
        self.source_path = str(source_path) if source_path is not None else None
        self.source_sha256 = source_sha256

    @classmethod
    def from_pickle(cls, path: os.PathLike[str] | str) -> "ProcessedCandidateProvider":
        with open(path, "rb") as handle:
            samples = pickle.load(handle)
        if not isinstance(samples, list):
            raise ValueError(f"Processed candidate file must contain a list: {path}")
        mapping: dict[str, Sequence[str]] = {}
        for sample in samples:
            if not isinstance(sample, dict) or "relation_list" not in sample:
                raise ValueError(f"Malformed processed sample in {path}")
            sample_id = sample.get("id", sample.get("query_id"))
            if sample_id is None:
                raise ValueError(f"Processed sample lacks id/query_id in {path}")
            mapping[str(sample_id)] = sample["relation_list"]
        return cls(mapping, source_path=path, source_sha256=sha256_file(path))

    def get(self, query_id: str) -> list[str] | None:
        values = self._candidates.get(str(query_id))
        return list(values) if values is not None else None

    def __len__(self) -> int:
        return len(self._candidates)

    def provenance(self) -> dict[str, Any]:
        return {
            "path": self.source_path,
            "sha256": self.source_sha256,
            "records": len(self),
        }


def get_candidate_relations(
    record: Mapping[str, Any],
    processed_candidates: ProcessedCandidateProvider | None = None,
) -> tuple[list[str], str]:
    candidates, source = _embedded_candidates(record)
    if candidates is not None:
        return candidates, source or "rich_record"
    if processed_candidates is not None:
        fallback = processed_candidates.get(get_record_id(record))
        if fallback is not None:
            return fallback, "processed_pickle_fallback"
    raise ValueError(f"Record {get_record_id(record)} has no candidate relation pool")


def normalize_score_modes(score_modes: Iterable[str]) -> tuple[str, ...]:
    if isinstance(score_modes, str):
        score_modes = score_modes.split(",")
    result: list[str] = []
    for raw_mode in score_modes:
        mode = raw_mode.strip().lower().replace("+", "_")
        aliases = {
            "sparse_dense": "dense_sparse",
            "dense_and_sparse": "dense_sparse",
            "all": "ensemble",
            "dense_sparse_colbert": "ensemble",
            "colbert_sparse_dense": "ensemble",
        }
        mode = aliases.get(mode, mode)
        if mode not in SUPPORTED_SCORE_MODES:
            choices = ", ".join(sorted(SUPPORTED_SCORE_MODES))
            raise ValueError(f"Unknown score mode {raw_mode!r}; choose from {choices}")
        if mode not in result:
            result.append(mode)
    if not result:
        raise ValueError("At least one score mode is required")
    return tuple(result)


def normalize_k_values(k_values: Iterable[int]) -> tuple[int, ...]:
    if isinstance(k_values, str):
        k_values = (value.strip() for value in k_values.split(",") if value.strip())
    values = sorted(set(int(value) for value in k_values))
    if not values or values[0] < 1:
        raise ValueError("K values must be positive integers")
    return tuple(values)


def stable_rank(scores: Sequence[float]) -> list[int]:
    converted: list[float] = []
    for value in scores:
        score = float(value)
        if not math.isfinite(score):
            raise ValueError("Scorer returned a NaN or infinite score")
        converted.append(score)
    return sorted(range(len(converted)), key=lambda index: (-converted[index], index))


@dataclass
class MetricAccumulator:
    k_values: tuple[int, ...]
    count: int = 0
    reciprocal_rank_sum: float = 0.0
    recall_sums: dict[int, float] = field(default_factory=dict)
    ndcg_sums: dict[int, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.recall_sums = {k: 0.0 for k in self.k_values}
        self.ndcg_sums = {k: 0.0 for k in self.k_values}

    def add(self, ranked_relations: Sequence[str], positives: set[str]) -> None:
        if not positives:
            raise ValueError("MetricAccumulator requires at least one positive")
        self.count += 1
        first_relevant_rank = next(
            (rank for rank, relation in enumerate(ranked_relations, start=1) if relation in positives),
            None,
        )
        if first_relevant_rank is not None:
            self.reciprocal_rank_sum += 1.0 / first_relevant_rank

        for k in self.k_values:
            top_k = ranked_relations[:k]
            relevant_count = sum(relation in positives for relation in top_k)
            self.recall_sums[k] += relevant_count / len(positives)
            dcg = sum(
                1.0 / math.log2(rank + 1)
                for rank, relation in enumerate(top_k, start=1)
                if relation in positives
            )
            ideal_count = min(k, len(positives))
            ideal_dcg = sum(
                1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1)
            )
            self.ndcg_sums[k] += dcg / ideal_dcg if ideal_dcg else 0.0

    def result(self) -> dict[str, Any]:
        if not self.count:
            return {
                "evaluable_queries": 0,
                "recall_at": {str(k): None for k in self.k_values},
                "macro_mrr": None,
                "ndcg_at": {str(k): None for k in self.k_values},
            }
        return {
            "evaluable_queries": self.count,
            "recall_at": {
                str(k): self.recall_sums[k] / self.count for k in self.k_values
            },
            "macro_mrr": self.reciprocal_rank_sum / self.count,
            "ndcg_at": {
                str(k): self.ndcg_sums[k] / self.count for k in self.k_values
            },
        }


def _record_partition(record: Mapping[str, Any]) -> str:
    value = record.get("training_partition", record.get("split", "unspecified"))
    if isinstance(value, dict):
        value = value.get("name", value.get("partition", "unspecified"))
    return str(value)


def _partition_matches(actual: str, requested: str) -> bool:
    if requested == "all":
        return True
    if requested == "train":
        return actual in {"train", "fit"}
    return actual == requested


def evaluate_records(
    records: Iterable[Mapping[str, Any]],
    scorer: RelationScorer,
    *,
    score_modes: Iterable[str] = ("dense",),
    k_values: Iterable[int] = DEFAULT_K_VALUES,
    relation_format: str | None = None,
    processed_candidates: ProcessedCandidateProvider | None = None,
    partition: str = "all",
    limit: int | None = None,
) -> dict[str, Any]:
    """Evaluate records without retaining predictions or records in memory."""

    modes = normalize_score_modes(score_modes)
    ks = normalize_k_values(k_values)
    unsupported = set(modes) - set(scorer.supported_modes)
    if unsupported:
        raise ValueError(f"Scorer does not support modes: {sorted(unsupported)}")
    if relation_format is not None and relation_format not in FORMATTERS:
        raise ValueError(f"Unknown relation format: {relation_format}")
    if not isinstance(partition, str) or not partition.strip():
        raise ValueError("partition must be all, train/fit, dev, or an exact partition name")
    partition = partition.strip()
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")

    accumulators = {mode: MetricAccumulator(ks) for mode in modes}
    counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    format_counts: Counter[str] = Counter()
    partition_counts: Counter[str] = Counter()

    for record in records:
        if limit is not None and counts["records_selected"] >= limit:
            break
        counts["records_seen"] += 1
        actual_partition = _record_partition(record)
        partition_counts[actual_partition] += 1
        if not _partition_matches(actual_partition, partition):
            counts["skipped_by_partition"] += 1
            continue
        counts["records_selected"] += 1
        query_id = get_record_id(record)
        query = get_query(record)
        positives_list = get_positive_relations(record)
        positives = set(positives_list)
        candidates, candidate_source = get_candidate_relations(record, processed_candidates)
        source_counts[candidate_source] += 1

        selected_format = relation_format or str(record.get("relation_format", "split"))
        if selected_format not in FORMATTERS:
            raise ValueError(
                f"Record {query_id} uses unknown relation_format {selected_format!r}"
            )
        format_counts[selected_format] += 1

        if not positives:
            counts["skipped_without_positive"] += 1
            continue
        counts["evaluable_queries"] += 1
        counts["positive_assignments"] += len(positives)
        missing = positives - set(candidates)
        counts["positive_relations_outside_candidate_pool"] += len(missing)
        if missing:
            counts["queries_with_missing_positive"] += 1
        if not candidates:
            counts["empty_candidate_queries"] += 1
            for accumulator in accumulators.values():
                accumulator.add([], positives)
            continue

        passages = [format_relation(relation, selected_format) for relation in candidates]
        # The no-leakage boundary is intentional: no record metadata is passed.
        scores_by_mode = scorer.score(query, passages, modes)
        counts["candidate_relations_scored"] += len(candidates)
        for mode in modes:
            if mode not in scores_by_mode:
                raise ValueError(f"Scorer omitted requested mode {mode!r} for {query_id}")
            scores = scores_by_mode[mode]
            if len(scores) != len(candidates):
                raise ValueError(
                    f"Scorer returned {len(scores)} {mode} scores for "
                    f"{len(candidates)} candidates in {query_id}"
                )
            ranked = [candidates[index] for index in stable_rank(scores)]
            accumulators[mode].add(ranked, positives)

    return {
        "schema_version": "phase11-tier1-v1",
        "protocol": {
            "score_modes": list(modes),
            "k_values": list(ks),
            "relation_format_override": relation_format,
            "partition_filter": partition,
            "encoder_inputs": ["query", "formatted_candidate_relation"],
            "answer_entities_passed_to_encoder": False,
            "tie_break": "candidate_input_order",
        },
        "counts": dict(sorted(counts.items())),
        "candidate_sources": dict(sorted(source_counts.items())),
        "relation_formats": dict(sorted(format_counts.items())),
        "training_partitions": dict(sorted(partition_counts.items())),
        "metrics": {mode: accumulators[mode].result() for mode in modes},
    }


def evaluate_jsonl(
    rich_jsonl: os.PathLike[str] | str,
    scorer: RelationScorer,
    *,
    score_modes: Iterable[str],
    k_values: Iterable[int] = DEFAULT_K_VALUES,
    relation_format: str | None = None,
    processed_candidates: ProcessedCandidateProvider | None = None,
    partition: str = "all",
    limit: int | None = None,
    model_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    report = evaluate_records(
        iter_jsonl(rich_jsonl),
        scorer,
        score_modes=score_modes,
        k_values=k_values,
        relation_format=relation_format,
        processed_candidates=processed_candidates,
        partition=partition,
        limit=limit,
    )
    report["input"] = {
        "rich_jsonl": str(rich_jsonl),
        "rich_jsonl_sha256": sha256_file(rich_jsonl),
        "processed_candidate_records": (
            len(processed_candidates) if processed_candidates is not None else 0
        ),
        "processed_candidates": (
            processed_candidates.provenance()
            if processed_candidates is not None
            else None
        ),
    }
    report["model"] = dict(model_provenance or {})
    return report


class BGEM3RelationScorer:
    """Efficient per-query BGE-M3 scorer with optional in-memory PEFT merge."""

    supported_modes = SUPPORTED_SCORE_MODES

    def __init__(
        self,
        model_name_or_path: str,
        *,
        adapter_path: str | None = None,
        m3_heads_path: str | None = None,
        device: str | None = None,
        use_fp16: bool = True,
        batch_size: int = 32,
        query_max_length: int = 128,
        passage_max_length: int = 64,
        passage_cache_size: int = 8192,
        passage_cache_max_bytes: int = 512 * 1024 * 1024,
        mode_weights: Sequence[float] = (1.0, 1.0, 1.0),
        trust_remote_code: bool = False,
    ):
        if batch_size < 1 or query_max_length < 1 or passage_max_length < 1:
            raise ValueError("batch size and sequence lengths must be positive")
        if passage_cache_size < 0:
            raise ValueError("passage_cache_size cannot be negative")
        if passage_cache_max_bytes < 0:
            raise ValueError("passage_cache_max_bytes cannot be negative")
        if len(mode_weights) != 3:
            raise ValueError("mode_weights must be [dense, sparse, colbert]")
        self.mode_weights = tuple(float(value) for value in mode_weights)
        if any(value < 0 or not math.isfinite(value) for value in self.mode_weights):
            raise ValueError("mode weights must be finite and non-negative")
        if not any(self.mode_weights):
            raise ValueError("at least one mode weight must be positive")

        try:
            import torch
            from FlagEmbedding import BGEM3FlagModel
        except ImportError as error:  # pragma: no cover - exercised only in BGE env
            raise RuntimeError(
                "BGEM3RelationScorer requires FlagEmbedding and torch; run it in "
                "the dedicated bge-m3-analysis environment"
            ) from error

        if device is None or device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested but torch cannot use CUDA: {device}")
        self.device = device
        self.batch_size = batch_size
        self.query_max_length = query_max_length
        self.passage_max_length = passage_max_length
        self.passage_cache_size = passage_cache_size
        self.passage_cache_max_bytes = passage_cache_max_bytes
        self._passage_cache_bytes = 0
        self._passage_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self.model = BGEM3FlagModel(
            model_name_or_path,
            use_fp16=bool(use_fp16 and device.startswith("cuda")),
            use_bf16=False,
            devices=device,
            batch_size=batch_size,
            query_max_length=query_max_length,
            passage_max_length=passage_max_length,
            trust_remote_code=trust_remote_code,
        )

        if adapter_path:
            try:
                from peft import PeftModel
            except ImportError as error:  # pragma: no cover - BGE env concern
                raise RuntimeError("Loading a PEFT adapter requires the peft package") from error
            wrapper = getattr(self.model, "model", None)
            backbone = getattr(wrapper, "model", None)
            if backbone is None:
                raise RuntimeError(
                    "Unsupported FlagEmbedding internals: cannot locate BGE backbone for PEFT"
                )
            peft_model = PeftModel.from_pretrained(backbone, adapter_path, is_trainable=False)
            if not hasattr(peft_model, "merge_and_unload"):
                raise RuntimeError("This PEFT adapter cannot be merged in memory")
            wrapper.model = peft_model.merge_and_unload()
            wrapper.model.eval()

        if m3_heads_path:
            wrapper = getattr(self.model, "model", None)
            if wrapper is None:
                raise RuntimeError(
                    "Unsupported FlagEmbedding internals: cannot locate sparse/ColBERT heads"
                )
            load_m3_heads(wrapper, m3_heads_path)

    @staticmethod
    def _representation_nbytes(value: Any) -> int:
        """Conservatively estimate cached representation payload bytes.

        ColBERT multi-vectors are much larger than dense vectors.  Bounding
        only the number of cached relations can therefore consume several GiB
        on a full Freebase vocabulary.  Array/tensor payload sizes are exact;
        sparse dictionaries and test doubles use a small recursive estimate.
        """

        nbytes = getattr(value, "nbytes", None)
        if nbytes is not None:
            return int(nbytes() if callable(nbytes) else nbytes)
        if isinstance(value, Mapping):
            return sum(
                BGEM3RelationScorer._representation_nbytes(key)
                + BGEM3RelationScorer._representation_nbytes(item)
                for key, item in value.items()
            )
        if isinstance(value, (list, tuple)):
            return sum(BGEM3RelationScorer._representation_nbytes(item) for item in value)
        if isinstance(value, str):
            return len(value.encode("utf-8"))
        if isinstance(value, (int, float, bool)):
            return 8
        return 0

    def _cache_representation(self, passage: str, representation: dict[str, Any]) -> None:
        if not self.passage_cache_size:
            return
        previous = self._passage_cache.pop(passage, None)
        if hasattr(self, "_passage_cache_bytes"):
            current_bytes = self._passage_cache_bytes
        else:
            current_bytes = sum(
                self._representation_nbytes(item)
                for item in self._passage_cache.values()
            )
        if previous is not None:
            current_bytes -= self._representation_nbytes(previous)
        self._passage_cache[passage] = representation
        current_bytes += self._representation_nbytes(representation)
        max_bytes = getattr(self, "passage_cache_max_bytes", 0)
        while self._passage_cache and (
            len(self._passage_cache) > self.passage_cache_size
            or (max_bytes and current_bytes > max_bytes)
        ):
            _, evicted = self._passage_cache.popitem(last=False)
            current_bytes -= self._representation_nbytes(evicted)
        self._passage_cache_bytes = max(0, current_bytes)

    def _passage_representations(
        self,
        passages: Sequence[str],
        *,
        need_dense: bool,
        need_sparse: bool,
        need_colbert: bool,
    ) -> list[dict[str, Any]]:
        required = {
            name
            for name, enabled in (
                ("dense", need_dense),
                ("sparse", need_sparse),
                ("colbert", need_colbert),
            )
            if enabled
        }
        fresh: dict[str, dict[str, Any]] = {}
        missing: list[str] = []
        seen_missing: set[str] = set()
        for passage in passages:
            cached = self._passage_cache.get(passage)
            if cached is not None and required <= set(cached):
                self._passage_cache.move_to_end(passage)
                fresh[passage] = cached
            elif passage not in seen_missing:
                missing.append(passage)
                seen_missing.add(passage)

        if missing:
            output = self.model.encode_corpus(
                missing,
                batch_size=self.batch_size,
                max_length=self.passage_max_length,
                return_dense=need_dense,
                return_sparse=need_sparse,
                return_colbert_vecs=need_colbert,
            )
            for index, passage in enumerate(missing):
                representation = dict(self._passage_cache.get(passage, {}))
                if need_dense:
                    representation["dense"] = output["dense_vecs"][index]
                if need_sparse:
                    representation["sparse"] = output["lexical_weights"][index]
                if need_colbert:
                    representation["colbert"] = output["colbert_vecs"][index]
                fresh[passage] = representation
                self._cache_representation(passage, representation)

        result: list[dict[str, Any]] = []
        for passage in passages:
            representation = self._passage_cache.get(passage, fresh.get(passage))
            if representation is None or not required <= set(representation):
                raise RuntimeError(f"Missing encoded representation for passage: {passage!r}")
            result.append(representation)
        return result

    def score(
        self,
        query: str,
        passages: Sequence[str],
        score_modes: Sequence[str],
    ) -> Mapping[str, Sequence[float]]:
        modes = normalize_score_modes(score_modes)
        if not passages:
            return {mode: [] for mode in modes}
        need_dense = any(mode in {"dense", "dense_sparse", "ensemble"} for mode in modes)
        need_sparse = any(mode in {"sparse", "dense_sparse", "ensemble"} for mode in modes)
        need_colbert = any(mode in {"colbert", "ensemble"} for mode in modes)

        query_output = self.model.encode_queries(
            query,
            batch_size=1,
            max_length=self.query_max_length,
            return_dense=need_dense,
            return_sparse=need_sparse,
            return_colbert_vecs=need_colbert,
        )
        passage_representations = self._passage_representations(
            passages,
            need_dense=need_dense,
            need_sparse=need_sparse,
            need_colbert=need_colbert,
        )

        components: dict[str, list[float]] = {}
        if need_dense:
            try:
                import numpy as np
            except ImportError as error:  # pragma: no cover
                raise RuntimeError("Dense BGE scoring requires numpy") from error
            query_dense = np.asarray(query_output["dense_vecs"])
            passage_dense = np.asarray(
                [representation["dense"] for representation in passage_representations]
            )
            components["dense"] = [
                float(value) for value in np.matmul(passage_dense, query_dense)
            ]
        if need_sparse:
            query_sparse = query_output["lexical_weights"]
            components["sparse"] = [
                float(self.model.compute_lexical_matching_score(query_sparse, passage_sparse))
                for passage_sparse in (
                    representation["sparse"]
                    for representation in passage_representations
                )
            ]
        if need_colbert:
            query_colbert = query_output["colbert_vecs"]
            components["colbert"] = [
                float(self.model.colbert_score(query_colbert, passage_colbert))
                for passage_colbert in (
                    representation["colbert"]
                    for representation in passage_representations
                )
            ]

        dense_weight, sparse_weight, colbert_weight = self.mode_weights
        result: dict[str, Sequence[float]] = {}
        for mode in modes:
            if mode in components:
                result[mode] = components[mode]
            elif mode == "dense_sparse":
                denominator = dense_weight + sparse_weight
                if denominator <= 0:
                    raise ValueError("dense_sparse requires a positive dense or sparse weight")
                result[mode] = [
                    (dense_weight * dense + sparse_weight * sparse) / denominator
                    for dense, sparse in zip(components["dense"], components["sparse"])
                ]
            elif mode == "ensemble":
                denominator = dense_weight + sparse_weight + colbert_weight
                result[mode] = [
                    (
                        dense_weight * dense
                        + sparse_weight * sparse
                        + colbert_weight * colbert
                    )
                    / denominator
                    for dense, sparse, colbert in zip(
                        components["dense"], components["sparse"], components["colbert"]
                    )
                ]
            else:  # pragma: no cover - normalized modes make this unreachable
                raise RuntimeError(f"Unhandled score mode: {mode}")
        return result


def build_model_provenance(
    model_reference: str,
    *,
    model_tag: str,
    adapter_reference: str | None = None,
    m3_heads_reference: str | None = None,
    checkpoint_sha256: str | None = None,
) -> dict[str, Any]:
    if checkpoint_sha256 is not None:
        checkpoint_sha256 = checkpoint_sha256.strip().lower()
        if len(checkpoint_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in checkpoint_sha256
        ):
            raise ValueError("checkpoint_sha256 must contain 64 hexadecimal characters")
    base = fingerprint_reference(model_reference)
    adapter = fingerprint_reference(adapter_reference) if adapter_reference else None
    m3_heads = fingerprint_reference(m3_heads_reference) if m3_heads_reference else None
    composite = {
        "base_sha256": base["sha256"],
        "adapter_sha256": adapter["sha256"] if adapter else None,
        "m3_heads_sha256": m3_heads["sha256"] if m3_heads else None,
    }
    active_sha = checkpoint_sha256 or canonical_json_sha256(composite)
    return {
        "model_tag": model_tag,
        "base_model": base,
        "adapter": adapter,
        "adapter_merged_in_memory": bool(adapter_reference),
        "m3_heads": m3_heads,
        "m3_heads_loaded": bool(m3_heads_reference),
        "active_checkpoint_sha256": active_sha,
        "checkpoint_sha256_source": "explicit" if checkpoint_sha256 else "composite",
    }
