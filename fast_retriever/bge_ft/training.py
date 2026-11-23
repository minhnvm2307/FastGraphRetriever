"""BGE-M3 relation-encoder fine-tuning (LoRA).

The trainable model owned by Hugging Face ``Trainer`` is the PEFT model itself,
so optimizer, gradient accumulation, checkpoint rotation, and resume all use
the stock Trainer lifecycle.  FlagEmbedding's official M3 wrapper remains the
encoder/scoring implementation; only the multi-positive loss is custom.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import random
import shutil
import subprocess
import tempfile
import time
import unicodedata
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from torch import Tensor
from torch.utils.data import Dataset
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments


GIB = 1024 ** 3
CONFIG_HASH_EXCLUDED_FIELDS = frozenset({"resume_from_checkpoint"})


def canonical_text(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"expected text, got {type(value).__name__}")
    return " ".join(unicodedata.normalize("NFKC", value).split()).casefold()


def canonical_identifier(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"expected identifier, got {type(value).__name__}")
    return unicodedata.normalize("NFKC", value).strip().casefold()


def _string_list(record: Mapping[str, Any], key: str) -> list[str]:
    value = record.get(key, [])
    if value is None:
        return []
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{key} must be a list of strings")
    result = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{key}[{index}] must be non-empty text")
        result.append(item)
    return result


def normalize_flat_record(record: Mapping[str, Any], *, source: str = "record") -> dict[str, Any]:
    query = record.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ValueError(f"{source}: query must be non-empty text")
    pos = _string_list(record, "pos")
    if not pos:
        raise ValueError(f"{source}: pos must contain at least one relation")
    neg = _string_list(record, "neg")
    pos_ids = _string_list(record, "pos_ids")
    neg_ids = _string_list(record, "neg_ids")
    if pos_ids and len(pos_ids) != len(pos):
        raise ValueError(f"{source}: pos and pos_ids must have equal length")
    if neg_ids and len(neg_ids) != len(neg):
        raise ValueError(f"{source}: neg and neg_ids must have equal length")
    levels = record.get("negative_levels", []) or []
    if isinstance(levels, (str, bytes)) or not isinstance(levels, Sequence):
        raise ValueError(f"{source}: negative_levels must be a list")
    if levels and len(levels) != len(neg):
        raise ValueError(f"{source}: neg and negative_levels must have equal length")

    known = _string_list(record, "known_positive_texts")
    seen = {canonical_text(text) for text in known}
    for text in pos:
        if canonical_text(text) not in seen:
            known.append(text)
            seen.add(canonical_text(text))
    known_ids = _string_list(record, "known_positive_ids")
    id_seen = {canonical_identifier(item) for item in known_ids}
    for item in pos_ids:
        if canonical_identifier(item) not in id_seen:
            known_ids.append(item)
            id_seen.add(canonical_identifier(item))

    ignored_texts = [
        *_string_list(record, "ignored_passage_texts"),
        *_string_list(record, "ignored_negative_texts"),
    ]
    ignored_ids = [
        *_string_list(record, "ignored_passage_ids"),
        *_string_list(record, "ignored_negative_ids"),
    ]
    return {
        **dict(record),
        "query_id": str(record.get("query_id", record.get("id", ""))),
        "query": query,
        "pos": pos,
        "neg": neg,
        "pos_ids": pos_ids,
        "neg_ids": neg_ids,
        "known_positive_texts": known,
        "known_positive_ids": known_ids,
        "ignored_passage_texts": ignored_texts,
        "ignored_passage_ids": ignored_ids,
        "negative_levels": list(levels),
    }


class RelationJsonlDataset(Dataset):
    """Small indexed JSONL dataset; only byte offsets are retained in RAM."""

    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(path)
        self.offsets: list[int] = []
        with self.path.open("rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if line.strip():
                    self.offsets.append(offset)
        if not self.offsets:
            raise ValueError(f"training JSONL is empty: {self.path}")

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, index: int) -> dict[str, Any]:
        with self.path.open("rb") as handle:
            handle.seek(self.offsets[index])
            raw = handle.readline()
        try:
            record = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON at {self.path}, record {index}") from error
        return normalize_flat_record(record, source=f"{self.path}:{index + 1}")

    def validate_all(self) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "records": len(self),
            "positive_mentions": 0,
            "negative_mentions": 0,
            "ignored_passage_mentions": 0,
            "relation_formats": set(),
        }
        for index in range(len(self)):
            row = self[index]
            summary["positive_mentions"] += len(row["pos"])
            summary["negative_mentions"] += len(row["neg"])
            summary["ignored_passage_mentions"] += len(row["ignored_passage_texts"])
            if row.get("relation_format"):
                summary["relation_formats"].add(str(row["relation_format"]))
        summary["relation_formats"] = sorted(summary["relation_formats"])
        return summary


class CollisionSafeCollator:
    """Tokenize one query and its explicit, deduplicated relation pool."""

    def __init__(
        self,
        tokenizer: Any,
        *,
        max_query_length: int,
        max_passage_length: int,
        max_negatives_per_query: int,
    ):
        self.tokenizer = tokenizer
        self.max_query_length = max_query_length
        self.max_passage_length = max_passage_length
        self.max_negatives_per_query = max_negatives_per_query

    def __call__(self, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        rows = [normalize_flat_record(record) for record in records]
        if len(rows) != 1:
            raise ValueError("the relation encoder requires micro_batch_size=1; accumulation does not merge passage pools")
        row = rows[0]
        passages: list[str] = []
        passage_keys: list[str] = []
        passage_ids: list[set[str]] = []
        index_by_text: dict[str, int] = {}

        def add(text: str, relation_id: str | None = None) -> None:
            key = canonical_text(text)
            index = index_by_text.get(key)
            if index is None:
                index = len(passages)
                index_by_text[key] = index
                passages.append(text)
                passage_keys.append(key)
                passage_ids.append(set())
            if relation_id:
                passage_ids[index].add(canonical_identifier(relation_id))

        for text, relation_id in zip(row["pos"], row["pos_ids"] or [None] * len(row["pos"])):
            add(text, relation_id)
        for text in row["known_positive_texts"]:
            add(text)
        for text in row["ignored_passage_texts"]:
            add(text)
        negatives = row["neg"]
        negative_ids = row["neg_ids"] or [None] * len(negatives)
        if self.max_negatives_per_query >= 0:
            negatives = negatives[: self.max_negatives_per_query]
            negative_ids = negative_ids[: self.max_negatives_per_query]
        for text, relation_id in zip(negatives, negative_ids):
            add(text, relation_id)

        positive_texts = {canonical_text(text) for text in row["known_positive_texts"]}
        positive_ids = {canonical_identifier(item) for item in row["known_positive_ids"]}
        ignored_texts = {canonical_text(text) for text in row["ignored_passage_texts"]}
        ignored_ids = {canonical_identifier(item) for item in row["ignored_passage_ids"]}
        positive_mask = torch.tensor(
            [[key in positive_texts or bool(ids & positive_ids) for key, ids in zip(passage_keys, passage_ids)]],
            dtype=torch.bool,
        )
        if not positive_mask.any():
            raise ValueError("query has no positive in its passage pool")
        candidate_mask = torch.tensor(
            [[not (key in ignored_texts or bool(ids & ignored_ids)) for key, ids in zip(passage_keys, passage_ids)]],
            dtype=torch.bool,
        )
        candidate_mask |= positive_mask
        query_tokens = dict(self.tokenizer(
            [row["query"]], padding=True, truncation=True,
            max_length=self.max_query_length, return_tensors="pt",
        ))
        passage_tokens = dict(self.tokenizer(
            passages, padding=True, truncation=True,
            max_length=self.max_passage_length, return_tensors="pt",
        ))
        return {
            "query_tokens": query_tokens,
            "passage_tokens": passage_tokens,
            "positive_mask": positive_mask,
            "candidate_mask": candidate_mask,
        }


def multi_positive_info_nce(
    scores: Tensor,
    positive_mask: Tensor,
    *,
    temperature: float = 0.1,
    candidate_mask: Tensor | None = None,
    reduction: str = "mean",
) -> Tensor:
    if scores.ndim != 2 or positive_mask.shape != scores.shape:
        raise ValueError("scores and positive_mask must have shape [queries, passages]")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    positive_mask = positive_mask.to(scores.device, dtype=torch.bool)
    if candidate_mask is None:
        candidate_mask = torch.ones_like(positive_mask)
    candidate_mask = candidate_mask.to(scores.device, dtype=torch.bool)
    if candidate_mask.shape != scores.shape:
        raise ValueError("candidate_mask shape mismatch")
    if (positive_mask & ~candidate_mask).any() or not positive_mask.any(dim=1).all():
        raise ValueError("each positive must be a valid candidate")
    scaled = scores.float() / temperature
    floor = torch.finfo(scaled.dtype).min
    denominator = torch.logsumexp(scaled.masked_fill(~candidate_mask, floor), dim=1)
    numerator = torch.logsumexp(scaled.masked_fill(~positive_mask, floor), dim=1)
    loss = denominator - numerator
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    if reduction != "mean":
        raise ValueError("reduction must be none, sum, or mean")
    return loss.mean()


@dataclass(frozen=True)
class TrainingConfig:
    train_data: str = ""
    dev_rich_data: str | None = None
    output_dir: str = ""
    model_name_or_path: str = "BAAI/bge-m3"
    cache_dir: str | None = None
    local_files_only: bool = True
    mode: str = "dense"
    seed: int = 42
    epochs: int = 3
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    temperature: float = 0.1
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    max_query_length: int = 96
    max_passage_length: int = 48
    max_negatives_per_query: int = 16
    num_workers: int = 0
    num_threads: int = 4
    fp16: bool = True
    bf16: bool = False
    gradient_checkpointing: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = ("query", "key", "value")
    max_grad_norm: float = 1.0
    save_steps: int = 100
    save_each_epoch: bool = True
    save_total_limit: int = 1
    log_steps: int = 10
    max_steps: int = -1
    early_stopping_patience: int = 0
    early_stopping_k: int = 10
    eval_passage_batch_size: int = 4
    resume_from_checkpoint: str | None = None
    require_cuda: bool = True
    min_free_gpu_gb: float = 8.0
    min_free_disk_gb: float = 6.0
    unified_sparse_score_weight: float = 0.3
    unified_colbert_score_weight: float = 1.0
    unified_sparse_loss_weight: float = 0.1

    def validate(self) -> None:
        if not self.train_data or not self.output_dir:
            raise ValueError("train_data and output_dir are required")
        if self.mode not in {"dense", "unified"}:
            raise ValueError("mode must be dense or unified")
        if self.micro_batch_size != 1:
            raise ValueError("the relation encoder requires micro_batch_size=1")
        if self.bf16 or not self.fp16:
            raise ValueError("this recipe trains in fp16; bf16 is not supported")
        if self.save_total_limit != 1:
            raise ValueError("save_total_limit must be 1")
        if self.eval_passage_batch_size > 4:
            raise ValueError("eval_passage_batch_size must be <= 4")
        if self.dev_rich_data is None and self.early_stopping_patience:
            raise ValueError("early stopping requires dev_rich_data")
        if self.dev_rich_data is not None and self.early_stopping_patience < 1:
            raise ValueError("dev_rich_data requires early_stopping_patience >= 1")
        if self.gradient_accumulation_steps < 1 or self.epochs < 1:
            raise ValueError("epochs and gradient_accumulation_steps must be positive")
        if self.max_steps == 0 or self.max_steps < -1:
            raise ValueError("max_steps must be -1 or positive")
        if self.temperature <= 0 or self.learning_rate <= 0:
            raise ValueError("temperature and learning_rate must be positive")
        if not {"query", "key", "value"}.issubset(self.target_modules):
            raise ValueError("target_modules must contain query, key, and value")


def resolved_config_dict(config: TrainingConfig, *, for_hash: bool = False) -> dict[str, Any]:
    value = asdict(config)
    value["target_modules"] = list(config.target_modules)
    if for_hash:
        for key in CONFIG_HASH_EXCLUDED_FIELDS:
            value.pop(key, None)
    return value


def config_sha256(config: TrainingConfig) -> str:
    config.validate()
    payload = json.dumps(
        resolved_config_dict(config, for_hash=True), sort_keys=True,
        separators=(",", ":"), ensure_ascii=False, allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def resolve_training_config(
    config_file: str | os.PathLike[str] | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> TrainingConfig:
    values = asdict(TrainingConfig())
    valid = {field.name for field in fields(TrainingConfig)}
    if config_file:
        loaded = json.loads(Path(config_file).read_text(encoding="utf-8"))
        unknown = set(loaded) - valid
        if unknown:
            raise ValueError(f"unknown config fields: {sorted(unknown)}")
        values.update(loaded)
    if overrides:
        unknown = set(overrides) - valid
        if unknown:
            raise ValueError(f"unknown config overrides: {sorted(unknown)}")
        values.update({key: item for key, item in overrides.items() if item is not None})
    target_modules = values["target_modules"]
    if isinstance(target_modules, str):
        target_modules = [item.strip() for item in target_modules.split(",") if item.strip()]
    values["target_modules"] = tuple(target_modules)
    config = TrainingConfig(**values)
    config.validate()
    return config


def file_sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _driver_version() -> str | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5, check=False,
        )
    except OSError:
        return None
    return result.stdout.strip() or None


def preflight_checks(config: TrainingConfig) -> tuple[torch.device, dict[str, Any]]:
    config.validate()
    for label, value in (("train", config.train_data), ("dev", config.dev_rich_data)):
        if value and (not Path(value).is_file() or Path(value).stat().st_size == 0):
            raise FileNotFoundError(f"{label} data missing or empty: {value}")
    output = Path(config.output_dir)
    if output.exists() and any(output.iterdir()) and config.resume_from_checkpoint is None:
        raise FileExistsError(f"non-empty output_dir without resume: {output}")
    anchor = output.parent
    while not anchor.exists():
        anchor = anchor.parent
    free_disk = shutil.disk_usage(anchor).free / GIB
    if free_disk < config.min_free_disk_gb:
        raise RuntimeError(f"disk guard failed: {free_disk:.2f} GiB free")
    available = torch.cuda.is_available()
    if config.require_cuda and not available:
        raise RuntimeError("CUDA guard failed in bge-m3-analysis")
    device = torch.device("cuda:0" if available else "cpu")
    report: dict[str, Any] = {
        "device": str(device), "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda, "free_disk_gb": round(free_disk, 3),
    }
    if available:
        free, _ = torch.cuda.mem_get_info(0)
        report.update({
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_free_gb": round(free / GIB, 3),
            "gpu_capability": list(torch.cuda.get_device_capability(0)),
            "nvidia_driver": _driver_version(),
        })
        if free / GIB < config.min_free_gpu_gb:
            raise RuntimeError(f"VRAM guard failed: {free / GIB:.2f} GiB free")
    return device, report


def set_deterministic_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_snapshot(config: TrainingConfig) -> str:
    path = Path(config.model_name_or_path)
    if path.exists():
        return str(path.resolve())
    from huggingface_hub import snapshot_download
    return snapshot_download(
        repo_id=config.model_name_or_path, cache_dir=config.cache_dir,
        local_files_only=config.local_files_only,
        ignore_patterns=["flax_model.msgpack", "rust_model.ot", "tf_model.h5"],
    )


def build_trainable_model(config: TrainingConfig) -> tuple[Any, Any, Any, dict[str, Any]]:
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoTokenizer
    from FlagEmbedding.finetune.embedder.encoder_only.m3.modeling import EncoderOnlyEmbedderM3Model
    from FlagEmbedding.finetune.embedder.encoder_only.m3.runner import EncoderOnlyEmbedderM3Runner

    if config.mode != "dense":
        raise NotImplementedError("unified v3 is deferred; smoke/v1 use dense mode")
    snapshot = _resolve_snapshot(config)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, cache_dir=config.cache_dir, use_fast=True)
    components = EncoderOnlyEmbedderM3Runner.get_model(
        snapshot, trust_remote_code=False, cache_dir=config.cache_dir, torch_dtype=torch.float16,
    )
    lora = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION, r=config.lora_r,
        lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout,
        target_modules=list(config.target_modules), bias="none",
    )
    peft_model = get_peft_model(components["model"], lora)
    if hasattr(peft_model.config, "use_cache"):
        peft_model.config.use_cache = False
    m3 = EncoderOnlyEmbedderM3Model(
        {**components, "model": peft_model}, tokenizer=tokenizer,
        negatives_cross_device=False, temperature=1.0, sub_batch_size=-1,
        sentence_pooling_method="cls", normalize_embeddings=True,
        unified_finetuning=False, use_self_distill=False,
    )
    trainable = sum(parameter.numel() for parameter in peft_model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in peft_model.parameters())
    if not trainable:
        raise RuntimeError("LoRA injection produced zero trainable parameters")
    return peft_model, m3, tokenizer, {
        "resolved_model_snapshot": snapshot,
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_fraction": trainable / total,
    }


class CurrentM3RelationScorer:
    supported_modes = frozenset({"dense"})

    def __init__(self, m3: Any, tokenizer: Any, config: TrainingConfig, device: torch.device):
        self.m3, self.tokenizer, self.config, self.device = m3, tokenizer, config, device

    def _tokens(self, texts: Sequence[str], maximum: int) -> dict[str, Tensor]:
        value = self.tokenizer(list(texts), padding=True, truncation=True, max_length=maximum, return_tensors="pt")
        return {key: tensor.to(self.device) for key, tensor in dict(value).items()}

    def score(self, query: str, passages: Sequence[str], score_modes: Sequence[str]) -> Mapping[str, Sequence[float]]:
        if set(score_modes) != {"dense"} and tuple(score_modes) != ("dense",):
            raise ValueError("dense trainer only supports dense evaluation")
        q_dense, _, _ = self.m3.encode(self._tokens([query], self.config.max_query_length))
        scores = []
        for start in range(0, len(passages), self.config.eval_passage_batch_size):
            p_dense, _, _ = self.m3.encode(self._tokens(
                passages[start:start + self.config.eval_passage_batch_size], self.config.max_passage_length
            ))
            scores.extend(self.m3.compute_dense_score(q_dense, p_dense).float().reshape(-1).cpu().tolist())
        return {"dense": scores}


class Phase11M3Trainer(Trainer):
    """Stock HF Trainer with only BGE-M3 relation-encoder loss/evaluation customized."""

    def __init__(self, *args: Any, phase11_config: TrainingConfig, m3_model: Any, **kwargs: Any):
        self.phase11_config = phase11_config
        self.m3_model = m3_model
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = False

    def compute_loss(
        self, model: Any, inputs: dict[str, Any], return_outputs: bool = False,
        num_items_in_batch: Tensor | int | None = None,
    ) -> Tensor | tuple[Tensor, Mapping[str, Tensor]]:
        del num_items_in_batch
        self.m3_model.model = model
        q_dense, _, _ = self.m3_model.encode(inputs["query_tokens"])
        p_dense, _, _ = self.m3_model.encode(inputs["passage_tokens"])
        scores = self.m3_model.compute_dense_score(q_dense, p_dense)
        loss = multi_positive_info_nce(
            scores, inputs["positive_mask"],
            candidate_mask=inputs["candidate_mask"],
            temperature=self.phase11_config.temperature,
        )
        outputs = {"loss": loss, "dense_scores": scores.detach()}
        return (loss, outputs) if return_outputs else loss

    def evaluate(
        self, eval_dataset: Dataset | None = None, ignore_keys: list[str] | None = None,
        metric_key_prefix: str = "eval",
    ) -> dict[str, float]:
        del eval_dataset, ignore_keys
        config = self.phase11_config
        if not config.dev_rich_data:
            return {}
        from .evaluation import evaluate_jsonl

        self.m3_model.model = self.model
        was_training = self.model.training
        self.model.eval()
        device = next(self.model.parameters()).device
        scorer = CurrentM3RelationScorer(self.m3_model, self.processing_class, config, device)
        autocast = torch.cuda.amp.autocast(dtype=torch.float16) if device.type == "cuda" else contextlib.nullcontext()
        try:
            with torch.inference_mode(), autocast:
                report = evaluate_jsonl(
                    config.dev_rich_data, scorer, score_modes=("dense",),
                    k_values=(config.early_stopping_k,), partition="dev",
                    model_provenance={"kind": "in_training_lora", "base": config.model_name_or_path},
                )
        finally:
            self.model.train(was_training)
        values = report["metrics"]["dense"]
        key = str(config.early_stopping_k)
        metrics = {
            f"{metric_key_prefix}_recall_at_{key}": float(values["recall_at"][key]),
            f"{metric_key_prefix}_mrr": float(values["macro_mrr"]),
            f"{metric_key_prefix}_ndcg_at_{key}": float(values["ndcg_at"][key]),
        }
        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        return metrics


def build_hf_training_arguments(config: TrainingConfig) -> TrainingArguments:
    config.validate()
    has_dev = config.dev_rich_data is not None
    kwargs: dict[str, Any] = {
        "output_dir": config.output_dir,
        "per_device_train_batch_size": config.micro_batch_size,
        "per_device_eval_batch_size": config.eval_passage_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "num_train_epochs": config.epochs,
        "max_steps": config.max_steps,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "warmup_ratio": config.warmup_ratio,
        "max_grad_norm": config.max_grad_norm,
        "fp16": config.fp16,
        "bf16": False,
        "gradient_checkpointing": config.gradient_checkpointing,
        "logging_steps": config.log_steps,
        "logging_first_step": True,
        "save_strategy": "epoch" if has_dev else "steps",
        "save_steps": config.save_steps,
        "save_total_limit": 1,
        "eval_strategy": "epoch" if has_dev else "no",
        "load_best_model_at_end": has_dev,
        "metric_for_best_model": "recall_at_10" if has_dev else None,
        "greater_is_better": True if has_dev else None,
        "remove_unused_columns": False,
        "dataloader_num_workers": config.num_workers,
        "dataloader_pin_memory": config.require_cuda,
        "report_to": [],
        "seed": config.seed,
        "data_seed": config.seed,
        "full_determinism": True,
        "save_only_model": False,
        "restore_callback_states_from_checkpoint": True,
        "use_cpu": not config.require_cuda,
        "optim": "adamw_torch",
    }
    signature = inspect.signature(TrainingArguments.__init__).parameters
    if "save_safetensors" in signature:
        kwargs["save_safetensors"] = True
    return TrainingArguments(**kwargs)


def _packages() -> dict[str, str | None]:
    result = {}
    for name in ("torch", "transformers", "FlagEmbedding", "peft", "accelerate"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def train(config: TrainingConfig) -> dict[str, Any]:
    device, resources = preflight_checks(config)
    torch.set_num_threads(config.num_threads)
    set_deterministic_seed(config.seed)
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    dataset = RelationJsonlDataset(config.train_data)
    data_summary = dataset.validate_all()
    config_hash = config_sha256(config)
    manifest = {
        "schema_version": "phase11.hf_trainer.v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config_sha256": config_hash,
        "training_data_sha256": file_sha256(config.train_data),
        "effective_config": resolved_config_dict(config),
        "resources": resources,
        "packages": _packages(),
        "data": data_summary,
        "framework": "transformers.Trainer + PEFT + FlagEmbedding EncoderOnlyEmbedderM3Model",
        "checkpoint_lifecycle": "native_hf_trainer",
    }
    _atomic_json(output / "run_manifest.json", manifest)
    _atomic_json(output / "resolved_config.json", resolved_config_dict(config))
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)

    peft_model, m3, tokenizer, model_metadata = build_trainable_model(config)
    manifest["model"] = model_metadata
    _atomic_json(output / "run_manifest.json", manifest)
    collator = CollisionSafeCollator(
        tokenizer, max_query_length=config.max_query_length,
        max_passage_length=config.max_passage_length,
        max_negatives_per_query=config.max_negatives_per_query,
    )
    callbacks = []
    if config.dev_rich_data:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience))
    trainer_kwargs: dict[str, Any] = {
        "model": peft_model,
        "args": build_hf_training_arguments(config),
        "train_dataset": dataset,
        "eval_dataset": [0] if config.dev_rich_data else None,
        "data_collator": collator,
        "callbacks": callbacks,
        "phase11_config": config,
        "m3_model": m3,
    }
    if "processing_class" in inspect.signature(Trainer.__init__).parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = Phase11M3Trainer(**trainer_kwargs)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    result = trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)
    final_dir = output / "final"
    trainer.save_model(str(final_dir))
    trainer.save_state()
    losses = [item["loss"] for item in trainer.state.log_history if "loss" in item]
    summary = {
        "config_sha256": config_hash,
        "training_data_sha256": manifest["training_data_sha256"],
        "global_step": trainer.state.global_step,
        "train_loss": float(result.training_loss),
        "logged_losses": losses,
        "loss_is_finite": math.isfinite(float(result.training_loss)),
        "peak_gpu_memory_gb": (
            round(torch.cuda.max_memory_allocated() / GIB, 3) if device.type == "cuda" else 0.0
        ),
        "final_artifact": str(final_dir),
        "contains_base_model_weights": False,
    }
    _atomic_json(output / "run_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BGE-M3 relation-encoder LoRA trainer")
    parser.add_argument("--config", required=True)
    parser.add_argument("--print_config", action="store_true")
    parser.add_argument("--resume_from_checkpoint")
    return parser


def cli_main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    config = resolve_training_config(
        args.config,
        {"resume_from_checkpoint": args.resume_from_checkpoint} if args.resume_from_checkpoint else None,
    )
    if args.print_config:
        print(json.dumps({
            "effective_config": resolved_config_dict(config),
            "config_sha256": config_sha256(config),
        }, indent=2, sort_keys=True))
        return 0
    train(config)
    return 0


__all__ = [
    "CollisionSafeCollator", "CONFIG_HASH_EXCLUDED_FIELDS", "Phase11M3Trainer",
    "RelationJsonlDataset", "TrainingConfig", "build_hf_training_arguments",
    "canonical_text", "cli_main", "config_sha256", "multi_positive_info_nce",
    "normalize_flat_record", "preflight_checks", "resolve_training_config",
    "resolved_config_dict", "train",
]
