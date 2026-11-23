"""Text encoders and cached features required by the GTSQA checkpoint."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data import DEFAULT_DATA_DIR, GraphRecord
from .features import lexical_features

GTE_MODEL = "Alibaba-NLP/gte-large-en-v1.5"
GTE_REVISION = "104333d6af6f97649377c2afbde10a7704870c7b"
GTE_CODE_REVISION = "40ced75c3017eb27626c9d4ea981bde21a2662f4"
BGE_MODEL = "BAAI/bge-m3"
BGE_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
DEFAULT_ADAPTER = Path("models/bge_adapter/final")
ADAPTER_REPO = "minhnvm/bge-m3-relation-lora"
ADAPTER_REVISION = "5db4f7929cc3cd3afa2e61d0705a36a9474db89b"
ADAPTER_SHA256 = "a7c7e919e8686edcefad0faa6bdf8047478daedd93ec049bfa33b4c976302cba"


def ensure_adapter(adapter_dir: Path, download_if_missing: bool = True) -> dict:
    """Fetch the exact public LoRA revision used by the GTSQA notebook."""
    config_path = adapter_dir / "adapter_config.json"
    weights_path = adapter_dir / "adapter_model.safetensors"
    missing = [path.name for path in (config_path, weights_path) if not path.is_file()]
    if missing and not download_if_missing:
        raise FileNotFoundError(f"BGE adapter files missing in {adapter_dir}: {', '.join(missing)}")
    if missing:
        from huggingface_hub import hf_hub_download

        adapter_dir.mkdir(parents=True, exist_ok=True)
        for filename in missing:
            hf_hub_download(
                repo_id=ADAPTER_REPO, filename=filename, revision=ADAPTER_REVISION,
                local_dir=adapter_dir,
            )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if (
        config.get("base_model_name_or_path") != BGE_MODEL
        or config.get("r") != 8
        or config.get("lora_alpha") != 16
        or set(config.get("target_modules", [])) != {"query", "key", "value"}
    ):
        raise ValueError("BGE adapter config differs from the notebook's expected LoRA")
    digest = hashlib.sha256()
    with weights_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    weights_sha256 = digest.hexdigest()
    if download_if_missing and weights_sha256 != ADAPTER_SHA256:
        raise ValueError("Downloaded BGE adapter checksum differs from the pinned checkpoint")
    known = weights_sha256 == ADAPTER_SHA256
    return {
        "repo": ADAPTER_REPO if known else "local",
        "revision": ADAPTER_REVISION if known else None,
        "weights_sha256": weights_sha256,
    }


class GTEEncoder:
    def __init__(self, device: str = "cpu") -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(
            GTE_MODEL, device=device, trust_remote_code=True, revision=GTE_REVISION,
            config_kwargs={"code_revision": GTE_CODE_REVISION},
            model_kwargs={"code_revision": GTE_CODE_REVISION},
        )
        self.model.max_seq_length = 128

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.empty((0, 1024), dtype=np.float32)
        return np.asarray(self.model.encode(
            texts, batch_size=batch_size, show_progress_bar=False,
            convert_to_numpy=True, normalize_embeddings=True,
        ), dtype=np.float32)


class BGEEncoder:
    def __init__(self, adapter_dir: Path = DEFAULT_ADAPTER, device: str = "cpu") -> None:
        from safetensors.torch import load_file
        from transformers import AutoModel, AutoTokenizer

        config_path = adapter_dir / "adapter_config.json"
        weights_path = adapter_dir / "adapter_model.safetensors"
        if not config_path.is_file() or not weights_path.is_file():
            raise FileNotFoundError(
                f"Fine-tuned BGE-M3 adapter is required at {adapter_dir}; "
                "expected adapter_config.json and adapter_model.safetensors"
            )
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(BGE_MODEL, revision=BGE_REVISION)
        self.model = AutoModel.from_pretrained(
            BGE_MODEL, revision=BGE_REVISION, torch_dtype=torch.float32
        )
        config = json.loads(config_path.read_text(encoding="utf-8"))
        weights = load_file(str(weights_path))
        modules = dict(self.model.named_modules())
        scale = config["lora_alpha"] / config["r"]
        merged = 0
        with torch.no_grad():
            for key, a in weights.items():
                if not key.endswith("lora_A.weight"):
                    continue
                name = key[len("base_model.model."):-len(".lora_A.weight")]
                b = weights[key.replace("lora_A", "lora_B")]
                modules[name].weight += scale * (b.float() @ a.float())
                merged += 1
        if not merged or merged * 2 != len(weights):
            raise ValueError("BGE adapter does not match the notebook's LoRA layout")
        self.model = self.model.to(device).eval()

    @torch.inference_mode()
    def encode(self, texts: list[str], max_length: int, batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.empty((0, 1024), dtype=np.float32)
        result = []
        for start in range(0, len(texts), batch_size):
            tokens = self.tokenizer(
                texts[start:start + batch_size], padding=True, truncation=True,
                max_length=max_length, return_tensors="pt",
            ).to(self.device)
            hidden = self.model(**tokens).last_hidden_state[:, 0]
            result.append(F.normalize(hidden.float(), dim=-1).cpu().numpy())
        return np.concatenate(result)


def prepare_embeddings(records: list[GraphRecord], directory: Path, adapter: Path, device: str) -> None:
    """Build compact, reproducible features for original test questions."""
    relations = sorted({relation for rec in records for relation in rec.relations})
    questions = [rec.question for rec in records]
    gte = GTEEncoder(device)
    gte_rel = gte.encode(relations).astype(np.float16)
    gte_q = gte.encode(questions).astype(np.float16)
    del gte
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    bge = BGEEncoder(adapter, device)
    bge_rel = bge.encode(
        [relation.replace(".", " ").replace("_", " ") for relation in relations],
        max_length=48,
    ).astype(np.float32)
    bge_q = bge.encode(questions, max_length=96).astype(np.float32)
    del bge
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    np.savez_compressed(
        directory / "embeddings.npz", relations=np.asarray(relations),
        graph_ids=np.asarray([rec.id for rec in records]),
        gte_rel=gte_rel, gte_q=gte_q, bge_rel=bge_rel, bge_q=bge_q,
    )


class FeatureStore:
    def __init__(self, directory: Path = DEFAULT_DATA_DIR) -> None:
        data = np.load(directory / "embeddings.npz", allow_pickle=False)
        self.relations = [str(x) for x in data["relations"]]
        self.relation_id = {name: i for i, name in enumerate(self.relations)}
        self.graph_id = {str(value): i for i, value in enumerate(data["graph_ids"])}
        self.gte_rel = data["gte_rel"]
        self.gte_q = data["gte_q"]
        self.bge_rel = data["bge_rel"]
        self.bge_q = data["bge_q"]

    def for_question(
        self, rec: GraphRecord, question: str,
        gte: GTEEncoder | None = None, bge: BGEEncoder | None = None,
    ) -> dict[str, np.ndarray]:
        relation_rows = [self.relation_id[name] for name in rec.relations]
        if question == rec.question:
            q_gte = np.asarray(self.gte_q[self.graph_id[rec.id]], dtype=np.float32)
            q_bge = np.asarray(self.bge_q[self.graph_id[rec.id]], dtype=np.float32)
        else:
            if gte is None or bge is None:
                raise ValueError("edited questions require GTE and fine-tuned BGE encoders")
            q_gte = gte.encode([question])[0].astype(np.float16).astype(np.float32)
            q_bge = bge.encode([question], max_length=96)[0]
        relation_gte = np.asarray(self.gte_rel[relation_rows], dtype=np.float32)
        prior = np.asarray(self.bge_rel[relation_rows], dtype=np.float32) @ q_bge
        prior = prior.astype(np.float16).astype(np.float32)
        prior_pair = np.stack([prior, prior - prior.max()], axis=1)
        return {
            "q": q_gte,
            "rel": relation_gte,
            "lex": np.asarray(lexical_features(question, rec.relations), dtype=np.float32),
            "prior": prior_pair,
        }
