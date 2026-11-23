"""RBF inference using the GTSQA notebook checkpoint and local graph records."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fast_retriever.rbf.rbf import RBF, log_to_logit

from .data import GraphRecord
from .encoders import BGEEncoder, FeatureStore, GTEEncoder

DEFAULT_CHECKPOINT = Path("gtsqa_rbf_ckpt/best.pt")


class Scorer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.rbf = RBF(emb=1024, hidden=256, T=3, mm=True, extra_prior=2, stop=True)

    def forward(self, inputs: dict) -> torch.Tensor:
        return log_to_logit(self.rbf(inputs) - math.log(2))


class Retriever:
    def __init__(self, checkpoint: Path = DEFAULT_CHECKPOINT, threads: int = 4) -> None:
        torch.set_num_threads(threads)
        self.model = Scorer().eval()
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state, strict=True)

    @torch.inference_mode()
    def retrieve(
        self, rec: GraphRecord, question: str, topic_entity_ids: list[int], k: int,
        features: FeatureStore, gte: GTEEncoder | None = None,
        bge: BGEEncoder | None = None,
        on_progress: Callable[[str, float], None] | None = None,
    ) -> dict:
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError("k must be a positive integer")
        if not question.strip():
            raise ValueError("question cannot be empty")
        topics = sorted(set(int(i) for i in topic_entity_ids))
        if not topics or any(i < 0 or i >= rec.n_entities for i in topics):
            raise ValueError("select at least one valid topic entity")
        if rec.n_triples == 0:
            return {
                "graph_id": rec.id, "question": question, "topics": topics, "k": k,
                "candidate_count": 0, "triples": [], "factors": [],
                "answer_recall": 0.0, "gold_triple_recall": 0.0,
                "timings_ms": {"features": 0.0, "rbf_score": 0.0, "ranking": 0.0},
                "_log_pi": np.empty((0, 2)), "_log_stop": np.empty((0, 2)),
            }
        started = time.perf_counter()
        if on_progress:
            on_progress("features", 0.30)
        feature = features.for_question(rec, question, gte, bge)
        encoded = time.perf_counter()
        if on_progress:
            on_progress("factors", 0.52)
        topic = torch.zeros(rec.n_entities, dtype=torch.long)
        topic[topics] = 1
        inputs = {
            "h": torch.from_numpy(rec.h.astype(np.int64)),
            "r": torch.from_numpy(rec.r.astype(np.int64)),
            "t": torch.from_numpy(rec.t.astype(np.int64)),
            "n_ent": rec.n_entities,
            "q": torch.from_numpy(feature["q"][None, :]),
            "rel": torch.from_numpy(feature["rel"]),
            "ent": torch.empty((rec.n_entities, 0)),
            "topic": F.one_hot(topic, num_classes=2).float(),
            "lex": torch.from_numpy(feature["lex"]),
            "rel_prior": torch.from_numpy(feature["prior"]),
        }
        logits = self.model.rbf.relation_logits(inputs)
        stop_logits = self.model.rbf.stop_logits(inputs)
        if on_progress:
            on_progress("paths", 0.72)
        scores = self.model(inputs).cpu().numpy()
        scored = time.perf_counter()
        if on_progress:
            on_progress("ranking", 0.91)
        order = np.lexsort((np.arange(rec.n_triples), -scores))[:k]
        triples = []
        for rank, index in enumerate(order, 1):
            h, r, t = rec.triple(int(index))
            triples.append({
                "id": f"T{rank}", "index": int(index), "head": h,
                "relation": r, "tail": t, "score": float(scores[index]),
                "gold": bool(rec.gold[index]),
            })
        found = set(rec.h[order].tolist()) | set(rec.t[order].tolist())
        answer_recall = len(found.intersection(rec.answers)) / max(len(rec.answers), 1)
        gold_recall = float(rec.gold[order].sum() / max(int(rec.gold.sum()), 1))
        factors = []
        step = torch.sigmoid(logits).cpu().numpy()
        stop = torch.sigmoid(stop_logits).cpu().numpy()
        for i, relation in enumerate(rec.relations):
            factors.append({
                "relation": relation,
                "step_forward": float(step[i, 0]), "step_reverse": float(step[i, 1]),
                "stop_forward": float(stop[i, 0]), "stop_reverse": float(stop[i, 1]),
            })
        result = {
            "graph_id": rec.id, "question": question, "topics": topics,
            "k": k, "candidate_count": rec.n_triples, "triples": triples,
            "factors": factors, "answer_recall": answer_recall,
            "gold_triple_recall": gold_recall,
            "timings_ms": {
                "features": (encoded - started) * 1000,
                "rbf_score": (scored - encoded) * 1000,
                "ranking": (time.perf_counter() - scored) * 1000,
            },
            "_log_pi": F.logsigmoid(logits).cpu().numpy(),
            "_log_stop": F.logsigmoid(stop_logits).cpu().numpy(),
        }
        if on_progress:
            on_progress("complete", 1.0)
        return result
