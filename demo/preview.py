"""Small synthetic graph for trying the interface without model downloads."""

from __future__ import annotations

import hashlib

import numpy as np

from .data import GraphRecord
from .features import lexical_features


def preview_graph() -> GraphRecord:
    names = [
        "Marie Curie", "Warsaw", "Poland", "Paris", "Sorbonne",
        "Pierre Curie", "Radium", "Nobel Prize", "Physics", "France",
    ]
    relations = [
        "born in", "located in", "moved to", "worked at", "married",
        "discovered", "awarded", "field of work",
    ]
    edges = [
        (0, 0, 1), (1, 1, 2), (0, 2, 3), (3, 1, 9),
        (0, 3, 4), (0, 4, 5), (0, 5, 6), (0, 6, 7),
        (0, 7, 8), (5, 3, 4), (6, 7, 8), (7, 7, 8),
    ]
    return GraphRecord(
        id="__preview__", question="Where was Marie Curie born?",
        names=names, relations=relations,
        h=np.asarray([edge[0] for edge in edges], dtype=np.int32),
        r=np.asarray([edge[1] for edge in edges], dtype=np.int32),
        t=np.asarray([edge[2] for edge in edges], dtype=np.int32),
        seeds=[0], answers=[1],
        gold=np.asarray([i == 0 for i in range(len(edges))], dtype=bool),
        n_hops=1, test_type=["synthetic preview"],
    )


class PreviewFeatures:
    """Deterministic stand-in vectors; RBF still uses the real checkpoint."""

    def for_question(self, rec, question, gte=None, bge=None):
        seed = int.from_bytes(hashlib.blake2b(question.encode(), digest_size=8).digest(), "big")
        rng = np.random.default_rng(seed)
        q = rng.normal(size=1024).astype(np.float32)
        q /= np.linalg.norm(q)
        rel_rng = np.random.default_rng(2026)
        rel = rel_rng.normal(size=(len(rec.relations), 1024)).astype(np.float32)
        rel /= np.linalg.norm(rel, axis=1, keepdims=True)
        lex = np.asarray(lexical_features(question, rec.relations), dtype=np.float32)
        prior = np.stack([lex, lex - lex.max()], axis=1)
        return {"q": q, "rel": rel, "lex": lex, "prior": prior.astype(np.float32)}
