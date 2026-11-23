import math
import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
import pyarrow as pa
import pyarrow.parquet as pq

from demo.data import GraphRecord, graph_from_row, read_graph, write_graph
from demo.evaluate import evaluate_records
from demo.encoders import ensure_adapter
from demo.features import clean_name, lexical_features, suggest_topics
from demo.prepare import prepare, sample_order
from demo.retriever import Retriever
from demo.trace import trace_path


def small_graph() -> GraphRecord:
    return GraphRecord(
        id="7", question="Who taught Ada?",
        names=["Ada (Q1)", "Teacher (Q2)", "School (Q3)", "Bob (Q4)"],
        relations=["educated by", "works at"],
        h=np.asarray([0, 1, 3], dtype=np.int32),
        r=np.asarray([0, 1, 0], dtype=np.int32),
        t=np.asarray([1, 2, 1], dtype=np.int32),
        seeds=[0], answers=[1], gold=np.asarray([1, 0, 0], dtype=bool),
        n_hops=1, test_type=["in-distribution"],
    )


class FixedFeatures:
    def for_question(self, rec, question, gte=None, bge=None):
        rng = np.random.default_rng(42)
        q = rng.normal(size=1024).astype(np.float32)
        q /= np.linalg.norm(q)
        rel = rng.normal(size=(len(rec.relations), 1024)).astype(np.float32)
        rel /= np.linalg.norm(rel, axis=1, keepdims=True)
        prior = np.linspace(0.4, 0.2, len(rec.relations), dtype=np.float32)
        return {
            "q": q, "rel": rel,
            "lex": np.asarray(lexical_features(question, rec.relations), dtype=np.float32),
            "prior": np.stack([prior, prior - prior.max()], axis=1),
        }


def test_graph_conversion_and_roundtrip(tmp_path):
    row = {
        "id": 7, "question": "Who taught Ada?", "seed_entities": ["Ada (Q1)"],
        "answer_node": "Teacher (Q2)",
        "answer_subgraph": [["Ada (Q1)", "educated by (P1)", "Teacher (Q2)"]],
        "graph": [
            ["Ada (Q1)", "educated by (P1)", "Teacher (Q2)"],
            ["Teacher (Q2)", "works at (P2)", "School (Q3)"],
        ],
        "n_hops": 1, "test_type": ["in-distribution"],
    }
    record = graph_from_row(row)
    assert record.relations == ["educated by", "works at"]
    assert record.gold.tolist() == [True, False]
    write_graph(record, tmp_path / "graphs")
    loaded = read_graph("7", tmp_path)
    assert loaded.triple(0) == ("Ada (Q1)", "educated by", "Teacher (Q2)")
    assert loaded.seeds == [0]


def test_topic_suggestions_and_sampler():
    assert clean_name("Ada (Q1)") == "Ada"
    assert suggest_topics(
        "Where did Ada Lovelace meet Bob?",
        ["Ada (Q1)", "Ada Lovelace (Q2)", "Bob (Q3)"],
    ) == [1, 2]
    rows = [
        {"id": i, "n_hops": 1 + i % 3,
         "seed_entities": ["a"] if i % 2 else ["a", "b"],
         "test_type": ["in-distribution" if i % 2 else "unseen_graph_type"]}
        for i in range(20)
    ]
    order = sample_order(rows)
    assert sorted(order) == list(range(20))
    assert order == sample_order(rows)


def test_prepare_local_parquet(tmp_path, monkeypatch):
    rows = []
    for i in range(4):
        rows.append({
            "id": i, "question": f"Who knows Person {i}?",
            "seed_entities": [f"Person {i} (Q{i})"],
            "answer_node": "Friend (Q99)",
            "answer_subgraph": [[f"Person {i} (Q{i})", "knows (P1)", "Friend (Q99)"]],
            "graph": [[f"Person {i} (Q{i})", "knows (P1)", "Friend (Q99)"]],
            "n_hops": 1 + i % 2, "test_type": ["in-distribution"],
        })
    parquet_path = tmp_path / "test.parquet"
    pq.write_table(pa.Table.from_pylist(rows), parquet_path)

    def fake_embeddings(records, directory, adapter, device):
        assert len(records) == 2
        np.savez_compressed(
            directory / "embeddings.npz", relations=np.asarray(["knows"]),
            graph_ids=np.asarray([rec.id for rec in records]),
            gte_rel=np.zeros((1, 1024), dtype=np.float16),
            gte_q=np.zeros((2, 1024), dtype=np.float16),
            bge_rel=np.zeros((1, 1024), dtype=np.float32),
            bge_q=np.zeros((2, 1024), dtype=np.float32),
        )

    monkeypatch.setattr("demo.prepare.prepare_embeddings", fake_embeddings)
    monkeypatch.setattr("demo.prepare.ensure_adapter", lambda *args, **kwargs: {
        "repo": "minhnvm/bge-m3-relation-lora", "revision": "test",
        "weights_sha256": "0" * 64,
    })
    directory = tmp_path / "out"
    manifest = prepare(Namespace(
        source_parquet=str(parquet_path), candidate_buffer=4, count=2,
        data_dir=str(directory), adapter="missing", device="cpu",
    ))
    assert manifest["count"] == 2 and manifest["features_ready"]
    assert len(manifest["source_sha256"]) == 64
    assert all((directory / "graphs" / f"{item['id']}.json.gz").exists()
               for item in manifest["questions"])


def test_local_adapter_validation(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(json.dumps({
        "base_model_name_or_path": "BAAI/bge-m3", "r": 8, "lora_alpha": 16,
        "target_modules": ["query", "key", "value"],
    }), encoding="utf-8")
    from safetensors.torch import save_file

    save_file({"dummy": torch.zeros(1)}, adapter / "adapter_model.safetensors")
    info = ensure_adapter(adapter, download_if_missing=False)
    assert len(info["weights_sha256"]) == 64
    assert info["repo"] == "local" and info["revision"] is None


@pytest.fixture(scope="module")
def retriever():
    return Retriever(Path("gtsqa_rbf_ckpt/best.pt"), threads=2)


@pytest.mark.parametrize("topics", [[0], [0, 3]])
def test_checkpoint_scores_and_path_trace(retriever, topics):
    rec = small_graph()
    result = retriever.retrieve(rec, rec.question, topics, 50, FixedFeatures())
    assert len(result["triples"]) == rec.n_triples
    assert result["triples"][0]["score"] >= result["triples"][-1]["score"]
    assert len(result["factors"]) == len(rec.relations)
    for item in result["triples"]:
        trace = trace_path(rec, result, item["index"])
        if not trace["path"]:
            assert item["score"] <= -9990
            continue
        assert item["index"] in [step["index"] for step in trace["path"]]
        model_log_score = -np.logaddexp(0, -item["score"]) + math.log(2)
        assert trace["combined_log_score"] == pytest.approx(model_log_score, abs=1e-4)


def test_checkpoint_matches_notebook_reference(retriever):
    """Run the training notebook's own RBF implementation on the same input."""
    notebook = json.loads(Path("gtsqa_rbf_ckpt/rbf-train-gtsqa.ipynb").read_text())
    reference_code = "".join(notebook["cells"][18]["source"])
    namespace = {"torch": torch, "nn": torch.nn, "F": F}
    exec(reference_code, namespace)
    reference = namespace["RBF"](
        emb=1024, hidden=256, T=3, mm=True, extra_prior=2, stop=True
    ).eval()
    reference.load_state_dict({
        key.removeprefix("rbf."): value
        for key, value in retriever.model.state_dict().items()
    })
    rec = small_graph()
    feature = FixedFeatures().for_question(rec, rec.question)
    topic = torch.tensor([1, 0, 0, 1], dtype=torch.long)
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
    with torch.inference_mode():
        reference_scores = reference(inputs).numpy()
    result = retriever.retrieve(rec, rec.question, [0, 3], 50, FixedFeatures())
    actual = {
        triple["index"]: -np.logaddexp(0, -triple["score"]) + math.log(2)
        for triple in result["triples"]
    }
    assert np.allclose(
        [actual[i] for i in range(rec.n_triples)], reference_scores,
        atol=1e-5,
    )


def test_retrieval_rejects_missing_topics(retriever):
    with pytest.raises(ValueError, match="topic"):
        retriever.retrieve(small_graph(), "Who taught Ada?", [], 200, FixedFeatures())


def test_arbitrary_budget_and_progress(retriever):
    rec = small_graph()
    updates = []
    result = retriever.retrieve(
        rec, rec.question, [0], 2, FixedFeatures(), on_progress=lambda stage, value: updates.append((stage, value))
    )
    assert len(result["triples"]) == 2
    assert [stage for stage, _ in updates] == ["features", "factors", "paths", "ranking", "complete"]
    assert [value for _, value in updates] == sorted(value for _, value in updates)
    with pytest.raises(ValueError, match="positive integer"):
        retriever.retrieve(rec, rec.question, [0], 0, FixedFeatures())


def test_empty_candidate_graph(retriever):
    rec = GraphRecord(
        id="8", question="Any relation?", names=["Ada (Q1)"], relations=[],
        h=np.empty(0, dtype=np.int32), r=np.empty(0, dtype=np.int32),
        t=np.empty(0, dtype=np.int32), seeds=[0], answers=[],
        gold=np.empty(0, dtype=bool), n_hops=0, test_type=[],
    )
    result = retriever.retrieve(rec, rec.question, [0], 200, FixedFeatures())
    assert result["triples"] == [] and result["candidate_count"] == 0


def test_sample_evaluator(retriever):
    report = evaluate_records([small_graph()], FixedFeatures(), retriever)
    assert report["questions"] == 1
    assert set(report["budgets"]) == {"50", "100", "200"}
    assert report["budgets"]["200"]["answer_node_recall"] == 1.0
    assert report["budgets"]["200"]["gold_triple_recall"] == 1.0


@pytest.mark.parametrize("seed", range(5))
def test_trace_matches_model_on_random_graphs(retriever, seed):
    rng = np.random.default_rng(seed)
    rec = GraphRecord(
        id=str(seed), question="Which relation connects the answer?",
        names=[f"Entity {i} (Q{i})" for i in range(8)],
        relations=["relation one", "relation two", "relation three"],
        h=rng.integers(0, 8, 15, dtype=np.int32),
        r=rng.integers(0, 3, 15, dtype=np.int32),
        t=rng.integers(0, 8, 15, dtype=np.int32),
        seeds=[0, 4], answers=[7], gold=np.zeros(15, dtype=bool),
        n_hops=3, test_type=["in-distribution"],
    )
    result = retriever.retrieve(rec, rec.question, rec.seeds, 50, FixedFeatures())
    for item in result["triples"]:
        trace = trace_path(rec, result, item["index"])
        if not trace["path"]:
            continue
        model_log_score = -np.logaddexp(0, -item["score"]) + math.log(2)
        assert trace["combined_log_score"] == pytest.approx(model_log_score, abs=1e-3)
