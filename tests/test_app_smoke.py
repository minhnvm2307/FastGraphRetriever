"""Small Streamlit interaction without downloading GTSQA or the text encoders."""

import json
from pathlib import Path

import numpy as np
import pytest

streamlit = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

import demo.data as data
from test_demo import small_graph


def test_choose_example_and_retrieve(tmp_path, monkeypatch):
    rec = small_graph()
    data.write_graph(rec, tmp_path / "graphs")
    (tmp_path / "manifest.json").write_text(json.dumps({
        "count": 1, "features_ready": True,
        "questions": [{
            "id": rec.id, "question": rec.question,
            "n_hops": rec.n_hops, "n_triples": rec.n_triples,
        }],
    }), encoding="utf-8")
    rng = np.random.default_rng(7)
    gte_rel = rng.normal(size=(2, 1024)).astype(np.float32)
    gte_q = rng.normal(size=(1, 1024)).astype(np.float32)
    bge_rel = rng.normal(size=(2, 1024)).astype(np.float32)
    bge_q = rng.normal(size=(1, 1024)).astype(np.float32)
    for array in (gte_rel, gte_q, bge_rel, bge_q):
        array /= np.linalg.norm(array, axis=1, keepdims=True)
    np.savez_compressed(
        tmp_path / "embeddings.npz", relations=np.asarray(rec.relations),
        graph_ids=np.asarray([rec.id]), gte_rel=gte_rel.astype(np.float16),
        gte_q=gte_q.astype(np.float16), bge_rel=bge_rel, bge_q=bge_q,
    )
    monkeypatch.setattr(data, "DEFAULT_DATA_DIR", tmp_path)
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "demo/app.py").run()
    assert not app.exception
    assert len(app.button) >= 1
    app.number_input[0].set_value(2).run()
    assert not app.exception
    app.button[0].click().run()
    assert not app.exception
    assert len(app.metric) == 6
    assert len(app.session_state["retrieval"][1]["triples"]) == 2
