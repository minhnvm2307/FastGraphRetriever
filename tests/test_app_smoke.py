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
    assert len(app.tabs) == 4
    assert len(app.button) >= 1
    app.number_input[0].set_value(2).run()
    assert not app.exception
    app.button[0].click().run()
    assert not app.exception
    assert len(app.metric) == 7
    assert len(app.session_state["retrieval"][1]["triples"]) == 2
    next(button for button in app.button if button.label.startswith("Chạy đánh giá")).click().run(timeout=20)
    assert not app.exception
    assert app.session_state["evaluation"]["questions"] == 1


def test_missing_data_shows_setup_status(tmp_path, monkeypatch):
    import demo.llm as llm

    monkeypatch.setattr(data, "DEFAULT_DATA_DIR", tmp_path)
    monkeypatch.setattr(llm, "deepseek_ready", lambda: False)
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "demo/app.py").run(timeout=20)
    assert not app.exception
    assert len(app.tabs) == 4
    assert any("Chế độ minh họa" in item.value for item in app.warning)
    assert len(app.session_state["retrieval"][1]["triples"]) == 12
    app.button[0].click().run(timeout=20)
    assert not app.exception
    assert len(app.session_state["retrieval"][1]["triples"]) == 12
    assert any("Chuẩn bị dữ liệu" in item.value for item in app.info)
    app.chat_input[0].set_value("Where was Marie Curie born?").run(timeout=20)
    assert not app.exception
    answer = app.session_state["chat_messages"][-1]
    assert answer["retrieval_ms"] >= answer["rbf_ms"] >= 0
    assert answer["answer_ms"] is None
    assert answer["total_ms"] >= answer["retrieval_ms"]
    assert answer["triples"]


def test_chat_answer_timing_and_history(tmp_path, monkeypatch):
    import demo.llm as llm

    monkeypatch.setattr(data, "DEFAULT_DATA_DIR", tmp_path)
    monkeypatch.setattr(llm, "deepseek_ready", lambda: True)
    seen = []

    def fake_answer(question, triples, topics, history=None):
        seen.append((question, len(triples), topics, history))
        return {
            "response": "Câu trả lời thử", "answers": ["Câu trả lời thử"],
            "repaired": None, "timing": {"api_s": 0.001},
            "usage": {"total_tokens": 20},
        }

    monkeypatch.setattr(llm, "answer_with_details", fake_answer)
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "demo/app.py").run(timeout=20)
    app.chat_input[0].set_value("Where was Marie Curie born?").run(timeout=20)
    app.chat_input[0].set_value("And where did she work?").run(timeout=20)
    assert not app.exception
    assert seen[0][3] == []
    assert seen[0][2] == ["Marie Curie"]
    assert [item["role"] for item in seen[1][3]] == ["user", "assistant"]
    answer = app.session_state["chat_messages"][-1]
    assert answer["content"] == "Câu trả lời thử"
    assert answer["answer_ms"] >= 0
    assert answer["total_ms"] >= answer["retrieval_ms"]
