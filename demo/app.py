"""Streamlit interface for paper-aligned local RBF retrieval."""

from __future__ import annotations

import hashlib
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
import streamlit.components.v1 as components

from demo.data import DEFAULT_DATA_DIR, read_graph, read_manifest
from demo.encoders import BGEEncoder, FeatureStore, GTEEncoder
from demo.features import suggest_topics
from demo.graph_view import render_graph
from demo.llm import answer_question
from demo.retriever import Retriever
from demo.trace import trace_path

st.set_page_config(page_title="RBF · GTSQA retrieval", layout="wide")


@st.cache_resource
def engine():
    return Retriever(threads=int(os.getenv("TORCH_THREADS", "4"))), FeatureStore(DEFAULT_DATA_DIR)


@st.cache_resource
def live_encoders():
    return GTEEncoder(), BGEEncoder(Path(os.getenv("BGE_ADAPTER_DIR", "models/bge_adapter/final")))


@st.cache_data(show_spinner=False)
def graph(graph_id: str):
    return read_graph(graph_id, DEFAULT_DATA_DIR)


st.title("RBF · Question → Evidence Graph")
st.caption("Trực quan hóa cách câu hỏi và topic entities chọn ra subgraph bằng RBF")
st.caption("Data: [Graphcore/GTSQA](https://huggingface.co/datasets/Graphcore/GTSQA), CC BY 4.0 · Cattaneo et al. (2025)")

for column, title, description in zip(
    st.columns(4),
    ("01 · Câu hỏi", "02 · Relation factors", "03 · RBF paths", "04 · Top‑k evidence"),
    ("Chọn query và topic entities", "MLP tính step/stop cho quan hệ",
     "Chấm điểm đường đi hoàn chỉnh", "Xếp hạng và dựng subgraph"),
):
    with column.container(border=True):
        st.markdown(f"**{title}**  \n{description}")
st.divider()

if not (DEFAULT_DATA_DIR / "manifest.json").is_file():
    st.info("Prepare the local GTSQA sample first: `python -m demo.prepare --count 100`.")
    st.stop()
manifest = read_manifest(DEFAULT_DATA_DIR)
if not manifest.get("features_ready"):
    st.error("Graph files exist, but embeddings are incomplete. Re-run `python -m demo.prepare` after adding the BGE adapter.")
    st.stop()

examples = manifest["questions"]
labels = {item["id"]: f"{item['question']} · {item['n_hops']} hops · {item['n_triples']:,} triples" for item in examples}
selected_id = st.selectbox("GTSQA test example", list(labels), format_func=lambda key: labels[key])
rec = graph(selected_id)
question = st.text_area("Question", value=rec.question, key=f"question_{selected_id}", height=80).strip()
original = question == rec.question
suggestions = rec.seeds if original else suggest_topics(question, rec.names)
query_key = hashlib.blake2b(question.encode(), digest_size=5).hexdigest()
topics = st.multiselect(
    "Topic entities", list(range(rec.n_entities)), default=suggestions,
    format_func=lambda index: rec.names[index], key=f"topics_{selected_id}_{query_key}",
    help="Original examples use dataset seeds. Edited questions suggest matching graph entities; correct them here.",
)
if not topics:
    st.info("Select at least one topic entity from this graph to run retrieval.")
k = int(st.number_input(
    "Top‑k triple budget", min_value=1, max_value=min(500, rec.n_triples),
    value=min(200, rec.n_triples), step=1,
    help="Chọn số triples chính xác sẽ trả về và hiển thị. Tối đa 500 để graph dễ quan sát.",
))
st.caption(f"Sẽ xếp hạng toàn bộ {rec.n_triples:,} triples ứng viên và hiển thị top {k:,}.")
if not original:
    st.caption("Edited questions run inside this example’s graph. Gold recall describes the original example only.")
signature = (selected_id, question, tuple(sorted(topics)), k)
if st.button("Retrieve subgraph", type="primary", disabled=not topics or not question):
    try:
        with st.status("Đang truy xuất evidence subgraph…", expanded=True) as status:
            bar = st.progress(0, text="Chuẩn bị graph và checkpoint")
            action_started = time.perf_counter()
            retriever, features = engine()
            bar.progress(12, text="Nạp checkpoint và encoder nếu cần")
            status.write("Đã nạp checkpoint RBF")
            gte, bge = live_encoders() if not original else (None, None)
            bar.progress(24, text="Chuẩn bị query và topic entities")
            models_ready = time.perf_counter()
            stages = {
                "features": "Mã hóa query và chuẩn bị đặc trưng quan hệ",
                "factors": "MLP tính step/stop factors cho từng quan hệ",
                "paths": "RBF chấm điểm các đường đi hoàn chỉnh",
                "ranking": f"Xếp hạng và cắt top {k:,} triples",
                "complete": "Hoàn tất subgraph và số liệu đánh giá",
            }

            def update_progress(stage: str, fraction: float) -> None:
                label = stages[stage]
                bar.progress(int(fraction * 100), text=label)
                status.write(label)

            result = retriever.retrieve(
                rec, question, topics, k, features, gte, bge,
                on_progress=update_progress,
            )
            result["timings_ms"]["model_setup"] = (models_ready - action_started) * 1000
            result["timings_ms"]["total_action"] = (time.perf_counter() - action_started) * 1000
            st.session_state["retrieval"] = (signature, result)
            status.update(label=f"Đã truy xuất {len(result['triples']):,} triples", state="complete", expanded=False)
    except Exception as error:
        st.error(str(error))

saved = st.session_state.get("retrieval")
if not saved or saved[0] != signature:
    st.stop()
result = saved[1]
triples = result["triples"]
metrics = st.columns(6)
metrics[0].metric("Candidate triples", f"{result['candidate_count']:,}")
metrics[1].metric("Returned", len(triples))
metrics[2].metric("Retrieve action", f"{result['timings_ms']['total_action']:.1f} ms")
metrics[3].metric("RBF score", f"{result['timings_ms']['rbf_score']:.1f} ms")
metrics[4].metric("Feature work", f"{result['timings_ms']['features']:.1f} ms")
metrics[5].metric("Gold triple recall", f"{result['gold_triple_recall']:.1%}" if original else "—")
if original:
    st.caption(f"Answer recall@{k}: {result['answer_recall']:.1%} · Gold triple recall@{k}: {result['gold_triple_recall']:.1%}")
st.caption(
    f"Model setup: {result['timings_ms']['model_setup']:.1f} ms · "
    "Feature work includes encoding but not first model load. "
    "The paper’s published scorer timings used one GPU."
)
with st.container(border=True):
    st.markdown("**Quá trình retrieval của lượt này**")
    stages = st.columns(4)
    stages[0].markdown(f"**1 · Query + topics**  \n{len(topics)} topic entities làm điểm bắt đầu")
    stages[1].markdown(
        f"**2 · Relation MLP**  \n{len(rec.relations):,} loại quan hệ · "
        f"feature {result['timings_ms']['features']:.1f} ms"
    )
    stages[2].markdown(
        f"**3 · RBF paths**  \n{result['candidate_count']:,} triples được chấm điểm · "
        f"{result['timings_ms']['rbf_score']:.1f} ms"
    )
    stages[3].markdown(
        f"**4 · Top‑k subgraph**  \n{len(triples):,} triples được chọn · "
        f"ranking {result['timings_ms']['ranking']:.1f} ms"
    )

st.subheader("Retrieved evidence graph")
st.caption("🟠 Topic entity / best path · 🟢 Annotated answer · 🔵 Node trên best path · Xám: evidence khác. Dùng ‘Vừa khung’ để xem toàn bộ subgraph.")
selected = st.selectbox(
    "Chọn triple để làm nổi bật best complete path", triples,
    format_func=lambda item: f"[{item['id']}] {item['head']} → {item['relation']} → {item['tail']}",
)
if selected:
    explanation = trace_path(rec, result, selected["index"])
    path_indices = {step["index"] for step in explanation["path"]}
else:
    explanation, path_indices = {}, set()
graph_triples = list(triples)
if explanation.get("path"):
    present = {item["index"] for item in graph_triples}
    for step in explanation["path"]:
        if step["index"] not in present:
            h, r, t = rec.triple(step["index"])
            graph_triples.append({"index": step["index"], "head": h, "relation": r, "tail": t})
            present.add(step["index"])
components.html(render_graph(
    graph_triples, path_indices,
    {rec.names[i] for i in topics}, {rec.names[i] for i in rec.answers} if original else set(),
    selected["index"] if selected else None,
), height=690, scrolling=False)
if explanation.get("path"):
    st.caption("Cạnh cam là best complete path qua triple đã chọn; một số cạnh của path có thể nằm ngoài top‑k và được thêm làm ngữ cảnh.")
    st.write(" → ".join(step["head"] for step in explanation["path"]) + " → " + explanation["path"][-1]["tail"])
    st.caption(
        f"Union log-score {explanation['union_log_score']:.3f} · "
        f"All-seeds contribution {explanation['intersection_log_score']:.3f}. "
        + explanation["note"]
    )
elif selected:
    st.info(explanation["note"])

left, right = st.columns([1.4, 1])
with left:
    st.subheader("Ranked triples")
    st.dataframe(
        [{key: value for key, value in item.items() if key not in ("index", "gold")}
         for item in triples],
        hide_index=True, width="stretch", height=390,
    )
with right:
    st.subheader("Relation step / stop factors")
    st.caption("Giá trị MLP theo từng hướng của quan hệ, dùng để chấm điểm path.")
    st.dataframe(result["factors"], hide_index=True, width="stretch", height=390)

if os.getenv("DEEPSEEK_API_KEY") and os.getenv("DEEPSEEK_MODEL"):
    st.subheader("Optional answer")
    if st.button("Generate answer from top 30 triples"):
        try:
            with st.spinner("Asking DeepSeek…"):
                answer, warnings = answer_question(question, triples)
            st.write(answer)
            if warnings:
                st.warning("; ".join(warnings))
        except Exception as error:
            st.error(str(error))
