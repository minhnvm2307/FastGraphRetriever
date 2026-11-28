"""Streamlit interface for paper-aligned local RBF retrieval."""

from __future__ import annotations

import hashlib
import os
import sys
import time
from pathlib import Path
from textwrap import dedent

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from demo.data import DEFAULT_DATA_DIR, read_graph, read_manifest
from demo.encoders import BGEEncoder, FeatureStore, GTEEncoder
from demo.evaluate import evaluate_records
from demo.features import suggest_topics
from demo.graph_view import render_graph
from demo.llm import answer_question, answer_with_details, deepseek_ready
from demo.preview import PreviewFeatures, preview_graph
from demo.retriever import Retriever
from demo.trace import trace_path

st.set_page_config(page_title="RBF · Evidence Explorer", page_icon="🕸️", layout="wide")
st.markdown("""<style>
  .block-container {max-width: 1500px; padding-top: 1.25rem; padding-bottom: 3rem}
  .rbf-hero {position:relative; overflow:hidden; border:1px solid #314967;
    border-radius:24px; padding:32px 38px 28px; margin-bottom:22px;
    background:radial-gradient(circle at 86% 18%,rgba(94,234,212,.19),transparent 28%),
      radial-gradient(circle at 74% 82%,rgba(113,104,235,.24),transparent 34%),
      linear-gradient(115deg,#0b172b 5%,#152945 58%,#10273e);
    box-shadow:0 18px 50px rgba(0,0,0,.16); color:#f7fbff}
  .rbf-hero::after {content:""; position:absolute; width:310px; height:310px;
    right:-95px; top:-170px; border:1px solid rgba(196,236,255,.25);
    border-radius:50%; box-shadow:0 0 0 44px rgba(196,236,255,.025),
      0 0 0 92px rgba(196,236,255,.018); pointer-events:none}
  .rbf-eyebrow {font-size:.73rem; letter-spacing:.18em; font-weight:800;
    text-transform:uppercase; color:#78e9d6; margin-bottom:12px}
  .rbf-hero h1 {font-size:clamp(2.15rem,3.5vw,3.4rem); line-height:1.08;
    letter-spacing:-.035em; margin:0 0 12px; color:#fff}
  .rbf-hero p {font-size:1rem; max-width:700px; line-height:1.55;
    margin:0 0 23px; color:#c4d4e9}
  .rbf-pills {display:flex; flex-wrap:wrap; gap:8px}
  .rbf-pill {display:inline-flex; align-items:center; gap:8px; padding:7px 12px;
    border-radius:999px; border:1px solid rgba(192,218,245,.24);
    background:rgba(255,255,255,.07); color:#e9f4ff; font-size:.78rem;
    font-weight:650; letter-spacing:.01em}
  .rbf-dot {width:7px; height:7px; border-radius:50%; background:#63e6be;
    box-shadow:0 0 0 3px rgba(99,230,190,.13)}
  .rbf-dot-muted {background:#fbbf75; box-shadow:0 0 0 3px rgba(251,191,117,.12)}
  .rbf-flow {display:flex; flex-wrap:wrap; gap:8px; margin:4px 0 22px}
  .rbf-flow span {padding:8px 12px; border:1px solid rgba(128,128,128,.28);
    border-radius:10px; background:rgba(128,128,128,.09); font-size:.78rem;
    font-weight:650}
  .rbf-flow b {color:#36bfa9; margin-right:5px}
  div[data-testid="stMetric"] {background:rgba(128,128,128,.08);
    border:1px solid rgba(128,128,128,.27); border-radius:14px; padding:13px 16px}
  div[data-testid="stMetricValue"] {font-weight:720; letter-spacing:-.02em}
  div[data-testid="stTabs"] button {font-weight:680}
  div[data-testid="stTabs"] button[aria-selected="true"] {color:#21bda7}
  div[data-testid="stChatMessage"] {border:1px solid rgba(128,128,128,.19);
    border-radius:16px; background:rgba(128,128,128,.035)}
  button[data-testid="baseButton-primary"] {background:linear-gradient(110deg,#17b9a5,#0f91b1);
    border:0; color:#fff; font-weight:700; box-shadow:0 5px 16px rgba(20,165,156,.2)}
  .rbf-speed {display:grid; gap:8px; margin:8px 0 12px}
  .rbf-speed-row {display:grid; grid-template-columns:82px minmax(0,1fr) 94px;
    gap:10px; align-items:center; font-size:.78rem; font-weight:600}
  .rbf-speed-track {height:8px; border-radius:999px; overflow:hidden;
    background:rgba(128,128,128,.19)}
  .rbf-speed-fill {height:100%; border-radius:999px; background:#23bda9}
  .rbf-speed-fill.answer {background:#efaa67}
  .rbf-speed-row strong {text-align:right; font-variant-numeric:tabular-nums}
</style>""", unsafe_allow_html=True)


@st.cache_resource
def engine():
    return Retriever(threads=int(os.getenv("TORCH_THREADS", "4"))), FeatureStore(DEFAULT_DATA_DIR)


@st.cache_resource
def live_encoders():
    return GTEEncoder(), BGEEncoder(Path(os.getenv("BGE_ADAPTER_DIR", "models/bge_adapter/final")))


@st.cache_data(show_spinner=False)
def graph(graph_id: str):
    return read_graph(graph_id, DEFAULT_DATA_DIR)


@st.cache_resource
def preview_engine():
    return Retriever(threads=int(os.getenv("TORCH_THREADS", "4")))


def timing_comparison(retrieval_ms: float, answer_ms: float) -> None:
    longest = max(retrieval_ms, answer_ms, 1)
    retrieval_width = max(5, min(100, retrieval_ms / longest * 100))
    answer_width = max(5, min(100, answer_ms / longest * 100))
    st.markdown(dedent(f"""
    <div class="rbf-speed" aria-label="So sánh thời gian truy xuất và trả lời">
      <div class="rbf-speed-row"><span>Truy xuất</span>
        <div class="rbf-speed-track"><div class="rbf-speed-fill" style="width:{retrieval_width:.1f}%"></div></div>
        <strong>{retrieval_ms:.1f} ms</strong></div>
      <div class="rbf-speed-row"><span>Trả lời</span>
        <div class="rbf-speed-track"><div class="rbf-speed-fill answer" style="width:{answer_width:.1f}%"></div></div>
        <strong>{answer_ms:.1f} ms</strong></div>
    </div>"""), unsafe_allow_html=True)


def retrieve_view(manifest: dict, preview: bool = False) -> None:
    examples = manifest["questions"]
    if not examples:
        st.info("Tập dữ liệu chưa có câu hỏi. Hãy chuẩn bị lại dữ liệu.")
        return
    if preview:
        st.warning("Chế độ minh họa: đồ thị và embedding được tạo sẵn để thử giao diện. "
                   "Điểm số và recall ở đây không phải kết quả GTSQA.")
    st.markdown(dedent("""<div class="rbf-flow" aria-label="Các bước truy xuất">
      <span><b>01</b> Chọn câu hỏi</span><span><b>02</b> Chọn thực thể</span>
      <span><b>03</b> RBF xếp hạng</span><span><b>04</b> Khám phá bằng chứng</span>
    </div>"""), unsafe_allow_html=True)
    controls, canvas = st.columns([1, 2.1], gap="large")
    with controls:
        st.subheader("Thiết lập truy xuất")
        st.caption("Chọn đồ thị, điều chỉnh câu hỏi và ngân sách bằng chứng.")
        labels = {item["id"]: f"{item['question']} · {item['n_hops']} hops · {item['n_triples']:,} triples" for item in examples}
        selected_id = st.selectbox("Câu hỏi GTSQA", list(labels), format_func=lambda key: labels[key])
        rec = preview_graph() if preview else graph(selected_id)
        st.caption(f"{rec.n_hops} hops · {rec.n_triples:,} triples ứng viên · {len(rec.seeds)} topic entities")
        question = st.text_area("Câu hỏi", value=rec.question, key=f"question_{selected_id}", height=80).strip()
        original = question == rec.question
        suggestions = rec.seeds if original else suggest_topics(question, rec.names)
        query_key = hashlib.blake2b(question.encode(), digest_size=5).hexdigest()
        topics = st.multiselect(
            "Topic entities", list(range(rec.n_entities)), default=suggestions,
            format_func=lambda index: rec.names[index], key=f"topics_{selected_id}_{query_key}",
            help="Original examples use dataset seeds. Edited questions suggest matching graph entities; correct them here.",
        )
        if not topics:
            st.info("Hãy chọn ít nhất một topic entity trong đồ thị này.")
        k = int(st.number_input(
            "Số triple cần lấy (top-k)", min_value=1, max_value=min(500, rec.n_triples),
            value=min(200, rec.n_triples), step=1,
            help="Chọn số triples chính xác sẽ trả về và hiển thị. Tối đa 500 để graph dễ quan sát.",
        ))
        st.caption(f"Sẽ xếp hạng toàn bộ {rec.n_triples:,} triples ứng viên và hiển thị top {k:,}.")
        if not original:
            st.caption("Câu hỏi đã sửa chạy trong đồ thị này; recall của câu hỏi gốc không còn áp dụng.")
        signature = (selected_id, question, tuple(sorted(topics)), k)
        auto_preview = preview and not st.session_state.get("preview_initialized", False) and bool(topics) and bool(question)
        if preview:
            st.session_state["preview_initialized"] = True
        if st.button("Truy xuất đồ thị", type="primary", disabled=not topics or not question) or auto_preview:
            try:
                with st.status("Đang truy xuất evidence subgraph…", expanded=True) as status:
                    bar = st.progress(0, text="Chuẩn bị graph và checkpoint")
                    action_started = time.perf_counter()
                    retriever, features = (preview_engine(), PreviewFeatures()) if preview else engine()
                    bar.progress(12, text="Nạp checkpoint và encoder nếu cần")
                    status.write("Đã nạp checkpoint RBF")
                    gte, bge = live_encoders() if not original and not preview else (None, None)
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
        with canvas:
            st.subheader("Đồ thị bằng chứng")
            st.info("Kết quả truy xuất sẽ xuất hiện ở đây.")
        return
    result = saved[1]
    triples = result["triples"]
    metrics = st.columns(6)
    metrics[0].metric("Triple ứng viên", f"{result['candidate_count']:,}")
    metrics[1].metric("Đã trả về", len(triples))
    metrics[2].metric("Tổng truy xuất", f"{result['timings_ms']['total_action']:.1f} ms")
    metrics[3].metric("Chấm điểm RBF", f"{result['timings_ms']['rbf_score']:.1f} ms")
    metrics[4].metric("Tạo đặc trưng", f"{result['timings_ms']['features']:.1f} ms")
    metrics[5].metric("Recall triple chuẩn", f"{result['gold_triple_recall']:.1%}" if original else "—")
    if original:
        st.caption(f"Answer recall@{k}: {result['answer_recall']:.1%} · Gold triple recall@{k}: {result['gold_triple_recall']:.1%}")
    st.caption(
        f"Nạp mô hình: {result['timings_ms']['model_setup']:.1f} ms · "
        "Tạo đặc trưng bao gồm mã hóa câu hỏi, không gồm lần nạp mô hình đầu tiên."
    )
    with st.container(border=True):
        st.markdown("**Các bước truy xuất của lượt này**")
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

    with controls:
        selected = st.selectbox(
            "Chọn triple để làm nổi bật best complete path", triples,
            format_func=lambda item: f"[{item['id']}] {item['head']} → {item['relation']} → {item['tail']}",
        )
        if selected:
            explanation = trace_path(rec, result, selected["index"])
            path_indices = {step["index"] for step in explanation["path"]}
        else:
            explanation, path_indices = {}, set()
    with canvas:
        st.subheader("Đồ thị bằng chứng")
        st.caption("🟠 Topic entity / best path · 🟢 Annotated answer · 🔵 Node trên best path · Xám: evidence khác. Dùng ‘Vừa khung’ để xem toàn bộ subgraph.")
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
    with controls:
        if selected:
            st.metric("Điểm RBF", f"{selected['score']:.4f}")
        if explanation.get("path"):
            st.markdown("**Đường đi tốt nhất**")
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
        st.subheader("Triple được xếp hạng")
        st.dataframe(
            [{key: value for key, value in item.items() if key not in ("index", "gold")}
             for item in triples],
            hide_index=True, width="stretch", height=390,
        )
    with right:
        st.subheader("Hệ số bước / dừng theo quan hệ")
        st.caption("Giá trị MLP theo từng hướng của quan hệ, dùng để chấm điểm path.")
        st.dataframe(result["factors"], hide_index=True, width="stretch", height=390)

    if deepseek_ready():
        st.subheader("Câu trả lời tùy chọn")
        if st.button("Sinh câu trả lời từ top 100 triples"):
            try:
                with st.spinner("Đang hỏi DeepSeek…"):
                    answer, warnings = answer_question(
                        question, triples, [rec.names[i] for i in topics],
                    )
                st.write(answer)
                if warnings:
                    st.warning("; ".join(warnings))
            except Exception as error:
                status = getattr(getattr(error, "response", None), "status_code", None)
                st.error("DeepSeek từ chối API key (HTTP 401). Cập nhật DEEPSEEK_API_KEY trong .env."
                         if status == 401 else str(error))


def chat_view(manifest: dict, preview: bool = False) -> None:
    st.subheader("Hỏi đáp trên đồ thị")
    st.caption("Mỗi lượt: RBF chọn bằng chứng → P2h dựng đường suy luận → DeepSeek tạo câu trả lời.")
    if preview:
        st.warning("Đang dùng đồ thị minh họa và embedding thay thế; kết quả không phải GTSQA.")

    examples = manifest["questions"]
    labels = {item["id"]: item["question"] for item in examples}
    settings, conversation = st.columns([1, 2], gap="large")
    with settings:
        st.markdown("**NGỮ CẢNH HỘI THOẠI**")
        selected_id = st.selectbox("Đồ thị hội thoại", list(labels), format_func=lambda key: labels[key], key="chat_graph")
        rec = preview_graph() if preview else graph(selected_id)
        topics = st.multiselect(
            "Topic entities cho chat", list(range(rec.n_entities)), default=rec.seeds,
            format_func=lambda index: rec.names[index], key=f"chat_topics_{selected_id}",
        )
        k = int(st.number_input("Top-k cho chatbot", min_value=1,
                                max_value=min(100, rec.n_triples),
                                value=min(30, rec.n_triples), key="chat_k"))
        llm_ready = deepseek_ready()
        st.caption("● DeepSeek đã kết nối" if llm_ready else "○ DeepSeek chưa cấu hình API key")
        st.caption("Câu hỏi gợi ý: " + rec.question)
        if st.button("Xóa hội thoại", key="clear_chat"):
            st.session_state["chat_messages"] = []

    if st.session_state.get("chat_graph_id") != selected_id:
        st.session_state["chat_graph_id"] = selected_id
        st.session_state["chat_messages"] = []
    messages = st.session_state.setdefault("chat_messages", [])

    with conversation:
        prompt = st.chat_input("Hỏi về đồ thị đang chọn…", disabled=not topics, key="graph_chat_input")
        if not topics:
            st.info("Chọn ít nhất một topic entity để bắt đầu chat.")
        if prompt:
            history = [
                {"role": item["role"], "content": item["content"]}
                for item in messages if item.get("llm_context")
            ]
            messages.append({"role": "user", "content": prompt, "llm_context": llm_ready})
            started = time.perf_counter()
            try:
                with st.spinner("Đang truy xuất và tạo câu trả lời…"):
                    retriever, features = (preview_engine(), PreviewFeatures()) if preview else engine()
                    gte, bge = (None, None) if preview or prompt == rec.question else live_encoders()
                    result = retriever.retrieve(rec, prompt, topics, k, features, gte, bge)
                    retrieval_ms = (time.perf_counter() - started) * 1000
                    answer_ms = None
                    warnings = []
                    details = None
                    if llm_ready:
                        answer_started = time.perf_counter()
                        try:
                            details = answer_with_details(
                                prompt, result["triples"], [rec.names[i] for i in topics], history,
                            )
                            answer = details["response"]
                            if details["repaired"]:
                                warnings.append(f"Đã chạy lượt sửa đáp án ({details['repaired']}).")
                        finally:
                            answer_ms = (time.perf_counter() - answer_started) * 1000
                    else:
                        answer = ("Đã truy xuất được bằng chứng. Để sinh câu trả lời, "
                                  "cấu hình DEEPSEEK_API_KEY rồi khởi động lại ứng dụng.")
                    messages.append({
                        "role": "assistant", "content": answer,
                        "llm_context": llm_ready,
                        "retrieval_ms": retrieval_ms, "answer_ms": answer_ms,
                        "total_ms": (time.perf_counter() - started) * 1000,
                        "rbf_ms": result["timings_ms"]["rbf_score"],
                        "triples": result["triples"], "warnings": warnings,
                        "details": details,
                    })
            except Exception as error:
                status = getattr(getattr(error, "response", None), "status_code", None)
                message = ("DeepSeek từ chối API key (HTTP 401). Cập nhật DEEPSEEK_API_KEY trong .env."
                           if status == 401 else f"Không hoàn tất lượt chat: {error}")
                messages.append({
                    "role": "assistant", "content": message,
                    "llm_context": False,
                    "retrieval_ms": locals().get("retrieval_ms"),
                    "answer_ms": locals().get("answer_ms"),
                    "total_ms": (time.perf_counter() - started) * 1000,
                    "triples": locals().get("result", {}).get("triples", []),
                    "warnings": [],
                })

        if not messages:
            with st.container(border=True):
                st.markdown("### Bắt đầu khám phá bằng chứng")
                st.write("Hỏi về đồ thị đang chọn. Mỗi lượt sẽ hiển thị đáp án, các triple dùng làm bằng chứng và thời gian của từng bước.")
                st.caption("Thử hỏi: " + rec.question)
        for item in messages:
            with st.chat_message(item["role"]):
                detail = item.get("details")
                if item["role"] == "assistant" and detail and detail.get("answers"):
                    st.caption("ĐÁP ÁN TỪ BẰNG CHỨNG")
                    st.markdown("### " + " · ".join(detail["answers"]))
                    with st.expander("Xem lập luận P2h"):
                        st.markdown(item["content"])
                else:
                    st.markdown(item["content"])
                if item["role"] == "assistant" and item.get("retrieval_ms") is not None:
                    speed = st.columns(3)
                    speed[0].metric("Truy xuất", f"{item['retrieval_ms']:.1f} ms")
                    speed[1].metric("Trả lời", f"{item['answer_ms']:.1f} ms" if item.get("answer_ms") is not None else "—")
                    speed[2].metric("Tổng", f"{item['total_ms']:.1f} ms")
                    if item.get("rbf_ms") is not None:
                        st.caption(f"Trong truy xuất: chấm điểm RBF {item['rbf_ms']:.1f} ms")
                    if item.get("answer_ms") is not None and item["retrieval_ms"] > 0:
                        st.caption(f"Thời gian trả lời / truy xuất: {item['answer_ms'] / item['retrieval_ms']:.1f}×")
                        timing_comparison(item["retrieval_ms"], item["answer_ms"])
                    if item.get("warnings"):
                        st.warning("; ".join(item["warnings"]))
                    if detail:
                        usage = detail["usage"]
                        st.caption(
                            f"DeepSeek API {detail['timing']['api_s']:.3f} s · "
                            f"{usage['total_tokens']:,} tokens"
                            + (f" · Ước tính ${usage['estimated_cost']:.6f}" if "estimated_cost" in usage else "")
                        )
                    with st.expander(f"Xem {len(item['triples'])} triple bằng chứng"):
                        st.dataframe([
                            {key: value for key, value in triple.items() if key not in ("index", "gold")}
                            for triple in item["triples"]
                        ], hide_index=True, width="stretch")


def evaluation_view(manifest: dict) -> None:
    st.subheader("Hiệu năng trên tập mẫu")
    st.caption("Recall trên các câu hỏi GTSQA đã chuẩn bị, không phải kết quả toàn bộ tập test của bài báo.")
    if st.button("Chạy đánh giá @50 / @100 / @200", type="primary"):
        try:
            with st.spinner("Đang chấm điểm tập mẫu…"):
                records = [graph(item["id"]) for item in manifest["questions"]]
                retriever, features = engine()
                st.session_state["evaluation"] = evaluate_records(records, features, retriever)
        except Exception as error:
            st.error(str(error))
    result = st.session_state.get("evaluation")
    if not result:
        st.info("Bấm Chạy đánh giá để xem recall và thời gian xử lý.")
        return
    overview = st.columns(3)
    overview[0].metric("Số câu hỏi", result["questions"])
    overview[1].metric("RBF median", f"{result['rbf_score_ms']['median']:.1f} ms")
    overview[2].metric("RBF p95", f"{result['rbf_score_ms']['p95']:.1f} ms")
    st.markdown("**Recall theo ngân sách top-k**")
    chart = pd.DataFrame([
        {"Top-k": int(k), "Answer node recall": scores["answer_node_recall"],
         "Gold triple recall": scores["gold_triple_recall"]}
        for k, scores in result["budgets"].items()
    ]).set_index("Top-k")
    st.bar_chart(chart, height=260)
    st.dataframe([
        {"top-k": k, "Answer node recall": f"{scores['answer_node_recall']:.1%}",
         "Gold triple recall": f"{scores['gold_triple_recall']:.1%}"}
        for k, scores in result["budgets"].items()
    ], hide_index=True, width="stretch")
    st.caption("Các giá trị đo trên CPU cục bộ, với nhãn gold từ answer_subgraph của GTSQA.")


def status_view(manifest: dict | None) -> None:
    st.subheader("Dữ liệu & trạng thái")
    adapter = Path(os.getenv("BGE_ADAPTER_DIR", "models/bge_adapter/final"))
    ready = bool(manifest and manifest.get("features_ready"))
    st.dataframe([
        {"Thành phần": "Checkpoint RBF", "Trạng thái": "Sẵn sàng" if Path("gtsqa_rbf_ckpt/best.pt").is_file() else "Thiếu"},
        {"Thành phần": "Dữ liệu GTSQA + embeddings", "Trạng thái": "Sẵn sàng" if ready else "Chưa sẵn sàng"},
        {"Thành phần": "BGE-M3 adapter", "Trạng thái": "Sẵn sàng" if (adapter / "adapter_model.safetensors").is_file() else "Thiếu"},
        {"Thành phần": "DeepSeek API key", "Trạng thái": "Đã cấu hình" if deepseek_ready() else "Thiếu"},
    ], hide_index=True, width="stretch")
    if manifest:
        st.caption(f"{manifest.get('count', len(manifest.get('questions', [])))} câu hỏi · "
                   f"revision {manifest.get('revision', 'không rõ')}")
    if not ready:
        st.info("Chuẩn bị dữ liệu trong terminal rồi tải lại trang. Lần đầu sẽ tải parquet và các mô hình lớn.")
        st.code(r".venv\Scripts\python.exe -m demo.prepare --count 100", language="powershell")


manifest_path = DEFAULT_DATA_DIR / "manifest.json"
manifest = read_manifest(DEFAULT_DATA_DIR) if manifest_path.is_file() else None
ready = bool(manifest and manifest.get("features_ready"))
dataset_label = f"{manifest.get('count', 0)} câu hỏi GTSQA" if ready else "Đồ thị minh họa"
dataset_dot = "rbf-dot" if ready else "rbf-dot rbf-dot-muted"
model_label = "DeepSeek đã cấu hình" if deepseek_ready() else "DeepSeek chưa cấu hình"
model_dot = "rbf-dot" if deepseek_ready() else "rbf-dot rbf-dot-muted"
st.markdown(dedent(f"""
<section class="rbf-hero">
  <div class="rbf-eyebrow">Knowledge Graph QA &nbsp; / &nbsp; Research Workbench</div>
  <h1>Evidence Explorer</h1>
  <p>Đặt câu hỏi, truy xuất các quan hệ liên quan và kiểm tra từng đường suy luận trên đồ thị bằng chứng.</p>
  <div class="rbf-pills">
    <span class="rbf-pill"><span class="{dataset_dot}"></span>{dataset_label}</span>
    <span class="rbf-pill"><span class="rbf-dot"></span>RBF checkpoint sẵn sàng</span>
    <span class="rbf-pill"><span class="{model_dot}"></span>{model_label}</span>
  </div>
</section>"""), unsafe_allow_html=True)
retrieve_tab, chat_tab, evaluation_tab, status_tab = st.tabs([
    "🔎 Truy xuất", "💬 Chatbot", "📊 Đánh giá", "⚙️ Dữ liệu & trạng thái",
])
preview_manifest = {"questions": [{
    "id": "__preview__", "question": "Where was Marie Curie born?",
    "n_hops": 1, "n_triples": 12,
}]}

with retrieve_tab:
    if ready:
        retrieve_view(manifest)
    else:
        retrieve_view(preview_manifest, preview=True)
with chat_tab:
    if ready:
        chat_view(manifest)
    else:
        chat_view(preview_manifest, preview=True)
with evaluation_tab:
    if ready:
        evaluation_view(manifest)
    else:
        st.subheader("Đánh giá tập mẫu")
        st.info("Cần chuẩn bị dữ liệu trước khi chạy đánh giá.")
with status_tab:
    status_view(manifest)

st.caption("Nguồn dữ liệu: Graphcore/GTSQA · CC BY 4.0 · Cattaneo et al. (2025)")
