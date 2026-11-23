"""Interactive, Neo4j Browser-inspired view of a ranked evidence subgraph."""

from __future__ import annotations

import json
from html import escape

from pyvis.network import Network


def _label(value: str, limit: int = 34) -> str:
    value = value.replace("<", "‹").replace(">", "›").strip()
    return value if len(value) <= limit else value[: limit - 1] + "…"


def render_graph(
    triples: list[dict], path_indices: set[int], topics: set[str],
    answers: set[str], selected_index: int | None = None,
) -> str:
    """Return a self-contained graph document for a Streamlit component iframe."""
    network = Network(height="600px", width="100%", directed=True, cdn_resources="in_line")
    path_nodes = {
        name for triple in triples if triple["index"] in path_indices
        for name in (triple["head"], triple["tail"])
    }
    names = list(dict.fromkeys(
        name for triple in triples for name in (triple["head"], triple["tail"])
    ))
    node_ids = {name: f"n{i}" for i, name in enumerate(names)}
    focus_node = next(
        (node_ids[triple["tail"]] for triple in triples if triple["index"] == selected_index),
        None,
    )
    for name in names:
        if name in topics:
            color, border, size, role = "#f59e5b", "#c66220", 27, "Topic entity"
        elif name in answers:
            color, border, size, role = "#77c9ae", "#248c70", 24, "Annotated answer"
        elif name in path_nodes:
            color, border, size, role = "#7eaee8", "#3576bd", 22, "Best path"
        else:
            color, border, size, role = "#b9c7d7", "#71849b", 15, "Retrieved entity"
        network.add_node(
            node_ids[name], label=_label(name),
            entity=name.replace("<", "‹").replace(">", "›"),
            title=f"<b>{escape(name)}</b><br>{role}",
            shape="dot", size=size,
            color={"background": color, "border": border,
                   "highlight": {"background": "#ffdd84", "border": "#cf8a18"}},
            borderWidth=2, font={"face": "Arial", "size": 18, "color": "#24334c",
                                 "strokeWidth": 3, "strokeColor": "#ffffff"},
        )
    for triple in triples:
        index = triple["index"]
        is_path = index in path_indices
        is_selected = index == selected_index
        relation = str(triple["relation"])
        score = triple.get("score")
        rank = triple.get("id", "Path context")
        detail = f"{escape(str(rank))} · {escape(relation)}"
        if score is not None:
            detail += f"<br>RBF score: {float(score):.4f}"
        network.add_edge(
            node_ids[triple["head"]], node_ids[triple["tail"]],
            id=f"e{index}", label=_label(relation, 25),
            title=detail, relation=relation.replace("<", "‹").replace(">", "›"),
            rank=str(rank),
            score=f"{float(score):.4f}" if score is not None else "—",
            color={"color": "#ed8b3c" if is_path else "#aab8c8",
                   "highlight": "#d6611b", "hover": "#d6611b"},
            width=4 if is_selected else 3 if is_path else 1.4,
            arrows={"to": {"enabled": True, "scaleFactor": 0.55}},
            font={"face": "Arial", "size": 13, "color": "#596b80",
                  "strokeWidth": 4, "strokeColor": "#ffffff", "align": "middle"},
            smooth={"enabled": True, "type": "dynamic", "roundness": 0.12},
        )
    network.set_options('''{
      "autoResize": true,
      "interaction": {"hover": true, "multiselect": false,
        "navigationButtons": false, "keyboard": true,
        "tooltipDelay": 140, "zoomView": true, "dragView": true},
      "nodes": {"chosen": true, "shadow": {"enabled": true,
        "color": "rgba(42, 66, 94, .13)", "size": 8, "x": 0, "y": 3}},
      "edges": {"selectionWidth": 2, "hoverWidth": 1.6,
        "smooth": {"enabled": true, "type": "dynamic"}},
      "physics": {"enabled": true,
        "barnesHut": {"gravitationalConstant": -4700, "centralGravity": 0.18,
          "springLength": 150, "springConstant": 0.035, "damping": 0.34},
        "stabilization": {"enabled": true, "iterations": 90, "updateInterval": 30}}
    }''')
    page = network.generate_html()
    css = """<style>
      body {margin:0; background:#f7f9fc; font-family:Arial,sans-serif; color:#26364c}
      #mynetwork {border:1px solid #dce4ee; border-radius:0 0 12px 12px;
        background:radial-gradient(circle at 50% 45%, #ffffff 0%, #f7f9fc 72%)}
      .graph-tools {display:flex; flex-wrap:wrap; align-items:center; gap:9px;
        padding:10px 12px; border:1px solid #dce4ee; border-bottom:0;
        border-radius:12px 12px 0 0; background:#fff}
      .graph-tools input[type=search] {flex:1; min-width:150px; padding:8px 10px;
        border:1px solid #cfdae6; border-radius:7px; font:inherit}
      .graph-tools button {padding:8px 11px; border:1px solid #cfdae6;
        border-radius:7px; background:#f7f9fc; color:#26364c; cursor:pointer}
      .graph-tools button:hover {background:#eaf1f8}
      .graph-tools label {font-size:12px; white-space:nowrap; color:#53657d}
      #graph-detail {min-height:18px; padding:8px 13px; font-size:12px;
        color:#53657d; border:1px solid #dce4ee; border-top:0;
        border-radius:0 0 10px 10px; background:#fff}
    </style>"""
    toolbar = """<div class="graph-tools">
      <input id="graph-search" type="search" placeholder="Tìm entity trong graph…" aria-label="Tìm entity">
      <button id="graph-zoom-in" type="button" aria-label="Phóng to">＋</button>
      <button id="graph-zoom-out" type="button" aria-label="Thu nhỏ">−</button>
      <button id="graph-path" type="button">◎ Best path</button>
      <button id="graph-fit" type="button">⛶ Vừa khung</button>
      <label><input id="graph-labels" type="checkbox" checked> Tên quan hệ</label>
    </div>"""
    script = """<div id="graph-detail">Chọn node hoặc cạnh để xem chi tiết · Kéo để di chuyển · Cuộn để zoom</div>
    <script>
    const detail = document.getElementById('graph-detail');
    const search = document.getElementById('graph-search');
    document.getElementById('graph-fit').onclick = () => network.fit({animation:{duration:350}});
    document.getElementById('graph-path').onclick = () => {
      if (focusNode) network.focus(focusNode, {scale:0.95, animation:{duration:350}});
    };
    document.getElementById('graph-zoom-in').onclick = () => network.moveTo({
      scale: network.getScale() * 1.25, animation:{duration:250}});
    document.getElementById('graph-zoom-out').onclick = () => network.moveTo({
      scale: network.getScale() / 1.25, animation:{duration:250}});
    search.oninput = () => {
      const query = search.value.trim().toLocaleLowerCase();
      if (!query) { network.unselectAll(); return; }
      const found = nodes.get().find(n => n.entity.toLocaleLowerCase().includes(query));
      if (found) {
        network.selectNodes([found.id]);
        network.focus(found.id, {scale:1.1, animation:{duration:350}});
        detail.textContent = found.entity;
      } else { detail.textContent = 'Không tìm thấy entity.'; }
    };
    document.getElementById('graph-labels').onchange = event => {
      const size = event.target.checked ? 13 : 0;
      edges.update(edges.get().map(edge => ({id:edge.id,
        font:{...edge.font, size:size}})));
    };
    network.on('click', event => {
      if (event.nodes.length) {
        const node = nodes.get(event.nodes[0]);
        detail.textContent = node.entity;
      } else if (event.edges.length) {
        const edge = edges.get(event.edges[0]);
        detail.textContent = `${edge.rank}: ${nodes.get(edge.from).entity} → ${edge.relation} → ${nodes.get(edge.to).entity} · score ${edge.score}`;
      } else { detail.textContent = 'Chọn node hoặc cạnh để xem chi tiết · Kéo để di chuyển · Cuộn để zoom'; }
    });
    network.once('stabilizationIterationsDone', () => {
      network.setOptions({physics:false});
      if (focusNode) network.focus(focusNode, {scale:0.95, animation:{duration:350}});
      else network.fit({animation:{duration:350}});
    });
    </script>"""
    script = script.replace("const detail =", f"const focusNode = {json.dumps(focus_node)};\n    const detail =", 1)
    return page.replace("</head>", css + "</head>").replace("<body>", "<body>" + toolbar).replace("</body>", script + "</body>")
