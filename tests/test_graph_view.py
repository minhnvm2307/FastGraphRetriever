"""The graph iframe preserves the evidence semantics and embeds its own assets."""

from demo.graph_view import render_graph


def test_render_graph_is_directed_and_interactive():
    triples = [
        {"index": 1, "id": "T1", "head": "Ada (Q1)", "relation": "educated by (P1)",
         "tail": "Teacher (Q2)", "score": 1.23},
        {"index": 2, "id": "T2", "head": "Teacher (Q2)", "relation": "works at (P2)",
         "tail": "School (Q3)", "score": 0.8},
    ]
    html = render_graph(triples, {1}, {"Ada (Q1)"}, {"School (Q3)"}, selected_index=1)
    assert '"arrows": {"to": {"enabled": true' in html
    assert '"label": "educated by (P1)"' in html
    assert '"rank": "T1"' in html
    assert '"color": "#ed8b3c"' in html
    assert 'id="graph-search"' in html and 'id="graph-fit"' in html
    assert 'network.on(\'click\'' in html
    assert "cdnjs.cloudflare.com" not in html
