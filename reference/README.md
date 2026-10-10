# Research reference code (read-only)

Verbatim copies or excerpts from the SubgraphRAG research repo (`github.com/minhnvm2307/Sub-HigraphRAG`, branch `rel_rag`,
commit `488f574` plus local changes, copied 2026-10-11). They define the behaviour the app must reproduce
(`docs/RETRIEVER_SPEC.md`). **Do not import them from the app.** Port them into `api/fgr/retriever/` and check the port with
the parity tests.

| File | Source | What to take from it |
|---|---|---|
| `retriever_model.py` | `retrieve/src/model/retriever.py` | `PEConv` (weighted-mean message passing), `DDE`, `Retriever.forward`: feature layout, topic PE, plain + weighted DDE, relation-topic mask, MLP. Uses torch_geometric; the app re-implements `PEConv` with `index_add_` |
| `sg_adr_infer.py` | `retrieve/sg_adr_infer.py` | The four modes (base / aer / adr / adr_aer), B = 3 bridges from the top-3 triples, anchor-set memo, z-score max fusion, top-500 output |
| `rt_features.py` | excerpt of `retrieve/src/dataset/retriever.py` | `STOPWORDS`, `SYNONYM_GROUPS`, `clean_text`, `get_token_similarity`, `compute_lexical_score`; static helpers for score mixing and the top-k mask (`RTHelpers`); the live-feature branch and topic one-hot (as comments) |
| `gte_large_en.py` | `retrieve/src/model/text_encoders/gte_large_en.py` | Encoder settings: CLS pooling, L2 normalisation, max_length 8192 |
| `eval_retrieval.py` | `retrieve/eval.py` | Metric definitions: `ans_recall@k` = \|answers ∩ entities of top-k triples\| / \|answers\| |

The WL checkpoint these files load is in the S3 bundle (`models/wl_effective_num_s42/`); see `docs/DATA.md`.
