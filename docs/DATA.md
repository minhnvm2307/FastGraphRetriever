# DATA — the FastGraphRetriever data bundle

Everything the app needs that is expensive to rebuild (embeddings, the trained retriever, the research outputs used as
a test oracle) ships as one **versioned bundle on S3**. Git only holds the manifest with sha256 checksums
(`data_manifest/bundle_v1.json`), so the code and the data it was tested against cannot drift apart.

## 1. Getting it

```bash
cp .env.example .env            # fill AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY (same keys as the research .env:
                                # AccessKey-Id / SecretAccessKey), AWS_DEFAULT_REGION=us-east-1
pip install boto3
python tools/s3_bundle.py get   # -> data/bundle/, every file sha256-verified, re-runnable (skips good files)
```

- Location: `s3://banktransactiondata/fastgraphretriever/bundle/v1/` (private bucket).
- Size: **1.40 GB** (1.2 GB of it is the entity embedding table). Egress is roughly $0.13 per full download.
- `data/` is gitignored. Never commit bundle files.

The GTE encoder weights are **not** in the bundle. They come from Hugging Face (public, not gated). Pin these revisions:

| Repo | Revision | Why |
|---|---|---|
| `Alibaba-NLP/gte-large-en-v1.5` | `104333d6af6f97649377c2afbde10a7704870c7b` | question / new-entity encoder (1024-d, CLS pooling, L2-normalised) |
| `Alibaba-NLP/new-impl` | `40ced75c3017eb27626c9d4ea981bde21a2662f4` | `trust_remote_code` model code that gte-large-en-v1.5 loads |

## 2. Contents

| Path | Size | What |
|---|---|---|
| `kg/entities.tsv.gz` | 17 MB | 935,050 rows: `eid  key  name  emb_row` |
| `kg/relations.tsv.gz` | <1 MB | 4,570 rows: `rid  name` (Freebase relation ids, e.g. `theater.theater_production.play`) |
| `kg/triples.tsv.gz` | 5 MB | 1,806,486 rows: `h_eid  rid  t_eid` (directed, deduplicated) |
| `emb/ent.f16.npy` | 1.2 GB | `[614324, 1024]` float16. GTE embedding of each **unique entity name** |
| `emb/rel.f16.npy` | 9 MB | `[4570, 1024]` float16. Row `rid` |
| `emb/q.f16.npy` | 2 MB | `[897, 1024]` float16. Row = `row` field of `benchmark/questions.jsonl` |
| `benchmark/test.pkl` | 75 MB | The 897 per-question subgraphs, SubgraphRAG `processed` format (unchanged research input) |
| `benchmark/questions.jsonl` | 0.5 MB | One line per question (fields below) |
| `benchmark/local2eid.npz` | 4 MB | `{question_id: int32[n_entities]}`: global eid of each local entity index of that question |
| `golden/adr_grailqa_oda_{base,adr,aer,adr_aer}.pth` | 4 × 18 MB | Research output (top-500 per question) for each retrieval mode: **the parity oracle** |
| `models/wl_effective_num_s42/cpt.pth` | 16 MB | WL retriever checkpoint (trained on WebQSP, seed 42) |
| `models/wl_effective_num_s42/config.json` | – | Its config (copied out of the checkpoint for readability) |
| `MANIFEST.json` | – | Sizes, sha256, stats, provenance |

### 2.1 TSV format

- UTF-8, gzip, a header line, tab-separated, `\n` line ends.
- No quoting. Tabs, `\r` and `\n` inside names were replaced by spaces at build time, so `line.rstrip('\n').split('\t')` is always correct.
- Open with `gzip.open(p, 'rt', encoding='utf-8', newline='\n')`. **Do not** use universal newlines, and do not use a CSV reader with quote handling: names contain `"`.

### 2.2 Entities: identity vs. name vs. embedding

- `eid`: dense int id, 0..935049. This is the graph identity.
- `key`: Freebase MID (`m.0yrlqjm`), or the literal value / schema id for non-MID nodes (`1999`, `cvg.game_version`). Unique per row.
- `name`: display label. **Not unique.** "The Illusion" is both a play (`m.0yrltsn`) and a production (`m.0yrlqjm`). Every CVT node is called `unnamed_entity`.
- `emb_row`: row in `emb/ent.f16.npy`. Entities with the same name share a row. This is exact, not an approximation: in the research inputs the same name always had the same vector (checked: max abs diff 0.0 over 411k repeats).

Why MIDs matter: merging by name collapses all CVTs into one node of degree 97,499 and fuses 48.6k other repeated names. With MIDs, the maximum degree is 6,092 (`Human genome`) and p99 is 26.

### 2.3 `benchmark/questions.jsonl`

```json
{"row": 0, "id": "2101960008000", "question": "which play is produced by the illusion?",
 "topic_entities": ["The Illusion"], "topic_eids": [0], "answers": ["The Illusion"], "answer_eids": [1],
 "n_triples": 275, "n_entities": 168, "n_hops": 1, "level": "i.i.d.", "function": "none", "domain": "theater"}
```

- `level` ∈ {i.i.d., compositional, zero-shot}.
- `n_hops` ∈ {1, 2, 3, 4} (number of edges in the gold query graph).
- `function` ∈ {none, count, superlative, comparative}.

### 2.4 Golden files (`golden/*.pth`)

These are `torch.load(..., weights_only=False)` dicts produced by `reference/sg_adr_infer.py` on the research GPU:

```python
{question_id: {'question': str,
               'scored_triples': [(h_name, r_name, t_name, score), ...],   # top-500, descending
               'a_entity_in_graph': [answer names],
               'target_relevant_triples': [...]}}                          # empty for GrailQA
```

Reference answer recall (ansR, `reference/eval_retrieval.py` definition: |answers ∩ entities of top-k triples| / |answers|, mean over 897 questions):

| Mode | ansR@50 | @100 | @200 | @50 by hops 1 / 2 / 3+ |
|---|---|---|---|---|
| base (WL) | 83.18 | 86.99 | 91.82 | 94.0 / 59.6 / 27.6 |
| adr | 83.18 | 87.15 | 91.93 | 94.0 / 59.1 / 30.7 |
| aer | 83.09 | 88.88 | 92.86 | 93.7 / 60.3 / 27.7 |
| **adr_aer (WL+ADR+AER)** | **83.42** | **89.16** | **92.97** | 93.7 / 60.3 / 36.8 |

For context (research `BENCHMARK.md`, same 897 questions): SubgraphRAG 71.7 / 79.1 / 86.9 and REL-RAG 70.0 / 76.5 / 79.8.
**Caveat:** this is the ODA dev set (897 questions), not the 874-question GrailQA set of the REL-RAG paper.

## 3. Provenance (how the bundle was built)

Built on 2026-10-11 on the research machine by `tools/build_bundle.py` from:

- `SubgraphRAG/retrieve/data_files/grailqa_oda/processed/test.pkl` and `emb/gte-large-en-v1.5/test.pth` (the exact research inputs).
- `SubgraphRAG/retrieve/webqsp_wl_effective_num_Oct05-15:23:05/` (checkpoint + golden outputs).
- The ODA sources: `hop_2_Freebase/grailqa_{triples,labels}_2_hop_all.json` from HF `CCaccept/Subgraph`, plus `QAdata/graliqa.json` from `github.com/lanjiuqing64/KGdata`. These are used only to recover MIDs: each question's subgraph is rebuilt exactly as the research converter did (`_grail_record`), and the build **aborts unless all 897 match the research file exactly** (names, relations, h/r/t).

Checks run against the finished bundle (`docs/evidence/`):

- `verify_bundle.json`: the original research model, fed only from bundle files, reproduces the golden top-50/top-200 **sets in all 4 modes** (30 sampled questions), with score difference ≤ 2.3e-5 position-wise; same order on 29/30 (one near-tie swap, GPU vs CPU).
- `bench_factorized.json`: the factorised scorer of `RETRIEVER_SPEC.md` §4 matches the original to 1.5e-5. GTE on CPU matches the bundle's question vectors to 6e-5.

Rebuilding (only needed for a v2):

```bash
python -I tools/oda_extract.py graliqa.json grailqa_triples_2_hop_all.json grailqa_labels_2_hop_all.json oda_subset.pkl
python tools/build_bundle.py --qa graliqa.json --oda oda_subset.pkl --out data/bundle
python tools/verify_bundle.py && python tools/bench_factorized.py --gte     # research machine (needs torch_geometric)
python tools/s3_bundle.py put    # then bump the prefix to v2 and commit data_manifest/bundle_v2.json
```

## 4. Runtime data (created by the app, not in the bundle)

| Path | Written by | What |
|---|---|---|
| Neo4j volume `neo4j_data` | loader | The graph (≈1.5 GB on disk for the full import) |
| `data/runtime/llm_ledger.jsonl` | API | Append-only LLM spend ledger (the budget guard reads it on start) |
| `data/runtime/overlay/` | ingest | Embeddings of entities/relations added by medical ingestion (`ent_extra.f16.npy`, `rel_extra.f16.npy`, `index.json`) |
| `data/runtime/extract_cache/` | ingest | Cached LLM extractions keyed by sha256(text), so re-ingesting costs $0 |
| `data/runtime/bench_cache/` | API | Cached benchmark runs |

## 5. Medical samples (in git, not S3)

`samples/medical_vi/` holds **synthetic** Vietnamese clinical notes written for this demo (no real patient data) and their
hand-made reference extractions (`*.extraction.json`), which follow the schema in `AGENT_SPEC.md` §5. The mock LLM replays
them, so the ingestion demo works at $0.
