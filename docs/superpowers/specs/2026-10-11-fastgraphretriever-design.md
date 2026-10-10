# FastGraphRetriever — technical design

*2026-10-11. Companion to `docs/PRD.md`. Algorithm details: `docs/RETRIEVER_SPEC.md`. Data: `docs/DATA.md`.
API contract: `docs/API.md`. Agent and extraction: `docs/AGENT_SPEC.md`. UI: `docs/UI_SPEC.md`.*

## 1. Decisions taken during brainstorming (with the user, 2026-10-11)

| Decision | Choice | Reason |
|---|---|---|
| Language | Vietnamese documents and chat; the LLM normalises entities, relations and questions to **English** | The WL checkpoint uses GTE-large-en-v1.5 (English only). No retraining |
| LLM | DeepSeek (`deepseek-flash`, OpenAI-compatible) behind a provider interface; **$0.05 hard cap**; mock by default | Budget; no local GPU contention |
| Main deliverable | The **fast graph retriever**; medical extraction is only demoed | User priority |
| Demo graph | GrailQA-ODA Freebase subgraphs (897 questions, 935k entities, 1.8M triples) | Gold answers make the method's advantage visible; GTE embeddings already existed (no re-encoding) |
| Architecture | **A + benchmark mode**: Neo4j is the single source of truth for the chat path (Cypher 2-hop with caps); a benchmark source replays the exact research subgraphs | Real GraphRAG on Neo4j, plus a faithful comparison with the paper numbers |
| Embedding store | float16 memmaps on disk, not Neo4j vector properties | 1.2 GB vs ~2.5 GB+index in Neo4j; gathers by row are faster than Cypher property reads |
| Data delivery | Versioned bundle on **S3** + checksummed manifest in git | User preference (S3 over Git LFS) |
| Dev location | Another machine. This repo holds docs, the data manifest, research reference code and tools only | User instruction |

## 2. Architecture

```
                 ┌──────────────────────────── browser (React SPA) ────────────────────────────┐
                 │ Chat · Retrieval Lab · Benchmark · Ingest · Graph explorer                  │
                 └───────────────┬───────────────────────────────────────────▲──────────────────┘
                                 │ REST + SSE (/api/*)                        │ static files
                 ┌───────────────▼────────────────────────────────┐   ┌──────┴──────┐
                 │ api  (FastAPI, Python 3.11, torch CPU)          │   │ web (nginx) │  proxies /api → api
                 │                                                │   └─────────────┘
                 │  routes ── agent ── llm (deepseek | mock) ── budget ledger
                 │     │         │
                 │     │         └── tools: link_entities · retrieve_subgraph · expand_entity
                 │     ▼
                 │  RetrievalPipeline
                 │   encoder(GTE, CPU) → linker → SubgraphSource → RT features → WLScorer(factorised)
                 │                                 │  (Neo4jSource | BenchmarkSource)    → MultiPass(ADR/AER)
                 │  EmbeddingStore (memmap bundle + overlay)    ingest (extract → commit)
                 └───────────────┬────────────────────────────────┘
                                 │ bolt
                 ┌───────────────▼──────────────┐      one-off job:  loader (api image)
                 │ neo4j 5.26 community          │◄──── bundle TSV → batched CREATE, constraints, full-text index
                 └──────────────────────────────┘
   volumes: ./data/bundle (ro) · ./data/runtime (rw) · neo4j_data · hf_cache (GTE weights)
```

### 2.1 Containers (`docker-compose.yml`, to be written in M0)

| Service | Image | Ports (host, bound to 127.0.0.1) | Limits | Notes |
|---|---|---|---|---|
| `neo4j` | `neo4j:5.26-community` | 7474 (browser), 7687 (bolt). Override with `NEO4J_HTTP_PORT`/`NEO4J_BOLT_PORT` | `mem_limit: 2g`; `NEO4J_server_memory_heap_max__size=1g`, `..._pagecache_size=512m` | Healthcheck `cypher-shell 'RETURN 1'` |
| `api` | built from `api/Dockerfile` (python:3.11-slim + CPU torch wheel) | 8000 | `mem_limit: 3.5g`, `cpus: 4` | `HF_HUB_OFFLINE=1` after the first download; mounts `data/bundle:ro`, `data/runtime`, `hf_cache` |
| `web` | built from `web/Dockerfile` (node:20 build → nginx:alpine) | 8080 | `mem_limit: 256m` | Serves the SPA; `/api` reverse proxy with SSE buffering off |
| `loader` | the `api` image, `profiles: [tools]` | – | `mem_limit: 4g` | `docker compose run --rm loader --full` |

The research machine is shared (other users' Neo4j, ES, etc.). On the dev machine the default ports are fine; everything
binds to 127.0.0.1 so nothing is exposed by accident.

### 2.2 Repository layout (target)

```
FastGraphRetriever/
├── docker-compose.yml  .env.example  Makefile  README.md
├── api/
│   ├── pyproject.toml            # fastapi, uvicorn, neo4j, torch (cpu), transformers==4.43.2, numpy<2, pydantic-settings, sse-starlette, openai
│   ├── Dockerfile
│   ├── fgr/
│   │   ├── config.py             # Settings (env), price table, caps
│   │   ├── store/embeddings.py   # EmbeddingStore: bundle memmaps + overlay
│   │   ├── store/graph.py        # Neo4j driver wrapper + Cypher
│   │   ├── encoder/gte.py        # GTE CPU encoder, warm-up, LRU
│   │   ├── retriever/features.py # RT features (port of reference/rt_features.py)
│   │   ├── retriever/dde.py      # PEConv / DDE in pure torch
│   │   ├── retriever/scorer.py   # WLScorer: load cpt, reference_forward, prepare/score_pass (factorised)
│   │   ├── retriever/multipass.py# base/adr/aer/adr_aer, memo, z-max fusion, provenance
│   │   ├── retriever/sources.py  # SubgraphSource protocol, Neo4jSource, BenchmarkSource
│   │   ├── retriever/pipeline.py # RetrievalPipeline (+ stage timings)
│   │   ├── linking.py            # full-text + exact boost + embedding re-rank
│   │   ├── llm/                  # provider protocol, deepseek, mock, budget ledger
│   │   ├── agent/                # prompts, tools, loop
│   │   ├── ingest/               # extract (LLM + schema), commit (embed + Neo4j + overlay)
│   │   ├── bench/                # ansR, batch runner, cache
│   │   ├── loader/__main__.py    # bundle → Neo4j
│   │   └── api/                  # FastAPI app + routers
│   └── tests/                    # unit, parity, integration, api, agent
├── web/                          # Vite + React + TS (see UI_SPEC.md)
├── data_manifest/bundle_v1.json  # checksums of the S3 bundle (in git)
├── reference/                    # research code excerpts, read-only
├── samples/medical_vi/           # synthetic notes + reference extractions
├── tools/                        # s3_bundle.py, build/verify/bench (research-machine provenance)
├── docs/                         # PRD, specs, plans, evidence
└── .claude/skills/               # frontend-design, webapp-testing (vendored, Apache-2.0)
```

Module rules: `retriever/*` never imports `api`, `agent` or `llm`. `agent` talks to retrieval only through the tool
functions. `store/graph.py` is the only module that contains Cypher.

## 3. Neo4j data model

```
(:Entity {eid:int, key:str, name:str, emb_row:int, kg:str, is_cvt:bool,
          name_vi:str?, etype:str?, source_doc:str?})                      # name_vi/etype/source_doc: medical only
(:Entity)-[:R {rid:int, name:str}]->(:Entity)                            # one rel type; rid indexes the relation table
(:Relation {rid:int, name:str, kg:str})                                  # 4,570 nodes: lookup + browsing, not traversed
```

- Constraints and indexes: `CREATE CONSTRAINT entity_eid FOR (e:Entity) REQUIRE e.eid IS UNIQUE`; `CREATE INDEX entity_key FOR (e:Entity) ON (e.key)`; `CREATE INDEX entity_kg FOR (e:Entity) ON (e.kg)`; `CREATE FULLTEXT INDEX entity_names FOR (e:Entity) ON EACH [e.name, e.name_vi]`; `CREATE CONSTRAINT relation_rid FOR (r:Relation) REQUIRE r.rid IS UNIQUE`.
- `is_cvt = (name == 'unnamed_entity')`. CVTs are excluded from linking and drawn as small hollow nodes.
- A single relationship type `:R` keeps Cypher simple. Freebase has 4,570 relation names with dots, which would need backticks as types.
- `kg` partitions the graph (`grailqa`, `medical`); every query filters on it.
- Load: batched `UNWIND $rows AS row CREATE ...` (10k rows/batch, parameters only, no string-built Cypher) into an empty database. Nodes first, then constraints, then relationships matched by `eid`. Expect ~3–6 min for the full graph on a laptop. `--slice N` loads only the union of N benchmark questions' subgraphs (via `local2eid`) for fast dev.

## 4. Main flows

### 4.1 Retrieve (`POST /api/retrieve`)

1. `encode`: GTE(question_en). Benchmark questions use `emb/q.f16.npy` instead, which makes them exact and fast.
2. `link`: if anchors are not given, link each mention (`linking.py`). Up to 3 anchors are used.
3. `fetch`: `SubgraphSource.fetch(anchors, kg, caps, priority_rids)` → entities, triples. Neo4jSource computes `priority_rids` from RT scores over the KG's relation table first.
4. `features`: RT features over the subgraph's relations.
5. `prepare`: the factorised `base` matrix (once).
6. `passes`: MultiPass runs the mode, memoising anchor sets, and records bridges and per-pass timings.
7. `fuse` + top-k → response with provenance and `timings_ms` per stage.

### 4.2 Chat (`POST /api/chat`, SSE)

```
user msg ─► [LLM call 1: normalise] ─► {lang, question_en, mentions_en[], intent}
         ─► link_entities(mentions) ─► retrieve_subgraph(question_en, anchors, mode, k)   (no LLM)
         ─► [LLM call 2: answer with top-N triples as [T#]] ─► tokens …
         └► if the answer model requests one tool (expand_entity / retrieve again): run it, [LLM call 3: final]
```

- At most 3 LLM calls per turn, normally 2.
- Before each call, the budget guard estimates the cost. If the call would exceed the cap, the turn ends with a **retrieval-only answer**: top candidate entities and their triples, rendered without an LLM.
- Details and prompts: `AGENT_SPEC.md`.

### 4.3 Ingest (medical)

`POST /api/ingest/extract` → cache hit (sha256 of the text) or a sample fixture → otherwise an LLM extraction (guarded)
→ JSON validated against the schema → preview. `POST /api/ingest/commit` → assign eids/rids above the current max →
GTE-embed new names and relations (batched) → append to the overlay memmaps → `UNWIND CREATE` nodes and relationships with
`kg='medical'` → return the counts. `DELETE /api/ingest/medical` removes `kg='medical'` and clears the overlay.

## 5. Configuration (`.env`)

| Variable | Default | Meaning |
|---|---|---|
| `NEO4J_URI` / `NEO4J_USER` / `NEO4J_PASSWORD` | `bolt://neo4j:7687` / `neo4j` / (required) | |
| `BUNDLE_DIR` | `/data/bundle` | Mounted read-only |
| `RUNTIME_DIR` | `/data/runtime` | Ledger, overlay, caches |
| `TORCH_THREADS` | 4 | `torch.set_num_threads` |
| `LLM_PROVIDER` | `mock` | `mock` or `deepseek` |
| `DEEPSEEK_API_KEY` / `DEEPSEEK_BASE_URL` / `DEEPSEEK_MODEL` | – / `https://api.deepseek.com` / `deepseek-flash` | |
| `LLM_BUDGET_USD` | 0.05 | Hard cap over the ledger's lifetime |
| `LLM_PRICE_IN_MISS` / `LLM_PRICE_IN_HIT` / `LLM_PRICE_OUT` | 0.30 / 0.006 / 1.20 | USD per 1M tokens (DeepSeek peak prices, 10-2026; check before a live demo) |
| `RETRIEVE_MAX_TRIPLES` / `HOP1_CAP` / `FANOUT_CAP` / `RT_PRIORITY` | 10000 / 5000 / 200 / 50 | Neo4jSource caps |
| `AWS_*` | – | Only for `tools/s3_bundle.py` |

## 6. Error handling

| Condition | Behaviour |
|---|---|
| Neo4j unreachable | `/health` shows `neo4j: down`. Graph routes return 503 `NEO4J_UNAVAILABLE`. The benchmark source still works (it needs no Neo4j). The UI shows a banner |
| No anchor linked | 422 `NO_ANCHORS` with the candidate list. Chat asks the user to pick an entity (chips) |
| Subgraph bigger than the caps | Truncate; `truncated: true`; the UI shows a notice |
| Subgraph empty | 200 with empty triples and a hint |
| Encoder not ready | 503 `ENCODER_LOADING` (warm-up takes ~10–30 s at start) |
| LLM error / timeout (30 s) | Retry once, then a retrieval-only answer with `llm_error` set |
| Over budget | 402 `BUDGET_EXCEEDED` for direct LLM routes; chat degrades to retrieval-only; the UI meter turns red |
| Invalid extraction JSON | One repair attempt, then 422 with the raw text in the preview for manual edit |

All errors use `{"error": {"code", "message", "hint"}}`. Messages are plain language, with VI/EN chosen by `Accept-Language`.

## 7. Testing strategy

| Layer | Tooling | What |
|---|---|---|
| Unit | pytest | features, dde, scorer (factorised vs reference_forward), multipass logic (toy graphs with known bridges), z-fusion, ledger maths, linking ranking, extraction schema |
| Parity | pytest + bundle | quick golden (30 questions × 4 modes) in CI; full 897 via `make parity-full` (`RETRIEVER_SPEC.md` §7) |
| Integration | pytest + compose Neo4j (`--slice 50`) | loader counts, search, Neo4jSource caps/order, `/api/retrieve` end to end, G4 check (`make g4`, 200 questions on the full graph) |
| API contract | pytest + httpx + pydantic | Every route against `API.md` shapes, including SSE event order |
| Agent | pytest + mock provider | Normalise → link → retrieve → answer; budget refusal → retrieval-only; tool round limit |
| Web | vitest + Testing Library | Mode toggle, citation ↔ triple highlight, budget meter states, i18n |
| UI smoke | Playwright via `.claude/skills/webapp-testing` | Screenshots of the 5 pages in light/dark, no console errors |
| Performance | `make bench-latency` | G2 numbers per stage |

TDD applies (superpowers:test-driven-development). Parity tests come first in M2, because every UI feature trusts the retriever.

## 8. Performance budget (G2, 4 threads, warm)

| Stage | Budget p50 | Evidence |
|---|---|---|
| GTE encode | 450 ms | 607 ms measured at 2 threads on a loaded machine |
| Linking (full-text, ≤ 3 mentions) | 50 ms | – |
| Neo4j fetch (≤ 10k triples) | 200 ms | measure in M3 |
| RT features + prepare | 60 ms | 9 + 41 ms measured |
| Passes (≤ 6) | 60 ms | 12 ms/pass measured |
| Fuse + serialise | 20 ms | – |
| **Total** | **≤ 800 ms** | |

## 9. Version pins (what the parity evidence was produced with)

| Component | Version | Note |
|---|---|---|
| Python | 3.10 on the research machine; 3.11 in the API image | No version-specific code |
| torch | 2.1.0 (research). Use a 2.x **CPU** wheel in the image | Re-run the quick golden test if the major/minor changes |
| transformers | **4.43.2** | The `new-impl` remote code of GTE is sensitive to transformers upgrades; keep this pin unless parity is re-checked |
| numpy | 1.26.4 (`numpy<2`) | Bundle `.npy` files load on any version; the pin avoids torch/numpy ABI warnings |
| Neo4j | 5.26 community | LTS line; full-text index syntax as in §3 |
| Node | 20 LTS (web build) | |
