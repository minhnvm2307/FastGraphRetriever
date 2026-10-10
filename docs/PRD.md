# PRD — FastGraphRetriever

*Version 1.0, 2026-10-11. Owner: minhnv. Status: approved for implementation on the dev machine.*

## 1. Summary

FastGraphRetriever turns the research retriever **WL + ADR + AER** (anchor-equivariant retrieval over an entity graph,
trained on WebQSP) into a working **GraphRAG chat application**:

- A knowledge graph lives in **Neo4j**.
- A question is linked to graph entities.
- The retriever selects the evidence subgraph **fast on CPU**.
- An LLM agent (DeepSeek) answers in Vietnamese or English and cites the triples it used.
- Every answer shows its evidence graph and *why* each triple was retrieved (anchor, bridge, per-anchor pass).

The demo graph is GrailQA's Freebase subgraphs: 935k entities and 1.8M triples, with 897 benchmark questions that have gold answers.
A second, small graph shows the medical use case: synthetic Vietnamese clinical notes turned into triples by an LLM.

## 2. Why

- The research result is strong but only visible in tables. On GrailQA, WL+ADR+AER reaches ansR@50/100/200 = **83.4 / 89.2 / 93.0**, against 71.7 / 79.1 / 86.9 for SubgraphRAG and 70.0 / 76.5 / 79.8 for REL-RAG.
- A live system shows three things a table can't:
  - the retriever works on a real graph database, outside per-question research files;
  - multi-pass retrieval (bridges, per-anchor passes) is fast enough to be interactive;
  - the evidence makes answers checkable, which matters most in a domain like medicine.

## 3. Users and use cases

| Persona | Goal | Key use cases |
|---|---|---|
| **Presenter** (the researcher: thesis defence, lab demo, paper demo) | Show the method working, and show it beats the baselines | UC1 run a benchmark question and compare modes side by side; UC2 show the aggregate recall table live; UC3 chat with evidence |
| **Evaluator** (examiner, colleague) | Try their own question and inspect the evidence | UC3 chat; UC4 pick anchors and modes by hand in the Retrieval Lab; UC5 browse the graph |
| **Domain user** (medical, demo persona) | Ask Vietnamese questions over clinical notes | UC6 ingest a note (VI) → graph; UC3 chat over the medical graph |

## 4. Goals and success metrics

| # | Goal | Metric (acceptance) |
|---|---|---|
| G1 | **Faithful** to the research | BenchmarkSource reproduces ansR@50/100/200 of all 4 modes within ±0.05 on 897 questions (`DATA.md` §2.4). The quick golden test passes (`RETRIEVER_SPEC.md` §7) |
| G2 | **Fast** on CPU | `POST /api/retrieve` (Neo4jSource, adr_aer, k = 100, ≤ 10k triples), 4 API threads, warm: **p50 ≤ 800 ms, p95 ≤ 2.5 s** end to end. Scorer per extra pass p50 ≤ 50 ms. Neo4j fetch p50 ≤ 200 ms. A latency report script prints these |
| G3 | **Explainable** | Each chat answer has its subgraph view, a triple list with score and provenance (pass kind, bridge), and a latency breakdown. Each claim in the answer cites ≥ 1 triple `[T#]` |
| G4 | **Works on the real graph** | Neo4jSource with gold anchors: ansR@100 ≥ BenchmarkSource ansR@100 − 3.0 on 200 benchmark questions |
| G5 | **Cheap** | LLM spend capped at **$0.05 total** by default, enforced by the server (hard refusal past the cap). All automated tests use the mock LLM ($0) |
| G6 | **Light and portable** | `docker compose up` after `tools/s3_bundle.py get`. CPU only. Total RAM ≤ 6 GB (Neo4j 2 GB, API 3.5 GB, web 0.25 GB) |

## 5. Scope

### In scope (v1)

1. Data bootstrap from S3 (bundle v1) with checksums.
2. Loader: bundle → Neo4j (full graph, or a dev slice of N questions).
3. Retriever service: WL scorer (pure torch, factorised fast path), DDE, RT features, base/ADR/AER/ADR+AER, BenchmarkSource + Neo4jSource, entity linking, GTE question encoding on CPU.
4. Agentic chat over SSE: question normalisation VI→EN, linking, retrieval, answer with citations, one optional follow-up tool round. Budget guard and mock LLM.
5. Web app (React): Chat, Retrieval Lab, Benchmark, Ingest (medical), Graph Explorer; Vietnamese and English UI.
6. Medical ingestion demo: Vietnamese note → LLM extraction (EN names + VI labels) → preview → commit to the `medical` graph. Synthetic samples with pre-made extractions ($0 path).
7. Tests: unit, golden parity, integration (Neo4j), API contract, agent with mock LLM, UI smoke screenshots.

### Out of scope (v1)

- Training or fine-tuning any model; GPU serving; the RBF-Ret / ASE research variants.
- WebQSP / CWQ graphs. The bundle format allows them later as `kg` values.
- Multi-user accounts, auth beyond a single shared password, rate limiting for the public internet.
- Real patient data, clinical decision support, medical advice. The UI states this.
- Mobile apps. "React" means a web app; a React Native client is a possible later step over the same API.
- Quality claims for the medical graph. WL was trained on WebQSP/Freebase, so medical retrieval is a zero-shot demo.

## 6. Functional requirements

| ID | Requirement | Priority |
|---|---|---|
| FR-1 | `tools/s3_bundle.py get` downloads the bundle and verifies every sha256 against `data_manifest/bundle_v1.json`. It is re-runnable | P0 |
| FR-2 | Loader CLI imports entities, relations and triples into Neo4j with a unique constraint on `eid` and a full-text index on names. Modes: `--slice N` (subgraphs of N benchmark questions, for dev) and `--full`. Idempotent: refuses to load into a non-empty DB without `--reset` | P0 |
| FR-3 | Entity search: full-text + exact-match boost, filtered by `kg`, returns eid, name, key, kg, degree | P0 |
| FR-4 | Retrieval API: input English question, anchors (eids) or mentions, `kg`, `mode` ∈ {base, adr, aer, adr_aer}, `k` ∈ {50, 100, 200, 500}, `source` ∈ {neo4j, benchmark}. Output per `API.md`: ranked triples with score, provenance, bridges, passes, subgraph size, `truncated`, per-stage timings | P0 |
| FR-5 | Compare: the same question in two modes, with a diff (triples gained/lost, rank changes) | P1 |
| FR-6 | Benchmark: list and filter the 897 questions (hops, level, function, domain, text); run one question in any mode with gold answers marked and ansR@k shown; batch-run N questions with progress (SSE) and aggregate ansR@k; cache results | P0 |
| FR-7 | Chat (SSE): accepts VI or EN; normalises to an English question + mentions (LLM); links; retrieves; answers in the user's language with `[T#]` citations; may do one follow-up tool call (`expand_entity` or a second `retrieve`). Streams status, retrieval result, tokens, citations, usage | P0 |
| FR-8 | LLM providers: `mock` (default, deterministic, $0) and `deepseek` (OpenAI-compatible). Switchable per request when the server allows it | P0 |
| FR-9 | Budget guard: append-only ledger; estimate before each call; refuse when spent + estimate > cap (`LLM_BUDGET_USD`, default 0.05); `GET /api/budget`; UI meter; chat falls back to **retrieval-only** answers when refused | P0 |
| FR-10 | Medical ingest: paste ≤ 6,000 chars of Vietnamese or pick a sample → extraction (cached by text hash; samples come from fixtures) → editable preview → commit to `kg=medical` (new eids, GTE embeddings in an overlay store) → reset button | P1 |
| FR-11 | Graph explorer: search an entity, show its 1-hop neighbourhood, expand on click, open any node in the Lab as an anchor | P1 |
| FR-12 | Health and config: `GET /api/health` (Neo4j, encoder, scorer, bundle version, graph stats), `GET /api/config` | P0 |
| FR-13 | i18n: Vietnamese default, English toggle. Answers follow the language of the user's message | P1 |
| FR-14 | Latency report: `make bench-latency` runs a fixed question set against the running API and prints p50/p95 per stage | P0 |

## 7. Non-functional requirements

| Area | Requirement |
|---|---|
| Performance | G2 targets. Encoder warm-up at start. No per-request model loads. Torch threads configurable (`TORCH_THREADS`, default 4) |
| Resources | Compose limits: neo4j `mem_limit 2g` (heap 1g, pagecache 512m), api `3.5g` + `cpus 4`, web `256m`. No GPU required. The loader runs as a one-off job capped at 4 GB |
| Reliability | Neo4j down → `/health` red, retrieval 503 with a clear message, UI banner. LLM error or over budget → retrieval-only answer. Any cap hit → `truncated=true`, never an error. Retrieval timeout 15 s |
| Security | Ports bound to `127.0.0.1` by default. Secrets only in `.env` (never committed, never logged). Neo4j password required. CORS limited to the web origin. LLM output rendered as sanitised Markdown. Uploads are plain text only, size-capped |
| Privacy and safety | Only synthetic medical text in the repo. A visible "research demo, not medical advice" notice on chat and ingest. No telemetry |
| Reproducibility | Pinned Python/npm dependencies; pinned HF revisions; bundle versioned and checksummed; golden tests in CI |
| Accessibility | WCAG 2.1 AA contrast; keyboard reachable; graph encodings use shape + colour; honours `prefers-reduced-motion` |
| Maintainability | Module boundaries as in the design spec; type hints + mypy on core modules; ruff; tests next to each module |

## 8. Release plan (milestones)

| M | Deliverable | Exit criterion |
|---|---|---|
| M0 | Repo scaffold, compose with Neo4j, bundle fetched | `docker compose up neo4j` healthy; `s3_bundle.py get` OK |
| M1 | Loader (slice + full) | Counts equal to MANIFEST (935,050 / 1,806,486 for full); full-text search works |
| M2 | Retriever core + BenchmarkSource + parity | Quick golden test green; full parity within ±0.05 |
| M3 | Neo4jSource, linking, `/api/retrieve`, latency report | G2 and G4 met |
| M4 | Web: Retrieval Lab + Benchmark + Graph explorer | Smoke screenshots; benchmark aggregate matches API |
| M5 | Agent + Chat (mock first, then DeepSeek within budget) | Mock conversation tests green; one live smoke ≤ $0.01 |
| M6 | Medical ingest (fixtures first, live extraction behind the budget) | Sample notes ingest at $0; chat over `medical` works |
| M7 | UI/UX polish, demo script, docs | Demo script runs end to end in ≤ 10 min |

The step-by-step build plans are `docs/superpowers/plans/2026-10-11-fgr-backend.md` (M0–M3, M5–M6 backend) and `docs/superpowers/plans/2026-10-11-fgr-web.md` (M4, M5 UI, M7).

## 9. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| GTE-large on CPU is the slowest stage (~0.6 s/question measured with 2 threads) | G2 | 4 threads; warm-up; LRU cache; the factorised scorer keeps the rest small. Fallback: ONNX Runtime export, only if parity stays ≤ 1e-3 |
| Hub nodes make 2-hop Cypher slow or truncate answers | G2, G4 | Per-node fan-out cap with RT-priority ordering (`RETRIEVER_SPEC.md` §6); the G4 check measures the loss |
| Entity linking fails on free-text questions (names are not unique; CVTs are `unnamed_entity`) | Chat quality | LLM proposes English mentions; full-text + exact boost; the user can fix anchors in the UI; CVTs are excluded from linking |
| WL is weak on the medical graph (zero-shot, different relation vocabulary) | Medical demo quality | Dotted Freebase-like relation names so the lexical RT feature works; scope the claim as a demo; show evidence so errors are visible |
| DeepSeek tool-calling or pricing changes | FR-7, G5 | Provider interface + mock; the plan → retrieve → answer flow needs only JSON output (no native tool calls); the price table is in config |
| $0.05 is ~25–30 chat turns | Demo length | Retrieval-only mode and benchmark pages need no LLM; raise `LLM_BUDGET_USD` deliberately for a live demo |

## 10. Open questions (decide during build, defaults given)

1. Public demo hosting: default is local only (127.0.0.1 + SSH/VSCode port forwarding).
2. A second graph (CWQ) for the transfer story: default is not in v1. The bundle format supports it as `bundle/v2`.
