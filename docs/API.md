# API — FastGraphRetriever HTTP contract

Base path `/api`. JSON, UTF-8. Streaming routes use Server-Sent Events (`text/event-stream`; each event is `event: <name>`
followed by `data: <json>`). Implement the shapes as pydantic models in `api/fgr/api/schemas.py`. The contract tests in
`api/tests/api/` check every route against this file.

Errors: HTTP 4xx/5xx with `{"error": {"code": "NO_ANCHORS", "message": "…", "hint": "…", "details": {}}}`.

## 1. Shared types

```ts
type KG = "grailqa" | "medical";
type Mode = "base" | "adr" | "aer" | "adr_aer";
type Source = "neo4j" | "benchmark";

interface EntityRef { eid: number; name: string; key?: string; kg?: KG; name_vi?: string; is_cvt?: boolean }
interface RelationRef { rid: number; name: string; label?: string }      // label: humanised last segment, e.g. "producing company"

interface Pass {
  id: number;                       // 0..P-1, order of execution
  kind: "base" | "bridge" | "single" | "single+bridge";
  anchors: number[];                // eids
  bridges_added: number[];          // eids added by this pass (bridge kinds only)
  ms: number;
}

interface ScoredTriple {
  rank: number;                     // 1-based
  h: EntityRef; r: RelationRef; t: EntityRef;
  score: number;                    // fused score (raw logit when there is a single pass, else max z)
  best_pass: number;                // Pass.id with the highest z for this triple
  is_answer?: boolean;              // benchmark only: h or t is a gold answer
}

interface Timings { encode: number; link: number; fetch: number; features: number; prepare: number;
                    passes: number; fuse: number; total: number }        // milliseconds

interface RetrievalResult {
  question_en: string; kg: KG; mode: Mode; k: number; source: Source;
  anchors: EntityRef[];
  bridges: EntityRef[];             // union over passes
  passes: Pass[];
  triples: ScoredTriple[];          // length ≤ k
  subgraph: { n_entities: number; n_triples: number; truncated: boolean; caps: Record<string, number> };
  timings_ms: Timings;
  gold?: { answers: EntityRef[]; ans_recall: { "50": number; "100": number; "200": number }; hit_at_k: boolean };
}
```

## 2. System

| Method | Path | Response |
|---|---|---|
| GET | `/health` | `{status: "ok"\|"degraded", neo4j: "up"\|"down", encoder: "ready"\|"loading"\|"error", scorer: "ready", bundle_version: "v1", graph: {grailqa: {entities, triples}, medical: {entities, triples}}}` |
| GET | `/config` | `{modes: Mode[], k_options: [50,100,200,500], default_mode: "adr_aer", default_k: 100, llm: {provider, model, live_allowed: bool}, budget: Budget, caps: {...}}` |
| GET | `/budget` | `Budget = {cap_usd, spent_usd, remaining_usd, calls, last_call_at}` |

## 3. Graph

| Method | Path | Request | Response |
|---|---|---|---|
| GET | `/entities/search` | `?q=&kg=grailqa&limit=10&include_cvt=false` | `{items: (EntityRef & {score: number; degree: number})[]}` |
| GET | `/entities/{eid}` | – | `EntityRef & {degree, relations: {rid, name, count}[]}` |
| GET | `/entities/{eid}/neighbors` | `?limit=50&rid=` | `{center: EntityRef, triples: {h, r, t}[], truncated}` |
| GET | `/kg/{kg}/stats` | – | `{entities, triples, relations, top_degree: EntityRef & {degree}[]}` |

## 4. Retrieval

`POST /retrieve`

```json
{ "question_en": "which play is produced by the illusion?",
  "kg": "grailqa",
  "anchors": [0],                 // optional; eids
  "mentions": ["The Illusion"],   // optional; used when anchors are absent
  "mode": "adr_aer", "k": 100,
  "source": "neo4j",              // "benchmark" requires benchmark_id
  "benchmark_id": null,
  "caps": {"max_triples": 10000}  // optional overrides within server limits
}
```

→ `200 RetrievalResult`. Errors: `422 NO_ANCHORS` (`details.candidates` = per-mention search results), `404 UNKNOWN_BENCHMARK_ID`,
`503 NEO4J_UNAVAILABLE`, `503 ENCODER_LOADING`.

`POST /retrieve/compare`: same body with `"modes": ["base", "adr_aer"]` → `{results: {base: RetrievalResult, adr_aer: RetrievalResult},
diff: {gained: ScoredTriple[], lost: ScoredTriple[], moved: {triple, from_rank, to_rank}[]}}`. The two modes share the
encode, fetch and prepare stages (computed once).

## 5. Benchmark

| Method | Path | Request | Response |
|---|---|---|---|
| GET | `/benchmark/questions` | `?q=&hops=1,2&level=zero-shot&function=none&domain=&offset=0&limit=50` | `{total, items: {id, row, question, topic_entities, answers, n_hops, level, function, domain, n_triples}[]}` |
| GET | `/benchmark/questions/{id}` | – | one item plus `topic_eids`, `answer_eids` |
| POST | `/benchmark/run` (SSE) | `{ids?: string[], n?: number, sample: "first"\|"linspace"\|"random", seed?: 0, modes: Mode[], source: Source, ks: [50,100,200]}` | events below |
| GET | `/benchmark/reference` | – | the reference table of DATA.md §2.4 + the baselines (SubgraphRAG, REL-RAG) from BENCHMARK.md |

`/benchmark/run` events: `progress {done, total, current_id}` … then `result {per_mode: {mode: {ans_recall: {"50","100","200"}, by_hops: {...}, n}}, elapsed_ms, cached: bool}`.
Results are cached under `RUNTIME_DIR/bench_cache/` by hash(ids, modes, source, bundle version).

## 6. Chat

`POST /chat` (SSE)

```json
{ "session_id": "uuid", "message": "Vở kịch nào do The Illusion sản xuất?", "kg": "grailqa",
  "mode": "adr_aer", "k": 100, "llm": "mock" | "deepseek", "history": [{"role": "user"|"assistant", "content": "…"}] }
```

Events, in order:

| Event | Data |
|---|---|
| `status` | `{stage: "normalising"\|"linking"\|"retrieving"\|"answering"\|"tool", message}` (repeatable) |
| `plan` | `{lang: "vi"\|"en", question_en, mentions_en: string[], intent}` |
| `anchors` | `{anchors: EntityRef[], candidates: {mention, items: EntityRef[]}[]}` |
| `retrieval` | `RetrievalResult` (the triples given to the LLM are the first `prompt_triples`, default 30, numbered T1..Tn) |
| `tool_call` | `{name, arguments}` / `tool_result {name, summary}` (only for the optional follow-up round) |
| `token` | `{text}` (streamed answer) |
| `citations` | `{map: {"T3": ScoredTriple, …}}` (sent before `done`) |
| `done` | `{answer, usage: {calls, prompt_tokens, completion_tokens, usd}, budget: Budget, degraded: null\|"budget"\|"llm_error"}` |
| `error` | `{code, message}`; the stream then closes |

History is client-held (stateless server). `session_id` only tags ledger entries.

## 7. Ingest (medical)

| Method | Path | Request | Response |
|---|---|---|---|
| GET | `/ingest/samples` | – | `{items: {id, title, chars}[]}` |
| POST | `/ingest/extract` | `{text?: string, sample_id?: string, llm: "mock"\|"deepseek"}` (text ≤ 6,000 chars) | `{extraction: Extraction, cached: bool, usage?, warnings: string[]}` |
| POST | `/ingest/commit` | `{extraction: Extraction, doc_title}` | `{added: {entities, triples, relations}, eids: number[], ms}` |
| DELETE | `/ingest/medical` | – | `{removed: {entities, triples}}` |

`Extraction` is defined in `AGENT_SPEC.md` §5.

## 8. Limits

- `k ≤ 500`; `max_triples ≤ 30000`; `mentions ≤ 5`; `anchors ≤ 5`; chat `message ≤ 2,000` chars; ingest text ≤ 6,000 chars.
- The server enforces `LLM_BUDGET_USD` regardless of client flags. `llm: "deepseek"` is rejected with 403 `LIVE_LLM_DISABLED` unless `LLM_PROVIDER=deepseek` is set server-side.
