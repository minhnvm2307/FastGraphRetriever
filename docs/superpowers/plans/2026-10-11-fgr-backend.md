# FastGraphRetriever Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A dockerised Python API that loads the S3 bundle into Neo4j, retrieves evidence subgraphs with WL + ADR + AER (exactly reproducing the research numbers, fast on CPU), and serves retrieval, benchmark, agentic chat and medical ingestion over REST/SSE.

**Architecture:** FastAPI app (`api/fgr`) around a source-agnostic `RetrievalPipeline`: GTE question encoder → entity linking → `SubgraphSource` (Neo4j 2-hop with caps, or the exact research subgraph) → relation-topic features → factorised WL scorer → multi-pass fusion. Neo4j 5.26 holds the graph; embeddings are float16 memmaps from the bundle plus a runtime overlay. The LLM (DeepSeek or mock) sits behind a budget-guarded provider and is used only by the chat agent and the extractor.

**Tech Stack:** Python 3.11, FastAPI, sse-starlette, pydantic v2 + pydantic-settings, torch 2.x CPU, transformers 4.43.2, numpy<2, neo4j Python driver 5.x, openai SDK (DeepSeek), pytest + httpx, Docker Compose, Neo4j 5.26 community.

**Spec:** `docs/superpowers/specs/2026-10-11-fastgraphretriever-design.md`. It cites `docs/PRD.md`, `docs/RETRIEVER_SPEC.md`, `docs/DATA.md`, `docs/API.md` and `docs/AGENT_SPEC.md`; read all of them. The UI is a separate plan: `docs/superpowers/plans/2026-10-11-fgr-web.md`.

## Global Constraints

- CPU only. No CUDA wheels in the image (`--index-url https://download.pytorch.org/whl/cpu`). `torch.set_num_threads(settings.torch_threads)`, default 4.
- Pins: `transformers==4.43.2`, `numpy<2`. GTE `Alibaba-NLP/gte-large-en-v1.5@104333d6af6f97649377c2afbde10a7704870c7b`; remote code `Alibaba-NLP/new-impl@40ced75c3017eb27626c9d4ea981bde21a2662f4`.
- Bundle v1 is read-only at `BUNDLE_DIR` (default `/data/bundle`, locally `data/bundle`); runtime writes go only to `RUNTIME_DIR`.
- Bundle facts: 935,050 entities; 614,324 embedding rows; 4,570 relations; 1,806,486 triples; 897 benchmark questions. Embeddings fp16 → cast to float32 before any maths.
- Retrieval constants: α = 0.5, RT top-k mask = 5, weighted-DDE floor = 0.1, DDE rounds 2 + 2, bridges B = 3, top-K ≤ 500, z = `(v - v.mean()) / (v.std() + 1e-6)` with `torch.std` (unbiased).
- Caps (Settings): `retrieve_max_triples=10000`, `hop1_cap=5000`, `fanout_cap=200`, `rt_priority=50`; hard limits `k ≤ 500`, `max_triples ≤ 30000`, `anchors ≤ 5`, `mentions ≤ 5`, chat message ≤ 2,000 chars, ingest text ≤ 6,000 chars.
- LLM: `llm_provider="mock"` by default; `llm_budget_usd=0.05`; prices per 1M tokens `0.006` (hit) / `0.30` (miss) / `1.20` (out); DeepSeek calls send `extra_body={"thinking": {"type": "disabled"}}`, `temperature=0`. **No automated test may hit the network or spend money.**
- Module boundaries: `fgr/retriever/*` never imports `fgr.api`, `fgr.agent` or `fgr.llm`. Cypher strings exist only in `fgr/store/graph.py` and always use parameters.
- Errors are `{"error": {"code", "message", "hint", "details"}}` with the codes of `docs/API.md`.
- Compose: every port bound to `127.0.0.1`; neo4j `mem_limit: 2g` (heap 1g, pagecache 512m); api `mem_limit: 3.5g`, `cpus: 4`; loader `mem_limit: 4g`. Secrets only via `.env`; never log keys.
- Test markers: `bundle` (needs `data/bundle`), `neo4j` (needs the compose Neo4j loaded with `--slice 50` unless stated), `gte` (needs GTE weights), `research` (needs `torch_geometric`, optional). Unmarked tests run anywhere in < 30 s.

## Review Focus

- Tiny or empty candidate subgraphs (m = 0, m = 1, m < 3, m < k): retrieval must return finite scores and an empty or short list, never NaN or a crash. Research `z()` gives NaN for a one-element vector; here a pass with m ≤ 1 is not z-normalised. Pinned in Task 5 and Task 10.
- Search text with Lucene special characters or Vietnamese diacritics ("amoxicillin/acid clavulanic", "HbA1c 8,1%", "tăng huyết áp", `"`, `:`): the search must escape them and still match. Pinned in Task 9.
- Duplicate display names ("The Illusion" ×2, `unnamed_entity`): everything user-facing keys on `eid`, never on name (`is_answer`, citations, highlights), and CVTs are never auto-linked. Pinned in Task 9 and Task 10.
- A chat client disconnecting mid-turn: no further LLM call may be made or recorded after the disconnect. Pinned in Task 12.
- Anchors that don't belong to the requested `kg` (a GrailQA eid with `kg=medical`), or that don't exist: 422 `NO_ANCHORS` with a hint, not a silent empty result. Pinned in Task 9.

---

### Task 1: Scaffold, settings, compose with Neo4j

**Files:**
- Create: `docker-compose.yml`, `.env.example`, `Makefile`, `api/pyproject.toml`, `api/Dockerfile`, `api/fgr/__init__.py`, `api/fgr/config.py`, `api/tests/conftest.py`
- Test: `api/tests/unit/test_config.py`

**Interfaces:**
- Produces: `fgr.config.Settings` (pydantic-settings, env names = upper-case field names) with fields `neo4j_uri: str = "bolt://neo4j:7687"`, `neo4j_user: str = "neo4j"`, `neo4j_password: str` (required), `bundle_dir: Path = Path("/data/bundle")`, `runtime_dir: Path = Path("/data/runtime")`, `torch_threads: int = 4`, `llm_provider: Literal["mock","deepseek"] = "mock"`, `deepseek_api_key: SecretStr | None`, `deepseek_base_url: str = "https://api.deepseek.com"`, `deepseek_model: str = "deepseek-flash"`, `llm_budget_usd: float = 0.05`, `llm_price_in_hit: float = 0.006`, `llm_price_in_miss: float = 0.30`, `llm_price_out: float = 1.20`, `retrieve_max_triples: int = 10000`, `hop1_cap: int = 5000`, `fanout_cap: int = 200`, `rt_priority: int = 50`, `prompt_triples: int = 30`, `link_margin: float = 0.8`; `get_settings() -> Settings` (cached).
- Produces: pytest fixtures `settings` and `bundle_dir` (skips `bundle`-marked tests when `data/bundle/MANIFEST.json` is absent), and auto-skip for the `neo4j`/`gte`/`research` markers when unavailable.

- [ ] **Step 1: Write the failing test**

```python
def test_defaults(monkeypatch):
    monkeypatch.setenv("NEO4J_PASSWORD", "x")
    s = Settings()
    assert s.llm_provider == "mock" and s.llm_budget_usd == 0.05 and s.torch_threads == 4
    assert (s.retrieve_max_triples, s.hop1_cap, s.fanout_cap, s.rt_priority) == (10000, 5000, 200, 50)

def test_password_required(monkeypatch):
    monkeypatch.delenv("NEO4J_PASSWORD", raising=False)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)
```

- [ ] **Step 2: Run** `cd api && pip install -e ".[dev]" && pytest tests/unit/test_config.py -v`. Expected: FAIL (`fgr.config` missing).
- [ ] **Step 3: Implement `Settings` / `get_settings`**, `pyproject.toml` (deps from the Tech Stack; extras `dev = [pytest, httpx, ruff, mypy]`, `research = [torch_geometric]`; pytest markers registered), `api/Dockerfile` (python:3.11-slim, CPU torch index, non-root user, `HF_HOME=/hf`), `docker-compose.yml` with the `neo4j` service per design §2.1 (healthcheck `cypher-shell -u neo4j -p $NEO4J_PASSWORD 'RETURN 1'`, volume `neo4j_data`) and placeholders for `api`/`web`/`loader` filled in Task 14, `.env.example` listing every variable of design §5 with no secret values, `Makefile` targets `test`, `test-all`, `up`, `down`, `fetch` (`python tools/s3_bundle.py get`).
- [ ] **Step 4: Run** `pytest tests/unit/test_config.py -v` → PASS; `docker compose up -d neo4j && docker compose ps` → neo4j `healthy`.
- [ ] **Step 5: Commit** `git commit -m "chore: scaffold api package, settings, compose neo4j"`

### Task 2: Bundle reader and embedding store

**Files:**
- Create: `api/fgr/store/__init__.py`, `api/fgr/store/bundle.py`, `api/fgr/store/embeddings.py`
- Test: `api/tests/unit/test_bundle_tsv.py`, `api/tests/bundle/test_bundle.py`, `api/tests/unit/test_embeddings_overlay.py`

**Interfaces:**
- Produces: `Bundle(root: Path)` with `entities: EntityTable` (`eid`-indexed numpy arrays `key: np.ndarray[object]`, `name: np.ndarray[object]`, `emb_row: np.ndarray[int64]`), `relation_names: list[str]` (index = rid), `iter_triples(batch: int) -> Iterator[np.ndarray]` (int64 `[b,3]`), `questions: list[dict]` (questions.jsonl), `question(qid: str) -> dict`, `record(qid: str) -> dict` (test.pkl record), `local2eid(qid: str) -> np.ndarray[int64]`, `manifest: dict`. All loaded lazily, cached.
- Produces: `read_tsv(path: Path) -> Iterator[list[str]]` (gzip, `newline='\n'`, header skipped).
- Produces: `EmbeddingStore(bundle_dir: Path, overlay_dir: Path)` with `ent(rows: np.ndarray) -> torch.Tensor` (float32 `[n,1024]`), `rel(rids: np.ndarray) -> torch.Tensor`, `q(row: int) -> torch.Tensor` (`[1,1024]`), `n_ent_rows: int`, `n_rel_rows: int`, `append_ent(vecs: torch.Tensor) -> np.ndarray` (new global rows, starting at 614,324), `append_rel(vecs: torch.Tensor) -> np.ndarray` (new rids, starting at 4,570), `reset_overlay() -> None`. Rows ≥ the bundle size are served from `overlay_dir/ent_extra.f16.npy` / `rel_extra.f16.npy`.

- [ ] **Step 1: Write the failing tests**

```python
def test_read_tsv_keeps_carriage_returns_inside_fields(tmp_path):  # names may contain \r (DATA.md §2.1)
    p = tmp_path / "e.tsv.gz"
    with gzip.open(p, "wt", encoding="utf-8", newline="\n") as f:
        f.write("eid\tkey\tname\temb_row\n0\tm.1\ta\rb\t0\n")
    assert list(read_tsv(p)) == [["0", "m.1", "a\rb", "0"]]

@pytest.mark.bundle
def test_bundle_counts(bundle_dir):
    b = Bundle(bundle_dir)
    assert len(b.entities.eid) == 935050 and len(b.relation_names) == 4570
    assert sum(len(x) for x in b.iter_triples(200_000)) == 1806486
    q = b.question("2101960008000")
    assert q["topic_eids"] == [0] and q["answer_eids"] == [1] and q["row"] == 0
    assert b.local2eid("2101960008000")[0] == 0 and b.entities.name[0] == "The Illusion"

@pytest.mark.bundle
def test_embedding_shapes(bundle_dir, tmp_path):
    e = EmbeddingStore(bundle_dir, tmp_path)
    assert e.ent(np.array([0, 614323])).shape == (2, 1024) and e.ent(np.array([0])).dtype == torch.float32
    assert e.q(0).shape == (1, 1024) and e.rel(np.array([4569])).shape == (1, 1024)

def test_overlay_append_and_read(tmp_path, fake_bundle_dir):   # fake bundle fixture: 3 ent rows, 2 rel rows
    e = EmbeddingStore(fake_bundle_dir, tmp_path)
    rows = e.append_ent(torch.ones(2, 1024))
    assert rows.tolist() == [3, 4] and torch.equal(e.ent(rows), torch.ones(2, 1024))
    assert EmbeddingStore(fake_bundle_dir, tmp_path).ent(np.array([4])).sum() == 1024   # survives a restart
```

- [ ] **Step 2: Run** `pytest tests/unit/test_bundle_tsv.py tests/unit/test_embeddings_overlay.py tests/bundle/test_bundle.py -v`. Expected: FAIL (import errors).
- [ ] **Step 3: Implement** `read_tsv`, `Bundle`, `EmbeddingStore` (`np.load(..., mmap_mode="r")`; overlay files rewritten atomically via `.tmp` + `os.replace`; add the `fake_bundle_dir` fixture to `conftest.py`).
- [ ] **Step 4: Run** the same command → PASS (bundle tests need `make fetch` first).
- [ ] **Step 5: Commit** `git commit -m "feat: bundle reader and embedding store with overlay"`

### Task 3: Relation-topic features

**Files:**
- Create: `api/fgr/retriever/__init__.py`, `api/fgr/retriever/features.py`
- Test: `api/tests/unit/test_features.py`, `api/tests/bundle/test_features_parity.py`

**Interfaces:**
- Produces: `clean_text(text: str) -> list[str]`, `compute_lexical_score(question_tokens: list[str], relation_name: str) -> float`, `rt_features(q_emb: Tensor, rel_emb: Tensor, question: str, rel_names: list[str], alpha: float = 0.5, topk: int = 5) -> Tensor` (`[R,2]` = `[score, mask]`), `rt_scores(q_emb, rel_emb, question, rel_names, alpha=0.5) -> Tensor` (`[R]`, no mask; reused by Neo4jSource for priority relations). Copy `STOPWORDS`, `SYNONYM_GROUPS` and the token rules verbatim from `reference/rt_features.py`.

- [ ] **Step 1: Write the failing tests**

```python
def test_lexical_example():
    q = clean_text("which play is produced by the illusion?")
    assert q == ["play", "produced", "illusion"]
    assert compute_lexical_score(q, "theater.theater_production.producing_company") == pytest.approx(0.375)

def test_mask_is_top5():
    s = torch.tensor([.1, .9, .5, .3, .7, .2])
    f = rt_features_from_scores(s, topk=5)          # helper used by rt_features
    assert f[:, 1].tolist() == [0., 1., 1., 1., 1., 1.]

@pytest.mark.bundle
def test_parity_with_reference(bundle_dir):         # 30 linspace questions; imports reference/rt_features.py in the test only
    for qid in linspace_ids(30):
        ours, ref = rt_features(...), reference_rt_features(...)
        assert torch.allclose(ours[:, 0], ref[:, 0], atol=1e-6) and torch.equal(ours[:, 1], ref[:, 1])
```

- [ ] **Step 2: Run** `pytest tests/unit/test_features.py tests/bundle/test_features_parity.py -v` → FAIL.
- [ ] **Step 3: Implement** the four functions plus `rt_features_from_scores(score: Tensor, topk: int) -> Tensor`. Add `linspace_ids(n) = [questions[i]["id"] for i in np.linspace(0, 896, n).astype(int)]` to `tests/helpers.py`.
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `git commit -m "feat: relation-topic features (port of research code)"`

### Task 4: DDE and the WL scorer (reference + factorised)

**Files:**
- Create: `api/fgr/retriever/types.py`, `api/fgr/retriever/dde.py`, `api/fgr/retriever/scorer.py`
- Test: `api/tests/unit/test_dde.py`, `api/tests/unit/test_scorer.py`, `api/tests/bundle/test_scorer_factorised.py`

**Interfaces:**
- Produces (`types.py`): `Mode = Literal["base","adr","aer","adr_aer"]`; `@dataclass ScoringInput(q_emb: Tensor[1,1024], ent_emb: Tensor[n_text,1024], n_non_text: int, rel_emb: Tensor[R,1024], h: Tensor[m], r: Tensor[m], t: Tensor[m], rtf: Tensor[R,2])`; `@dataclass Subgraph(eids: np.ndarray, names: list[str], emb_rows: np.ndarray, n_text: int, rids: np.ndarray, rel_names: list[str], h: Tensor, r: Tensor, t: Tensor, anchors: list[int], truncated: bool = False, is_cvt: np.ndarray | None = None)` (local ids; text entities first; `anchors` local).
- Produces (`dde.py`): `pe_conv(src: Tensor, dst: Tensor, x: Tensor, w: Tensor) -> Tensor` (weighted mean over in-edges, 0 when weight sum ≤ 1e-12); `dde(topic_oh: Tensor, h: Tensor, t: Tensor, w: Tensor | None) -> list[Tensor]` (`[f1, f2, b1, b2]`, each `[n,2]`).
- Produces (`scorer.py`): `WLScorer.load(path: Path) -> WLScorer`; `reference_forward(inp: ScoringInput, anchors: frozenset[int]) -> Tensor[m]` (the unfactorised `Retriever.forward`); `prepare(inp: ScoringInput) -> Prepared`; `score_pass(prep: Prepared, anchors: frozenset[int]) -> Tensor[m]`. `W1` column blocks per RETRIEVER_SPEC §3.2; anchor feature width 18 derived from the weight shape.

- [ ] **Step 1: Write the failing tests**

```python
def test_pe_conv_weighted_mean():
    x = torch.tensor([[1., 0.], [0., 1.], [0., 0.]])
    out = pe_conv(src=torch.tensor([0, 1]), dst=torch.tensor([2, 2]), x=x, w=torch.tensor([1., 3.]))
    assert torch.allclose(out[2], torch.tensor([.25, .75])) and out[0].abs().sum() == 0   # no in-edges -> 0

def test_dde_rounds_and_reverse(): ...           # chain 0→1→2, anchor {0}: f1[1]=[0,1], f2[2]=[0,1], b1 of node 0 = mean of x[1]

def test_factorised_equals_reference_random():   # random weights, n=7, m=12, R=3, every anchor set of size 1..2
    s = WLScorer.random(seed=0)
    for A in anchor_sets:
        assert torch.allclose(s.score_pass(s.prepare(inp), A), s.reference_forward(inp, A), atol=1e-4)

@pytest.mark.bundle
def test_factorised_equals_reference_on_benchmark(bundle_dir):   # 30 linspace questions, topic anchors
    assert max_abs_diff <= 1e-4
```

- [ ] **Step 2: Run** `pytest tests/unit/test_dde.py tests/unit/test_scorer.py tests/bundle/test_scorer_factorised.py -v` → FAIL.
- [ ] **Step 3: Implement** `pe_conv` with `index_add_` and `dde` per RETRIEVER_SPEC §3.1. Implement `WLScorer`: `reference_forward` per §3.2; `prepare`/`score_pass` per §4 (prepare computes `base [m,1024]` once; `score_pass` adds `(feat @ W_hf.T)[h] + (feat @ W_tf.T)[t]`, ReLU, `@ w2 + b2`). Add `WLScorer.random(seed)` for tests. Load the checkpoint with `torch.load(..., weights_only=False)` and read `model_state_dict`.
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `git commit -m "feat: DDE and WL scorer with factorised fast path"`

### Task 5: Multi-pass retrieval (base / ADR / AER / ADR+AER)

**Files:**
- Create: `api/fgr/retriever/multipass.py`
- Test: `api/tests/unit/test_multipass.py`

**Interfaces:**
- Consumes: `Mode` (Task 4).
- Produces: `run_mode(score: Callable[[frozenset[int]], Tensor], h: Tensor, t: Tensor, anchors: Sequence[int], mode: Mode, bridges_b: int = 3) -> MultiPassResult`; `@dataclass PassInfo(id: int, kind: Literal["base","bridge","single","single+bridge"], anchors: tuple[int, ...], bridges_added: tuple[int, ...], ms: float)`; `@dataclass MultiPassResult(fused: Tensor[m], best_pass: Tensor[m] (int64, PassInfo.id), passes: list[PassInfo])`. `passes` lists **distinct** anchor sets in first-use order. `fused` is computed over the research pass list of RETRIEVER_SPEC §5 (identical result).

- [ ] **Step 1: Write the failing tests**

```python
def test_base_single_pass_is_raw(): r = run_mode(score, h, t, [0], "base"); assert torch.equal(r.fused, score(frozenset({0})))
def test_adr_one_anchor_equals_base(): assert torch.equal(run_mode(..., [0], "adr").fused, run_mode(..., [0], "base").fused)
def test_aer_bridges_are_endpoints_of_top3_triples():   # toy scores with known top-3 triples
    r = run_mode(score, h, t, [0], "aer")
    assert r.passes[1].kind == "bridge" and set(r.passes[1].bridges_added) == {2, 5, 7}
def test_fused_is_max_of_unbiased_z():
    p = [score(frozenset({0})), score(frozenset({0, 2, 5, 7}))]
    z = lambda v: (v - v.mean()) / (v.std() + 1e-6)
    assert torch.allclose(run_mode(..., "aer").fused, torch.stack([z(x) for x in p]).max(0).values)
def test_memo_scores_each_anchor_set_once():            # counting score fn; adr_aer with anchors [0, 1]
    assert calls == len({frozenset(p.anchors) for p in r.passes})
def test_adr_aer_kinds(): assert [p.kind for p in r.passes][:2] == ["base", "bridge"] and "single" in kinds
def test_tiny_subgraphs_are_finite():                   # Review Focus: m = 0, 1, 2 with 2 anchors, all modes
    for m in (0, 1, 2):
        r = run_mode(score_m, h[:m], t[:m], [0, 1], "adr_aer")
        assert r.fused.shape == (m,) and torch.isfinite(r.fused).all()
```

- [ ] **Step 2: Run** `pytest tests/unit/test_multipass.py -v` → FAIL.
- [ ] **Step 3: Implement** `run_mode` per RETRIEVER_SPEC §5. Pass timing via `time.perf_counter`. When `m ≤ 1`, skip z-normalisation (use raw scores). `best_pass` = argmax over the z-stack mapped to distinct-pass ids (single pass → all 0).
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `git commit -m "feat: multi-pass ADR/AER with provenance"`

### Task 6: Shared schemas, BenchmarkSource, pipeline, golden parity

**Files:**
- Create: `api/fgr/schemas.py`, `api/fgr/retriever/sources.py`, `api/fgr/retriever/pipeline.py`, `api/fgr/bench/__init__.py`, `api/fgr/bench/metrics.py`, `api/scripts/parity_full.py`
- Test: `api/tests/unit/test_metrics.py`, `api/tests/bundle/test_golden_quick.py`

**Interfaces:**
- Consumes: Tasks 2–5.
- Produces (`schemas.py`, pydantic, exactly the types of `docs/API.md` §1): `EntityRef`, `RelationRef`, `Pass`, `ScoredTriple`, `Timings`, `RetrievalResult`, `RetrieveRequest(question_en: str = "", kg: Literal["grailqa","medical"] = "grailqa", anchors: list[int] | None, mentions: list[str] | None, mode: Mode = "adr_aer", k: Literal[50,100,200,500] = 100, source: Literal["neo4j","benchmark"] = "neo4j", benchmark_id: str | None, caps: dict[str,int] | None)`.
- Produces: `class SubgraphSource(Protocol): def fetch(self, anchors: list[int], kg: str, question_en: str, q_emb: Tensor) -> Subgraph`; `BenchmarkSource(bundle: Bundle)` with `fetch_question(qid: str) -> Subgraph` (research local order; anchors = `q_entity_id_list`).
- Produces: `RetrievalPipeline(bundle, emb: EmbeddingStore, scorer: WLScorer, settings, encoder: Encoder | None = None, neo4j_source: SubgraphSource | None = None, linker=None)` with `retrieve(req: RetrieveRequest) -> RetrievalResult` and `retrieve_many(req, modes: list[Mode]) -> dict[Mode, RetrievalResult]` (shared encode/fetch/prepare). `Encoder` protocol: `encode(texts: list[str]) -> Tensor`.
- Produces (`metrics.py`): `ans_recall(scored: list[tuple[str,str,str,float]], answers: set[str], ks=(50,100,200)) -> dict[int,float]` (research definition, by name).

- [ ] **Step 1: Write the failing tests**

```python
def test_ans_recall_by_name():
    s = [("A", "r", "B", 1.0), ("C", "r", "D", .5)]
    assert ans_recall(s, {"B", "D"}, ks=(1, 2)) == {1: .5, 2: 1.0}

@pytest.mark.bundle
@pytest.mark.parametrize("mode", ["base", "adr", "aer", "adr_aer"])
def test_golden_quick(pipeline, golden, mode):        # RETRIEVER_SPEC §7 quick criterion
    same_set = same_order = 0
    for qid in linspace_ids(30):
        res = pipeline.retrieve(RetrieveRequest(source="benchmark", benchmark_id=qid, mode=mode, k=500))
        ours = [(x.h.name, x.r.name, x.t.name, x.score) for x in res.triples]
        ref = golden[mode][qid]["scored_triples"]
        same_set += {x[:3] for x in ours[:50]} == {x[:3] for x in ref[:50]}
        if [x[:3] for x in ours[:50]] == [x[:3] for x in ref[:50]]:
            same_order += 1
            assert max(abs(a[3] - b[3]) for a, b in zip(ours[:50], ref[:50])) <= 1e-3
    assert same_set >= 29

@pytest.mark.bundle
def test_benchmark_result_has_gold_and_eid_based_answers(pipeline):   # Review Focus: duplicate names
    res = pipeline.retrieve(RetrieveRequest(source="benchmark", benchmark_id="2101960008000", mode="adr_aer", k=100))
    assert res.gold and set(res.gold.ans_recall) == {"50", "100", "200"}
    assert all(x.is_answer == (x.h.eid == 1 or x.t.eid == 1) for x in res.triples)   # eid 1 is the answer, eid 0 shares its name
```

- [ ] **Step 2: Run** `pytest tests/unit/test_metrics.py tests/bundle/test_golden_quick.py -v` → FAIL.
- [ ] **Step 3: Implement** `schemas.py`, `BenchmarkSource`, `RetrievalPipeline` (benchmark source uses `emb.q(row)`, neo4j source uses `encoder`; stage timings into `Timings`; `ScoredTriple.score` = fused value; `gold` only for the benchmark source; `is_answer` by eid), `ans_recall`, and `scripts/parity_full.py` (all 897 × 4 modes, prints ansR@50/100/200 next to the DATA.md §2.4 table, exit code 1 if any |Δ| > 0.05). Add `make parity` (quick test) and `make parity-full`.
- [ ] **Step 4: Run** the tests → PASS; `make parity-full` → prints `adr_aer 83.42 / 89.16 / 92.97` (±0.05), exit 0.
- [ ] **Step 5: Commit** `git commit -m "feat: retrieval pipeline with benchmark source; golden parity"`

### Task 7: GTE question encoder

**Files:**
- Create: `api/fgr/encoder/__init__.py`, `api/fgr/encoder/gte.py`
- Test: `api/tests/gte/test_gte.py`

**Interfaces:**
- Produces: `GTEEncoder(model_id: str, revision: str, code_revision: str, threads: int, cache_size: int = 256)` implementing `Encoder.encode(texts: list[str]) -> Tensor[n,1024]` (L2-normalised CLS; batch 32; LRU for single texts); `ready: bool`; `warmup() -> None`.

- [ ] **Step 1: Write the failing test**

```python
@pytest.mark.gte
def test_matches_bundle_question_vectors(bundle_dir):
    enc = GTEEncoder(...); b = Bundle(bundle_dir); e = EmbeddingStore(bundle_dir, tmp)
    v = enc.encode([b.questions[19]["question"]])
    assert torch.allclose(v, e.q(19), atol=1e-3)          # measured 6e-5 on the research machine
    assert torch.allclose(v.norm(dim=1), torch.ones(1), atol=1e-4)
```

- [ ] **Step 2: Run** `pytest tests/gte -v` → FAIL.
- [ ] **Step 3: Implement** with `AutoTokenizer`/`AutoModel.from_pretrained(model_id, revision=..., trust_remote_code=True, code_revision=...)` on CPU, **without** `unpad_inputs`/`use_memory_efficient_attention`; tokenizer `max_length=8192, padding=True, truncation=True`; `torch.inference_mode()`.
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `git commit -m "feat: GTE CPU question encoder"`

### Task 8: Graph store and loader

**Files:**
- Create: `api/fgr/store/graph.py`, `api/fgr/loader/__init__.py`, `api/fgr/loader/__main__.py`
- Test: `api/tests/neo4j/test_loader.py`

**Interfaces:**
- Consumes: `Bundle` (Task 2).
- Produces: `GraphStore(uri: str, user: str, password: str)` with `ensure_schema() -> None` (design §3 constraints and indexes), `is_empty() -> bool`, `create_entities(rows: list[dict]) -> None`, `create_relations(rows: list[dict]) -> None`, `create_triples(rows: list[dict]) -> None` (rows `{h, rid, t, name}`; batched `UNWIND`, 10,000 per batch), `counts(kg: str) -> dict` (`entities`, `triples`), `max_ids() -> tuple[int, int]` (max eid, max rid), `delete_kg(kg: str) -> dict`, `ping() -> bool`, `close()`. (Search/neighbours/subgraph are added in Task 9.)
- Produces CLI: `python -m fgr.loader (--full | --slice N) [--reset] [--batch 10000]`. Slice = the first N questions by `row`: union of their `local2eid` entities and their triples (via `record(qid)` + `relation_names.index`). Exit code 2 if the DB is not empty and `--reset` is absent.

- [ ] **Step 1: Write the failing tests**

```python
@pytest.mark.neo4j
def test_slice_counts_match_bundle(graph_empty, bundle_dir):     # fixture: wipes the DB
    run_loader(["--slice", "5"])
    exp_e, exp_t = expected_slice_counts(Bundle(bundle_dir), 5)  # computed from the bundle in the test
    assert graph_empty.counts("grailqa") == {"entities": exp_e, "triples": exp_t}

@pytest.mark.neo4j
def test_refuses_non_empty_without_reset(graph_loaded):
    assert run_loader(["--slice", "5"]).returncode == 2

@pytest.mark.neo4j
def test_schema_and_cvt_flag(graph_loaded):
    assert "entity_eid" in constraint_names(graph_loaded) and "entity_names" in index_names(graph_loaded)
    assert node(graph_loaded, eid_of_name("unnamed_entity"))["is_cvt"] is True
```

- [ ] **Step 2: Run** `docker compose up -d neo4j && pytest tests/neo4j/test_loader.py -v` → FAIL.
- [ ] **Step 3: Implement** `GraphStore` (official `neo4j` driver; `execute_write` with parameters only; node props `eid, key, name, emb_row, kg='grailqa', is_cvt = name == 'unnamed_entity'`; `:Relation` nodes for all relations; relationships `:R {rid, name}`) and the loader CLI with a progress line per batch.
- [ ] **Step 4: Run** → PASS. Then `python -m fgr.loader --full --reset` against a fresh volume and check `counts("grailqa") == {"entities": 935050, "triples": 1806486}` (manual, takes minutes).
- [ ] **Step 5: Commit** `git commit -m "feat: Neo4j graph store and bundle loader"`

### Task 9: Neo4jSource and entity linking

**Files:**
- Modify: `api/fgr/store/graph.py` (add `search`, `neighbors`, `entity`, `subgraph`), `api/fgr/retriever/sources.py` (add `Neo4jSource`)
- Create: `api/fgr/linking.py`, `api/scripts/g4_check.py`
- Test: `api/tests/neo4j/test_search.py`, `api/tests/neo4j/test_neo4j_source.py`, `api/tests/unit/test_linking.py`

**Interfaces:**
- Produces (`GraphStore`): `search(text: str, kg: str, limit: int = 10, include_cvt: bool = False) -> list[dict]` (`eid, name, key, kg, score, degree`; full-text on `entity_names` with the input escaped via the Lucene escape list `+ - && || ! ( ) { } [ ] ^ " ~ * ? : \ /`); `neighbors(eid: int, limit: int, rid: int | None) -> list[tuple[int,int,int]]`; `entity(eid: int) -> dict | None`; `subgraph(anchors: list[int], kg: str, hop1_cap: int, fanout_cap: int, max_triples: int, priority_rids: list[int]) -> tuple[np.ndarray, dict[int, dict], bool]` (triples `[m,3]` global ids sorted by `(h, rid, t)`; nodes `eid → {name, emb_row, key, is_cvt}`; truncated flag) implementing RETRIEVER_SPEC §6 steps 1–4.
- Produces: `Neo4jSource(graph: GraphStore, emb: EmbeddingStore, relation_names: list[str], settings)` implementing `SubgraphSource.fetch` (priority rids = top `rt_priority` relations of the kg by `rt_scores`; entity order = anchors first, then by eid; text entities first). Raises `NoAnchors` when an anchor is not in `kg` or does not exist.
- Produces: `Linker(graph: GraphStore)` with `link(mentions: list[str], kg: str) -> LinkResult(anchors: list[EntityRef], candidates: list[MentionCandidates(mention: str, items: list[EntityRef])])`. Rule: case-insensitive exact name matches win (up to 2 per mention, highest degree first). Otherwise take the top candidate when the second's score ≤ `link_margin` × the top's. CVTs are never returned. At most 3 anchors in total.

- [ ] **Step 1: Write the failing tests**

```python
@pytest.mark.neo4j
@pytest.mark.parametrize("text", ['amoxicillin/acid clavulanic', 'HbA1c 8,1%', 'a "quoted" (name): x', 'tăng huyết áp'])
def test_search_escapes_special_characters(graph_loaded, text):     # Review Focus
    graph_loaded.search(text, "grailqa")                             # must not raise

@pytest.mark.neo4j
def test_search_finds_both_homonyms_and_skips_cvts(graph_loaded):
    eids = [x["eid"] for x in graph_loaded.search("The Illusion", "grailqa")]
    assert {0, 1} <= set(eids)
    assert all(not x["name"] == "unnamed_entity" for x in graph_loaded.search("unnamed entity", "grailqa"))

@pytest.mark.neo4j
def test_subgraph_caps_and_order(graph_loaded):
    tri, nodes, truncated = graph_loaded.subgraph([0], "grailqa", 5000, 200, 50, [])
    assert len(tri) == 50 and truncated and tri.tolist() == sorted(tri.tolist())

@pytest.mark.neo4j
def test_priority_relations_survive_fanout_cap(graph_loaded): ...   # fanout_cap=1: each kept hop-2 edge has a priority rid when one exists

@pytest.mark.neo4j
def test_anchor_from_other_kg_is_rejected(neo4j_source):             # Review Focus
    with pytest.raises(NoAnchors):
        neo4j_source.fetch([0], "medical", "q", q_emb)

def test_linker_exact_match_and_margin(fake_graph):                  # fake GraphStore with canned search results
    assert [a.eid for a in Linker(fake_graph).link(["The Illusion"], "grailqa").anchors] == [1, 0]   # exact, by degree
```

- [ ] **Step 2: Run** `pytest tests/neo4j/test_search.py tests/neo4j/test_neo4j_source.py tests/unit/test_linking.py -v` → FAIL.
- [ ] **Step 3: Implement** the methods above, then `scripts/g4_check.py`: 200 linspace questions, gold anchors, Neo4jSource vs BenchmarkSource ansR@100, prints both, exit 1 if the gap is > 3.0. Add `make g4`.
- [ ] **Step 4: Run** the tests → PASS. On the full graph, `make g4` → exit 0.
- [ ] **Step 5: Commit** `git commit -m "feat: Neo4j candidate subgraphs and entity linking"`

### Task 10: HTTP API: system, graph, retrieval, benchmark

**Files:**
- Create: `api/fgr/api/__init__.py`, `api/fgr/api/app.py`, `api/fgr/api/deps.py`, `api/fgr/api/errors.py`, `api/fgr/api/routes_system.py`, `api/fgr/api/routes_graph.py`, `api/fgr/api/routes_retrieve.py`, `api/fgr/api/routes_benchmark.py`, `api/fgr/bench/runner.py`, `api/scripts/bench_latency.py`
- Test: `api/tests/api/test_system.py`, `api/tests/api/test_retrieve.py`, `api/tests/api/test_benchmark.py`

**Interfaces:**
- Consumes: Tasks 2–9.
- Produces: `create_app(deps: Deps | None = None) -> FastAPI`; `@dataclass Deps(settings, bundle, emb, scorer, pipeline, graph: GraphStore | None, linker, encoder, llm=None, budget=None)` built by `build_deps(settings)` at startup (encoder warm-up in a background thread; `/health` reports `loading` until ready). `ApiError(code: str, status: int, message: str, hint: str = "", details: dict | None = None)` mapped to the error JSON. Routes and shapes exactly as `docs/API.md` §2–5.
- Produces: `run_benchmark(pipeline, ids: list[str], modes: list[Mode], source, ks) -> Iterator[dict]` (progress dicts, then the result dict), cached under `runtime_dir/bench_cache/<sha256>.json`.

- [ ] **Step 1: Write the failing tests** (TestClient; `Deps` with BenchmarkSource and a fake GraphStore unless marked `neo4j`)

```python
def test_health_shape(client): assert set(client.get("/api/health").json()) >= {"status", "neo4j", "encoder", "scorer", "bundle_version", "graph"}

@pytest.mark.bundle
def test_retrieve_benchmark(client):
    r = client.post("/api/retrieve", json={"source": "benchmark", "benchmark_id": "2101960008000", "mode": "adr_aer", "k": 100}).json()
    assert len(r["triples"]) == 100 and r["triples"][0]["rank"] == 1 and r["gold"]["ans_recall"]["100"] >= 0
    assert set(r["timings_ms"]) == {"encode", "link", "fetch", "features", "prepare", "passes", "fuse", "total"}

def test_errors(client):
    assert client.post("/api/retrieve", json={"source": "benchmark", "benchmark_id": "nope"}).json()["error"]["code"] == "UNKNOWN_BENCHMARK_ID"
    assert client.post("/api/retrieve", json={"question_en": "x", "mentions": ["zzzz-no-match"]}).status_code == 422
    assert client.post("/api/retrieve", json={"question_en": "x", "anchors": [0], "k": 1000}).status_code == 422

def test_tiny_subgraph_returns_short_list(client_with_fake_graph):   # Review Focus: fake graph returns m = 1
    r = client_with_fake_graph.post("/api/retrieve", json={"question_en": "x", "anchors": [5], "mode": "adr_aer"}).json()
    assert len(r["triples"]) == 1 and math.isfinite(r["triples"][0]["score"])

@pytest.mark.bundle
def test_compare_diff_is_consistent(client):
    d = client.post("/api/retrieve/compare", json={..., "modes": ["base", "adr_aer"]}).json()["diff"]
    assert not ({key(x) for x in d["gained"]} & {key(x) for x in d["lost"]})

@pytest.mark.bundle
def test_benchmark_filters_and_run(client):
    assert all(x["n_hops"] == 2 for x in client.get("/api/benchmark/questions?hops=2&limit=20").json()["items"])
    events = sse_events(client.post("/api/benchmark/run", json={"n": 3, "sample": "linspace", "modes": ["base"], "source": "benchmark", "ks": [50]}))
    assert events[-1][0] == "result" and events[-1][1]["per_mode"]["base"]["n"] == 3
```

- [ ] **Step 2: Run** `pytest tests/api -v` → FAIL.
- [ ] **Step 3: Implement** the app (CORS limited to `WEB_ORIGIN`, default `http://localhost:8080`; SSE via sse-starlette), the routes, the benchmark runner with cache, and `scripts/bench_latency.py` (20 fixed questions: 10 benchmark ids with gold anchors via `source=neo4j` + 10 free-text with mentions; 2 warm-up calls; prints p50/p95 per `timings_ms` stage and total; exit 1 if total p50 > 800 ms or p95 > 2,500 ms). Add `make bench-latency`.
- [ ] **Step 4: Run** `pytest tests/api -v` → PASS. With the full stack up, `make bench-latency` → exit 0.
- [ ] **Step 5: Commit** `git commit -m "feat: REST API for retrieval, graph and benchmark"`

### Task 11: LLM providers and the budget guard

**Files:**
- Create: `api/fgr/llm/__init__.py`, `api/fgr/llm/base.py`, `api/fgr/llm/budget.py`, `api/fgr/llm/deepseek.py`, `api/fgr/llm/mock.py`, `api/fgr/api/routes_budget.py`
- Test: `api/tests/unit/test_budget.py`, `api/tests/unit/test_deepseek_request.py`, `api/tests/unit/test_mock_llm.py`

**Interfaces:**
- Produces: `@dataclass Usage(prompt_hit: int, prompt_miss: int, completion: int)`; `LLMResponse(text: str, tool_calls: list[dict], usage: Usage)`; `class LLMProvider(Protocol)` per AGENT_SPEC §2; `class BudgetExceeded(Exception)`; `Budget(ledger: Path, cap_usd: float, prices: Prices)` with `spent() -> float`, `estimate(chars_in: int, max_tokens: int) -> float` (`ceil(chars/2.5)` tokens at the miss price + `max_tokens` at the out price), `check(estimate_usd: float) -> None`, `record(usage: Usage, *, provider, model, purpose, session_id, estimate_usd) -> float`, `status() -> dict` (`cap_usd, spent_usd, remaining_usd, calls, last_call_at`); `GuardedLLM(provider, budget)` with the same `chat(...)` signature plus `session_id` (check → call → record under one lock); `DeepSeekProvider(api_key, base_url, model, client_factory=OpenAI)`; `MockProvider(bundle, samples_dir)`; `GET /api/budget`.

- [ ] **Step 1: Write the failing tests**

```python
def test_cost_formula(tmp_path):
    b = Budget(tmp_path / "l.jsonl", 0.05, Prices(hit=0.006, miss=0.30, out=1.20))
    assert b.record(Usage(1000, 2000, 500), provider="deepseek", model="m", purpose="t", session_id="s", estimate_usd=0) == pytest.approx(0.001206)
    assert Budget(tmp_path / "l.jsonl", 0.05, prices).spent() == pytest.approx(0.001206)   # survives restart

def test_refuses_before_calling(tmp_path):
    prov = CountingProvider(); g = GuardedLLM(prov, Budget(tmp_path / "l.jsonl", 0.0, prices))
    with pytest.raises(BudgetExceeded):
        g.chat([{"role": "user", "content": "hi"}], max_tokens=10, purpose="t", session_id="s")
    assert prov.calls == 0

def test_deepseek_request_disables_thinking():
    fake = FakeOpenAI(); DeepSeekProvider("k", "u", "deepseek-flash", client_factory=lambda **kw: fake).chat(msgs, json_mode=True, max_tokens=50, purpose="t")
    kw = fake.last_kwargs
    assert kw["extra_body"] == {"thinking": {"type": "disabled"}} and kw["temperature"] == 0
    assert kw["response_format"] == {"type": "json_object"}

def test_mock_normalise_benchmark_question(mock_llm):
    out = json.loads(mock_llm.chat(normalise_messages("which play is produced by the illusion?"), json_mode=True, purpose="normalise").text)
    assert out["mentions_en"] == ["The Illusion"] and out["lang"] == "en"
```

- [ ] **Step 2: Run** `pytest tests/unit/test_budget.py tests/unit/test_deepseek_request.py tests/unit/test_mock_llm.py -v` → FAIL.
- [ ] **Step 3: Implement** per AGENT_SPEC §2–3 and §6. The mock recognises the normalise/answer/extract purposes from the `purpose` argument. DeepSeek reads `usage.prompt_cache_hit_tokens` / `prompt_cache_miss_tokens` / `completion_tokens`. A 30 s timeout and one retry on 5xx/timeout.
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `git commit -m "feat: LLM providers with server-side budget guard"`

### Task 12: Chat agent and `/api/chat`

**Files:**
- Create: `api/fgr/agent/__init__.py`, `api/fgr/agent/prompts.py`, `api/fgr/agent/render.py`, `api/fgr/agent/tools.py`, `api/fgr/agent/loop.py`, `api/fgr/api/routes_chat.py`
- Test: `api/tests/unit/test_render.py`, `api/tests/agent/test_loop.py`, `api/tests/api/test_chat_sse.py`

**Interfaces:**
- Consumes: `RetrievalPipeline`, `Linker`, `GuardedLLM`, schemas.
- Produces: `render_triples(triples: list[ScoredTriple], start: int = 1) -> str`; `parse_citations(text: str, n: int) -> list[str]`; `degraded_answer(result: RetrievalResult, lang: str) -> str` (AGENT_SPEC §4.5); `ChatRequest` pydantic per API.md §6; `async def run_turn(req: ChatRequest, deps: Deps, is_disconnected: Callable[[], Awaitable[bool]]) -> AsyncIterator[tuple[str, dict]]` yielding the API.md §6 events in order. The prompts are AGENT_SPEC §4.1/§4.3 verbatim.

- [ ] **Step 1: Write the failing tests**

```python
def test_render_format():
    assert render_triples([triple("The Illusion", "m.0yrlqjm", "theater.theater_production.play", "The Illusion")]) == \
        "[T1] The Illusion (m.0yrlqjm) —theater production › play→ The Illusion"
    assert "[event#42]" in render_triples([triple_with_cvt_tail(eid=42)])

def test_parse_citations_drops_unknown(): assert parse_citations("x [T2][T9] y [T2]", n=5) == ["T2"]

async def test_event_order_mock(deps_mock):
    names = [e for e, _ in await collect(run_turn(chat("which play is produced by the illusion?"), deps_mock, never))]
    assert names[:4] == ["status", "plan", "anchors", "retrieval"] and names[-2:] == ["citations", "done"]

async def test_budget_exhausted_degrades(deps_live_zero_budget):
    done = last_event(await collect(run_turn(chat(..., llm="deepseek"), deps_live_zero_budget, never)))
    assert done["degraded"] == "budget" and "[T" in done["answer"]

async def test_one_tool_round_only(deps_tool_happy_provider):        # provider asks for a tool every time
    evs = await collect(run_turn(...)); assert sum(e == "tool_call" for e, _ in evs) == 1

async def test_disconnect_stops_spending(deps_counting_provider):    # Review Focus
    evs = await collect(run_turn(chat(...), deps_counting_provider, disconnect_after("retrieval")))
    assert deps_counting_provider.llm.provider.calls == 1                   # only normalise ran
    assert deps_counting_provider.budget.status()["calls"] == 1

async def test_chitchat_skips_retrieval(deps_mock_chitchat):
    assert "retrieval" not in [e for e, _ in await collect(run_turn(chat("xin chào"), deps_mock_chitchat, never))]
```

- [ ] **Step 2: Run** `pytest tests/unit/test_render.py tests/agent tests/api/test_chat_sse.py -v` → FAIL.
- [ ] **Step 3: Implement** the loop per AGENT_SPEC §4 (history truncated to the last 2 turns; `llm: "deepseek"` → 403 `LIVE_LLM_DISABLED` unless `settings.llm_provider == "deepseek"`; check `is_disconnected()` before every LLM call and between streamed chunks) and the SSE route.
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `git commit -m "feat: bounded chat agent over SSE"`

### Task 13: Medical ingestion

**Files:**
- Create: `api/fgr/ingest/__init__.py`, `api/fgr/ingest/schema.py`, `api/fgr/ingest/extract.py`, `api/fgr/ingest/commit.py`, `api/fgr/api/routes_ingest.py`
- Modify: `api/fgr/store/graph.py` (add `merge_medical(nodes, relations, triples) -> dict`)
- Test: `api/tests/unit/test_extraction_schema.py`, `api/tests/unit/test_extract.py`, `api/tests/neo4j/test_commit.py`

**Interfaces:**
- Produces: `Extraction` pydantic (AGENT_SPEC §5.1; validators: unique ids, h/t exist, `r` in `RELATIONS` (§5.2), `type` in `TYPES`, ≤ 60 entities, ≤ 120 triples); `split_note(text: str, max_chars: int = 3000) -> list[str]` (split on blank-line sections, ≤ 2 chunks); `extract(text: str | None, sample_id: str | None, llm: GuardedLLM, cache_dir: Path, samples_dir: Path) -> ExtractResult(extraction, cached: bool, usage: Usage | None, warnings: list[str])`; `commit(extraction: Extraction, text_sha: str, graph: GraphStore, emb: EmbeddingStore, encoder: Encoder) -> CommitResult(added: dict, eids: list[int], ms: float)`; node keys `med:{type}:{slug(name_en)}` for shared types, `med:{sha8}:{id}` for Patient/LabResult/Value; routes per API.md §7.

- [ ] **Step 1: Write the failing tests**

```python
@pytest.mark.parametrize("n", ["01", "02", "03"])
def test_samples_validate(n): Extraction.model_validate_json(read(f"samples/medical_vi/note_{n}.extraction.json"))

def test_rejects_unknown_relation(): with pytest.raises(ValidationError): Extraction.model_validate(sample_with_relation("medical.patient.likes"))

def test_mock_extract_replays_sample_and_rejects_other(mock_guarded, tmp_path):
    assert extract(read("samples/medical_vi/note_01.txt"), None, mock_guarded, tmp_path, SAMPLES).extraction.doc_title.startswith("Ca 1")
    with pytest.raises(ApiError, match="MOCK_CANNOT_EXTRACT"): extract("văn bản khác", None, mock_guarded, tmp_path, SAMPLES)

def test_text_limit(client): assert client.post("/api/ingest/extract", json={"text": "a" * 6001, "llm": "mock"}).status_code == 422
def test_split_note(): assert len(split_note(long_note_7000_chars, 3000)) == 2   # long note above 3,000 chars

@pytest.mark.neo4j
def test_commit_shares_concepts_and_resets(graph_loaded, emb, fake_encoder):
    commit(load("01"), "aaaaaaaa", ...); commit(load("03"), "bbbbbbbb", ...)
    assert count_nodes(graph_loaded, key="med:Condition:hypertension") == 1
    assert min(new_eids) > 935049 and emb.n_ent_rows > 614324
    assert graph_loaded.delete_kg("medical")["entities"] > 0 and graph_loaded.counts("medical")["entities"] == 0
```

- [ ] **Step 2: Run** `pytest tests/unit/test_extraction_schema.py tests/unit/test_extract.py tests/neo4j/test_commit.py -v` → FAIL.
- [ ] **Step 3: Implement** per AGENT_SPEC §5 (extraction cache `runtime_dir/extract_cache/<sha256>.json`; one JSON repair call; `commit` embeds new names and relations in batches of 32 through the encoder and appends them to the overlay; `DELETE /api/ingest/medical` also calls `emb.reset_overlay()`).
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `git commit -m "feat: medical note ingestion (extract, preview, commit)"`

### Task 14: Full stack in Docker, demo checks, docs

**Files:**
- Modify: `docker-compose.yml` (add `api`, `loader`; `web` is added by the web plan), `Makefile`, `README.md`
- Create: `api/scripts/smoke_live.py`, `docs/DEMO_SCRIPT.md`
- Test: `api/tests/neo4j/test_end_to_end.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `docker compose up -d` (neo4j + api); `docker compose --profile tools run --rm loader --full`; make targets `fetch`, `load-slice`, `load-full`, `up`, `parity`, `parity-full`, `g4`, `bench-latency`, `smoke-live` (manual, live DeepSeek: one normalise + one answer + extraction of `note_03`; prints spend; refuses unless `LLM_PROVIDER=deepseek`; must stay ≤ $0.01).

- [ ] **Step 1: Write the failing test**

```python
@pytest.mark.neo4j
def test_end_to_end_neo4j_retrieve(live_api):        # against the running compose api (slice 50)
    q = first_slice_question(); r = httpx.post(f"{live_api}/api/retrieve", json={"question_en": q["question"], "anchors": q["topic_eids"], "mode": "adr_aer"}).json()
    assert r["source"] == "neo4j" and r["triples"] and r["timings_ms"]["total"] > 0
```

- [ ] **Step 2: Run** `pytest tests/neo4j/test_end_to_end.py -v` → FAIL (api service missing).
- [ ] **Step 3: Implement** the compose `api` service (design §2.1: mounts `./data/bundle:/data/bundle:ro`, `./data/runtime:/data/runtime`, `hf_cache:/hf`; healthcheck `GET /api/health`; `depends_on: neo4j: condition: service_healthy`) and the `loader` service (`profiles: [tools]`). Write `scripts/smoke_live.py`. Rewrite `README.md` quickstart (fetch → up → load → open) and `docs/DEMO_SCRIPT.md` (a 10-minute demo: benchmark question in the Lab Base vs ADR+AER, aggregate table, a chat in Vietnamese, ingesting a note, the 2-hop medical question).
- [ ] **Step 4: Run** `make load-slice && docker compose up -d && pytest tests/neo4j -v` → PASS; `make load-full && make g4 && make bench-latency` → exit 0.
- [ ] **Step 5: Commit** `git commit -m "chore: full backend stack in compose, demo script"`
