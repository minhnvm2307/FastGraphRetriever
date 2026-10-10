# RETRIEVER SPEC — WL + ADR + AER, exact and fast

This document defines the retrieval algorithm precisely enough to reimplement it **without** the research repo. The
verbatim research code it is derived from is in `reference/` (read-only; do not import it from the app). Anything marked
**MUST** is checked by a parity test against `golden/` (see §7).

Notation: a question `q`; a candidate subgraph with `n` entities (local ids 0..n-1) and `m` directed triples
`(h_i, r_i, t_i)`; `R` distinct relations in it (local relation ids 0..R-1); an anchor set `A ⊆ {0..n-1}`.

## 1. Inputs

| Input | Shape | Source |
|---|---|---|
| `q_emb` | `[1, 1024]` | GTE-large-en-v1.5 of the **English** question (bundle `emb/q.f16.npy` for benchmark questions) |
| `ent_emb` | `[n_text, 1024]` | `emb/ent.f16.npy[emb_row]` for the text entities, in local order |
| `n_non_text` | int | Entities without an embedding. They use the model's learned `non_text_entity_emb`. Always 0 for the GrailQA bundle; medical overlay entities always have an embedding |
| `rel_emb` | `[R, 1024]` | `emb/rel.f16.npy[rid]` |
| `h, r, t` | `[m]` int64 | Local ids. Entities are ordered **text first, then non-text** |
| `question_text` | str | English question (for the lexical score) |
| anchors | set of local ids | The topic entities (benchmark: `q_entity_id_list`; chat: linked entities) |

All embeddings are cast to **float32** before any computation (they are stored as fp16).

## 2. Relation-topic (RT) features (WL config: `hybrid`, α = 0.5, top-5 mask, no calibration)

Computed once per question, independent of the anchors. **MUST** match `reference/rt_features.py`.

```
semantic[j] = rel_emb[j] · q_emb                       # both already L2-normalised: cosine
lexical[j]  = compute_lexical_score(clean_text(question_text), relation_name[j])
score[j]    = 0.5 * lexical[j] + 0.5 * semantic[j]
mask[j]     = 1 if j in torch.topk(score, min(5, R)).indices else 0
rtf         = stack([score, mask], dim=1)               # [R, 2]
```

- `clean_text`: lowercase, strip ASCII punctuation (`string.punctuation`), split on whitespace, drop `STOPWORDS`.
- `compute_lexical_score`: split the relation name on `.`. The parts get weights 1.0 (last), 0.5 (second to last), 0.2 (others). Split each part on `_`, skip tokens shorter than 2 characters and stopwords. Each remaining token takes its best `get_token_similarity` against the question tokens (1.0 exact; 0.9 same `SYNONYM_GROUPS` group; 0.9/0.8 shared prefix rules). Return the weighted mean, or 0 if there are no tokens. Copy the token tables verbatim.
- Ties in `topk`: use `torch.topk` on CPU float32 exactly as written (tie order affects the mask on ~1/30 questions).

## 3. The WL scorer (one "pass" for one anchor set)

Checkpoint: `models/wl_effective_num_s42/cpt.pth`, i.e. `model_state_dict` with
`non_text_entity_emb.weight [1,1024]`, `pred.0.weight [1024,4134]`, `pred.0.bias [1024]`, `pred.2.weight [1,1024]`,
`pred.2.bias [1]`. Config: `topic_pe=True`, `dde_mode='both'`, `DDE_kwargs={num_rounds:2, num_reverse_rounds:2}`,
`weighted_dde_floor=0.1`, `use_relation_topic_mask=True`.

### 3.1 DDE (directional distance encoding)

`PEConv(edge_index, x, w)`: a **weighted mean** of the in-neighbours' features:
`out[v] = Σ_{(u→v)} w_e x[u] / Σ_{(u→v)} w_e`, and `out[v] = 0` when the weight sum is ≤ 1e-12. In pure torch this is
`index_add_` on the destination, then a division with a mask. No torch_geometric is needed.

```
topic_oh = one_hot(1[v ∈ A], 2)                 # [n, 2]: column 1 = "is anchor"
fwd edges: h→t ; rev edges: t→h
DDE(x0, w):  f1 = PEConv(fwd, x0, w); f2 = PEConv(fwd, f1, w)
             b1 = PEConv(rev, x0, w); b2 = PEConv(rev, b1, w)
             return [f1, f2, b1, b2]               # 4 × [n, 2]
dde_plain    = DDE(topic_oh, w = 1)
dde_weighted = DDE(topic_oh, w = clamp(rtf[r, 0], min=0.1))    # per-edge weight from its relation's RT score
```

### 3.2 Triple features and MLP

```
ent_full = cat(ent_emb, non_text_entity_emb.expand(n_non_text, 1024))          # [n, 1024]
h_e      = cat(ent_full, topic_oh, *dde_plain, *dde_weighted)                   # [n, 1024 + 2 + 8 + 8] = [n, 1042]
x_i      = cat(q_emb, h_e[h_i], rel_emb[r_i], h_e[t_i], one_hot(rtf[r_i,1], 2)) # [m, 4134]
logit_i  = W2 · relu(W1 x_i + b1) + b2                                          # [m]
```

The input layout of `W1` (column offsets) is
`q 0:1024 | h_emb 1024:2048 | h_feat 2048:2066 | r 2066:3090 | t_emb 3090:4114 | t_feat 4114:4132 | mask 4132:4134`.

## 4. Fast path: the factorised first layer (the "fast" in FastGraphRetriever)

Only 38 of the 4,134 input dimensions depend on the anchor set (`h_feat`, `t_feat`: 18 each; the mask does not depend on
anchors). So `W1 x_i + b1` splits into a per-question part, computed once, and a small per-pass part:

```
once per question:
  base = (W_q q + b1) + (ent_full W_h^T)[h] + (rel_emb W_r^T + onehot(mask) W_m^T)[r] + (ent_full W_t^T)[t]   # [m, 1024]
per pass (anchor set A):
  feat = cat(topic_oh, *dde_plain, *dde_weighted)                                                             # [n, 18]
  logit = relu(base + (feat W_hf^T)[h] + (feat W_tf^T)[t]) · w2 + b2
```

Measured on the research machine (2 CPU threads, machine shared, 30 GrailQA questions; `docs/evidence/bench_factorized.json`):

| | p50 | max (29k triples) |
|---|---|---|
| Original forward, one pass | 117 ms | 2,020 ms |
| Factorised: once-per-question `prepare` | 41 ms | 966 ms |
| Factorised: each pass | **12 ms** | 418 ms |
| Difference vs original | max 1.5e-5 | top-500 set identical on 29/30 (tie at the cut) |

ADR+AER uses 2 distinct passes at p50 and up to 6, so the scorer goes from about `passes × 117 ms` to about `41 + passes × 12 ms`.
The implementation **MUST** keep the unfactorised forward (`reference_forward`) for tests and assert agreement to ≤ 1e-4.

Further optional speed-ups (only if the latency targets in the PRD are missed; measure first):

1. Cache `ent_full W_h^T` and `ent_full W_t^T` per `emb_row` in an LRU (entities recur across questions).
2. Precompute them offline for all 614k names: 2 × 1.2 GB fp16, which turns `prepare` into gathers.
3. Reduce `torch.set_num_threads` contention: API threads 4, Neo4j on separate cores.

## 5. Multi-pass retrieval: base, ADR, AER, ADR+AER

**MUST** match `reference/sg_adr_infer.py` (B = 3, max_K = 500).

```
S(A)      = scores of a pass with anchor set A (memoised by sorted(A): identical sets are scored once)
z(v)      = (v - mean(v)) / (std(v) + 1e-6)          # torch.std: unbiased (n-1)
chain(A)  = [S(A)] + ([S(A ∪ new)] if new else [])
            where top = topk(S(A), min(3, m)).indices, new = ({h[top]} ∪ {t[top]}) \ A      # "bridges"
tops      = anchors;  single = [{a} for a in sorted(tops)] if |tops| > 1 else []

base    : passes = [S(tops)]
adr     : passes = [S(tops)] + [S(x) for x in single]
aer     : passes = chain(tops)
adr_aer : passes = chain(tops) + [p for x in single for p in chain(x)]

fused = passes[0] if len(passes) == 1 else max_over_passes(z(p))      # elementwise max of z-scores
result = topk(fused, min(K, m))
```

Notes that matter for parity:
- A single pass is **not** z-normalised (raw logits are returned). With several passes, the fused scores are z-scores.
- The memo key is the sorted anchor tuple. `adr_aer` on a one-anchor question equals `aer`.
- Bridges come from the top-3 **triples** of that pass (both endpoints), not the top-3 entities.

Provenance the app adds (not in the research code, it does not change scores): for each returned triple, the pass
whose z-score was the max (`best_pass`), and for each pass its anchor set and kind (`base`, `bridge`, `single`,
`single+bridge`). The UI uses this to show *why* a triple was retrieved.

## 6. Candidate subgraph sources

The scorer is source-agnostic. A source returns `(entities[eid, name, emb_row], triples[h_eid, rid, t_eid], rel names)`, and the
pipeline maps them to local ids (entities ordered text first).

| Source | Use | Definition |
|---|---|---|
| `BenchmarkSource` | Benchmark page, parity tests | The question's research subgraph: `benchmark/test.pkl` record, global ids via `local2eid`. Local order, relation order and triple order **identical to the research input** (required for exact parity, because tie-breaks depend on order) |
| `Neo4jSource` | Chat, Retrieval Lab | 2-hop neighbourhood of the anchors in the global graph, restricted to `kg`, capped (below) |

`Neo4jSource` algorithm (deterministic):

1. `E1` = all edges incident to an anchor, either direction (cap `hop1_cap`, default 5,000).
2. `N1` = endpoints of `E1` that are not anchors.
3. `E2` = for each `b ∈ N1`, up to `fanout_cap` (default 200) incident edges. **Priority order:** edges whose relation is in the question's top-`rt_priority` (default 50) relations by RT score (§2, computed over all relations of the KG, ~5 ms), then the rest. Ties are broken by `(rid, other_eid)`.
4. The result is `E1 ∪ E2`, truncated to `max_triples` (default 10,000 for chat, 30,000 for the Lab "full" setting) with `E1` first. Report `truncated=true` when any cap bit.
5. Triple order: sorted by `(h_eid, rid, t_eid)`. Entity order: anchors first, then by eid.

Product-path check (acceptance): with gold anchors, `Neo4jSource` ansR@100 on 200 benchmark questions must be ≥ BenchmarkSource
ansR@100 − 3.0. The global graph is a superset of each question's subgraph, so the caps are the only reason it can lose.

## 7. Parity tests (MUST pass before any UI work depends on the retriever)

| Test | Data | Pass criterion |
|---|---|---|
| RT features | 30 sampled questions | `score` equal to `reference/rt_features.py` within 1e-6; mask identical |
| DDE | toy graph + 30 questions | equal to the reference PEConv/DDE math within 1e-6 |
| Factorised vs unfactorised | 30 questions × 4 modes | max abs diff ≤ 1e-4 |
| Golden, quick (CI) | 30 questions (`np.linspace(0, 896, 30).astype(int)`), BenchmarkSource, 4 modes | top-50 set equal on ≥ 29/30; where order is equal, position-wise |Δscore| ≤ 1e-3 |
| Golden, full (manual, `make parity-full`) | 897 questions, 4 modes | ansR@50/100/200 within ±0.05 of DATA.md §2.4 |

The quick-golden criterion is exactly what `tools/verify_bundle.py` measured with the original research class (29/30 order,
30/30 sets, Δ ≤ 2.3e-5), so it is achievable on CPU.

## 8. Question encoding

- Same as `reference/gte_large_en.py`: tokenizer `max_length=8192`, `padding=True`, `truncation=True`; take the CLS vector (`last_hidden_state[:, 0]`); L2-normalise.
- On CPU, load with plain `AutoModel.from_pretrained(..., trust_remote_code=True)`. Do not pass `unpad_inputs` / `use_memory_efficient_attention`: those need xformers/GPU. This was verified to match the bundle vectors to 6e-5.
- Measured: 607 ms p50, 1.27 s max per question (2 threads, shared machine). Warm up on start. LRU-cache by text.
- New entity and relation names (medical ingestion) use the same function, batched (batch 32).
