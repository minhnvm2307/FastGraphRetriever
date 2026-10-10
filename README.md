# FastGraphRetriever

A GraphRAG chat application built on the research retriever **WL + ADR + AER** (anchor-equivariant, multi-pass
retrieval over an entity graph):

- the graph lives in **Neo4j**;
- the retriever picks the evidence subgraph **fast on CPU**;
- a budget-guarded LLM agent (DeepSeek) answers in Vietnamese or English, citing the triples it used;
- every answer shows *why* each triple was retrieved: anchor, bridge, or per-anchor pass.

> **Status (2026-10-11): specification stage.** This repository holds the product and technical docs, the implementation
> plans, the data manifest, the research reference code and the provenance tools. The application is built from the plans
> in `docs/superpowers/plans/`.

## Why this retriever

On GrailQA (897 questions, Freebase subgraphs; retriever trained only on WebQSP), answer recall at 50 / 100 / 200 triples:

| Retriever | ansR@50 | @100 | @200 |
|---|---|---|---|
| SubgraphRAG | 71.7 | 79.1 | 86.9 |
| REL-RAG (line graph) | 70.0 | 76.5 | 79.8 |
| WL (base) | 83.2 | 87.0 | 91.8 |
| **WL + ADR + AER** | **83.4** | **89.2** | **93.0** |

On 3+-hop questions, ADR + AER lifts ansR@50 from 27.6 to 36.8. Source: research `BENCHMARK.md`, reproduced from the golden files in the
data bundle (`docs/DATA.md` §2.4).

## Documents

| Doc | For |
|---|---|
| [`docs/PRD.md`](docs/PRD.md) | What and why: users, goals with measurable acceptance, requirements, milestones, risks |
| [`docs/superpowers/specs/2026-10-11-fastgraphretriever-design.md`](docs/superpowers/specs/2026-10-11-fastgraphretriever-design.md) | Architecture, containers, Neo4j model, flows, configuration, errors, testing |
| [`docs/RETRIEVER_SPEC.md`](docs/RETRIEVER_SPEC.md) | The exact algorithm (RT features, DDE, WL scorer, factorised fast path, ADR/AER) and the parity criteria |
| [`docs/DATA.md`](docs/DATA.md) | The S3 data bundle: contents, formats, provenance, reference numbers |
| [`docs/API.md`](docs/API.md) | REST/SSE contract |
| [`docs/AGENT_SPEC.md`](docs/AGENT_SPEC.md) | Chat agent, prompts, budget guard, mock LLM, medical extraction schema |
| [`docs/UI_SPEC.md`](docs/UI_SPEC.md) | Visual design (tokens, type, layout), pages, components |
| [`docs/superpowers/plans/2026-10-11-fgr-backend.md`](docs/superpowers/plans/2026-10-11-fgr-backend.md) | Backend build plan (14 TDD tasks) |
| [`docs/superpowers/plans/2026-10-11-fgr-web.md`](docs/superpowers/plans/2026-10-11-fgr-web.md) | Web build plan (7 TDD tasks) |
| [`docs/evidence/`](docs/evidence/) | Parity and latency measurements made on the research machine |

## Getting the data (dev machine)

```bash
git clone git@github.com:minhnvm2307/FastGraphRetriever.git && cd FastGraphRetriever
cp .env.example .env                 # set AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY
pip install boto3
python tools/s3_bundle.py get        # 1.4 GB -> data/bundle/, sha256-verified, re-runnable
```

Nothing needs re-embedding: the bundle carries the GTE embeddings of all 614k entity names, 4,570 relations and 897
questions, plus the WL checkpoint and the research outputs used as the test oracle.

## Repository layout

```
docs/              PRD, specs, plans, evidence
data_manifest/     checksums of the S3 bundle (bundle_v1.json)
reference/         research code excerpts (read-only, see reference/README.md)
samples/medical_vi synthetic Vietnamese clinical notes + reference extractions
tools/             bundle build/verify/upload/download (see tools/README.md)
.claude/skills/    frontend-design, webapp-testing (Anthropic, Apache-2.0)
api/, web/         created by the build plans
```

## Working with Claude Code on the dev machine

Open the repo and ask Claude Code to execute `docs/superpowers/plans/2026-10-11-fgr-backend.md` with the superpowers
`subagent-driven-development` (or `executing-plans`) skill. Then do the same with the web plan. The vendored skills in
`.claude/skills/` load automatically.
