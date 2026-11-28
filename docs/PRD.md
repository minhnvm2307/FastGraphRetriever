# PRD — RBF GTSQA retrieval demo

*Version 2.0 · 2026-11-24 · Local research demo*

## Purpose

Show how the RBF retriever in [Beyond Seen Paths](RBF-ACL-paper.pdf) turns a question and its topic entities into a ranked evidence subgraph. Researchers and engineers can inspect the method. Retrieval is the main result; an LLM answer is optional.

## Demo flow

1. Select one of 100 prepared GTSQA test examples. Show its question, dataset seed entities, candidate count, hop count, and test type.
2. Keep or edit the question. For edited questions, suggest topic entities by matching names in the selected graph and let the user correct them. Require a confirmed topic entity.
3. Set any integer top-k budget from 1 to 500 (or the candidate count if smaller). Score the full candidate graph, then show exactly the requested number of ranked triples with IDs, heads, relations, tails, and RBF scores.
4. Select a triple to highlight its best complete path and inspect relation-local step and stop factors. With multiple topic entities, report union and all-seeds score contributions; the highlighted path explains the union term.
5. Report feature and RBF scoring times separately. On original questions, report answer and annotated-gold-triple recall at the selected budget. Edited questions are exploratory and are not scored against the original answer.
6. If DeepSeek is configured, offer a separate P2h answer action using up to 100 ranked triples and the selected topic entities. Show API latency and token usage. Retrieval works without LLM credentials.
7. Show live, stage-based progress during model setup, feature encoding, relation-factor computation, RBF path scoring, and ranking. The progress report reflects actual computation boundaries.
8. Render the evidence graph as an interactive, Neo4j Browser-inspired canvas with visible entity captions, directed and labelled relations, semantic colors for topic/answer/path, zoom, fit, entity search, and click details. Highlight the selected triple's best path, marking any path-only context edge outside top-k.

## Data and model

- Use the official `Graphcore/GTSQA` `gtsqa-with-graphs` **test split**. Keep its question-specific candidate graphs separate and select 100 usable examples deterministically across hop counts, seed counts, and test types. Record the source revision and attribution in a local manifest. See [DATA.md](DATA.md).
- Load `gtsqa_rbf_ckpt/best.pt` with 1024-dimensional GTE question/relation vectors, the notebook's lexical feature, and two fine-tuned BGE-M3 similarity features. Preparation downloads the pinned LoRA adapter from `minhnvm/bge-m3-relation-lora` to `models/bge_adapter/final/`; edited questions use that local copy.
- Match the notebook's feature ordering, forward/reverse relation factors, three non-backtracking max-product rounds, stop factor, and two-step complete-path continuation. Score the full candidate graph before top-k selection.
- Cache original-question embeddings and relation vectors. Edited questions encode on demand; the UI distinguishes encoder time from RBF time.

## Implementation and interfaces

- `python -m demo.prepare --count 100` downloads the official test parquet or accepts an existing copy, validates selected graphs, and writes a manifest, graph files, and embeddings under ignored `data/gtsqa_demo/`.
- `python -m demo.evaluate` computes sampled-test answer-node and annotated-gold-triple recall at 50, 100, and 200 triples, with scorer timing.
- A Python service accepts a graph record, question, topic-entity IDs, and any positive integer `k`; it returns ranked triples, factors, recall, timing, and data for a selected-triple path trace. Streamlit calls it directly. The UI limits graph rendering to 500 triples for usable interaction.
- The UI shows clear messages for missing preparation/model files, no matched topic entity, empty/short graphs, and unavailable optional LLM.
- Do not commit dataset graphs, model caches, API keys, or adapter weights.

## Acceptance

- With the checkpoint and adapter supplied, documented commands prepare 100 examples and launch the UI.
- Checkpoint keys and shapes load strictly. Fixed-graph CPU scores and ordering match the GTSQA notebook within floating-point tolerance, including reverse edges and multiple seeds. Path traces reproduce the best-path union score.
- Original examples return at most the chosen budget and report answer/gold-triple recall. Sample aggregate recall is not presented as the paper's full-test result.
- An arbitrary UI budget such as 37 returns 37 ranked triples. The graph preserves edge direction and relation labels, and the selected triple's best path remains identifiable.
- Report measured local CPU timings. The paper's 4.7–4.9 ms scorer medians used one GPU and are not a CPU acceptance target.

## Later work

Neo4j integration, uploaded graphs, medical ingestion, multi-dataset benchmarks, and a general chat agent can be planned separately.
