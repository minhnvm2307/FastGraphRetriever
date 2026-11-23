# RBF GTSQA retrieval demo

This repository contains the RBF research code and a local Streamlit demonstration of [Beyond Seen Paths: Relation-Factorised Retrieval for Knowledge Graph Question Answering](docs/RBF-ACL-paper.pdf). The demo ranks triples from official, question-specific GTSQA graphs with `gtsqa_rbf_ckpt/best.pt`.

## Setup

Use Python 3.10–3.12 in a virtual environment:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements-demo.txt
```

If you already use the `medgraphrag` Conda environment, activate it and run `pip install -r requirements-demo.txt` there. The pinned Transformers 4.x version is required for the GTE model; Transformers 5.x does not load this checkpoint's GTE architecture.

Preparation downloads the pinned [fine-tuned BGE-M3 LoRA adapter](https://huggingface.co/minhnvm/bge-m3-relation-lora) into ignored `models/bge_adapter/final/`. It validates the base model and LoRA configuration before processing GTSQA. For an existing copy, use `--adapter /path/to/adapter`.

Prepare a 100-question test sample and launch the UI:

```bash
python -m demo.prepare --count 100
streamlit run demo/app.py
```

To measure recall on the prepared sample, run `python -m demo.evaluate`. It reports answer-node and annotated-gold-triple recall at 50, 100, and 200 triples, plus CPU scoring times. These are sample results, not the paper's full-test results.

The first preparation downloads the official GTSQA test parquet (about 832 MB), GTE-large-en-v1.5, and BGE-M3. For an existing test parquet, add `--source-parquet /path/to/test-00000-of-00001.parquet`. Prepared data and caches live beneath ignored `data/gtsqa_demo/`. Use `--device cuda` for faster one-time feature preparation when CUDA is available; retrieval defaults to CPU.

The UI lets you choose an original test question and its dataset topic entities. You can edit the question, review suggested entities in the same graph, choose any top-k budget from 1 to 500, and highlight a selected triple's best complete path. The interactive graph supports search, zoom, fit, directed relation labels, and click details. Retrieval shows live progress through model setup, feature encoding, RBF scoring, and ranking. To enable optional DeepSeek answers, export `DEEPSEEK_API_KEY` and `DEEPSEEK_MODEL`; the answer action is separate from retrieval.

## Research code and specifications

- `fast_retriever/rbf/rbf.py`: relation-local step/stop scorer and max-product path inference.
- `gtsqa_rbf_ckpt/rbf-train-gtsqa.ipynb`: GTSQA training and feature-preparation reference.
- [PRD](docs/PRD.md) and [data notes](docs/DATA.md): current requirements and data semantics.
