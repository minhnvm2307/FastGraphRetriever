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

The UI has four tabs: **Retrieval**, **Chatbot**, **Evaluation**, and **Data & status**. Retrieval places question, topic-entity, and top-k controls beside the interactive evidence graph. You can edit the question, review suggested entities in the same graph, choose any top-k budget from 1 to 500, and inspect a selected triple's best complete path. The graph supports search, zoom, fit, directed relation labels, and click details. Chatbot retrieves fresh evidence for each message, keeps conversation history within the selected graph, and shows retrieval time, answer time, total time, and RBF scoring time separately. Evaluation runs sampled-test recall at 50, 100, and 200 triples on demand. Data & status shows whether the checkpoint and prepared files are available and gives the preparation command when they are missing. DeepSeek answers use the supplied P2h reasoning-path format and report API latency and token usage. A local ignored `.env` with `DEEPSEEK_API_KEY` enables answers. Without it, Chatbot still retrieves and displays evidence.

Without prepared GTSQA data, Retrieval opens a small **synthetic preview** automatically. It uses the real RBF checkpoint with stand-in embeddings so the controls, graph, and path explanation can be tried immediately. Preview scores and recall are illustrative and are not GTSQA results. Preparing GTSQA replaces the preview with the real sample.

On Windows PowerShell, you can also enable DeepSeek answers for the current terminal before starting the app:

```powershell
$env:DEEPSEEK_API_KEY = "your-key"
$env:DEEPSEEK_MODEL = "deepseek-flash"
.venv\Scripts\python.exe -m streamlit run demo/app.py
```

## Research code and specifications

- `fast_retriever/rbf/rbf.py`: relation-local step/stop scorer and max-product path inference.
- `gtsqa_rbf_ckpt/rbf-train-gtsqa.ipynb`: GTSQA training and feature-preparation reference.
- [PRD](docs/PRD.md) and [data notes](docs/DATA.md): current requirements and data semantics.
