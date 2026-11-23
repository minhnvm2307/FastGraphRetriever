# GTSQA demo data

The demo uses the [Graphcore/GTSQA dataset](https://huggingface.co/datasets/Graphcore/GTSQA), configuration `gtsqa-with-graphs`, **test** split. It includes official candidate graphs of up to 30,000 triples per question. The full graph configuration is about 14.5 GB; its single test parquet is about 832 MB. Preparation downloads only that test file and stores 100 selected graphs separately under `data/gtsqa_demo/`.

The dataset is Creative Commons Attribution 4.0. Cite Cattaneo, Luschi, and Justus, *Ground-Truth Subgraphs for Better Training and Evaluation of Knowledge Graph Augmented LLMs* (2025) when sharing the demo or results. `manifest.json` records the source revision and example IDs.

## Prepare

1. Preparation fetches the fine-tuned BGE-M3 LoRA from [`minhnvm/bge-m3-relation-lora`](https://huggingface.co/minhnvm/bge-m3-relation-lora) at pinned revision `5db4f7929cc3cd3afa2e61d0705a36a9474db89b`. It stores `adapter_config.json` and `adapter_model.safetensors` under ignored `models/bge_adapter/final/`. This is the encoder referenced by `BGE_ADAPTER` in `gtsqa_rbf_ckpt/rbf-train-gtsqa.ipynb`.
2. Install dependencies as described in the root README.
3. Run `python -m demo.prepare --count 100`. For an existing official parquet, use `--source-parquet /path/to/test-00000-of-00001.parquet`.

The command writes `graphs/<id>.json.gz`, `embeddings.npz`, and `manifest.json` beneath `data/gtsqa_demo/`. The manifest records the adapter revision and weights checksum. It caches GTE and fine-tuned BGE-M3 embeddings for original questions and graph relations. Edited questions need both encoder models at runtime. Downloads and caches stay outside Git.

The text encoders are pinned to GTE-large-en-v1.5 revision `104333d6af6f97649377c2afbde10a7704870c7b` (remote code revision `40ced75c3017eb27626c9d4ea981bde21a2662f4`) and BGE-M3 revision `5617a9f61b028005a4858fdac845db406aefb181`. Transformers 5.x cannot load the GTE revision used here; install the pinned dependencies in `requirements-demo.txt`.

Sampling uses a fixed seed and round-robin groups by test type, hop count, and whether a question has multiple seed entities. Only graphs containing a seed, answer, and annotated gold triple are kept. Increase `--candidate-buffer` if fewer than 100 candidates are usable.

## Evaluation semantics

The UI's gold-triple recall follows the GTSQA training notebook's `answer_subgraph` labels. The dataset card also provides `full_answer_subgraph_wikikg2` for full-graph ground truth; these are distinct targets. Report the chosen target with any exported result. Original examples use dataset seeds. Edited questions use confirmed topic entities and have no valid gold recall against the original question.
