"""Question--relation similarity of the fine-tuned BGE-M3 relation encoder (LoRA r=8 on q/k/v, dense mode, trained on
WebQSP train only with retrieve/train_bge_m3_lora.py --config retrieve/configs/bge_ft/v1_dense_entity_disjoint.json).
Encodes every question (max 96 tokens) and every unique relation text ('split' format of bge_ft/relation_text.py:
'.' and '_' -> spaces; max 48 tokens) with CLS pooling + L2 normalisation (as FlagEmbedding's M3 dense training), and
writes the cosine of each question with each of its candidate relations.
Input: relq/<name>.jsonl rows {id, question, relations}. Output: relq_scores/<name>.pkl = {id: float32 [n_rel]}.
  python relscore_bge.py --make_inputs data/bge/relq                       # one <dataset>_<split>.jsonl per record file
  python relscore_bge.py --adapter <run>/final --inputs data/bge/relq --out data/bge/scores_ft [--base] (base = no adapter)
train_retriever.py --rel_prior bge reads data/bge/scores_ft/<dataset>_<split>.pkl.
"""
import argparse, json, os, pickle
import numpy as np
import torch

from transformers import AutoModel, AutoTokenizer
from peft import PeftModel


def split_relation(r):
    return r.replace('.', ' ').replace('_', ' ')


@torch.no_grad()
def encode(model, tok, texts, max_len, bs=128, dev='cuda'):
    out = []
    order = np.argsort([len(t) for t in texts])
    embs = [None] * len(texts)
    for i in range(0, len(texts), bs):
        idx = order[i:i + bs]
        b = tok([texts[j] for j in idx], max_length=max_len, truncation=True, padding=True, return_tensors='pt').to(dev)
        with torch.autocast('cuda', dtype=torch.float16, enabled=dev == 'cuda'):
            h = model(**b).last_hidden_state[:, 0]
        h = torch.nn.functional.normalize(h.float(), dim=-1).cpu()
        for k, j in enumerate(idx):
            embs[j] = h[k]
    return torch.stack(embs)


def make_inputs(out_dir, data_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')):
    """relq/<dataset>_<split>.jsonl from the records of prepare_full.py (relations in the order of rec['relation_list'])"""
    os.makedirs(out_dir, exist_ok=True)
    for dataset in ('webqsp', 'cwq'):
        for split in ('train', 'val', 'test', 'test_extra'):
            path = f'{data_dir}/{dataset}/{split}.pkl'
            if not os.path.exists(path):
                continue
            with open(f'{out_dir}/{dataset}_{split}.jsonl', 'w') as f:
                for rec in pickle.load(open(path, 'rb')):
                    f.write(json.dumps({'id': str(rec['id']), 'question': rec['question'], 'relations': list(rec['relation_list'])}) + '\n')
            print(f'{dataset}_{split}.jsonl written', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--make_inputs', default=None, help='write the relq/*.jsonl inputs to this directory and exit')
    ap.add_argument('--adapter', default=None); ap.add_argument('--base', action='store_true')
    ap.add_argument('--inputs'); ap.add_argument('--out')
    ap.add_argument('--model', default='BAAI/bge-m3')
    ap.add_argument('--device', default='cuda', help="'cpu' runs fp32 (no autocast)")
    args = ap.parse_args()
    if args.make_inputs:
        return make_inputs(args.make_inputs)
    assert args.inputs and args.out, '--inputs and --out are required'

    tok = AutoTokenizer.from_pretrained(args.model)
    dt = torch.float16 if args.device == 'cuda' else torch.float32
    model = AutoModel.from_pretrained(args.model, torch_dtype=dt).to(args.device).eval()
    if not args.base:
        model = PeftModel.from_pretrained(model, args.adapter).eval()
    os.makedirs(args.out, exist_ok=True)

    for fn in sorted(os.listdir(args.inputs)):
        if not fn.endswith('.jsonl'):
            continue
        rows = [json.loads(l) for l in open(os.path.join(args.inputs, fn))]
        rel_vocab = {}
        for r in rows:
            for rel in r['relations']:
                rel_vocab.setdefault(rel, len(rel_vocab))
        R = encode(model, tok, [split_relation(r) for r in rel_vocab], 48, dev=args.device)
        Q = encode(model, tok, [r['question'] for r in rows], 96, dev=args.device)
        out = {}
        for k, r in enumerate(rows):
            idx = torch.tensor([rel_vocab[x] for x in r['relations']], dtype=torch.long)
            out[r['id']] = (R[idx] @ Q[k]).numpy().astype(np.float32)
        pickle.dump(out, open(os.path.join(args.out, fn.replace('.jsonl', '.pkl')), 'wb'))
        print(fn, len(rows), 'questions,', len(rel_vocab), 'unique relations', flush=True)


if __name__ == '__main__':
    main()
