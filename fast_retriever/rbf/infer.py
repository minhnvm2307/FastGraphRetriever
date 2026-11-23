"""Test-time retrieval of a trained run (RBF or a triple-MLP baseline) on EVERY test question, in the format of
retrieve/eval.py. RBF runs are scored with the complete-path max-marginals switched on (rbf.py, mm), as in every number
of the paper. The topic entities of a question are its anchors.

Questions: data/<ds>/test.pkl (answer in the subgraph) + data/<ds>/test_extra.pkl (the other test questions), i.e. the
full test set (WebQSP 1,638 of 1,639: one question has no subgraph; CWQ 3,531); zero-shot sets: data_ext/<name>.

  python infer.py --run abl_bce_stop_scratch_webqsp --dataset cwq
Writes runs/<run>/pred_<dataset>_base.pth = {question id: {'question', 'q_entity', 'a_entity_in_graph',
'scored_triples': [(head, relation, tail, score)] top 500}} -> evaluate with retrieve/eval.py.
"""
import argparse, json, os
import torch
from tqdm import tqdm

from train_retriever import EXTERNAL, ROOT, attach_bge_prior, build_scorer, load_split, to_device

RUNS = f'{ROOT}/rbf/runs'
TOP_K = 500


def main():
    ap = argparse.ArgumentParser(description='Retrieve top-500 triples for every test question (see the docstring).')
    ap.add_argument('--run', required=True, help='run directory name in rbf/runs (or a path)')
    ap.add_argument('--dataset', required=True, choices=['webqsp', 'cwq', *EXTERNAL], help='test set to retrieve on')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--top_k', type=int, default=TOP_K, help='triples kept per question')
    ap.add_argument('--limit', type=int, default=0, help='first N questions only (smoke test)')
    args = ap.parse_args()
    run_dir = args.run if os.path.isdir(args.run) else f'{RUNS}/{args.run}'
    train_args = json.load(open(f'{run_dir}/result.json'))['args']
    model = build_scorer(train_args)
    model.load_state_dict(torch.load(f'{run_dir}/best.pt', map_location=args.device))
    model.to(args.device).eval()
    if hasattr(model, 'rbf'):
        model.rbf.mm = True                                      # complete-path scoring at test time
    questions = []
    for split in ('test', 'test_extra'):
        records, emb = load_split(args.dataset, split)
        if not records:                                        # external sets have no test_extra
            continue
        attach_bge_prior(records, args.dataset, split, train_args.get('rel_prior', 'none'))
        questions += [(rec, emb) for rec in records]
    if args.limit:
        questions = questions[:args.limit]
    out = {}
    with torch.no_grad():
        for rec, emb in tqdm(questions, desc=f'{os.path.basename(run_dir)} on {args.dataset}'):
            scores = model(to_device(rec, emb, args.device, anchors=tuple(sorted(set(rec['q_ent'])))))
            names, relations = rec['entity_names'], rec['relation_list']
            top = torch.topk(scores, min(args.top_k, len(scores))).indices.tolist(); scores = scores.cpu()
            out[rec['id']] = {
                'question': rec['question'], 'q_entity': [names[i] for i in rec['q_ent']],
                'a_entity_in_graph': [names[i] for i in rec.get('a_ent', [])],
                'scored_triples': [(names[rec['h'][i]], relations[rec['r'][i]], names[rec['t'][i]], float(scores[i])) for i in top]}
    path = f'{run_dir}/pred_{args.dataset}_base' + ('_smoke' if args.limit else '') + '.pth'
    torch.save(out, path); print(f'{len(out)} questions -> {path}')


if __name__ == '__main__':
    main()
