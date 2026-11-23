"""Retrieval numbers of the paper's main table (Table "main") and of the stop-factor ablation.

Metrics, top-K triples (K = 100 in the paper, also 50), on the SubgraphRAG test questions whose answer is in the subgraph:
  Ans   answer recall: share of the gold answers found as head / tail of the top-K triples (protocol of Li et al. 2025)
  GQ    gold-query triple recall: recall of the SPARQL v2 label triples of the test question (labels.py: the triples on
        the gold query's relation chain that lead to a gold answer, plus its constraint triples), on the questions with
        such a label (WebQSP 1,496; CWQ 2,510)
  Path  share of those questions whose top-K contains one complete gold path topic -> answer (eval.py PathHit)
One configuration per retriever, one seed (42). Ours = RBF + BGE + WL + stop factor, SPARQL v2 labels
(rbf/runs/abl_bce_stop_scratch_{ds}). Run from retrieve/:  python paper_main_table.py
"""
import os, pickle, sys
import numpy as np
import torch
from eval import load_targets

ROOT = os.environ.get('SGRAG_ROOT', os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
RR = f'{ROOT}/relrag/runs/authors_ckpt'
RUNS = f'{ROOT}/retrieve/rbf/runs'
DATA = f'{ROOT}/retrieve/rbf/data'
KS = (50, 100)
# setting -> (test set, {retriever: prediction file})
SETTINGS = {
    'WebQSP': ('webqsp', {
        'SubgraphRAG (official release)': f'{ROOT}/reason/scored_triples/webqsp_240912_unidir_test.pth',
        'SubgraphRAG (our training run)': f'{RUNS}/webqsp_dde/adr_webqsp_base.pth',
        'REL-RAG (authors K_80_M_120)': f'{ROOT}/relrag/data/teacher_GPT4o_labels/webqsp_K80_M120.pth',
        'REL-RAG (authors ckpt)': f'{RR}/webqsp_model/retrieval_test_path_bf/retrieval_result_100.pth',
        'Ours': f'{RUNS}/abl_bce_stop_scratch_webqsp/pred_webqsp_base.pth',
        'Ours - stop': f'{RUNS}/abl_pi_full_webqsp/pred_webqsp_base.pth'}),
    'CWQ': ('cwq', {
        'SubgraphRAG (official release)': f'{ROOT}/reason/scored_triples/cwq_240907_unidir_test.pth',
        'SubgraphRAG (our training run)': f'{RUNS}/cwq_dde/adr_cwq_base.pth',
        'REL-RAG (authors K_80_M_120)': f'{ROOT}/relrag/data/teacher_GPT4o_labels/cwq_K80_M120.pth',
        'REL-RAG (authors ckpt)': f'{RR}/cwq_model/retrieval_cwq_test_path_bf/retrieval_result_100.pth',
        'Ours': f'{RUNS}/abl_bce_stop_scratch_cwq/pred_cwq_base.pth',
        'Ours - stop': f'{RUNS}/abl_pi_full_cwq/pred_cwq_base.pth'}),
    'CWQ->WebQSP': ('webqsp', {
        'SubgraphRAG (our training run)': f'{RUNS}/cwq_dde/adr_webqsp_base.pth',
        'REL-RAG (authors ckpt)': f'{RR}/cwq_model/retrieval_test_path_bf/retrieval_result_100.pth',
        'Ours': f'{RUNS}/abl_bce_stop_scratch_cwq/pred_webqsp_base.pth',
        'Ours - stop': f'{RUNS}/abl_pi_full_cwq/pred_webqsp_base.pth'}),
    'WebQSP->CWQ': ('cwq', {
        'SubgraphRAG (our training run)': f'{RUNS}/webqsp_dde/adr_cwq_base.pth',
        'REL-RAG (authors ckpt)': f'{RR}/webqsp_model/retrieval_cwq_test_path_bf/retrieval_result_100.pth',
        'Ours': f'{RUNS}/abl_bce_stop_scratch_webqsp/pred_cwq_base.pth',
        'Ours - stop': f'{RUNS}/abl_pi_full_webqsp/pred_cwq_base.pth'}),
}


def gold_query_labels(dataset):
    """{question id: set of (head name, relation, tail name)} = SPARQL v2 label (chain + constraints) of the test split"""
    labels = pickle.load(open(f'{DATA}/{dataset}/sparqlv2_test.pkl', 'rb'))
    out = {}
    for rec in pickle.load(open(f'{DATA}/{dataset}/test.pkl', 'rb')):
        lab = labels.get(rec['id'])
        if lab is None:
            continue
        names, rels = rec['entity_names'], rec['relation_list']
        h, r, t = rec['h'].tolist(), rec['r'].tolist(), rec['t'].tolist()
        triples = {(names[h[k]], rels[r[k]], names[t[k]]) for k in lab['chain'] + lab['cons']}
        if triples:
            out[rec['id']] = triples
    return out


cache = {}
for setting, (ds, runs) in SETTINGS.items():
    if ds not in cache:
        qids, answers, _, _, gold = load_targets(ds)
        cache[ds] = (answers, gold, gold_query_labels(ds))
    answers, gold, gq = cache[ds]
    gq = {q: v for q, v in gq.items() if q in answers}
    paths = {q: v[2] for q, v in gold.items() if q in answers}
    print(f'\n### {setting} (test {ds}: answer in subgraph {len(answers)}, with GQ label {len(gq)}, with gold path {len(paths)})')
    print('| retriever | #triple@100 | ' + ' | '.join(f'{m}@{k}' for k in KS for m in ('Ans', 'GQ', 'Path')) + ' |')
    print('|---|---|' + '---|' * (3 * len(KS)))
    for name, path in runs.items():
        pred = torch.load(path, weights_only=False)
        v = {f'{m}{k}': [] for m in ('Ans', 'GQ', 'Path') for k in KS}
        size = []
        for q in answers:
            ranked = pred[q]['scored_triples'] if q in pred else []
            size.append(min(len(ranked), 100))
            for k in KS:
                top = {tuple(x[:3]) for x in ranked[:k]}
                ents = {x[0] for x in top} | {x[2] for x in top}
                v[f'Ans{k}'].append(len(answers[q] & ents) / len(answers[q]))
                if q in gq:
                    v[f'GQ{k}'].append(len(gq[q] & top) / len(gq[q]))
                if q in paths:
                    v[f'Path{k}'].append(any(t <= top for t in paths[q]))
        m = {c: 100 * np.mean(x) for c, x in v.items()}
        print(f'| {name} | {np.mean(size):.1f} | ' + ' | '.join(f"{m[f'{c}{k}']:.1f}" for k in KS for c in ('Ans', 'GQ', 'Path')) + ' |',
              flush=True)
