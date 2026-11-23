"""Max-product vs sum-product path aggregation (rbf/abl_sum_product.py), same checkpoints, test time only (paper Tables "ablation", "sum-max").

Metrics as paper_main_table.py (Ans / GQ / Path at K = 50, 100 on the questions with the answer in the subgraph), plus
answer recall@50 on the WebQSP->CWQ groups of the controlled table (seen path / novel composition / unseen relation,
CWQ test questions split against the WebQSP training labels v2: seen path / novel composition / unseen relation).
Run from retrieve/:  python paper_ablation_sum.py > runs_paper_ablation_sum.txt
"""
import os, pickle, sys
import numpy as np
import torch
from eval import load_targets

ROOT = os.environ.get('SGRAG_ROOT', os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
RUNS = f'{ROOT}/retrieve/rbf/runs'
DATA = f'{ROOT}/retrieve/rbf/data'
sys.path.insert(0, f'{ROOT}/retrieve/rbf')
from labels import gold_queries_for

KS = (50, 100)
SETTINGS = [('WebQSP', 'webqsp', 'webqsp'), ('CWQ', 'cwq', 'cwq'), ('CWQ->WebQSP', 'cwq', 'webqsp'), ('WebQSP->CWQ', 'webqsp', 'cwq')]
MODELS = {'Ours': 'abl_bce_stop_scratch', 'Ours - stop': 'abl_pi_full'}


def gold_query_labels(dataset):
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


def cwq_groups():
    """CWQ test question ids by relation-path group w.r.t. the WebQSP training labels (paper Section "Seen paths and novel compositions")"""
    tr = {r['id']: r for r in pickle.load(open(f'{DATA}/webqsp/train.pkl', 'rb'))}
    lab = pickle.load(open(f'{DATA}/webqsp/sparqlv2_train.pkl', 'rb'))
    seen_rel = set()
    for qid, l in lab.items():
        r = tr.get(qid)
        if r is not None:
            seen_rel |= {r['relation_list'][r['r'][j]] for j in l['chain'] + l['cons']}
    seen_chain = {tuple(h['predicate'] for h in q['chain']) for q in gold_queries_for('webqsp', tr).values()}
    recs = {r['id']: r for r in pickle.load(open(f'{DATA}/cwq/test.pkl', 'rb'))}
    groups = {'Seen': [], 'Novel': [], 'Unseen rel': []}
    for qid, q in gold_queries_for('cwq', recs).items():
        rels = [h['predicate'] for h in q['chain']]
        if not recs[qid]['a_ent']:
            continue
        g = 'Unseen rel' if any(p not in seen_rel for p in rels) else ('Seen' if tuple(rels) in seen_chain else 'Novel')
        groups[g].append(qid)
    return groups


cache, groups = {}, cwq_groups()
for setting, train_ds, test_ds in SETTINGS:
    if test_ds not in cache:
        qids, answers, _, _, gold = load_targets(test_ds)
        cache[test_ds] = (answers, gold, gold_query_labels(test_ds))
    answers, gold, gq = cache[test_ds]
    gq = {q: v for q, v in gq.items() if q in answers}
    paths = {q: v[2] for q, v in gold.items() if q in answers}
    print(f'\n### {setting}')
    head = [f'{m}@{k}' for k in KS for m in ('Ans', 'GQ', 'Path')] + ([f'{g} ({len(v)}) Ans@50' for g, v in groups.items()] if setting == 'WebQSP->CWQ' else [])
    print('| model | semiring | ' + ' | '.join(head) + ' |'); print('|---|---|' + '---|' * len(head))
    for name, run in MODELS.items():
        for semiring, suffix in (('max', 'base'), ('sum', 'sum')):
            pred = torch.load(f'{RUNS}/{run}_{train_ds}/pred_{test_ds}_{suffix}.pth', weights_only=False)
            v = {f'{m}{k}': [] for m in ('Ans', 'GQ', 'Path') for k in KS}
            ans50 = {}
            for q in answers:
                ranked = pred[q]['scored_triples'] if q in pred else []
                for k in KS:
                    top = {tuple(x[:3]) for x in ranked[:k]}
                    ents = {x[0] for x in top} | {x[2] for x in top}
                    v[f'Ans{k}'].append(len(answers[q] & ents) / len(answers[q]))
                    if k == 50:
                        ans50[q] = v['Ans50'][-1]
                    if q in gq:
                        v[f'GQ{k}'].append(len(gq[q] & top) / len(gq[q]))
                    if q in paths:
                        v[f'Path{k}'].append(any(t <= top for t in paths[q]))
            cells = [f"{100 * np.mean(v[f'{m}{k}']):.1f}" for k in KS for m in ('Ans', 'GQ', 'Path')]
            if setting == 'WebQSP->CWQ':
                for g, ids in groups.items():       # groups use the answers of test.pkl, as in novel_group.py
                    vals = []
                    for qid in ids:
                        top = pred[qid]['scored_triples'][:50]
                        ents = {x[0] for x in top} | {x[2] for x in top}
                        a = set(pred[qid]['a_entity_in_graph'])
                        vals.append(len(a & ents) / len(a))
                    cells.append(f'{100 * np.mean(vals):.1f}')
            print(f'| {name} | {semiring} | ' + ' | '.join(cells) + ' |', flush=True)
