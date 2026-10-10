"""Evidence for the "fast scorer" design (docs/RETRIEVER_SPEC.md §4): the WL MLP's first layer factorised so that
each extra ADR/AER pass only touches the 36 anchor-dependent input dims. Research machine only.

For sampled questions it compares, pass by pass, the ORIGINAL model output with the factorised computation
(max abs score difference, identical top-K), and times both. Optionally times GTE question encoding on CPU.

  python tools/bench_factorized.py --bundle data/bundle --research /home/minhnv/Lab/SubgraphRAG/retrieve --n 30 [--gte]
"""
import argparse, gzip, json, os, pickle, sys, time
import numpy as np
import torch
import torch.nn.functional as F


def read_tsv(path):
    with gzip.open(path, 'rt', encoding='utf-8', newline='\n') as f:
        next(f)
        return [line.rstrip('\n').split('\t') for line in f]


class Factorised:
    """pre = W1·[q, h_emb, h_feat, r, t_emb, t_feat, mask] + b1, split by input block (layout of Retriever.forward)."""
    def __init__(self, model):
        W, self.b1 = model.pred[0].weight.detach(), model.pred[0].bias.detach()
        self.w2, self.b2 = model.pred[2].weight.detach()[0], model.pred[2].bias.detach()
        D = W.shape[0]
        nf = (W.shape[1] - 4 * D - 2) // 2                      # anchor-dependent entity dims per slot (18 for WL)
        o = np.cumsum([0, D, D, nf, D, D, nf, 2])
        self.Wq, self.Wh, self.Whf, self.Wr, self.Wt, self.Wtf, self.Wm = (W[:, o[i]:o[i + 1]] for i in range(7))
        self.model = model

    def prepare(self, q, ent_full, rel, rtf, h, r, t):
        """once per question: everything that does not depend on the anchor set"""
        mask_oh = F.one_hot(rtf[:, 1].long(), 2).float()
        Pr = rel @ self.Wr.T + mask_oh @ self.Wm.T                   # relation block + relation-topic mask
        base = (q @ self.Wq.T + self.b1) + (ent_full @ self.Wh.T)[h] + Pr[r] + (ent_full @ self.Wt.T)[t]
        return base

    def feats(self, topic_oh, h, r, t, rtf):
        m = self.model
        ei, rei = torch.stack([h, t]), torch.stack([t, h])
        parts = [topic_oh] + m.dde(topic_oh, ei, rei) + m.dde(topic_oh, ei, rei, edge_weight=torch.clamp(rtf[r, 0], min=m.weighted_dde_floor))
        return torch.cat(parts, dim=1)

    def score(self, base, feat, h, t):
        pre = base + (feat @ self.Whf.T)[h] + (feat @ self.Wtf.T)[t]
        return torch.relu(pre) @ self.w2 + self.b2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bundle', default='data/bundle')
    ap.add_argument('--research', default='/home/minhnv/Lab/SubgraphRAG/retrieve')
    ap.add_argument('--n', type=int, default=30)
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--gte', action='store_true')
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(here, '..', 'reference')); sys.path.insert(0, args.research)
    from rt_features import clean_text, compute_lexical_score, RTHelpers
    from src.model.retriever import Retriever

    b = args.bundle
    P = pickle.load(open(f'{b}/benchmark/test.pkl', 'rb'))
    l2e = np.load(f'{b}/benchmark/local2eid.npz')
    emb_row = np.array([int(r[3]) for r in read_tsv(f'{b}/kg/entities.tsv.gz')])
    rid = {r[1]: int(r[0]) for r in read_tsv(f'{b}/kg/relations.tsv.gz')}
    ENT = np.load(f'{b}/emb/ent.f16.npy', mmap_mode='r'); REL, Q = np.load(f'{b}/emb/rel.f16.npy'), np.load(f'{b}/emb/q.f16.npy')
    golden = torch.load(f'{b}/golden/adr_grailqa_oda_base.pth', weights_only=False)
    cpt = torch.load(f'{b}/models/wl_effective_num_s42/cpt.pth', map_location='cpu', weights_only=False)
    cfg = cpt['config']['retriever']
    model = Retriever(1024, **{k: v for k, v in cfg.items() if k != 'pos_weight'}); model.load_state_dict(cpt['model_state_dict']); model.eval()
    fz = Factorised(model)

    stats = {'n_triples': [], 'orig_pass_ms': [], 'prep_ms': [], 'fact_pass_ms': [], 'max_abs_diff': [], 'top500_same_set': 0,
             'golden_top50_same_order': 0, 'golden_max_abs_diff_top50': []}
    with torch.no_grad():
        for k in np.linspace(0, len(P) - 1, args.n).astype(int):
            s = P[k]; nt = len(s['text_entity_list']); n_nt = len(s['non_text_entity_list'])
            loc = l2e[s['id']]
            ent = torch.from_numpy(np.asarray(ENT[emb_row[loc[:nt]]])).float()
            rel = torch.from_numpy(REL[[rid[x] for x in s['relation_list']]]).float()
            q = torch.from_numpy(Q[k:k + 1]).float()
            h, r, t = (torch.tensor(s[x]) for x in ('h_id_list', 'r_id_list', 't_id_list'))
            sem = (rel @ q.T).squeeze(1)
            lex = torch.tensor([compute_lexical_score(clean_text(s['question']), x) for x in s['relation_list']], dtype=torch.float32)
            score = RTHelpers._select_relation_score('hybrid', sem, lex, cfg.get('relation_score_alpha', 0.5), len(rel))
            rtf = torch.stack([score, RTHelpers._select_relation_topic_mask(score, cfg['topk_rt'])], dim=1)
            m = torch.zeros(nt + n_nt, dtype=torch.long); m[s['q_entity_id_list']] = 1
            topic_oh = F.one_hot(m, 2).float()

            t0 = time.perf_counter(); orig = model(h, r, t, q, ent, n_nt, rel, topic_oh, rtf).reshape(-1); t1 = time.perf_counter()
            ent_full = torch.cat([ent, model.non_text_entity_emb.weight.expand(n_nt, -1)], 0)
            base = fz.prepare(q, ent_full, rel, rtf, h, r, t); t2 = time.perf_counter()
            fact = fz.score(base, fz.feats(topic_oh, h, r, t, rtf), h, t); t3 = time.perf_counter()

            stats['n_triples'].append(len(h)); stats['orig_pass_ms'].append((t1 - t0) * 1e3)
            stats['prep_ms'].append((t2 - t1) * 1e3); stats['fact_pass_ms'].append((t3 - t2) * 1e3)
            stats['max_abs_diff'].append(float((orig - fact).abs().max()))
            K = min(500, len(orig))
            stats['top500_same_set'] += set(torch.topk(orig, K).indices.tolist()) == set(torch.topk(fact, K).indices.tolist())
            g = golden[s['id']]['scored_triples'][:50]
            top = torch.topk(fact, min(50, len(fact)))
            names = s['text_entity_list'] + s['non_text_entity_list']
            mine = [(names[h[j]], s['relation_list'][r[j]], names[t[j]]) for j in top.indices.tolist()]
            stats['golden_top50_same_order'] += mine == [x[:3] for x in g]
            stats['golden_max_abs_diff_top50'].append(max(abs(a - x[3]) for a, x in zip(top.values.tolist(), g)))
    nt_ = np.array(stats['n_triples']); n = len(nt_)
    rep = {'questions': n, 'threads': args.threads, 'triples_p50': int(np.median(nt_)), 'triples_max': int(nt_.max()),
           'factorised_vs_original_max_abs_diff': max(stats['max_abs_diff']),
           'factorised_top500_same_set': f"{stats['top500_same_set']}/{n}",
           'factorised_vs_golden_top50_same_order': f"{stats['golden_top50_same_order']}/{n}",
           'factorised_vs_golden_max_abs_diff_top50': max(stats['golden_max_abs_diff_top50']),
           'orig_pass_ms_p50': round(float(np.median(stats['orig_pass_ms'])), 1), 'orig_pass_ms_max': round(max(stats['orig_pass_ms']), 1),
           'prep_ms_p50': round(float(np.median(stats['prep_ms'])), 1), 'prep_ms_max': round(max(stats['prep_ms']), 1),
           'fact_pass_ms_p50': round(float(np.median(stats['fact_pass_ms'])), 1), 'fact_pass_ms_max': round(max(stats['fact_pass_ms']), 1)}
    if args.gte:
        from transformers import AutoModel, AutoTokenizer
        tok = AutoTokenizer.from_pretrained('Alibaba-NLP/gte-large-en-v1.5')
        enc = AutoModel.from_pretrained('Alibaba-NLP/gte-large-en-v1.5', trust_remote_code=True).eval()
        qs = [P[i]['question'] for i in range(20)]
        with torch.no_grad():
            enc(**tok(qs[:1], return_tensors='pt'))                   # warm-up
            ts = []
            for x in qs:
                t0 = time.perf_counter()
                e = F.normalize(enc(**tok([x], max_length=8192, padding=True, truncation=True, return_tensors='pt')).last_hidden_state[:, 0], p=2, dim=1)
                ts.append((time.perf_counter() - t0) * 1e3)
            diff = float((e[0] - torch.from_numpy(Q[19]).float()).abs().max())
        rep.update({'gte_question_ms_p50': round(float(np.median(ts)), 1), 'gte_question_ms_max': round(max(ts), 1),
                    'gte_cpu_vs_bundle_q_max_abs_diff': diff})
    print(json.dumps(rep, indent=1))
    json.dump(rep, open(f'{b}/../bench_factorized.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
