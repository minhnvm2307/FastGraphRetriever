"""Check that the bundle reproduces the research retriever, and measure CPU latency (research machine only).

Rebuilds each sampled question's inputs from the bundle alone (global embedding tables + benchmark/test.pkl +
local2eid), computes the relation-topic features live (reference/rt_features.py), scores with the ORIGINAL research
model class (SubgraphRAG retrieve/src/model/retriever.py, needs torch_geometric) using the ADR/AER logic of
reference/sg_adr_infer.py, and compares with golden/adr_grailqa_oda_<mode>.pth (computed on GPU by the research code).

  python tools/verify_bundle.py --bundle data/bundle --research /home/minhnv/Lab/SubgraphRAG/retrieve --n 30
"""
import argparse, gzip, json, os, pickle, sys, time
import numpy as np
import torch
import torch.nn.functional as F

z = lambda v: (v - v.mean()) / (v.std() + 1e-6)


def read_tsv(path):
    with gzip.open(path, 'rt', encoding='utf-8', newline='\n') as f:
        next(f)
        return [line.rstrip('\n').split('\t') for line in f]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bundle', default='data/bundle')
    ap.add_argument('--research', default='/home/minhnv/Lab/SubgraphRAG/retrieve')
    ap.add_argument('--n', type=int, default=30)
    ap.add_argument('--threads', type=int, default=2)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(here, '..', 'reference'))
    sys.path.insert(0, args.research)
    from rt_features import clean_text, compute_lexical_score, RTHelpers
    from src.model.retriever import Retriever

    b = args.bundle
    P = pickle.load(open(f'{b}/benchmark/test.pkl', 'rb'))
    l2e = np.load(f'{b}/benchmark/local2eid.npz')
    emb_row = np.array([int(r[3]) for r in read_tsv(f'{b}/kg/entities.tsv.gz')])
    rid = {r[1]: int(r[0]) for r in read_tsv(f'{b}/kg/relations.tsv.gz')}
    ENT = np.load(f'{b}/emb/ent.f16.npy', mmap_mode='r')
    REL, Q = np.load(f'{b}/emb/rel.f16.npy'), np.load(f'{b}/emb/q.f16.npy')
    golden = {m: torch.load(f'{b}/golden/adr_grailqa_oda_{m}.pth', weights_only=False) for m in ('base', 'adr', 'aer', 'adr_aer')}
    cpt = torch.load(f'{b}/models/wl_effective_num_s42/cpt.pth', map_location='cpu', weights_only=False)
    cfg = cpt['config']['retriever']
    model = Retriever(1024, **{k: v for k, v in cfg.items() if k != 'pos_weight'})
    model.load_state_dict(cpt['model_state_dict']); model.eval()

    idx = np.linspace(0, len(P) - 1, args.n).astype(int)
    rep = {m: {'top50_jacc': [], 'top200_jacc': [], 'max_abs_score_diff': [], 'exact_top50_order': 0} for m in golden}
    lat = {'n_triples': [], 'feat_ms': [], 'pass_ms': [], 'adr_aer_ms': [], 'n_passes': []}
    with torch.no_grad():
        for k in idx:
            s = P[k]
            nt = len(s['text_entity_list'])
            loc = l2e[s['id']]
            ent = torch.from_numpy(np.asarray(ENT[emb_row[loc[:nt]]])).float()
            rel = torch.from_numpy(REL[[rid[r] for r in s['relation_list']]]).float()
            q = torch.from_numpy(Q[k:k + 1]).float()
            h, r, t = (torch.tensor(s[x]) for x in ('h_id_list', 'r_id_list', 't_id_list'))
            t0 = time.perf_counter()
            sem = (rel @ q.T).squeeze(1)
            lex = torch.tensor([compute_lexical_score(clean_text(s['question']), x) for x in s['relation_list']], dtype=torch.float32)
            score = RTHelpers._select_relation_score('hybrid', sem, lex, cfg.get('relation_score_alpha', 0.5), len(rel))
            mask = RTHelpers._select_relation_topic_mask(score, cfg['topk_rt'])
            rtf = torch.stack([score, mask], dim=1)
            lat['feat_ms'].append((time.perf_counter() - t0) * 1e3)
            n_ent = nt + len(s['non_text_entity_list'])
            names = s['text_entity_list'] + s['non_text_entity_list']
            cache, ptimes = {}, []

            def S(anchors):
                key = tuple(sorted(anchors))
                if key not in cache:
                    m = torch.zeros(n_ent, dtype=torch.long); m[list(key)] = 1
                    t1 = time.perf_counter()
                    cache[key] = model(h, r, t, q, ent, len(s['non_text_entity_list']), rel, F.one_hot(m, 2).float(), rtf).reshape(-1)
                    ptimes.append((time.perf_counter() - t1) * 1e3)
                return cache[key]

            def chain(anchors, B=3):
                sc = S(anchors)
                top = torch.topk(sc, min(B, len(sc))).indices
                new = (set(h[top].tolist()) | set(t[top].tolist())) - set(anchors)
                return [sc] + ([S(set(anchors) | new)] if new else [])

            tops = set(s['q_entity_id_list'])
            single = [{x} for x in sorted(tops)] if len(tops) > 1 else []
            t2 = time.perf_counter()
            res = {}
            for mode in ('base', 'aer', 'adr', 'adr_aer'):
                passes = {'base': lambda: [S(tops)], 'aer': lambda: chain(tops),
                          'adr': lambda: [S(tops)] + [S(x) for x in single],
                          'adr_aer': lambda: chain(tops) + [p for x in single for p in chain(x)]}[mode]()
                sc = passes[0] if len(passes) == 1 else torch.stack([z(p) for p in passes]).max(0).values
                top = torch.topk(sc, min(500, len(sc)))
                res[mode] = [(names[h[j]], s['relation_list'][r[j]], names[t[j]], float(v))
                             for v, j in zip(top.values.tolist(), top.indices.tolist())]
            lat['adr_aer_ms'].append((time.perf_counter() - t2) * 1e3)
            lat['n_passes'].append(len(cache)); lat['pass_ms'] += ptimes; lat['n_triples'].append(len(h))
            for mode, mine in res.items():
                g = golden[mode][s['id']]['scored_triples']
                for kk in (50, 200):
                    a, bb = {x[:3] for x in mine[:kk]}, {x[:3] for x in g[:kk]}
                    rep[mode][f'top{kk}_jacc'].append(len(a & bb) / max(1, len(a | bb)))
                # position-wise: names are not unique, so a name-keyed lookup would compare different triples
                rep[mode]['max_abs_score_diff'].append(max([abs(a[3] - c[3]) for a, c in zip(mine[:50], g[:50])] or [0]))
                rep[mode]['exact_top50_order'] += [x[:3] for x in mine[:50]] == [x[:3] for x in g[:50]]
    out = {m: {'top50_jaccard_mean': float(np.mean(v['top50_jacc'])), 'top50_jaccard_min': float(np.min(v['top50_jacc'])),
               'top200_jaccard_mean': float(np.mean(v['top200_jacc'])), 'max_abs_score_diff': float(np.max(v['max_abs_score_diff'])),
               'exact_top50_order': f"{v['exact_top50_order']}/{len(idx)}"} for m, v in rep.items()}
    nt_ = np.array(lat['n_triples'])
    out['latency_cpu'] = {'threads': args.threads, 'questions': len(idx), 'triples_p50': int(np.median(nt_)), 'triples_max': int(nt_.max()),
                          'rt_features_ms_p50': round(float(np.median(lat['feat_ms'])), 1),
                          'one_pass_ms_p50': round(float(np.median(lat['pass_ms'])), 1),
                          'one_pass_ms_max': round(float(np.max(lat['pass_ms'])), 1),
                          'all_modes_ms_p50': round(float(np.median(lat['adr_aer_ms'])), 1),
                          'distinct_passes_p50': int(np.median(lat['n_passes'])), 'distinct_passes_max': int(np.max(lat['n_passes'])),
                          'ms_per_1k_triples_per_pass': round(float(np.sum(lat['pass_ms']) / (np.sum(nt_ * np.array(lat['n_passes'])) / 1e3)), 1)}
    print(json.dumps(out, indent=1))
    json.dump(out, open(f'{b}/../verify_report.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
