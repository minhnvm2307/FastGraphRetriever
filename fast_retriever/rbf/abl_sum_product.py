"""Ablation (paper Tables "ablation", "sum-max"): sum-product instead of max-product path aggregation, at TEST time only, on a trained RBF run.

Same checkpoint, same factors pi / phi, same non-backtracking constraint, same complete-path (mm) scoring and anchor
fusion as infer.py --modes base; only the semiring of the two recursions changes:
  max-product: score(e) = max over complete non-backtracking walks through e of prod pi (x phi of the last step)
  sum-product: score(e) = sum over the same walks of the same products
Non-backtracking sums are exact: per node we keep the total S(v) and the per-last-step-tag sums S_g(v), and a step
over relation r forwards reads S(h) - S_{r reverse}(h) (and symmetrically), the sum analogue of the Viterbi top-2.
Computed in probability space (float64); walks have at most T + COMPLETION_STEPS = 5 factors.

  python abl_sum_product.py --run abl_pi_full_webqsp --dataset cwq
Writes runs/<run>/pred_<dataset>_sum.pth (--semiring max: pred_<dataset>_maxcheck.pth) (same format as infer.py) -> evaluate like pred_<dataset>_base.pth.
  python abl_sum_product.py --selftest      # brute-force check of the two recursions on random small graphs
"""
import argparse, itertools, json, os, random
import torch
from tqdm import tqdm

import rbf
from rbf import COMPLETION_STEPS, NEG
from train_retriever import attach_bge_prior, build_scorer, load_split, to_device

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs')


def _to_log(x):
    return torch.where(x > 0, torch.log(x.clamp(min=1e-300)), torch.full_like(x, NEG)).float()


def _tag_tables(h, t, r, n):
    """candidate steps: [forward steps (dest t, tag 2r) ; reverse steps (dest h, tag 2r+1)] -> index of each
    candidate's (dest, tag) group. The group a forward step over e must exclude (walks at h whose last step is r
    reversed) is the group of e's own reverse candidate, and vice versa."""
    ntag = 2 * (int(r.max()) + 1)
    keys = torch.cat([t * ntag + 2 * r, h * ntag + 2 * r + 1])
    _, inv = torch.unique(keys, return_inverse=True)
    E = len(h)
    return inv, inv[E:], inv[:E]                    # group of every candidate; excluded group for fwd / rev steps


def _scatter(values, index, size):
    out = torch.zeros(values.shape[0], size, dtype=values.dtype, device=values.device)
    return out.index_add_(1, index, values)


def propagate_sum(log_p, h, t, r, log_pi_fwd, log_pi_rev, n, T):
    """sum-product analogue of rbf.propagate_non_backtracking (same signature, log-space outputs)"""
    init = torch.exp(log_p.double()) * (log_p > NEG / 2)
    pf, pr = torch.exp(log_pi_fwd.double()), torch.exp(log_pi_rev.double())
    group, excl_fwd, excl_rev = _tag_tables(h, t, r, n)
    n_group = int(group.max()) + 1
    S, Sg = init, torch.zeros(init.shape[0], n_group, dtype=init.dtype, device=init.device)
    dest = torch.cat([t, h])
    for _ in range(T):
        c = torch.cat([(S[:, h] - Sg[:, excl_fwd]).clamp(min=0) * pf, (S[:, t] - Sg[:, excl_rev]).clamp(min=0) * pr], 1)
        S, Sg = init + _scatter(c, dest, n), _scatter(c, group, n_group)
    path_fwd = (S[:, h] - Sg[:, excl_fwd]).clamp(min=0) * pf
    path_rev = (S[:, t] - Sg[:, excl_rev]).clamp(min=0) * pr
    return _to_log(path_fwd), _to_log(path_rev), _to_log(S)


def completion_sum(h, t, r, log_pi_fwd, log_pi_rev, n, n_text, log_stop_fwd=None, log_stop_rev=None):
    """sum-product analogue of rbf.best_completion: sum over non-backtracking continuations (<= COMPLETION_STEPS)
    after a forward step into t / a reverse step into h that stop at a text entity"""
    dev = h.device
    pf, pr = torch.exp(log_pi_fwd.double()), torch.exp(log_pi_rev.double())
    init = torch.zeros(1, n, dtype=torch.float64, device=dev)
    if log_stop_fwd is None:
        init[:, :n_text] = 1.0                                     # stopping at a text entity is free
        stop_fwd = stop_rev = torch.zeros(1, len(h), dtype=torch.float64, device=dev)
    else:                                                          # stop right after the step, paying phi
        stop_fwd = torch.where(t < n_text, torch.exp(log_stop_fwd.double()), torch.zeros_like(pf))[None, :]
        stop_rev = torch.where(h < n_text, torch.exp(log_stop_rev.double()), torch.zeros_like(pr))[None, :]
    # tag of a continuation = its first step; destination = the node it starts from
    ntag = 2 * (int(r.max()) + 1)
    keys = torch.cat([h * ntag + 2 * r, t * ntag + 2 * r + 1])
    _, group = torch.unique(keys, return_inverse=True)
    E = len(h)
    excl_fwd, excl_rev = group[E:], group[:E]                      # after a fwd step into t: not r reversed from t
    n_group = int(group.max()) + 1
    C, Cg = init, torch.zeros(1, n_group, dtype=torch.float64, device=dev)
    dest = torch.cat([h, t])
    after = lambda C, Cg: ((C[:, t] - Cg[:, excl_fwd]).clamp(min=0) + stop_fwd,
                           (C[:, h] - Cg[:, excl_rev]).clamp(min=0) + stop_rev)
    for _ in range(COMPLETION_STEPS):
        a_fwd, a_rev = after(C, Cg)
        c = torch.cat([a_fwd * pf, a_rev * pr], 1)
        C, Cg = init + _scatter(c, dest, n), _scatter(c, group, n_group)
    a_fwd, a_rev = after(C, Cg)
    return _to_log(a_fwd), _to_log(a_rev)


def use_sum_product():
    rbf.propagate_non_backtracking, rbf.best_completion = propagate_sum, completion_sum


# ------------------------------------------------------------------ brute-force self-test
def _walks(n, edges, start, max_len):
    """all non-backtracking walks from `start` with <= max_len steps: lists of (edge index, direction)"""
    out, frontier = [[]], [[]]
    for _ in range(max_len):
        nxt = []
        for w in frontier:
            node = start if not w else (edges[w[-1][0]][2] if w[-1][1] == 0 else edges[w[-1][0]][0])
            last = None if not w else (edges[w[-1][0]][1], w[-1][1])
            for k, (a, rel, b) in enumerate(edges):
                for d, (u, v) in enumerate(((a, b), (b, a))):
                    if u == node and not (last and last[0] == rel and last[1] != d):
                        nxt.append(w + [(k, d)])
        out += nxt; frontier = nxt
    return out


def selftest(trials=30):
    torch.manual_seed(0); random.seed(0)
    for _ in range(trials):
        n, R, E = random.randint(3, 6), random.randint(1, 3), random.randint(2, 8)
        edges = list({(random.randrange(n), random.randrange(R), random.randrange(n)) for _ in range(E)})
        edges = [e for e in edges if e[0] != e[2]] or [(0, 0, 1)]
        h, r, t = (torch.tensor([e[i] for e in edges]) for i in range(3))
        lpf, lpr = torch.log(torch.rand(len(edges)) * .9 + .05), torch.log(torch.rand(len(edges)) * .9 + .05)
        lsf, lsr = torch.log(torch.rand(len(edges)) * .9 + .05), torch.log(torch.rand(len(edges)) * .9 + .05)
        n_text, T, a = random.randint(1, n), 3, 0
        log_p = torch.full((1, n), NEG); log_p[0, a] = 0.
        pf_, pr_, _ = propagate_sum(log_p, h, t, r, lpf, lpr, n, T)
        for use_stop in (False, True):
            cf, cr = completion_sum(h, t, r, lpf, lpr, n, n_text, lsf if use_stop else None, lsr if use_stop else None)
            want_pre = torch.zeros(2, len(edges), dtype=torch.float64)
            want_full = torch.zeros(2, len(edges), dtype=torch.float64)
            fac = lambda k, d: float(torch.exp(lpf[k] if d == 0 else lpr[k]))
            end = lambda k, d: edges[k][2] if d == 0 else edges[k][0]
            for w in _walks(n, edges, a, T + 1):
                if not w: continue
                k, d = w[-1]
                prod = 1.0
                for kk, dd in w: prod *= fac(kk, dd)
                if len(w) <= T + 1: want_pre[d, k] += prod
            # complete walks: prefix ending with e (<= T+1 steps) + continuation (<= 2 steps) ending at a text entity
            for w in _walks(n, edges, a, T + 1 + COMPLETION_STEPS):
                for i in range(min(len(w), T + 1)):
                    k, d = w[i]; cont = w[i + 1:]
                    if len(cont) > COMPLETION_STEPS: continue
                    last = w[-1]
                    if end(*last) >= n_text: continue
                    prod = 1.0
                    for kk, dd in w: prod *= fac(kk, dd)
                    if use_stop: prod *= float(torch.exp(lsf[last[0]] if last[1] == 0 else lsr[last[0]]))
                    want_full[d, k] += prod
            got_pre = torch.stack([torch.exp(pf_[0].double()), torch.exp(pr_[0].double())]) * (torch.stack([pf_[0], pr_[0]]) > NEG / 2)
            assert torch.allclose(got_pre, want_pre, rtol=1e-4, atol=1e-6), (got_pre, want_pre)
            gf = torch.exp((pf_[0] + cf[0]).double()) * ((pf_[0] > NEG / 2) & (cf[0] > NEG / 2))
            gr = torch.exp((pr_[0] + cr[0]).double()) * ((pr_[0] > NEG / 2) & (cr[0] > NEG / 2))
            assert torch.allclose(torch.stack([gf, gr]), want_full, rtol=1e-4, atol=1e-6), (torch.stack([gf, gr]), want_full)
    print(f'selftest OK ({trials} random graphs)')


def main():
    ap = argparse.ArgumentParser(description='Sum-product ablation of a trained RBF run (see the docstring).')
    ap.add_argument('--run'); ap.add_argument('--dataset', choices=['webqsp', 'cwq'])
    ap.add_argument('--device', default='cuda:0'); ap.add_argument('--top_k', type=int, default=500)
    ap.add_argument('--limit', type=int, default=0); ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--semiring', default='sum', choices=['sum', 'max'], help='max = sanity check, must match infer.py')
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if args.semiring == 'sum':
        use_sum_product()
    run_dir = args.run if os.path.isdir(args.run) else f'{RUNS}/{args.run}'
    train_args = json.load(open(f'{run_dir}/result.json'))['args']
    model = build_scorer(train_args)
    model.load_state_dict(torch.load(f'{run_dir}/best.pt', map_location=args.device))
    model.to(args.device).eval()
    model.rbf.mm = True
    questions = []
    for split in ('test', 'test_extra'):
        records, emb = load_split(args.dataset, split)
        attach_bge_prior(records, args.dataset, split, train_args.get('rel_prior', 'none'))
        questions += [(rec, emb) for rec in records]
    if args.limit:
        questions = questions[:args.limit]
    out = {}
    with torch.no_grad():
        for rec, emb in tqdm(questions, desc=f'{args.semiring}-product {os.path.basename(run_dir)} on {args.dataset}'):
            # raw log score: Scorer.forward's log_to_logit clamps log scores at 0, and sums of walks can exceed 1
            scores = model.rbf(to_device(rec, emb, args.device, anchors=tuple(sorted(set(rec['q_ent'])))))
            names, relations = rec['entity_names'], rec['relation_list']
            top = torch.topk(scores, min(args.top_k, len(scores))).indices.tolist(); scores = scores.cpu()
            out[rec['id']] = {
                'question': rec['question'], 'q_entity': [names[i] for i in rec['q_ent']],
                'a_entity_in_graph': [names[i] for i in rec.get('a_ent', [])],
                'scored_triples': [(names[rec['h'][i]], relations[rec['r'][i]], names[rec['t'][i]], float(scores[i])) for i in top]}
    path = f'{run_dir}/pred_{args.dataset}_' + ('sum' if args.semiring == 'sum' else 'maxcheck') + ('_smoke' if args.limit else '') + '.pth'
    torch.save(out, path); print(f'{len(out)} questions -> {path}')


if __name__ == '__main__':
    main()
