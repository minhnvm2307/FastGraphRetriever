"""Shortest-path triple labels of SubgraphRAG (Li et al., 2025), as computed by its RetrieverDataset: a triple is
positive if it lies on a shortest directed path between a topic entity and an answer entity (forward or backward; when
both directions have paths, only the globally shortest are kept).
Reads data_files/<dataset>/processed/<split>.pkl (emb.py) and writes data_files/<dataset>/triple_scores/<split>.pth =
{id: {'triple_scores': float tensor [n_triples], 'max_path_length': int or None}}.
  python sp_labels.py -d webqsp         (all splits)
"""
import argparse, os, pickle
import networkx as nx
import torch
from tqdm import tqdm


def nx_graph(sample):
    g = nx.DiGraph()
    for i, (h, r, t) in enumerate(zip(sample['h_id_list'], sample['r_id_list'], sample['t_id_list'])):
        g.add_edge(h, t, triple_id=i, relation_id=r)
    return g


def shortest_paths(g, q, a):
    try:
        forward = list(nx.all_shortest_paths(g, q, a))
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        forward = []
    try:
        backward = list(nx.all_shortest_paths(g, a, q))
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        backward = []
    paths = forward + backward
    if not forward or not backward:
        return paths
    shortest = min(len(p) for p in paths)
    return [p for p in paths if len(p) == shortest]


def score_sample(sample):
    g = nx_graph(sample)
    paths = [p for q in sample['q_entity_id_list'] for a in sample['a_entity_id_list'] for p in shortest_paths(g, q, a)]
    scores = torch.zeros(len(sample['h_id_list']))
    max_len = max((len(p) - 1 for p in paths), default=None)
    for p in paths:
        for u, v in zip(p[:-1], p[1:]):
            scores[g[u][v]['triple_id']] = 1.
    return {'triple_scores': scores, 'max_path_length': max_len}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('-d', '--dataset', required=True, choices=['webqsp', 'cwq'])
    ap.add_argument('--splits', default='train,val,test')
    args = ap.parse_args()
    out_dir = f'data_files/{args.dataset}/triple_scores'; os.makedirs(out_dir, exist_ok=True)
    for split in args.splits.split(','):
        out = f'{out_dir}/{split}.pth'
        if os.path.exists(out):
            print(out, 'exists'); continue
        samples = pickle.load(open(f'data_files/{args.dataset}/processed/{split}.pkl', 'rb'))
        torch.save({s['id']: score_sample(s) for s in tqdm(samples, desc=f'{args.dataset} {split}')}, out)
        print('->', out)


if __name__ == '__main__':
    main()
