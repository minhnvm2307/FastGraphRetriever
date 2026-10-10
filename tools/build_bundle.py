"""Build the FastGraphRetriever data bundle from the SubgraphRAG research artefacts.

Provenance script: it runs once, on the research machine, with numpy + torch. Output goes to data/bundle/
(gitignored); tools/s3_bundle.py uploads it, and the dev machine downloads it with tools/fetch_data.py.

  python -I tools/oda_extract.py graliqa.json grailqa_triples_2_hop_all.json grailqa_labels_2_hop_all.json oda_subset.pkl
  python tools/build_bundle.py --src /home/minhnv/Lab/SubgraphRAG --qa graliqa.json --oda oda_subset.pkl --out data/bundle

Node identity. The research files keep only display names, and names are not unique: ODA labels every CVT node
"unnamed_entity" (267k repeats) and 48.6k other names repeat inside a question ("Length", "Measurement System"...).
Merging by name would fuse them into hubs (one node of degree 97k). So the per-question subgraphs are rebuilt from
the ODA source exactly as retrieve/exp_qrdde/ext_data.py::_grail_record did, every question is checked to match
the processed file (names, relations, h/r/t), and the global graph is keyed by Freebase MID (or literal value).

Embeddings stay keyed by name: the same name always has the same GTE vector in the source file (checked 10-11: max
abs diff 0.0 over 411k repeats), so the deduplicated table reproduces the research inputs exactly.

Outputs (schema in docs/DATA.md):
  kg/entities.tsv.gz      eid, key (MID or literal), name, emb_row
  kg/relations.tsv.gz     rid, name                     (row rid of emb/rel.f16.npy)
  kg/triples.tsv.gz       h_eid, rid, t_eid             deduplicated union of the question subgraphs
  emb/ent.f16.npy         [n_names, 1024]  GTE-large-en-v1.5, one row per unique entity name
  emb/rel.f16.npy         [n_rel, 1024]
  emb/q.f16.npy           [n_q, 1024]      row = "row" field of benchmark/questions.jsonl
  benchmark/test.pkl      per-question subgraphs (SubgraphRAG processed format), unchanged copy
  benchmark/questions.jsonl  id, question, topic/answer names and eids, subgraph size, hops, level, domain
  benchmark/local2eid.npz    per question: global eid of each local entity index (benchmark mode -> global graph)
  golden/adr_grailqa_oda_{base,adr,aer,adr_aer}.pth   top-500 output of the research code (parity oracle)
  models/wl_effective_num_s42/cpt.pth + config.json   WL retriever checkpoint (trained on WebQSP)
  MANIFEST.json           sizes, sha256, stats
"""
import argparse, gzip, hashlib, json, os, pickle, re, shutil, time
import numpy as np
import torch

CKPT = 'retrieve/webqsp_wl_effective_num_Oct05-15:23:05'
GQ = 'retrieve/data_files/grailqa_oda'
MID = re.compile(r'^[mg]\.[0-9a-z_]+$')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 24), b''):
            h.update(block)
    return h.hexdigest()


def write_tsv(path, header, rows):
    with gzip.open(path, 'wt', encoding='utf-8', newline='\n') as f:   # names may contain \r: never translate
        f.write('\t'.join(header) + '\n')
        for row in rows:
            f.write('\t'.join(' '.join(str(x).split('\t')).replace('\r', ' ').replace('\n', ' ') for x in row) + '\n')


def oda_records(rows, oda, max_triples=30000):
    """Same construction as ext_data.convert_grailqa_oda/_grail_record, keeping the node keys it discarded."""
    trip, lab, out = oda['trip'], oda['lab'], {}
    for row in rows:
        tops = [n['id'] for n in row['graph_query']['nodes'] if n['node_type'] == 'entity' and n['id'] in trip]
        if not tops:
            continue
        T, names = [], {}
        for tp in tops:
            T += [tuple(x) for x in trip[tp]]
            names.update(lab.get(tp, {}))
        T = list(dict.fromkeys(T))
        if len(T) > max_triples:
            continue
        ents = list(dict.fromkeys([x for h, _, t in T for x in (h, t)]))
        text_e = [e for e in ents if e in names or not MID.match(e)]
        non_text = [e for e in ents if not (e in names or not MID.match(e))]
        order = text_e + non_text
        eid = {e: k for k, e in enumerate(order)}
        rels = list(dict.fromkeys(r for _, r, _ in T)); rid = {r: k for k, r in enumerate(rels)}
        out[str(row['qid'])] = {'keys': order, 'names': [names.get(e, e) for e in order], 'rels': rels,
                                'h': [eid[h] for h, _, _ in T], 'r': [rid[r] for _, r, _ in T],
                                't': [eid[t] for _, _, t in T], 'n_hops': int(row['num_edge']),
                                'level': row['level'], 'function': row['function'],
                                'domain': (row.get('domains') or ['-'])[0]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='/home/minhnv/Lab/SubgraphRAG')
    ap.add_argument('--qa', required=True, help='graliqa.json (ODA KGdata/QAdata)')
    ap.add_argument('--oda', required=True, help='output of tools/oda_extract.py')
    ap.add_argument('--out', default='data/bundle')
    args = ap.parse_args()
    t0 = time.time()
    out = args.out
    for d in ('kg', 'emb', 'benchmark', 'golden', 'models/wl_effective_num_s42'):
        os.makedirs(f'{out}/{d}', exist_ok=True)

    P = pickle.load(open(f'{args.src}/{GQ}/processed/test.pkl', 'rb'))
    E = torch.load(f'{args.src}/{GQ}/emb/gte-large-en-v1.5/test.pth', map_location='cpu', mmap=True, weights_only=False)
    R = oda_records(json.load(open(args.qa)), pickle.load(open(args.oda, 'rb')))

    # 1. every processed question must be reproduced exactly by the ODA rebuild
    bad = []
    for s in P:
        r = R.get(s['id'])
        names = s['text_entity_list'] + s['non_text_entity_list']
        if (r is None or r['names'] != names or r['rels'] != s['relation_list'] or r['h'] != list(s['h_id_list'])
                or r['r'] != list(s['r_id_list']) or r['t'] != list(s['t_id_list'])):
            bad.append(s['id'])
    if bad:
        raise SystemExit(f'{len(bad)} questions do not match the ODA rebuild, e.g. {bad[:5]}')
    print(f'{len(P)} questions reproduced exactly from the ODA source', flush=True)

    # 2. global graph keyed by MID / literal; embeddings keyed by name
    ent, ent_name = {}, []                       # key -> eid; eid -> display name
    name_row, name_src = {}, []                  # name -> emb row; row -> (qid, local text index)
    rel, rel_src = {}, {}
    trip_parts, local2eid, n_inst = [], {}, 0
    for s in P:
        r = R[s['id']]
        nt = len(s['text_entity_list'])
        loc = np.empty(len(r['keys']), dtype=np.int64)
        for i, (k, n) in enumerate(zip(r['keys'], r['names'])):
            if k not in ent:
                ent[k] = len(ent); ent_name.append(n)
            loc[i] = ent[k]
            if i < nt and n not in name_row:
                name_row[n] = len(name_src); name_src.append((s['id'], i))
        rloc = np.empty(len(r['rels']), dtype=np.int64)
        for j, rn in enumerate(r['rels']):
            if rn not in rel:
                rel[rn] = len(rel); rel_src[rel[rn]] = (s['id'], j)
            rloc[j] = rel[rn]
        h, rr, t = (np.asarray(r[k], dtype=np.int64) for k in ('h', 'r', 't'))
        trip_parts.append(np.stack([loc[h], rloc[rr], loc[t]], axis=1))
        local2eid[s['id']] = loc.astype(np.int32)
        n_inst += len(h)
    T = np.unique(np.concatenate(trip_parts), axis=0)
    print(f'entities {len(ent)}, unique names {len(name_row)}, relations {len(rel)}, triples {len(T)} '
          f'from {n_inst} instances', flush=True)

    # 3. embedding tables, gathered question by question from the mmapped source
    ent_tab = np.lib.format.open_memmap(f'{out}/emb/ent.f16.npy', mode='w+', dtype=np.float16,
                                        shape=(len(name_src), 1024))
    by_q = {}
    for row, (qid, i) in enumerate(name_src):
        by_q.setdefault(qid, []).append((i, row))
    for qid, pairs in by_q.items():
        src_i, dst = (np.array(x) for x in zip(*pairs))
        ent_tab[dst] = E[qid]['entity_embs'][torch.from_numpy(src_i)].numpy()
    ent_tab.flush(); del ent_tab
    rel_tab = np.zeros((len(rel), 1024), dtype=np.float16)
    for rid, (qid, j) in rel_src.items():
        rel_tab[rid] = E[qid]['relation_embs'][j].numpy()
    np.save(f'{out}/emb/rel.f16.npy', rel_tab)
    np.save(f'{out}/emb/q.f16.npy', np.stack([E[s['id']]['q_emb'][0].numpy() for s in P]).astype(np.float16))

    keys = [None] * len(ent)
    for k, e in ent.items():
        keys[e] = k
    write_tsv(f'{out}/kg/entities.tsv.gz', ['eid', 'key', 'name', 'emb_row'],
              ((e, keys[e], ent_name[e], name_row.get(ent_name[e], -1)) for e in range(len(ent))))
    write_tsv(f'{out}/kg/relations.tsv.gz', ['rid', 'name'], ((r, n) for n, r in sorted(rel.items(), key=lambda x: x[1])))
    write_tsv(f'{out}/kg/triples.tsv.gz', ['h_eid', 'rid', 't_eid'], map(tuple, T.tolist()))

    # 4. benchmark files, golden outputs, checkpoint
    with open(f'{out}/benchmark/questions.jsonl', 'w', encoding='utf-8') as f:
        for k, s in enumerate(P):
            r = R[s['id']]; loc = local2eid[s['id']]
            f.write(json.dumps({'row': k, 'id': s['id'], 'question': s['question'],
                                'topic_entities': [r['names'][i] for i in s['q_entity_id_list']],
                                'topic_eids': [int(loc[i]) for i in s['q_entity_id_list']],
                                'answers': [r['names'][i] for i in s['a_entity_id_list']],
                                'answer_eids': [int(loc[i]) for i in s['a_entity_id_list']],
                                'n_triples': len(s['h_id_list']), 'n_entities': len(loc), 'n_hops': r['n_hops'],
                                'level': r['level'], 'function': r['function'], 'domain': r['domain']},
                               ensure_ascii=False) + '\n')
    np.savez_compressed(f'{out}/benchmark/local2eid.npz', **local2eid)
    shutil.copy2(f'{args.src}/{GQ}/processed/test.pkl', f'{out}/benchmark/test.pkl')
    for m in ('base', 'adr', 'aer', 'adr_aer'):
        shutil.copy2(f'{args.src}/{CKPT}/adr_grailqa_oda_{m}.pth', f'{out}/golden/adr_grailqa_oda_{m}.pth')
    shutil.copy2(f'{args.src}/{CKPT}/cpt.pth', f'{out}/models/wl_effective_num_s42/cpt.pth')
    cpt = torch.load(f'{args.src}/{CKPT}/cpt.pth', map_location='cpu', weights_only=False)
    json.dump({'config': cpt['config'], 'epoch': cpt.get('epoch'), 'best_val_metric': cpt.get('best_val_metric'),
               'seed': cpt.get('seed'), 'source': f'{CKPT}/cpt.pth'},
              open(f'{out}/models/wl_effective_num_s42/config.json', 'w'), indent=1, default=str)

    deg = np.bincount(np.concatenate([T[:, 0], T[:, 2]]), minlength=len(ent))
    top = np.argsort(-deg)[:10]
    files = {}
    for d, _, fs in os.walk(out):
        for fn in sorted(fs):
            p = os.path.join(d, fn); rp = os.path.relpath(p, out)
            if rp != 'MANIFEST.json':
                files[rp] = {'bytes': os.path.getsize(p), 'sha256': sha256(p)}
    manifest = {'name': 'fastgraphretriever-bundle', 'version': 'v1', 'created': time.strftime('%Y-%m-%d %H:%M'),
                'source': {'repo': args.src, 'dataset': GQ, 'checkpoint': CKPT,
                           'oda': 'huggingface.co/datasets/CCaccept/Subgraph hop_2_Freebase + KGdata/QAdata/graliqa.json'},
                'stats': {'questions': len(P), 'entities': len(ent), 'unique_entity_names': len(name_row),
                          'relations': len(rel), 'triples': int(len(T)), 'triple_instances': int(n_inst),
                          'degree_p50': int(np.percentile(deg, 50)), 'degree_p99': int(np.percentile(deg, 99)),
                          'degree_max': int(deg.max()),
                          'top_degree': [[ent_name[e], keys[e], int(deg[e])] for e in top]},
                'files': files}
    json.dump(manifest, open(f'{out}/MANIFEST.json', 'w'), indent=1, ensure_ascii=False)
    print(json.dumps(manifest['stats'], indent=1, ensure_ascii=False))
    print(f'done in {time.time() - t0:.0f}s, total {sum(v["bytes"] for v in files.values()) / 2**20:.0f} MB')


if __name__ == '__main__':
    main()
