"""Stream-filter the ODA GrailQA 2-hop Freebase files down to the topic entities of the questions in graliqa.json.

The raw files (HF dataset CCaccept/Subgraph, hop_2_Freebase/, 851 MB) are too large to json.load on a shared
machine; ijson keeps one top-level value in memory at a time. Needs only ijson (no torch).

  python -I tools/oda_extract.py <graliqa.json> <grailqa_triples_2_hop_all.json> <grailqa_labels_2_hop_all.json> <out.pkl>

Sources: graliqa.json from github.com/lanjiuqing64/KGdata (QAdata/), the two hop_2_Freebase files from
huggingface.co/datasets/CCaccept/Subgraph. Output: {'trip': {topic_mid: [(h, r, t), ...]}, 'lab': {topic_mid: {mid: label}}}.
"""
import json, pickle, sys
from sys import intern
import ijson


def main(qa_path, trip_path, lab_path, out_path):
    rows = json.load(open(qa_path))
    need = {n['id'] for row in rows for n in row['graph_query']['nodes'] if n['node_type'] == 'entity'}
    trip, lab = {}, {}
    with open(trip_path, 'rb') as f:
        for k, v in ijson.kvitems(f, '', use_float=True):
            if k in need:
                trip[k] = [tuple(intern(str(y)) for y in x) for x in v]   # MIDs/relations repeat: intern
    with open(lab_path, 'rb') as f:
        for k, v in ijson.kvitems(f, '', use_float=True):
            if k in need:
                lab[k] = {intern(a): b for a, b in v.items()}
    with open(out_path, 'wb') as f:
        pickle.dump({'trip': trip, 'lab': lab}, f, protocol=5)
    print(f'{len(rows)} questions, {len(need)} topic entities; kept triples for {len(trip)}, labels for {len(lab)}')


if __name__ == '__main__':
    main(*sys.argv[1:5])
