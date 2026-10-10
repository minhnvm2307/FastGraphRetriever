# tools/

| Script | Where it runs | What |
|---|---|---|
| `s3_bundle.py` | dev machine (and research machine for `put`) | `get` downloads `data/bundle/` and checks every sha256 against `data_manifest/bundle_v1.json`; `--only PREFIX` fetches a subset; `ls` lists the S3 prefix. Needs `boto3` and the AWS keys in `.env` |
| `oda_extract.py` | research machine, once | Stream-filters the 851 MB ODA source files down to the 879 topic entities that are needed (ijson). Run with `python -I` (downloaded data) |
| `build_bundle.py` | research machine, once | Builds `data/bundle/` from the research inputs. Rebuilds each question's subgraph from ODA to recover MIDs and aborts unless all 897 match the research file exactly |
| `verify_bundle.py` | research machine (`torch_geometric`) | The original research model, fed only from bundle files, against the golden outputs + CPU latency → `docs/evidence/verify_bundle.json` |
| `bench_factorized.py` | research machine (`torch_geometric`) | Factorised scorer vs original (difference, speed) and GTE-on-CPU latency → `docs/evidence/bench_factorized.json` |

Results of the 2026-10-11 run are in `docs/evidence/` and summarised in `docs/DATA.md` §3 and `docs/RETRIEVER_SPEC.md` §4.
