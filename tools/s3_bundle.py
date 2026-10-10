"""Upload / download the FastGraphRetriever data bundle (docs/DATA.md) to / from S3, verified by sha256.

  python tools/s3_bundle.py get  [--dest data/bundle] [--only PREFIX ...]   # dev machine: download + verify
  python tools/s3_bundle.py put  [--src data/bundle]             # research machine: upload (done once, 2026-10-11)
  python tools/s3_bundle.py ls

The file list and checksums come from data_manifest/bundle_v1.json (committed to git), so a download is checked
against what was uploaded. Files that already exist with the right sha256 are skipped, so `get` can be re-run.
Credentials: standard AWS variables, from the environment or from .env at the repo root:
  AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION (default us-east-1)
Optional: FGR_S3_BUCKET (default banktransactiondata), FGR_S3_PREFIX (default fastgraphretriever/bundle/v1/).
Needs only boto3. Egress: a full download is ~1.4 GB (about $0.13 at AWS internet egress prices).
"""
import argparse, hashlib, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(ROOT, 'data_manifest', 'bundle_v1.json')


def load_env():
    path = os.path.join(ROOT, '.env')
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                k, v = line.split('=', 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 24), b''):
            h.update(block)
    return h.hexdigest()


def client():
    import boto3
    from boto3.s3.transfer import TransferConfig
    load_env()
    s3 = boto3.client('s3', region_name=os.environ.get('AWS_DEFAULT_REGION', 'us-east-1'))
    return s3, TransferConfig(multipart_chunksize=64 * 2**20, max_concurrency=4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['get', 'put', 'ls'])
    ap.add_argument('--dest', default=os.path.join(ROOT, 'data', 'bundle'))
    ap.add_argument('--src', default=os.path.join(ROOT, 'data', 'bundle'))
    ap.add_argument('--only', nargs='*', default=None, help='get: only files whose path starts with one of these')
    args = ap.parse_args()
    s3, cfg = client()
    bucket = os.environ.get('FGR_S3_BUCKET', 'banktransactiondata')
    prefix = os.environ.get('FGR_S3_PREFIX', 'fastgraphretriever/bundle/v1/')
    if args.cmd == 'ls':
        for page in s3.get_paginator('list_objects_v2').paginate(Bucket=bucket, Prefix=prefix):
            for o in page.get('Contents', []):
                print(f"{o['Size'] / 2**20:9.1f} MB  {o['Key'][len(prefix):]}")
        return
    if args.cmd == 'put':
        files = json.load(open(os.path.join(args.src, 'MANIFEST.json')))['files']
        for rel in list(files) + ['MANIFEST.json']:
            s3.upload_file(os.path.join(args.src, rel), bucket, prefix + rel, Config=cfg)
            remote = s3.head_object(Bucket=bucket, Key=prefix + rel)['ContentLength']
            ok = remote == os.path.getsize(os.path.join(args.src, rel))
            print(f"{'OK ' if ok else 'SIZE MISMATCH'} {rel}", flush=True)
            if not ok:
                sys.exit(1)
        return
    files = json.load(open(MANIFEST))['files']
    bad = 0
    for rel, meta in list(files.items()) + [('MANIFEST.json', None)]:
        if args.only and not any(rel.startswith(p) for p in args.only):
            continue
        dst = os.path.join(args.dest, rel)
        if meta and os.path.exists(dst) and os.path.getsize(dst) == meta['bytes'] and sha256(dst) == meta['sha256']:
            print(f'skip {rel}'); continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        s3.download_file(bucket, prefix + rel, dst + '.part', Config=cfg)
        if meta and sha256(dst + '.part') != meta['sha256']:
            print(f'CHECKSUM MISMATCH {rel}'); bad += 1; continue
        os.replace(dst + '.part', dst)
        print(f'OK   {rel}', flush=True)
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
