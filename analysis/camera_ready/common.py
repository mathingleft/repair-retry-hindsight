"""Strict readers and artifact-relative paths for the submitted experiments."""
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path

from artifact_path import ARTIFACT, GENERATED, HERE

DATA = ARTIFACT / 'experiments/decomp_rerun'
OUTPUT = HERE / 'out'
QDIRS = ['out','out_ext','out_ext2','out_ext3','out_noi']
DDIRS = ['out_m2','out_m2_ext','out_m2_ext2','out_m2_ext3','out_m2_noi']


def read(path):
    path = Path(path)
    op = gzip.open if path.suffix == '.gz' else open
    with op(path, 'rt', encoding='utf-8') as f:
        return [json.loads(l) for l in f if l.strip()]


def dump(name, value):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def csvout(name, rows):
    rows=list(rows)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def normalize(r):
    return {'id':r.get('id',r.get('problem_id')), 'traj':r.get('traj',r.get('trajectory_id')),
            'attempt':r.get('attempt',r.get('attempt_id')), 'status':r.get('status',r.get('verification_result')),
            'code':r.get('code',r.get('extracted_proof')), 'error':r.get('error',r.get('compiler_output'))}
