"""Run every camera-ready analysis and compare its output with the committed reference.

    python run_all.py [--skip-slow] [--check]

Reads the frozen records (run scripts/reproduce.py first so they are unpacked and
generated/ exists). --check copies the committed out/ aside first and reports
any output file whose content differs. Not run: r20_reverify.py (needs a Lean
service; see README), ce.py and r16_seq.py (kept for the record, not used in
the paper), and drift.py unless the Mathlib sources from fetch_mathlib.sh exist.
"""
import argparse
import filecmp
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORDER = ['as_ce', 'r1', 'r2', 'r4', 'r8', 'r9', 'r10', 'r13', 'r13b', 'r16', 'r18',
         'parseable', 'per_sample', 'lean3', 'loo_table', 'multiplicity']
SLOW = ['r6', 'token_matched']   # several minutes each
DRIFT = 'drift'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--skip-slow', action='store_true')
    p.add_argument('--check', action='store_true')
    a = p.parse_args()
    scripts = ORDER + ([] if a.skip_slow else [s for s in SLOW if (HERE/f'{s}.py').exists()])
    if Path(os.environ.get('DRIFT_CACHE', HERE/'mathlib-drift-cache')).is_dir():
        scripts.append(DRIFT)   # needs the libraries fetched by fetch_mathlib.sh
    reference = None
    if a.check:
        reference = Path(tempfile.mkdtemp())/'out'
        shutil.copytree(HERE/'out', reference)
    failed = []
    for name in scripts:
        print(f'== {name}', flush=True)
        r = subprocess.run([sys.executable, str(HERE/f'{name}.py')], cwd=HERE)
        if r.returncode:
            failed.append(name)
    if reference:
        changed = []
        for ref in sorted(reference.rglob('*')):
            new = HERE/'out'/ref.relative_to(reference)
            if ref.is_file() and (not new.exists() or not filecmp.cmp(ref, new, shallow=False)):
                changed.append(str(ref.relative_to(reference)))
        print('outputs differing from the committed reference:', changed or 'none')
        if changed:
            failed.append('check')
    print('failed:', failed or 'none')
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
