"""Locate the released artifact: paper/artifact in the paper repo, the root of the public release.

Kept separate from common.py so scripts that also import the artifact's own
analysis modules (which have a common.py of their own) can use it without a clash.
"""
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _artifact():
    if os.environ.get('VERICODE_ARTIFACT'):
        return Path(os.environ['VERICODE_ARTIFACT'])
    for root in (HERE.parents[1]/'artifact', HERE.parents[1]):
        if (root/'experiments/decomp_rerun').is_dir():
            return root
    raise SystemExit('artifact not found; set VERICODE_ARTIFACT')


ARTIFACT = _artifact()
GENERATED = Path(os.environ.get('GENERATED', ARTIFACT/'generated'))  # outputs of scripts/reproduce.py
