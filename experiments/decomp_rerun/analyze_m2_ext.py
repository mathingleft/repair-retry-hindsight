"""Prover extension arms (F_E_shuf, N; out_m2_ext/) against the prover's six core arms (out_m2/). Problems = bank2.

    python analyze_m2_ext.py [m2_outdir] [m2ext_outdir]
"""
import collections, json, os, random, sys
from analyze_ext import load, contrast
HERE = os.path.dirname(os.path.abspath(__file__))
def main(m2="out_m2", ext="out_m2_ext"):
    npass = collections.defaultdict(collections.Counter); solved = collections.defaultdict(set); seen = collections.Counter()
    load(m2, npass, solved, seen); load(ext, npass, solved, seen)
    pids = sorted(json.loads(l)["id"] for l in open(os.path.join(HERE, m2, "bank2.jsonl")))
    for a in ("bare", "I", "N", "F", "F_E", "F_E_shuf", "F_C", "F_CE"): solved[a] &= set(pids)
    print(f"DeepSeek-Prover-V2-7B, {len(pids)} problems\narm        solved  passes")
    for a in ("bare", "I", "N", "F", "F_E", "F_E_shuf", "F_C", "F_CE"):
        if seen[a]: print(f"{a:9s} {len(solved[a]):6d} {sum(npass[a][p] for p in pids):7d}")
    rng = random.Random(1)
    print("\nerror content on the prover: is the compiler error interchangeable with a wrong one?")
    for x, y in (("F_E_shuf", "F_E"), ("F_E_shuf", "F"), ("F_E", "F"), ("F_E_shuf", "I")):
        if seen[x] and seen[y]: print(contrast(x, y, npass, solved, pids, rng))
    print("\nnonsense on the prover: does N ~= I and does the meaningful sentence cost?")
    for x, y in (("N", "I"), ("N", "F"), ("N", "bare"), ("F", "I")):
        if seen[x] and seen[y]: print(contrast(x, y, npass, solved, pids, rng))
if __name__ == "__main__":
    main(*sys.argv[1:3])
