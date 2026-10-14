"""Prover no-instruction ladder (out_m2_noi/) against the prover's instruction-bearing arms (out_m2/, out_m2_ext/). Problems = bank2.

    python analyze_m2_noi.py
"""
import collections, json, os, random
from analyze_ext import load, contrast
HERE = os.path.dirname(os.path.abspath(__file__))
def main():
    npass = collections.defaultdict(collections.Counter); solved = collections.defaultdict(set); seen = collections.Counter()
    for d in ("out_m2", "out_m2_ext", "out_m2_noi"): load(d, npass, solved, seen)
    pids = sorted(json.loads(l)["id"] for l in open(os.path.join(HERE, "out_m2", "bank2.jsonl")))
    P = set(pids)
    for a in list(solved): solved[a] &= P
    order = ("bare", "I", "N", "F", "F_noI", "F_E", "F_E_noI", "F_E_shuf", "F_E_shuf_noI", "F_C", "F_C_noI", "F_CE", "F_CE_noI")
    print(f"DeepSeek-Prover-V2-7B, {len(pids)} problems\narm            solved  passes")
    for a in order:
        if seen[a]: print(f"{a:13s} {len(solved[a]):6d} {sum(npass[a][p] for p in pids):7d}")
    rng = random.Random(1)
    def C(x, y):
        if seen[x] and seen[y]: print(contrast(x, y, npass, solved, pids, rng))
    print("\nfeedback content with NO instruction (vs bare):")
    for x in ("F_noI", "F_E_noI", "F_E_shuf_noI", "F_C_noI", "F_CE_noI"): C(x, "bare")
    print("\nthe instruction's worth on top of each scaffold (X vs X_noI):")
    for x in ("F", "F_E", "F_E_shuf", "F_C", "F_CE"): C(x, x + "_noI")
    print("\nthe ladder inside the no-instruction regime:")
    for x, y in (("F_noI", "I"), ("F_E_noI", "F_noI"), ("F_E_shuf_noI", "F_E_noI"), ("F_E_shuf_noI", "F_noI"), ("F_C_noI", "F_noI"), ("F_CE_noI", "F_E_noI"), ("F_CE_noI", "F_C_noI")): C(x, y)
    print("\ninstruction x content 2x2s (solved; interaction = (X - X_noI) - (I - bare)):")
    for x in ("F", "F_E", "F_C", "F_CE"):
        if seen[x] and seen[x + "_noI"]:
            s = lambda a: len(solved[a])
            print(f"  {x:5s}: bare {s('bare')}  I {s('I')}  {x}_noI {s(x+'_noI')}  {x} {s(x)}   instruction alone {s('I')-s('bare'):+d}, on top of {x} {s(x)-s(x+'_noI'):+d}, interaction {(s(x)-s(x+'_noI'))-(s('I')-s('bare')):+d}")
if __name__ == "__main__":
    main()
