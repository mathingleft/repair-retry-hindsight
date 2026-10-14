"""No-instruction arms (arms.NOI_ARMS, out_noi/) against the main and extension arms.

    python analyze_noi.py [main_outdir] [ext_outdir] [noi_outdir]     # default: out/ out_ext/ out_noi/

For every X in {F, F_E, F_R, F_C, F_CE, F_E_shuf}: X_noI vs bare (feedback content with no instruction), X vs X_noI
(the instruction's worth on top of X), the ladder inside the no-instruction regime (F_E_noI vs F_noI, F_E_shuf_noI vs
F_E_noI, F_R_noI vs F_noI, F_C_noI vs F_noI, F_CE_noI vs F_E_noI, F_CE_noI vs F_C_noI, F_noI vs N_noI), and the
instruction x content 2x2 per content type. Exact paired sign test on per-problem pass counts (two-sided), McNemar on
the solved indicator, paired bootstrap 95% CI on the solved-rate difference (all from analyze_ext).
"""
import collections, json, os, random, sys
from analyze_ext import load, contrast

HERE = os.path.dirname(os.path.abspath(__file__))
ORDER = ("bare", "I", "N", "N_noI", "F", "F_noI", "F_E", "F_E_noI", "F_E_shuf", "F_E_shuf_noI",
         "F_R", "F_R_noI", "F_C", "F_C_noI", "F_CE", "F_CE_noI")

def main(main_out="out", ext_out="out_ext", noi_out="out_noi"):
    npass = collections.defaultdict(collections.Counter); solved = collections.defaultdict(set); seen = collections.Counter()
    for d in (main_out, ext_out, noi_out): load(d, npass, solved, seen)
    pids = sorted(json.loads(l)["id"] for l in open(os.path.join(HERE, "bank.jsonl")) if not json.loads(l).get("is_framing_holdout"))
    have = [a for a in ORDER if seen[a]]
    print(f"{len(pids)} problems\narm            rows  solved  passes")
    for a in have: print(f"{a:13s} {seen[a]:5d} {len(solved[a]):6d} {sum(npass[a].values()):7d}")
    rng = random.Random(1)
    def C(x, y):
        if seen[x] and seen[y]: print(contrast(x, y, npass, solved, pids, rng))
    print("\nfeedback content with NO instruction (vs bare):")
    for x in ("N_noI", "F_noI", "F_E_noI", "F_E_shuf_noI", "F_R_noI", "F_C_noI", "F_CE_noI"): C(x, "bare")
    print("\nthe instruction's worth on top of each scaffold (X vs X_noI):")
    for x in ("N", "F", "F_E", "F_E_shuf", "F_R", "F_C", "F_CE"): C(x, x + "_noI")
    print("\nthe ladder inside the no-instruction regime:")
    for x, y in (("F_E_noI", "F_noI"), ("F_E_shuf_noI", "F_E_noI"), ("F_E_shuf_noI", "F_noI"), ("F_R_noI", "F_noI"),
                 ("F_C_noI", "F_noI"), ("F_CE_noI", "F_E_noI"), ("F_CE_noI", "F_C_noI"), ("F_noI", "N_noI")): C(x, y)
    print("\ninstruction x content 2x2s (solved counts; interaction = (X - X_noI) - (I - bare)):")
    for x in ("N", "F", "F_E", "F_R", "F_C", "F_CE"):
        if seen[x] and seen[x + "_noI"] and seen["I"]:
            s = lambda a: len(solved[a])
            print(f"  {x:5s}: bare {s('bare')}  I {s('I')}  {x}_noI {s(x+'_noI')}  {x} {s(x)}   instruction alone {s('I')-s('bare'):+d}, "
                  f"on top of {x} {s(x)-s(x+'_noI'):+d}, interaction {(s(x)-s(x+'_noI'))-(s('I')-s('bare')):+d}")

if __name__ == "__main__":
    main(*sys.argv[1:4])
