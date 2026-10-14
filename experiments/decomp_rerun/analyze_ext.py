"""Extension-arm analysis (post-hoc, 2026-08-28): F_E_shuf and N_noI (arms.EXT_ARMS) against the main arms.

    python analyze_ext.py [main_outdir] [ext_outdir]     # default: out/ out_ext/

Reads verified.jsonl from both dirs. Reports per-arm solved/passes, then:
  - error-content contrasts: F_E_shuf vs F_E (the test), F_E_shuf vs F, F_E vs F
  - the 2x2 {sentence none/N} x {instruction no/yes}: bare, I, N_noI, N, with main effects and interaction
Tests: exact paired sign test on per-problem pass counts (two-sided; this is the preregistered primary
statistic), McNemar on the solved indicator, paired bootstrap 95% CI on the solved-rate difference.
"""
import collections, json, os, random, sys
from math import comb

def sign(b, c):
    n = b + c
    if n == 0: return 1.0
    k = min(b, c); return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)

def load(outdir, npass, solved, arms_seen):
    p = os.path.join(outdir, "verified.jsonl")
    if not os.path.exists(p): return
    for l in open(p):
        try: d = json.loads(l)
        except Exception: continue
        arms_seen[d["arm"]] += 1
        if d["status"] == "PASS":
            npass[d["arm"]][d["id"]] += 1; solved[d["arm"]].add(d["id"])

def contrast(x, y, npass, solved, pids, rng, B=20000):
    b = sum(npass[x][p] > npass[y][p] for p in pids); c = sum(npass[x][p] < npass[y][p] for p in pids)
    bx = len(solved[x] - solved[y]); cy = len(solved[y] - solved[x])
    obs = (len(solved[x]) - len(solved[y])) / len(pids); ds = []
    for _ in range(B):
        S = [pids[rng.randrange(len(pids))] for _ in pids]
        ds.append(sum((p in solved[x]) - (p in solved[y]) for p in S) / len(pids))
    ds.sort()
    return (f"  {x:9s} vs {y:6s}: solved {len(solved[x]):3d} vs {len(solved[y]):3d}  ({100*obs:+5.1f}pp "
            f"[{100*ds[int(.025*B)]:+5.1f}, {100*ds[int(.975*B)]:+5.1f}])  pass-count sign {b}/{c} p={sign(b,c):.3g}   "
            f"solved McNemar {bx}/{cy} p={sign(bx,cy):.3g}")

def main(main_dir, ext_dir):
    npass = collections.defaultdict(collections.Counter); solved = collections.defaultdict(set)
    seen = collections.Counter()
    load(main_dir, npass, solved, seen); load(ext_dir, npass, solved, seen)
    bank = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bank.jsonl")
    pids = sorted(json.loads(l)["id"] for l in open(bank) if not json.loads(l).get("is_framing_holdout"))
    rng = random.Random(1)
    print("arm        rows  solved  passes")
    for a in ("bare", "I", "N", "N_noI", "F", "F_E", "F_E_shuf", "F_R", "F_C", "F_CE"):
        if seen[a]: print(f"  {a:9s} {seen[a]:4d}  {len(solved[a]):4d}   {sum(npass[a].values()):4d}")
    def have(*a):
        for x in a:
            if seen[x] != 4 * len(pids):
                print(f"  [skip: {x} has {seen[x]} rows, need {4*len(pids)}]"); return False
        return True
    print("\nerror content: is the compiler error interchangeable with a wrong one?")
    for x, y in (("F_E_shuf", "F_E"), ("F_E_shuf", "F"), ("F_E", "F")):
        if have(x, y): print(contrast(x, y, npass, solved, pids, rng))
    print("\n2x2 sentence x instruction line")
    print("                 no instruction   instruction")
    if have("bare", "I"): print(f"  no sentence     {len(solved['bare']):4d}            {len(solved['I']):4d}")
    if have("N_noI", "N"): print(f"  nonsense (N)    {len(solved['N_noI']):4d}            {len(solved['N']):4d}")
    for x, y in (("I", "bare"), ("N_noI", "bare"), ("N", "I"), ("N", "N_noI"), ("N_noI", "I")):
        if have(x, y): print(contrast(x, y, npass, solved, pids, rng))
    if have("bare", "I", "N_noI", "N"):
        # interaction on the solved indicator: (N - I) - (N_noI - bare), paired bootstrap over problems
        def inter(S):
            return (sum((p in solved["N"]) - (p in solved["I"]) for p in S) - sum((p in solved["N_noI"]) - (p in solved["bare"]) for p in S)) / len(pids)
        obs = inter(pids); ds = sorted(inter([pids[rng.randrange(len(pids))] for _ in pids]) for _ in range(20000))
        print(f"  interaction (sentence effect with instruction - without): {100*obs:+.1f}pp [{100*ds[500]:+.1f}, {100*ds[19499]:+.1f}]")
        print("  reading: I ~= N ~= N_noI >> bare -> any text; I ~= bare < N_noI ~= N -> the sentence, not the instruction;")
        print("           I ~= N >> N_noI ~= bare -> the instruction line does the work")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "out", sys.argv[2] if len(sys.argv) > 2 else "out_ext")
