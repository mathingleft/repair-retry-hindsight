"""F_CE_shuf analysis (post-hoc, 2026-08-31): the F_CE scaffold with a WRONG problem's compiler error
(same seeded derangement as F_E_shuf). Does error content matter once the failed code is present?

    python analyze_ext2.py [main_outdir] [ext_outdir] [ext2_outdir] [bank]
    # Qwen  : python analyze_ext2.py out out_ext out_ext2
    # prover: python analyze_ext2.py out_m2 out_m2_ext out_m2_ext2 out_m2/bank2.jsonl

Contrasts (exact paired sign test on per-problem pass counts, McNemar on solved, paired bootstrap 95% CI):
  F_CE_shuf vs F_CE     : the test — wrong error vs own error, code present (twin of F_E_shuf vs F_E)
  F_CE_shuf vs F_C      : does ANY error block add/hurt on top of the code?
  F_CE      vs F_C      : does the OWN error add/hurt on top of the code? (context row)
  F_CE_shuf vs F_E_shuf : the code's effect holding the (wrong) error fixed
"""
import collections, json, os, random, re, sys
from math import comb

def body(code):
    # same normalisation as copy_rate.py: proof body after :=, comments stripped, whitespace collapsed
    if code is None: return None
    i = code.find(":="); b = code[i + 2:] if i >= 0 else code
    b = re.sub(r"--[^\n]*", "", b); b = re.sub(r"/-.*?-/", "", b, flags=re.S)
    return re.sub(r"\s+", " ", b).strip()

def sign(b, c):
    n = b + c
    if n == 0: return 1.0
    k = min(b, c); return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)

def load(outdir, npass, solved, seen):
    p = os.path.join(outdir, "verified.jsonl")
    if not os.path.exists(p): return
    for l in open(p):
        try: d = json.loads(l)
        except Exception: continue
        seen[d["arm"]] += 1
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
    return (f"  {x:9s} vs {y:9s}: solved {len(solved[x]):3d} vs {len(solved[y]):3d}  ({100*obs:+5.1f}pp "
            f"[{100*ds[int(.025*B)]:+5.1f}, {100*ds[int(.975*B)]:+5.1f}])  pass-count sign {b}/{c} p={sign(b,c):.3g}   "
            f"solved McNemar {bx}/{cy} p={sign(bx,cy):.3g}")

def main(main_dir, ext_dir, ext2_dir, bank=None):
    npass = collections.defaultdict(collections.Counter); solved = collections.defaultdict(set)
    seen = collections.Counter()
    for d in (main_dir, ext_dir, ext2_dir): load(d, npass, solved, seen)
    bank = bank or os.path.join(os.path.dirname(os.path.abspath(__file__)), "bank.jsonl")
    pids = sorted(json.loads(l)["id"] for l in open(bank) if not json.loads(l).get("is_framing_holdout"))
    rng = random.Random(1)
    print(f"bank {bank}: {len(pids)} problems\narm         rows  solved  passes")
    for a in ("F", "F_C", "F_E", "F_E_shuf", "F_CE", "F_CE_shuf"):
        if seen[a]: print(f"  {a:9s} {seen[a]:4d}  {len(solved[a]):4d}   {sum(npass[a].values()):4d}")
    def have(*a):
        for x in a:
            if seen[x] != 4 * len(pids):
                print(f"  [skip: {x} has {seen[x]} rows, need {4*len(pids)}]"); return False
        return True
    print("\nerror content with code present: is the compiler error interchangeable with a wrong one?")
    for x, y in (("F_CE_shuf", "F_CE"), ("F_CE_shuf", "F_C"), ("F_CE", "F_C"), ("F_CE_shuf", "F_E_shuf")):
        if have(x, y): print(contrast(x, y, npass, solved, pids, rng))
    if have("F_E", "F_E_shuf", "F_CE", "F_CE_shuf"):
        # does the own-vs-wrong error gap depend on code presence?
        # interaction = (F_CE - F_CE_shuf) - (F_E - F_E_shuf) on the solved indicator, paired bootstrap over problems
        def inter(S):
            return (sum((p in solved["F_CE"]) - (p in solved["F_CE_shuf"]) for p in S)
                    - sum((p in solved["F_E"]) - (p in solved["F_E_shuf"]) for p in S)) / len(pids)
        obs = inter(pids); ds = sorted(inter([pids[rng.randrange(len(pids))] for _ in pids]) for _ in range(20000))
        print(f"  interaction (own-vs-wrong error gap, with code - without): {100*obs:+.1f}pp "
              f"95% CI [{100*ds[500]:+.1f}, {100*ds[19499]:+.1f}]")
        print("  reading: >0 with CI excluding 0 -> error CONTENT matters only when the failed code is present")
    # copy channel: does the wrong error still suppress copying of the failed proof? (copy_rate.py normalisation)
    pred = {json.loads(l)["id"]: body(json.loads(l)["predecessor_code"]) for l in open(bank)}
    ncopy = collections.defaultdict(collections.Counter); ntot = collections.Counter()
    for d in (main_dir, ext_dir, ext2_dir):
        p = os.path.join(d, "verified.jsonl")
        if not os.path.exists(p): continue
        for l in open(p):
            r = json.loads(l)
            if r["arm"] not in ("F_C", "F_CE", "F_CE_shuf") or r["id"] not in pred: continue
            ntot[r["arm"]] += 1
            if body(r.get("lean_code")) == pred[r["id"]]: ncopy[r["arm"]][r["id"]] += 1
    print("\nexact copies of the failed proof (body-normalised)")
    for a in ("F_C", "F_CE", "F_CE_shuf"):
        if ntot[a]:
            c = sum(ncopy[a].values())
            print(f"  {a:9s} {c}/{ntot[a]} = {100*c/ntot[a]:.1f}%")
    for x, y in (("F_CE_shuf", "F_CE"), ("F_CE_shuf", "F_C")):
        if ntot[x] and ntot[y]:
            b_ = sum(ncopy[x][p] > ncopy[y][p] for p in pids); c_ = sum(ncopy[x][p] < ncopy[y][p] for p in pids)
            print(f"  {x} vs {y} copies per problem: {b_}/{c_} p={sign(b_, c_):.3g}")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "out",
         sys.argv[2] if len(sys.argv) > 2 else "out_ext",
         sys.argv[3] if len(sys.argv) > 3 else "out_ext2",
         sys.argv[4] if len(sys.argv) > 4 else None)
