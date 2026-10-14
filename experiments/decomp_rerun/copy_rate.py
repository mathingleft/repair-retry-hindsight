"""One-step copy-rate / length analysis for the decomposition rerun (PAPER_OUTLINE_2026-08-28 Figs 1-2).

For each arm: fraction of rollouts whose proof body is byte-identical (after comment/whitespace
normalisation) to the frozen predecessor's body, mean similarity to it, pass rate among copies vs
non-copies, per-problem sign tests, bootstrap CIs on problems-solved, and output length.

    python copy_rate.py [outdir]     # default: out/
"""
import collections, difflib, itertools, json, os, random, re, statistics, sys
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = ("bare", "I", "F", "F_E", "F_C", "F_CE", "F_R", "N", "N2", "N3", "N4", "N5", "S", "P", "D")

def body(code):
    if code is None: return None
    i = code.find(":="); b = code[i + 2:] if i >= 0 else code
    b = re.sub(r"--[^\n]*", "", b); b = re.sub(r"/-.*?-/", "", b, flags=re.S)
    return re.sub(r"\s+", " ", b).strip()

def sign(b, c):
    n = b + c
    if n == 0: return 1.0
    k = min(b, c); return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)

def main(outdir):
    bank = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(HERE, "bank.jsonl"))}
    pred = {pid: body(d["predecessor_code"]) for pid, d in bank.items()}
    rows = [json.loads(l) for l in open(os.path.join(outdir, "verified.jsonl"))]
    rows = [d for d in rows if d["arm"] in ARMS]
    pids = sorted({d["id"] for d in rows})
    copies = collections.defaultdict(collections.Counter)
    solved = collections.defaultdict(set)
    print("arm    n    exact-copy   mean-sim   pass|copy   pass|noncopy")
    for a in ARMS:
        R = [d for d in rows if d["arm"] == a]; e = 0; ss = 0.0; cp = [0, 0]; nc = [0, 0]
        if not R: continue
        for d in R:
            b = body(d.get("lean_code")); p = pred[d["id"]]
            c = b is not None and b == p
            if c: e += 1; copies[a][d["id"]] += 1
            if b is not None: ss += difflib.SequenceMatcher(None, b[:3000], p[:3000]).ratio()
            ok = d["status"] == "PASS"
            if ok: solved[a].add(d["id"])
            t = cp if c else nc; t[0] += 1; t[1] += ok
        print(f"{a:5s} {len(R):4d}   {e:4d} ({100*e/len(R):4.1f}%)   {ss/len(R):.3f}     "
              f"{cp[1]}/{cp[0]}        {nc[1]}/{nc[0]} ({100*nc[1]/max(nc[0],1):.1f}%)")
    print("\nper-problem copy-rate sign tests")
    for x, y in [("bare", "F"), ("bare", "I"), ("bare", "N"), ("F_C", "F"), ("F_CE", "F_E"), ("F_C", "F_CE")]:
        b = sum(copies[x][p] > copies[y][p] for p in pids); c = sum(copies[x][p] < copies[y][p] for p in pids)
        print(f"  {x}>{y}: {b} vs {c}, p={sign(b, c):.2e}")
    print("\nproblems-solved contrasts: diff, paired bootstrap 95% CI, McNemar on solved indicator")
    rng = random.Random(1); B = 20000
    for x, y in [("F", "bare"), ("I", "bare"), ("F", "I"), ("N", "I"), ("F_E", "F"), ("F_R", "F"), ("F_C", "F"), ("F_C", "I"), ("F_CE", "F_E"), ("F_CE", "I"), ("N", "bare"), ("N", "F")]:
        obs = (len(solved[x]) - len(solved[y])) / len(pids); ds = []
        for _ in range(B):
            S = [pids[rng.randrange(len(pids))] for _ in pids]
            ds.append(sum((p in solved[x]) - (p in solved[y]) for p in S) / len(pids))
        ds.sort(); bx = len(solved[x] - solved[y]); cy = len(solved[y] - solved[x])
        print(f"  {x:5s}-{y:5s}: {100*obs:+5.1f}pp  [{100*ds[int(.025*B)]:+5.1f}, {100*ds[int(.975*B)]:+5.1f}]   "
              f"{bx}/{cy} p={sign(bx, cy):.3g}")
    # Inter-sample diversity (feedback on outline v2): distinct normalised bodies among the k=4 samples
    # per problem, and mean pairwise similarity within a problem. Tests whether N's lead over F comes
    # with a more diverse sample set (a second mechanism) or not (noise).
    byp = collections.defaultdict(lambda: collections.defaultdict(list))
    for d in rows: byp[d["arm"]][d["id"]].append(body(d.get("lean_code")))
    distinct = {a: {} for a in ARMS}; psim = {a: {} for a in ARMS}
    print("\ninter-sample diversity: mean distinct proofs of 4, frac all-4-distinct, mean within-problem pairwise sim")
    for a in ARMS:
        if not byp[a]: continue
        for p in pids:
            bs = [b for b in byp[a][p] if b is not None]
            distinct[a][p] = len(set(bs))
            sims = [difflib.SequenceMatcher(None, x[:2000], y[:2000]).ratio() for x, y in itertools.combinations(bs, 2)]
            psim[a][p] = statistics.mean(sims) if sims else 1.0
        dv = list(distinct[a].values())
        print(f"{a:5s} {statistics.mean(dv):.2f}   {sum(v == 4 for v in dv)/len(dv):.2f}   {statistics.mean(psim[a].values()):.3f}")
    print("per-problem sign tests: distinct x>y / x<y ; pairwise-sim x<y / x>y (first = x more diverse)")
    for x, y in [("N", "F"), ("N", "I"), ("F", "I"), ("N", "F_E"), ("F", "bare"), ("I", "bare"), ("N", "bare"), ("F_R", "F"), ("F_C", "F"), ("F_CE", "F_E")]:
        if not distinct[x] or not distinct[y]: continue
        b = sum(distinct[x][p] > distinct[y][p] for p in pids); c = sum(distinct[x][p] < distinct[y][p] for p in pids)
        b2 = sum(psim[x][p] < psim[y][p] for p in pids); c2 = sum(psim[x][p] > psim[y][p] for p in pids)
        print(f"  {x:5s} vs {y:5s}: distinct {b}/{c} p={sign(b, c):.2g}   sim-lower {b2}/{c2} p={sign(b2, c2):.2g}")
    print("\npass concentration among solved problems: mean passes of 4, problems with 4/4")
    npass = collections.defaultdict(collections.Counter)
    for d in rows:
        if d["status"] == "PASS": npass[d["arm"]][d["id"]] += 1
    for a in ARMS:
        s = list(npass[a].values())
        if s: print(f"  {a:5s} solved={len(s):3d} passes={sum(s):3d} mean={statistics.mean(s):.2f} 4of4={sum(v == 4 for v in s)}")
    roll = os.path.join(outdir, "rollouts.jsonl")
    if os.path.exists(roll):
        L = collections.defaultdict(list)
        for l in open(roll):
            try: d = json.loads(l)
            except Exception: continue
            if d["arm"] in ARMS: L[d["arm"]].append(len(d.get("text") or ""))
        print("\narm   median output chars   p95")
        for a in ARMS:
            if not L[a]: continue
            v = sorted(L[a]); print(f"{a:5s} {int(statistics.median(v)):6d}   {v[int(.95*len(v))]:7d}")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "out"))
