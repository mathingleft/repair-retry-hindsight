"""Analysis for the decomposition rerun. Problem = unit throughout.

PREREGISTERED ADDITIONS (committed 2026-08-27, before any rerun data existed):
 - TOST equivalence on H2 (F_E vs F) and H3 (F_R vs F): margin +/- 8.0pp of
   problems-solved rate (= half the ledger framing effect of +16.8pp), 90% CI.
 - Confirmatory interaction: the F_CE-vs-F_E code effect differs between
   syntactic predecessors (buckets unknown_ident, parse) and non-syntactic
   (tactic, unsolved_goals, no_goals, type, other). Permutation test, 20k
   label flips within problem, two-sided. Grouping fixed HERE, pre-data,
   from the independent probe-v2 dataset.

Primary contrasts (preregistered here, before any data):
  H1  F vs bare        (does the content-free failure sentence help?)
  H2  F_E vs F         (does the compiler error add anything?)
  H3  F_R vs F         (does a reflection add anything?)
  H4  F_C vs F         (does the failed code alone add anything?)
  H5  F_CE vs F_E      (does adding code to error help/hurt? THE new cell)
Multiplicity (corrected 2026-08-28): the family frozen in commit original_comparison_plan was SIX tests (H1, H2, H2b,
H3, H4, H5; the old docstring said "H1-H5"). Holm is reported twice: over the original B1
preregistered three-test family (H1, H2, H3) and over the six-test rerun family.
TOST (corrected 2026-08-28): the frozen version used an independent-proportions SE although arms are
paired by problem; it now uses the paired per-problem difference in the solved indicator with a paired
SE (two one-sided z tests, margin +/-8pp) plus a paired percentile bootstrap CI (20k resamples).
Secondary: per-bucket breakdown of H5 (exploratory).
"""
import collections, itertools, json, os, sys
from math import factorial

def comb(n, k): return factorial(n) // (factorial(k) * factorial(n - k))
def sign_test(pos, neg, one_sided=False):
    """Exact paired sign test over non-tied problem-level n_pass differences."""
    n = pos + neg
    if n == 0: return 1.0
    p1 = sum(comb(n, i) for i in range(pos, n + 1)) / 2 ** n
    return min(1.0, p1 if one_sided else 2 * min(p1, 1 - p1 + comb(n, pos) / 2 ** n))

def mcnemar(b, c):
    """Exact McNemar: two-sided binomial test on the discordant pairs (b vs c, p=0.5).
    (Was referenced by the exploratory per-bucket section but never defined.)"""
    from math import comb
    n = b + c
    if n == 0: return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)

def main(outdir, exclude=()):
    """exclude: problem ids dropped from every count/contrast (e.g. to restrict to the frozen prereg bank)."""
    exclude = set(exclude)
    solved = collections.defaultdict(set)   # arm -> {problem ids with >=1 PASS}
    npass = collections.defaultdict(collections.Counter)  # arm -> {pid: n_pass of 4}
    counts = collections.defaultdict(collections.Counter)
    bucket = {}
    bank = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bank.jsonl")
    for l in open(bank):
        d = json.loads(l)
        if d["id"] in exclude: continue
        bucket[d["id"]] = d["bucket"]
    for l in open(os.path.join(outdir, "verified.jsonl")):
        d = json.loads(l)
        if d["id"] in exclude: continue
        counts[d["arm"]][d["status"] == "PASS"] += 1
        if d["status"] == "PASS":
            solved[d["arm"]].add(d["id"]); npass[d["arm"]][d["id"]] += 1
    pids = sorted({d for a in solved for d in solved[a]} | set(bucket))
    arms = sorted(counts)
    print("arm      problems-solved   passes/rollouts")
    for a in arms:
        tot = sum(counts[a].values())
        print(f"  {a:6s} {len(solved[a]):4d}            {counts[a][True]}/{tot}")
    # prereg primaries one-sided (F>bare, F_E>F, F_E>bare); extension + reflection two-sided
    H = [("H1", "F", "bare", True), ("H2", "F_E", "F", True), ("H2b", "F_E", "bare", True),
         ("H3", "F_R", "F", False), ("H4", "F_C", "F", False), ("H5", "F_CE", "F_E", False)]
    res = []
    for name, x, y, one in H:
        if x not in counts or y not in counts: continue
        diffs = [npass[x][p] - npass[y][p] for p in pids]
        b = sum(1 for d in diffs if d > 0); c = sum(1 for d in diffs if d < 0)
        res.append((name, x, y, len(solved[x]), len(solved[y]), b, c, sign_test(b, c, one)))
    def holm(rows):
        rows = sorted(rows, key=lambda r: r[-1]); m = len(rows); out = []; prev = 0
        for i, r in enumerate(rows):
            adj = max(prev, min(1.0, (m - i) * r[-1])); prev = adj; out.append((r, adj))
        return out
    for label, fam in (("original B1 prereg family (H1, H2, H3)", ("H1", "H2", "H3")),
                       ("rerun family frozen in original_comparison_plan (H1, H2, H2b, H3, H4, H5)", ("H1", "H2", "H2b", "H3", "H4", "H5"))):
        print(f"\nHolm over the {label}:")
        print("contrast   arms          solved     discordant   p        Holm-adj")
        for (name, x, y, sx, sy, b, c, p), adj in holm([r for r in res if r[0] in fam]):
            print(f"  {name:4s} {x:>5s} vs {y:5s}  {sx:3d} vs {sy:3d}   {b:3d} / {c:3d}   {p:8.2e}  {adj:8.2e}")
    # --- preregistered TOST (equivalence, +/-8pp margin, problems-solved rate) ---
    import math, random
    def tost(x, y, margin=0.08):
        """Paired TOST on the problem-level solved indicator: d_i = 1[x solves i] - 1[y solves i]."""
        d = [(1 if p in solved[x] else 0) - (1 if p in solved[y] else 0) for p in pids]
        n = len(d); mean = sum(d) / n
        var = sum((v - mean) ** 2 for v in d) / (n - 1); se = math.sqrt(var / n) if var > 0 else 1e-9
        phi = lambda t: .5 * (1 + math.erf(t / math.sqrt(2)))
        p = max(1 - phi((mean + margin) / se), 1 - phi((margin - mean) / se))
        rng = random.Random(22072600); boots = []
        for _ in range(20000):
            boots.append(sum(d[rng.randrange(n)] for _ in range(n)) / n)
        boots.sort()
        ci90 = (boots[int(.05 * len(boots))], boots[int(.95 * len(boots)) - 1])
        ci95 = (boots[int(.025 * len(boots))], boots[int(.975 * len(boots)) - 1])
        return mean, se, p, ci90, ci95
    for x, y in (("F_E", "F"), ("F_R", "F")):
        if x in solved and y in solved:
            mean, se, p, ci90, ci95 = tost(x, y)
            print(f"\nTOST (paired) {x} vs {y}: mean paired diff in solved rate {mean:+.3f} (SE {se:.3f}); "
                  f"margin +/-8pp: p = {p:.4f} {'EQUIVALENT' if p < .05 else 'not shown equivalent'}; "
                  f"paired bootstrap 90% CI [{ci90[0]:+.3f}, {ci90[1]:+.3f}], 95% CI [{ci95[0]:+.3f}, {ci95[1]:+.3f}]")
    # --- preregistered interaction: code effect x predecessor failure type ---
    SYN = {"unknown_ident", "parse"}
    if "F_CE" in npass and "F_E" in npass:
        random.seed(0)
        syn = [p for p in pids if bucket.get(p) in SYN]
        non = [p for p in pids if bucket.get(p) not in SYN]
        def eff(sub):
            return sum(npass["F_CE"][p]-npass["F_E"][p] for p in sub)/max(1,len(sub))
        obs = eff(non) - eff(syn)
        perm = []
        for _ in range(20000):
            d1=d2=0
            for p in syn:
                a,b = npass["F_CE"][p], npass["F_E"][p]
                if random.random()<.5: a,b=b,a
                d1 += a-b
            for p in non:
                a,b = npass["F_CE"][p], npass["F_E"][p]
                if random.random()<.5: a,b=b,a
                d2 += a-b
            perm.append(d2/max(1,len(non)) - d1/max(1,len(syn)))
        pv = sum(1 for v in perm if abs(v)>=abs(obs))/len(perm)
        print(f"\nINTERACTION (prereg): code effect non-syntactic minus syntactic = {obs:+.3f}"
              f" n_pass/problem, permutation p = {pv:.4f} (syn n={len(syn)}, non n={len(non)})")
    if "F_CE" in counts and "F_E" in counts:
        posthoc = [a for a in ("I", "N", "N2", "N3", "N4", "N5", "D", "P", "S") if a in npass]
        if posthoc:
            print("\nexploratory (post-hoc failure-sentence ablations; F scaffold, 13-token sentences; see arms.py):")
            pairs = [(f"{a} vs bare", a, "bare") for a in posthoc] + [(f"F vs {a}", "F", a) for a in posthoc]
            if "N" in npass and "I" in npass: pairs.append(("N vs I", "N", "I"))
            for lab, a, b in pairs:
                pos = len([q for q in pids if npass[a][q] > npass[b][q]])
                neg = len([q for q in pids if npass[a][q] < npass[b][q]])
                print(f"  {lab:10s} solved {len(solved[a]):3d} vs {len(solved[b]):3d}   discordant {pos:3d}/{neg:3d}   sign-test p={sign_test(pos, neg):.3g}")
        print("\nexploratory: H5 by predecessor failure bucket")
        for bk in sorted(set(bucket.values())):
            sub = [p for p in bucket if bucket[p] == bk]
            b = len([p for p in sub if p in solved["F_CE"] and p not in solved["F_E"]])
            c = len([p for p in sub if p in solved["F_E"] and p not in solved["F_CE"]])
            print(f"  {bk:16s} n={len(sub):3d}  F_CE-only={b}  F_E-only={c}  p={mcnemar(b,c):.3f}")

if __name__ == "__main__":
    # usage: analyze.py [outdir] [--exclude id1,id2,...]
    args = [a for a in sys.argv[1:]]
    excl = []
    if "--exclude" in args:
        i = args.index("--exclude"); excl = args[i + 1].split(","); del args[i:i + 2]
    if excl: print(f"EXCLUDED problems: {excl}")
    main(args[0] if args else os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"), exclude=excl)
