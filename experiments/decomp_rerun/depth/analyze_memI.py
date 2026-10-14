"""Pre-registered analysis for the memoryless instruction-only retry control (results/PREREG_memoryless_I_2026-09-06.md).

    python depth/analyze_memI.py <memI_outdir> [<depth-format jsonl or outdir> ...] [--repair name=path ...]

Memoryless I: n_i = 640 independent draws per problem, c_i passes. Pooled without-replacement estimator
    p_I,i(k) = 1 - C(n_i - c_i, k) / C(n_i, k).
Repair arms (sequential, early-stopped trajectories; depth_*.jsonl schema: arm,id,traj,attempt,status):
    p_repair,i(k) = #{trajectories with a PASS at attempt <= k} / #trajectories.
Delta_k = mean_i [p_repair,i(k) - p_I,i(k)] at k in {1,2,4,8,16,32,64}; paired PROBLEM bootstrap (problems resampled
with replacement, all trajectories and all 640 draws carried; 10,000 resamples; percentile 95% CI). Primary endpoint:
CE - I at k = 64. Sensitivity: conditional on attempt-1 failure (repair: trajectories whose attempt 1 failed, success by
k; I: the pooled estimator over k-1 draws). Exploratory: token cost, per-problem heterogeneity.
Repair arms are whatever depth-format files are passed: the same-stack local depth arms (out/depth_*.jsonl) and/or
exported code-bearing trajectories (--repair CE=path). Nothing is compared that is not passed in.
"""
import collections, glob, json, math, os, random, sys

KS = (1, 2, 4, 8, 16, 32, 64)
B = 10000

def comb(n, k): return math.comb(n, k) if 0 <= k <= n else 0

def p_pooled(n, c, k):
    if n < k: return float("nan")
    return 1.0 - comb(n - c, k) / comb(n, k)

def load_memI(outdir):
    n = collections.Counter(); c = collections.Counter(); tok = collections.Counter(); infra = collections.Counter()
    first_fail = collections.Counter()      # attempt-1 rows that failed (for the conditional analysis; informational)
    for l in open(os.path.join(outdir, "memI_verified.jsonl")):
        d = json.loads(l); p = d["problem_id"]; n[p] += 1
        tok[p] += d.get("response_token_count") or 0
        if d["verification_result"] == "PASS": c[p] += 1
        if str(d.get("infrastructure_status", "ok")) != "ok": infra[p] += 1
    return n, c, tok, infra

def load_repair(path, name=None):
    """depth-format jsonl -> {arm: {pid: {traj: first_pass_attempt or None}}} (plus attempts seen per traj)."""
    arms = collections.defaultdict(lambda: collections.defaultdict(dict)); att1fail = collections.defaultdict(lambda: collections.defaultdict(set))
    for l in open(path):
        try: d = json.loads(l)
        except Exception: continue
        a = name or d.get("arm"); pid = d.get("id") or d.get("problem_id"); t = d.get("traj", d.get("trajectory_id")); k = d.get("attempt", d.get("attempt_id"))
        st = d.get("status") or d.get("verification_result")
        e = arms[a][pid].setdefault(t, None)
        if st == "PASS" and (e is None or k < e): arms[a][pid][t] = k
        if k == 1 and st != "PASS": att1fail[a][pid].add(t)
    return arms, att1fail

def p_repair(trajs, k, only=None):
    ts = [t for t in trajs if only is None or t in only]
    if not ts: return float("nan")
    return sum(1 for t in ts if trajs[t] is not None and trajs[t] <= k) / len(ts)

def boot_ci(pids, f, rng):
    ds = []
    for _ in range(B):
        S = [pids[rng.randrange(len(pids))] for _ in pids]
        ds.append(sum(f(p) for p in S) / len(S))
    ds.sort(); return ds[int(.025 * B)], ds[int(.975 * B)]

def main():
    args = sys.argv[1:]; outdir = args[0]; rest = args[1:]
    repair_files = []; i = 0
    while i < len(rest):
        if rest[i] == "--repair": name, path = rest[i + 1].split("=", 1); repair_files.append((name, path)); i += 2
        elif os.path.isdir(rest[i]): repair_files += [(None, f) for f in sorted(glob.glob(os.path.join(rest[i], "depth_*.jsonl")))]; i += 1
        else: repair_files.append((None, rest[i])); i += 1
    n, c, tok, infra = load_memI(outdir)
    pids = sorted(n); rng = random.Random(22072606)
    print(f"memoryless_I: {len(pids)} problems, {sum(n.values())} draws, {sum(c.values())} passes, "
          f"{sum(infra.values())} infra-error rows")
    print("\nper problem: draws  passes  rate    pass@k (pooled estimator)  " + "  ".join(f"k={k}" for k in KS))
    for p in pids:
        print(f"  {p:32s} {n[p]:4d} {c[p]:5d} {c[p]/n[p]:6.3f}   " + "  ".join(f"{p_pooled(n[p], c[p], k):.3f}" for k in KS)
              + f"   mean resp tokens {tok[p]/n[p]:.0f}")
    print("\nmemoryless_I mean pass@k over problems:")
    for k in KS:
        lo, hi = boot_ci(pids, lambda p: p_pooled(n[p], c[p], k), rng)
        print(f"  k={k:2d}: {sum(p_pooled(n[p], c[p], k) for p in pids)/len(pids):.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
    arms = {}; a1f = {}
    for name, path in repair_files:
        A, F = load_repair(path, name)
        for a in A: arms[a] = A[a]; a1f[a] = F[a]
    if not arms:
        print("\n(no repair arms passed; pass depth-format jsonl files or --repair NAME=path)"); return
    for a in sorted(arms):
        miss = [p for p in pids if p not in arms[a]]
        if miss: print(f"\n[{a}: missing problems {miss}; contrast restricted to the overlap]")
        P = [p for p in pids if p in arms[a]]
        print(f"\n=== {a} (repair, {sum(len(arms[a][p]) for p in P)} trajectories) minus memoryless_I, paired problem bootstrap (n={len(P)} problems)")
        print("   k   repair   memI    delta_k   95% CI")
        for k in KS:
            f = lambda p, k=k: p_repair(arms[a][p], k) - p_pooled(n[p], c[p], k)
            obs = sum(f(p) for p in P) / len(P); lo, hi = boot_ci(P, f, rng)
            tag = "  <- PRIMARY" if (k == 64 and a.upper().startswith("CE") and not a.upper().startswith("CE_")) else ""
            verdict = ("repair underperforms retrying" if hi < 0 else "repair beats retrying" if lo > 0 else "no advantage established")
            print(f"  {k:3d}   {sum(p_repair(arms[a][p], k) for p in P)/len(P):.3f}   {sum(p_pooled(n[p], c[p], k) for p in P)/len(P):.3f}   "
                  f"{100*obs:+6.1f}pp  [{100*lo:+6.1f}, {100*hi:+6.1f}]  {verdict if k == 64 else ''}{tag}")
        # sensitivity: conditional on attempt-1 failure
        print(f"   conditional on attempt 1 failing (repair: trajectories with a failed attempt 1; memI: pooled estimator over k-1 draws)")
        for k in (2, 4, 8, 16, 32, 64):
            f = lambda p, k=k: p_repair(arms[a][p], k, only=a1f[a][p]) - p_pooled(n[p], c[p], k - 1)
            vals = [f(p) for p in P if not math.isnan(f(p))]
            if not vals: continue
            Pk = [p for p in P if not math.isnan(f(p))]
            lo, hi = boot_ci(Pk, f, rng)
            print(f"     k={k:2d}: delta {100*sum(vals)/len(vals):+6.1f}pp  [{100*lo:+6.1f}, {100*hi:+6.1f}]  (n={len(Pk)})")
    # pairwise repair contrasts at k=64 (secondary), same bootstrap
    names = sorted(arms)
    if len(names) > 1:
        print("\n=== repair-arm pairwise contrasts at k=64 (secondary; paired problem bootstrap)")
        for i, x in enumerate(names):
            for y in names[i + 1:]:
                P = [p for p in pids if p in arms[x] and p in arms[y]]
                f = lambda p: p_repair(arms[x][p], 64) - p_repair(arms[y][p], 64)
                obs = sum(f(p) for p in P) / len(P); lo, hi = boot_ci(P, f, rng)
                print(f"  {x} - {y}: {100*obs:+6.1f}pp  [{100*lo:+6.1f}, {100*hi:+6.1f}]  (n={len(P)})")
    print("\nper-problem heterogeneity at k=64 (repair solved-fraction | memI pooled pass@64):")
    for p in pids:
        print(f"  {p:32s} " + "  ".join(f"{a}={p_repair(arms[a][p], 64):.2f}" if p in arms[a] else f"{a}=n/a" for a in names) + f"  | memI={p_pooled(n[p], c[p], 64):.2f}")

if __name__ == "__main__":
    main()
