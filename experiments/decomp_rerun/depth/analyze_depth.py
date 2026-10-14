"""Depth arms analysis: assert (content-free "## Previous attempt N: failed."), f (compiler error) and noise (random-word
sentence per block) on the
same stack and the same deterministic seeds; base-iid null from the bank's cold pass counts.

    python depth/analyze_depth.py [outdir=out] [fp8_outdir=out_depth_fp8]
"""
import collections, json, os, random, sys
from math import comb

def sign_test(pos, neg):
    n = pos + neg
    if n == 0: return 1.0
    k = min(pos, neg)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)

def load(path):
    first, mx = {}, collections.defaultdict(int)
    if not os.path.exists(path): return first, mx
    for l in open(path):
        x = json.loads(l); k = (x["id"], x["traj"]); mx[k] = max(mx[k], x["attempt"])
        if x["status"] == "PASS": first[k] = min(first.get(k, 10**9), x["attempt"])
    return first, mx

def main(outdir="out", fp8="out_depth_fp8"):
    here = os.path.dirname(os.path.abspath(__file__))
    bank = [json.loads(l) for l in open(os.path.join(here, "depth_bank.jsonl"))]
    pids = [b["id"] for b in bank]; M = 10; B = 64
    arms = {"assert": load(os.path.join(outdir, "depth_assert.jsonl")), "f": load(os.path.join(outdir, "depth_f.jsonl")),
            "noise": load(os.path.join(outdir, "depth_noise.jsonl")),
            "blind": load(os.path.join(outdir, "depth_blind.jsonl")),   # measured: 64 independent bare draws per trajectory
            "blind_I": load(os.path.join(outdir, "depth_blind_I.jsonl")),   # measured: 64 independent instruction-prompt draws
            "blind_F": load(os.path.join(outdir, "depth_blind_F.jsonl")),   # measured: 64 independent sentence-only draws
            "assert_fp8": load(os.path.join(fp8, "depth_assert.jsonl"))}
    ks = (1, 2, 4, 8, 16, 32, 64)
    print(f"depth arms: {len(pids)} problems x {M} trajectories x <= {B} attempts; discovery@k = trajectories with a first PASS by attempt k")
    print(f"{'arm':11}{'trajs':>6}{'solved':>7}  " + "".join(f"@{k:<5}" for k in ks))
    for a, (first, mx) in arms.items():
        if not mx: continue
        print(f"{a:11}{len(mx):6}{len(first):7}  " + "".join(f"{sum(1 for v in first.values() if v <= k):<6}" for k in ks))
    # synthetic base-iid null from the recovered pool, under BOTH surviving transcriptions of its per-problem rates:
    # depth_bank.jsonl base_pass/base_n (used for `blind` in earlier drafts) and bank.jsonl base_pass_rate_k256.
    # P(first pass within k iid draws) = 1 - (1 - p)^k. Band = the two rows; the measured `blind` arm supersedes both.
    bank1 = {r["id"]: r for r in map(json.loads, open(os.path.join(os.path.dirname(here), "bank.jsonl")))}
    for label, rate in (("pool-iid(A)", lambda b: (b.get("base_pass") or 0) / (b.get("base_n") or 1)),
                        ("pool-iid(B)", lambda b: bank1.get(b["id"], {}).get("base_pass_rate_k256") or 0.0)):
        null = [sum(M * (1 - (1 - rate(b)) ** k) for b in bank) for k in ks]
        print(f"{label:11}{110:6}{null[-1]:7.1f}  " + "".join(f"{v:<6.1f}" for v in null) + f"   (synthetic; {'depth_bank.jsonl' if label.endswith('(A)') else 'bank.jsonl'} transcription)")
    print("  per-problem pool rates (A):", {b["id"]: f"{b.get('base_pass')}/{b.get('base_n')}" for b in bank})
    for a, lab in (("blind", "bare prompt"), ("blind_I", "instruction prompt"), ("blind_F", "sentence-only prompt")):
        if not arms[a][1]: continue
        cnt = collections.Counter(); tot = collections.Counter()
        for l in open(os.path.join(outdir, f"depth_{a}.jsonl")):
            x = json.loads(l); tot[x["id"]] += 1; cnt[x["id"]] += x["status"] == "PASS"
        print(f"  measured per-draw pass rates on this stack ({a} = {lab}, passes/draws):", {p: f"{cnt[p]}/{tot[p]}" for p in pids})
    def pair(x, y):
        fx, mxx = arms[x]; fy, mxy = arms[y]
        if not mxx or not mxy: return
        keys = sorted(set(mxx) & set(mxy))
        print(f"\npaired by (problem, trajectory), {x} vs {y} — same seeds in both arms:")
        for k in ks:
            xo = sum(1 for t in keys if fx.get(t, 10**9) <= k and not fy.get(t, 10**9) <= k)
            yo = sum(1 for t in keys if fy.get(t, 10**9) <= k and not fx.get(t, 10**9) <= k)
            both = sum(1 for t in keys if fx.get(t, 10**9) <= k and fy.get(t, 10**9) <= k)
            print(f"  @{k:<3} {x}-only {xo:3}  {y}-only {yo:3}  both {both:3}   sign-test p = {sign_test(xo, yo):.3g}")
        pos = sum(1 for p in pids if sum(1 for (q,t) in fx if q==p) > sum(1 for (q,t) in fy if q==p))
        neg = sum(1 for p in pids if sum(1 for (q,t) in fx if q==p) < sum(1 for (q,t) in fy if q==p))
        print(f"  problem-level sign test ({x} vs {y}): {pos}/{neg}, p = {sign_test(pos, neg):.3g}")
        # crossing: ({y}-{x})@64 - ({y}-{x})@8; trajectory-level bootstrap, seed 0, 4000 resamples
        def cross(ks):
            return (sum(1 for t in ks if fy.get(t, 10**9) <= 64) - sum(1 for t in ks if fx.get(t, 10**9) <= 64)) - \
                   (sum(1 for t in ks if fy.get(t, 10**9) <= 8) - sum(1 for t in ks if fx.get(t, 10**9) <= 8))
        random.seed(0)
        bs = sorted(cross(random.choices(keys, k=len(keys))) for _ in range(4000))
        print(f"  crossing ({y}-{x})@64 - ({y}-{x})@8 = {cross(keys):+d}, trajectory bootstrap 95% [{bs[100]:+d}, {bs[3899]:+d}]")
    for x, y in (("assert", "f"), ("noise", "f"), ("assert", "noise"), ("assert", "blind"), ("noise", "blind"), ("f", "blind"),
                 ("assert", "blind_I"), ("noise", "blind_I"), ("f", "blind_I"), ("blind_I", "blind"),
                 ("assert", "blind_F"), ("f", "blind_F"), ("blind_F", "blind_I"), ("blind_F", "blind")):
        pair(x, y)
    names = [a for a in ("assert", "f", "noise", "blind", "blind_I", "blind_F", "assert_fp8") if arms[a][1]]
    print("\nper problem (solved trajectories /10): " + " | ".join(names))
    for p in pids:
        print(f"  {p:34} " + " | ".join(f"{sum(1 for (q, t) in arms[a][0] if q == p):2}" for a in names))
    print("\nDECISION RULE (HANDOFF item 2): assert ~= f -> content-free claim holds at depth; assert ~= base-iid -> content matters at depth.")
    print("noise arm: if noise ~= f and noise ~= assert, the early assert lead is not a general content-free effect; if noise ~= assert > f at k<=16, it is.")

if __name__ == "__main__":
    main(*(sys.argv[1:3]))
