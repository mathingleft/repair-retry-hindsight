"""Preregistered analysis for the confirmatory error-content precision run.

  PREREG: PREREG_CONFIRM_ERROR_CONTENT_2026-08-30.md

Committed BEFORE generation. Run ONCE, after all blocks are generated and verified.

  python analyze_confirm_error.py out_confirm_error > out_confirm_error/ANALYSIS_CONFIRM.txt

Primary estimand (A): difference in the expected number of problems solved at least once in k=4,
on the FIXED 167-problem bank, F_E minus F_E_shuf, in percentage points.
Primary CI: problem-clustered, block-averaged (PREREG section 11).
Rollouts are never treated as independent observations; problem is always the cluster.
"""
import collections, json, math, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ARM_A, ARM_B = "F_E", "F_E_shuf"
K = 4
CRASHY = re.compile(r"verifier crash|no result returned|header command timed out|Failed to read from REPL", re.I)
TRUNC = "length"

# ---------- small stats helpers (no scipy on this stack) ----------
def norm_q(p):
    lo, hi = -12.0, 12.0
    for _ in range(300):
        m = (lo + hi) / 2
        if 0.5 * (1 + math.erf(m / math.sqrt(2))) < p: lo = m
        else: hi = m
    return (lo + hi) / 2

# Exact two-sided t quantiles for small df, where the expansion below is not accurate enough.
# (Only the secondary block-level interval uses small df; the primary uses df = n(R-1) >= 167.)
_T975 = {1: 12.7062, 2: 4.3027, 3: 3.1824, 4: 2.7764, 5: 2.5706, 6: 2.4469, 7: 2.3646, 8: 2.3060,
         9: 2.2622, 10: 2.2281, 11: 2.2010, 12: 2.1788, 13: 2.1604, 14: 2.1448, 15: 2.1314,
         16: 2.1199, 17: 2.1098, 18: 2.1009, 19: 2.0930, 20: 2.0860, 25: 2.0595, 30: 2.0423}
_T95 = {1: 6.3138, 2: 2.9200, 3: 2.3534, 4: 2.1318, 5: 2.0150, 6: 1.9432, 7: 1.8946, 8: 1.8595,
        9: 1.8331, 10: 1.8125, 11: 1.7959, 12: 1.7823, 13: 1.7709, 14: 1.7613, 15: 1.7531,
        16: 1.7459, 17: 1.7396, 18: 1.7341, 19: 1.7291, 20: 1.7247, 25: 1.7081, 30: 1.6973}

def t_q(p, df):
    """Two-sided quantile of Student t. Exact table for small df, expansion for large df."""
    if df <= 0: return float("nan")
    tab = _T975 if abs(p - 0.975) < 1e-9 else (_T95 if abs(p - 0.95) < 1e-9 else None)
    if tab is not None and df in tab: return tab[df]
    z = norm_q(p)
    g1 = (z**3 + z) / 4
    g2 = (5*z**5 + 16*z**3 + 3*z) / 96
    g3 = (3*z**7 + 19*z**5 + 17*z**3 - 15*z) / 384
    return z + g1/df + g2/df**2 + g3/df**3

def sign_test(b, c):
    from math import comb
    n = b + c
    if n == 0: return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)

# ---------- load ----------
def load_blocks(root):
    """root/bN/verified.jsonl -> {block: {arm: {pid: passes}}}, plus integrity counters."""
    blocks = {}
    integ = collections.defaultdict(collections.Counter)
    for d in sorted(os.listdir(root)):
        p = os.path.join(root, d, "verified.jsonl")
        if not (d.startswith("b") and os.path.exists(p)): continue
        npass = collections.defaultdict(collections.Counter)
        for l in open(p):
            try: r = json.loads(l)
            except Exception: continue
            arm = r["arm"]
            integ[(d, arm)]["rows"] += 1
            if str(r.get("finish")) == TRUNC: integ[(d, arm)]["trunc"] += 1
            if CRASHY.search(str(r.get("error") or "")): integ[(d, arm)]["infra"] += 1
            if r["status"] == "PASS":
                npass[arm][r["id"]] += 1
                integ[(d, arm)]["pass"] += 1
        blocks[d] = npass
    return blocks, integ

def pids_of(bank_path):
    return sorted(json.loads(l)["id"] for l in open(bank_path)
                  if l.strip() and not json.loads(l).get("is_framing_holdout"))

# ---------- primary estimator ----------
def primary(blocks, pids, arm_a=ARM_A, arm_b=ARM_B):
    """d[i][r] = solved_a - solved_b for problem i, block r."""
    bl = sorted(blocks)
    R = len(bl)
    d = {i: [] for i in pids}
    for r in bl:
        for i in pids:
            sa = 1 if blocks[r][arm_a][i] > 0 else 0
            sb = 1 if blocks[r][arm_b][i] > 0 else 0
            d[i].append(sa - sb)
    n = len(pids)
    D = sum(sum(v) for v in d.values()) / (n * R)
    # problem-clustered, block-averaged variance (PREREG 11)
    if R >= 2:
        var = 0.0
        for i in pids:
            m = sum(d[i]) / R
            var += sum((x - m) ** 2 for x in d[i]) / (R - 1)
        V = var / (n * n * R)
        df = n * (R - 1)
    else:
        # single block: fall back to the paired across-problem variance (df = n-1)
        m = D
        V = sum((sum(d[i]) - m) ** 2 for i in pids) / (n - 1) / n
        df = n - 1
    return D, V, df, R, d, bl

def block_level(blocks, pids, arm_a=ARM_A, arm_b=ARM_B):
    bl = sorted(blocks); out = []
    for r in bl:
        sa = sum(1 for i in pids if blocks[r][arm_a][i] > 0)
        sb = sum(1 for i in pids if blocks[r][arm_b][i] > 0)
        out.append((r, sa, sb, (sa - sb) / len(pids)))
    return out

def ci(D, V, df, conf=0.95):
    se = math.sqrt(max(V, 0.0))
    t = t_q(0.5 + conf / 2, df)
    return D - t * se, D + t * se, se, t

def boot_population(blocks, pids, arm_a=ARM_A, arm_b=ARM_B, B=20000, seed=1):
    """Population (problem-resampling) percentile bootstrap on the block-averaged solved rate.
    This is the interval the paper currently quotes (analyze_ext.contrast); it additionally treats
    the 167 problems as a sample from a population of problems and does NOT shrink to 0 with R."""
    import random
    rng = random.Random(seed)
    bl = sorted(blocks); R = len(bl)
    fa = {i: sum(1 for r in bl if blocks[r][arm_a][i] > 0) / R for i in pids}
    fb = {i: sum(1 for r in bl if blocks[r][arm_b][i] > 0) / R for i in pids}
    n = len(pids)
    obs = sum(fa[i] - fb[i] for i in pids) / n
    ds = []
    for _ in range(B):
        S = [pids[rng.randrange(n)] for _ in range(n)]
        ds.append(sum(fa[i] - fb[i] for i in S) / n)
    ds.sort()
    return obs, ds[int(.025 * B)], ds[int(.975 * B)]

# ---------- report ----------
def main(root):
    bank = os.path.join(HERE, "bank.jsonl")
    pids = pids_of(bank)
    n = len(pids)
    blocks, integ = load_blocks(root)
    if not blocks:
        print(f"no blocks found under {root}/b*/verified.jsonl"); sys.exit(1)
    P = print
    P("=" * 94)
    P("CONFIRMATORY ERROR-CONTENT PRECISION RUN — preregistered analysis")
    P("PREREG: PREREG_CONFIRM_ERROR_CONTENT_2026-08-30.md")
    P("=" * 94)
    P(f"problems n={n}   blocks R={len(blocks)}   arms {ARM_A} vs {ARM_B}   k={K}")
    P()
    P("--- INTEGRITY (per block x arm) ---")
    P(f"{'block':>6} {'arm':>10} {'rows':>6} {'expect':>7} {'pass':>6} {'trunc':>6} {'trunc%':>7} {'infra':>6}")
    bad = []
    for (b, arm) in sorted(integ):
        c = integ[(b, arm)]
        P(f"{b:>6} {arm:>10} {c['rows']:>6} {n*K:>7} {c['pass']:>6} {c['trunc']:>6} "
          f"{100*c['trunc']/max(c['rows'],1):>6.1f}% {c['infra']:>6}")
        if c["rows"] != n * K: bad.append(f"{b}/{arm}: {c['rows']} rows, expected {n*K}")
    if bad:
        P()
        for x in bad: P("  ROW-COUNT FAILURE: " + x)
        P("  PREREG section 10.3: missing generations are not tolerated. Analysis aborted.")
        sys.exit(2)
    P()
    P("--- PER-BLOCK SOLVED COUNTS (descriptive) ---")
    for r, sa, sb, dd in block_level(blocks, pids):
        P(f"  {r}: {ARM_A} {sa:3d}/{n}   {ARM_B} {sb:3d}/{n}   diff {100*dd:+6.2f} pp")
    P()
    D, V, df, R, d, bl = primary(blocks, pids)
    lo, hi, se, t = ci(D, V, df)
    P("=" * 94)
    P("PRIMARY RESULT — Estimand A, fixed 167-problem bank, problems solved at k=4")
    P("=" * 94)
    P(f"  point estimate      {100*D:+.2f} pp   ({ARM_A} minus {ARM_B}; positive = correct diagnostic better)")
    P(f"  primary 95% CI      [{100*lo:+.2f}, {100*hi:+.2f}] pp    width {100*(hi-lo):.2f} pp")
    P(f"    problem-clustered, block-averaged; SE {100*se:.2f} pp, t_{{0.975,{df}}} = {t:.3f}")
    lo90, hi90, _, t90 = ci(D, V, df, 0.90)
    P(f"  primary 90% CI      [{100*lo90:+.2f}, {100*hi90:+.2f}] pp   (used for the TOST below)")
    P()
    # equivalence
    DELTA = 0.05
    P(f"  PREREGISTERED EQUIVALENCE TEST (TOST), margin delta = +/-{100*DELTA:.0f} pp")
    inside = (lo90 > -DELTA) and (hi90 < DELTA)
    P(f"    90% CI [{100*lo90:+.2f}, {100*hi90:+.2f}] vs margin [{-100*DELTA:+.0f}, {100*DELTA:+.0f}]")
    if inside:
        P(f"    -> EQUIVALENCE ESTABLISHED at +/-{100*DELTA:.0f} pp. The correct diagnostic and a shuffled")
        P( "       diagnostic differ by less than the margin. This is NOT a claim that the effect is zero.")
    else:
        P(f"    -> EQUIVALENCE NOT ESTABLISHED at +/-{100*DELTA:.0f} pp.")
    P(f"    largest benefit still compatible with the data (95% upper limit): {100*hi:+.2f} pp"
      f"  = {(hi*n):+.1f} of {n} problems")
    P(f"    largest harm   still compatible with the data (95% lower limit): {100*lo:+.2f} pp"
      f"  = {(lo*n):+.1f} of {n} problems")
    # smallest margin that WOULD be established
    marg = max(abs(lo90), abs(hi90))
    P(f"    smallest margin this data establishes: +/-{100*marg:.2f} pp")
    P()
    # companion tests
    P("  COMPANION TESTS (secondary, unadjusted)")
    tot_a = sum(blocks[r][ARM_A][i] for r in bl for i in pids)
    tot_b = sum(blocks[r][ARM_B][i] for r in bl for i in pids)
    pa_i = {i: sum(blocks[r][ARM_A][i] for r in bl) for i in pids}
    pb_i = {i: sum(blocks[r][ARM_B][i] for r in bl) for i in pids}
    bpos = sum(pa_i[i] > pb_i[i] for i in pids); bneg = sum(pa_i[i] < pb_i[i] for i in pids)
    P(f"    pass counts summed over blocks: {ARM_A} {tot_a}, {ARM_B} {tot_b} (of {n*K*R} rollouts each)")
    P(f"    exact paired sign test on per-problem pass counts: {bpos}/{bneg}  p = {sign_test(bpos,bneg):.4g}")
    ca = sum(1 for i in pids if sum(d[i]) > 0); cb = sum(1 for i in pids if sum(d[i]) < 0)
    P(f"    per-problem discordance on the block-summed solved indicator: "
      f"{ARM_A}-favoured {ca}, {ARM_B}-favoured {cb}, tied {n-ca-cb}  p = {sign_test(ca,cb):.4g}")
    P()
    P("  DISCORDANCE PER BLOCK (how many problems flip, block by block)")
    for r in bl:
        pa = sum(1 for i in pids if blocks[r][ARM_A][i] > 0 and blocks[r][ARM_B][i] == 0)
        pb = sum(1 for i in pids if blocks[r][ARM_B][i] > 0 and blocks[r][ARM_A][i] == 0)
        P(f"    {r}: {ARM_A}-only {pa:3d}   {ARM_B}-only {pb:3d}   total discordant {pa+pb:3d}")
    P()
    P("--- SECONDARY / SENSITIVITY (all unadjusted; none replaces the primary) ---")
    bls = block_level(blocks, pids)
    if R >= 2:
        mb = sum(x[3] for x in bls) / R
        sd = math.sqrt(sum((x[3] - mb) ** 2 for x in bls) / (R - 1))
        tb = t_q(0.975, R - 1)
        P(f"  6. block-level t-interval (R-1={R-1} df): {100*mb:+.2f} pp "
          f"[{100*(mb-tb*sd/math.sqrt(R)):+.2f}, {100*(mb+tb*sd/math.sqrt(R)):+.2f}]")
    obs, blo, bhi = boot_population(blocks, pids)
    P(f"  7. population (problem-bootstrap) 95% CI: {100*obs:+.2f} pp [{100*blo:+.2f}, {100*bhi:+.2f}]")
    P( "     wider by construction: also treats the 167 problems as a sample of problems.")
    # estimand B
    da = [(pa_i[i] - pb_i[i]) / (K * R) for i in pids]
    mB = sum(da) / n
    vB = sum((x - mB) ** 2 for x in da) / (n - 1) / n
    loB, hiB, seB, _ = ci(mB, vB, n - 1)
    P(f"  5. ESTIMAND B, per-generation success (problem-clustered paired): {100*mB:+.2f} pp "
      f"[{100*loB:+.2f}, {100*hiB:+.2f}]  (DIFFERENT SCALE: per rollout, not per problem)")
    P(f"     raw rates: {ARM_A} {tot_a}/{n*K*R} = {100*tot_a/(n*K*R):.2f}%, "
      f"{ARM_B} {tot_b}/{n*K*R} = {100*tot_b/(n*K*R):.2f}%")
    # sensitivity 3/4: length + reuse strata
    sys.path.insert(0, HERE)
    import arms as _arms
    bankd = {}
    for l in open(bank):
        x = json.loads(l); bankd[x["id"]] = x
    shuf = {i: _arms.shuffled_error(i) for i in bankd}
    mult = collections.Counter(shuf[i] for i in pids)
    def stratum(mask, label):
        sub = [i for i in pids if mask(i)]
        if len(sub) < 10: P(f"     {label:52s} n={len(sub)} (too few)"); return
        ns = len(sub)
        m = sum(sum(d[i]) for i in sub) / (ns * R)
        if R >= 2:
            var = sum(sum((x - sum(d[i])/R) ** 2 for x in d[i]) / (R - 1) for i in sub) / (ns ** 2 * R)
            l2, h2, _, _ = ci(m, var, ns * (R - 1))
        else:   # single block: paired across-problem variance within the stratum
            var = sum((sum(d[i]) - m) ** 2 for i in sub) / (ns - 1) / ns
            l2, h2, _, _ = ci(m, var, ns - 1)
        P(f"     {label:52s} n={ns:3d}  {100*m:+6.2f} pp [{100*l2:+6.2f}, {100*h2:+6.2f}]")
    P("  3. length-mismatch strata (shuffled vs own diagnostic length):")
    stratum(lambda i: 0.25 <= len(shuf[i]) / max(len(bankd[i]["predecessor_error"]), 1) <= 4,
            "within 4x (well matched)")
    stratum(lambda i: not (0.25 <= len(shuf[i]) / max(len(bankd[i]["predecessor_error"]), 1) <= 4),
            "outside 4x (mismatched)")
    stratum(lambda i: len(shuf[i]) < 20000, "shuffled diagnostic < 20k chars (no monster)")
    P("  4. shuffle-reuse strata:")
    stratum(lambda i: mult[shuf[i]] < 5, "assigned a rarely-reused diagnostic (<5x)")
    stratum(lambda i: mult[shuf[i]] >= 5, "assigned a heavily-reused diagnostic (>=5x)")
    P()
    tot_infra = sum(integ[(b, a)]["infra"] for (b, a) in integ)
    P(f"  1. unresolved infra rows across all blocks/arms: {tot_infra}")
    P( "     PREREG 10.4: counted as FAIL in the primary. If 0, the sensitivity is vacuous.")
    tot_tr = sum(integ[(b, a)]["trunc"] for (b, a) in integ)
    P(f"  2. truncated rows (finish == 'length'): {tot_tr} "
      f"({100*tot_tr/sum(integ[(b,a)]['rows'] for (b,a) in integ):.2f}%); "
      f"reference at baseline_snapshot was ~1.5% in these arms")
    P()
    json.dump({"n": n, "R": R, "D_pp": 100*D, "ci95": [100*lo, 100*hi], "ci90": [100*lo90, 100*hi90],
               "se_pp": 100*se, "df": df, "equivalence_5pp": bool(inside),
               "smallest_margin_pp": 100*marg,
               "per_block": [{"block": r, ARM_A: sa, ARM_B: sb} for r, sa, sb, _ in bls],
               "estimand_B_pp": 100*mB, "estimand_B_ci95": [100*loB, 100*hiB],
               "infra_unresolved": tot_infra, "truncated": tot_tr},
              open(os.path.join(root, "RESULT_CONFIRM.json"), "w"), indent=2)
    P(f"machine-readable summary -> {os.path.join(root,'RESULT_CONFIRM.json')}")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "out_confirm_error"))
