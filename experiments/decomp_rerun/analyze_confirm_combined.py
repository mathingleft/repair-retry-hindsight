"""Preregistered COMBINED (old + new) estimate and heterogeneity table.

  PREREG: PREREG_CONFIRM_ERROR_CONTENT_2026-08-30.md sections 16 and 17

Committed before generation. Two batches, NEVER pooled at the rollout level:
  batch 1 = the baseline_snapshot block   (vLLM 0.28, seeds 22072601..604)   out/ + out_ext/
  batch 2 = the 4 new blocks    (vLLM 0.21, seeds 220730/1/2/3 01..04)  out_confirm_error/b1..b4

Fixed-effect inverse-variance meta-analysis of the two batch contrasts, with Cochran's Q / I^2 as a
heterogeneity (stack-effect) diagnostic. If the batches disagree, the NEW-ONLY estimate stays primary.

  python analyze_confirm_combined.py > out_confirm_error/ANALYSIS_COMBINED.txt
"""
import collections, json, math, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from analyze_confirm_error import (norm_q, t_q, load_blocks, pids_of, primary, ci, ARM_A, ARM_B, K)

Z95 = norm_q(0.975)

def batch_old(pids):
    """The baseline_snapshot block, assembled as a single k=4 block from the committed files."""
    npass = collections.defaultdict(collections.Counter)
    for d, arm in (("out", ARM_A), ("out_ext", ARM_B)):
        p = os.path.join(HERE, d, "verified.jsonl")
        for l in open(p):
            try: r = json.loads(l)
            except Exception: continue
            if r["arm"] == arm and r["status"] == "PASS": npass[arm][r["id"]] += 1
    return {"b0": npass}

def contrast(blocks, pids):
    D, V, df, R, d, bl = primary(blocks, pids)
    return D, V, df, R

def main():
    pids = pids_of(os.path.join(HERE, "bank.jsonl"))
    n = len(pids)
    P = print
    P("=" * 94)
    P("COMBINED OLD + NEW ESTIMATE — preregistered (PREREG section 16)")
    P("=" * 94)

    old = batch_old(pids)
    D1, V1, df1, R1 = contrast(old, pids)
    lo1, hi1, se1, _ = ci(D1, V1, df1)
    P(f"  batch 1 (baseline_snapshot, vLLM 0.28, R=1): {100*D1:+.2f} pp  SE {100*se1:.2f}  "
      f"95% CI [{100*lo1:+.2f}, {100*hi1:+.2f}]")

    newroot = os.path.join(HERE, "out_confirm_error")
    new, _ = load_blocks(newroot)
    if not new:
        P("\n  batch 2 not present yet (out_confirm_error/b*/verified.jsonl missing).")
        P("  Run this only after the confirmatory run is fully verified.")
        return
    D2, V2, df2, R2 = contrast(new, pids)
    lo2, hi2, se2, _ = ci(D2, V2, df2)
    P(f"  batch 2 (new, vLLM 0.21, R={R2}):     {100*D2:+.2f} pp  SE {100*se2:.2f}  "
      f"95% CI [{100*lo2:+.2f}, {100*hi2:+.2f}]   <- PRIMARY")

    # fixed-effect inverse-variance
    w1, w2 = 1 / V1, 1 / V2
    Dc = (w1 * D1 + w2 * D2) / (w1 + w2)
    Vc = 1 / (w1 + w2)
    sec = math.sqrt(Vc)
    P()
    P(f"  fixed-effect inverse-variance combination (weights {w1/(w1+w2):.3f} / {w2/(w1+w2):.3f}):")
    P(f"    {100*Dc:+.2f} pp  SE {100*sec:.2f}  95% CI [{100*(Dc-Z95*sec):+.2f}, {100*(Dc+Z95*sec):+.2f}]"
      f"  width {200*Z95*sec:.2f} pp")
    # Cochran's Q on 2 batches, 1 df
    Q = w1 * (D1 - Dc) ** 2 + w2 * (D2 - Dc) ** 2
    I2 = max(0.0, (Q - 1) / Q) if Q > 0 else 0.0
    # p for chi2 with 1 df = 2*(1-Phi(sqrt(Q)))
    pQ = 2 * (1 - 0.5 * (1 + math.erf(math.sqrt(Q) / math.sqrt(2)))) if Q >= 0 else float("nan")
    P(f"    heterogeneity: Cochran Q = {Q:.3f} (1 df), p = {pQ:.3f}, I^2 = {100*I2:.1f}%")
    dif = D1 - D2; sedif = math.sqrt(V1 + V2)
    P(f"    batch difference (old - new): {100*dif:+.2f} pp  95% CI "
      f"[{100*(dif-Z95*sedif):+.2f}, {100*(dif+Z95*sedif):+.2f}]  (PREREG 15.8, the stack-effect check)")
    if pQ < 0.05:
        P("    -> batches DISAGREE beyond chance. Reporting the combined estimate, but the NEW-ONLY")
        P("       estimate remains primary and the discrepancy is reported as a stack effect.")
    else:
        P("    -> no evidence of a batch/stack effect; the combined estimate is reportable as the")
        P("       prespecified secondary.")
    P()
    P("  Raw rollouts were NOT pooled across stacks; each batch's SE is computed within batch.")

    P()
    P("=" * 94)
    P("HETEROGENEITY ACROSS MODEL AND INSTRUCTION REGIME (descriptive; NEVER averaged)")
    P("=" * 94)
    def cell(dirs, a, b, bankpath, label):
        pd = pids_of(bankpath)
        cnt = {a: collections.Counter(), b: collections.Counter()}
        rows = collections.Counter()
        for d in dirs:
            p = os.path.join(HERE, d, "verified.jsonl")
            if not os.path.exists(p): return
            for l in open(p):
                try: r = json.loads(l)
                except Exception: continue
                if r["arm"] in cnt:
                    rows[r["arm"]] += 1
                    if r["status"] == "PASS": cnt[r["arm"]][r["id"]] += 1
        sa = sum(1 for i in pd if cnt[a][i] > 0); sb = sum(1 for i in pd if cnt[b][i] > 0)
        dd = [(1 if cnt[a][i] > 0 else 0) - (1 if cnt[b][i] > 0 else 0) for i in pd]
        m = sum(dd) / len(pd)
        v = sum((x - m) ** 2 for x in dd) / (len(pd) - 1) / len(pd)
        s = math.sqrt(v)
        P(f"  {label:44s} n={len(pd):3d}  {sa:3d} vs {sb:3d}  {100*m:+6.2f} pp "
          f"[{100*(m-Z95*s):+6.2f}, {100*(m+Z95*s):+6.2f}]")
    BANK = os.path.join(HERE, "bank.jsonl"); BANK2 = os.path.join(HERE, "out_m2", "bank2.jsonl")
    P("  (correct minus shuffled; bank.jsonl = 167 problems, bank2.jsonl = 137 DIFFERENT problems)")
    cell(["out", "out_ext"], "F_E", "F_E_shuf", BANK, "Qwen3.6-27B  instruction     [bank 1]")
    cell(["out_noi"], "F_E_noI", "F_E_shuf_noI", BANK, "Qwen3.6-27B  no instruction  [bank 1]")
    cell(["out_m2", "out_m2_ext"], "F_E", "F_E_shuf", BANK2, "DeepSeek-Prover-V2-7B instruction [bank 2]")
    cell(["out_m2_noi"], "F_E_noI", "F_E_shuf_noI", BANK2, "DeepSeek-Prover-V2-7B no instr    [bank 2]")
    P("  The two banks are NOT interchangeable: bank2 was built from the prover's OWN failures")
    P("  (build_bank2.py) and has 137 problems. Their percentage points are not commensurable and")
    P("  no pooled average across banks is reported.")
    P()
    P("=" * 94)
    P("PRESENCE vs CONTENT (PREREG section 18) — reported separately, never merged")
    P("=" * 94)
    cell(["out"], "F_E", "F", BANK, "PRESENCE: F_E vs F (error block vs none)")
    cell(["out", "out_ext"], "F_E", "F_E_shuf", BANK, "CONTENT:  F_E vs F_E_shuf (baseline_snapshot block)")
    P("  The presence contrast confounds the diagnostic's information with its presence, formatting,")
    P("  length and token count. Only the content contrast isolates problem-specific correctness.")

if __name__ == "__main__":
    main()
