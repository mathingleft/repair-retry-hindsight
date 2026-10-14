"""Build a second model's frozen-predecessor bank from its own verified `bare` rollouts.

    python build_bank2.py <outdir> --model <name> [--bank bank.jsonl] [--k 4]

For every non-holdout problem in the main bank: take the model's k `bare` samples (verified), pick the
first FAILING one with a real proof body as the frozen predecessor (fallback: any failing sample);
its spliced file is predecessor_code and kimina's message is predecessor_error, exactly the main bank's
shape. Problems the model solves cold in all k samples, or whose failures all lack a parseable proof body,
have no real failing predecessor and are dropped.
Writes <outdir>/bank2.jsonl. Refuses to run unless all k*len(bank) bare rows are verified.
"""
import argparse, collections, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
DEGENERATE = ("no code extracted", "no proof body parsed")

def bucket(err):
    e = (err or "")
    el = e.lower()
    if "unknown identifier" in el or "unknown constant" in el: return "unknown_ident"
    if "unexpected token" in el or "expected command" in el or "expected term" in el: return "parse"
    if "unsolved goals" in el: return "unsolved_goals"
    if "no goals" in el: return "no_goals"
    if "type mismatch" in el or "has type" in el: return "type"
    if any(t in el for t in ("failed", "tactic", "linarith", "omega", "simp", "rewrite", "motive", "not defined", "error")): return "tactic"
    return "other"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir"); ap.add_argument("--model", required=True)
    ap.add_argument("--bank", default=os.path.join(HERE, "bank.jsonl")); ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    bank = [json.loads(l) for l in open(a.bank) if l.strip()]
    bank = [b for b in bank if not b["is_framing_holdout"]]
    rows = collections.defaultdict(list)
    for l in open(os.path.join(a.outdir, "verified.jsonl")):
        try: d = json.loads(l)
        except Exception: continue
        if d["arm"] == "bare": rows[d["id"]].append(d)
    n = sum(len(rows[b["id"]]) for b in bank)
    if n != a.k * len(bank):
        print(f"bare not fully verified: {n} rows, need {a.k*len(bank)}"); sys.exit(2)
    out = a.out or os.path.join(a.outdir, "bank2.jsonl")
    kept, dropped, degenerate = [], 0, 0
    for b in bank:
        R = sorted(rows[b["id"]], key=lambda r: r["sample_idx"])
        fails = [r for r in R if r["status"] != "PASS"]
        if not fails: dropped += 1; continue
        real = [r for r in fails if not any((r.get("error") or "").startswith(t) for t in DEGENERATE)]
        if not real: degenerate += 1; continue      # no real failing proof to show: drop (documented)
        pred = real[0]
        kept.append({"id": b["id"], "statement": b["statement"], "predecessor_code": pred["lean_code"],
                     "predecessor_error": pred.get("error") or "", "predecessor_rollout_idx": pred["sample_idx"],
                     "bucket": bucket(pred.get("error")), "base_pass_rate_k4": sum(r["status"] == "PASS" for r in R) / len(R),
                     "is_framing_holdout": False, "source_model": a.model, "main_bank_bucket": b["bucket"]})
    with open(out, "w") as f:
        for r in kept: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    bc = collections.Counter(r["bucket"] for r in kept)
    print(f"bank2: {len(kept)} problems kept, {dropped} dropped (solved cold {a.k}/{a.k}), {degenerate} dropped (only degenerate failures, no proof body); "
          f"base-zero {sum(r['base_pass_rate_k4']==0 for r in kept)}; buckets {dict(bc)} -> {out}")

if __name__ == "__main__":
    main()
