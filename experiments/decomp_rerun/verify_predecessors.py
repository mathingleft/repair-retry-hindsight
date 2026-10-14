"""Re-verify every frozen predecessor in the bank on the CURRENT stack.
Closes the provenance question for the paper: 'each frozen failing attempt fails under the exact
verifier used for all experiments.' Expected result: 0 PASS. Any PASS is a red flag — report it.

Usage (kimina-lean-server running, same as for reverify_pass.py):
    python verify_predecessors.py --kimina-url http://localhost:8050
    # optional: --bank out_m2/bank2.jsonl  (prover bank; its predecessors are current-run rows already,
    #           so this is a sanity pass, not a gap-closer)

Writes out/PREDECESSOR_REVERIFY.txt (summary) and out/predecessor_reverify.jsonl (one line per problem).
Run from experiments/decomp_rerun/.
"""
import argparse, json, os, sys, threading
from concurrent.futures import ThreadPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from verify_run import verify_codes, verdict
from lean_extract import extract_lean_code, get_lean_preamble

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kimina-url", required=True)
    ap.add_argument("--bank", default=os.path.join(HERE, "bank.jsonl"))
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("LEAN_SERVER_MAX_REPLS", 8)))
    ap.add_argument("--include-holdouts", action="store_true")
    a = ap.parse_args()
    bank = [json.loads(l) for l in open(a.bank) if l.strip()]
    if not a.include_holdouts:
        bank = [b for b in bank if not b.get("is_framing_holdout")]
    log = os.path.join(HERE, "out", "predecessor_reverify.jsonl")
    txt = os.path.join(HERE, "out", "PREDECESSOR_REVERIFY.txt")
    lock = threading.Lock(); results = {}

    def proof_file(b):
        code = b["predecessor_code"] or ""
        # rows' lean_code is already the spliced full file; older banks may store the body only
        if "import Mathlib" in code: return code
        spliced = extract_lean_code(code, b["statement"])
        return spliced if "import Mathlib" in spliced else get_lean_preamble() + spliced

    def work(b):
        pid = b["id"]
        res = verify_codes(a, [{"custom_id": pid, "proof": proof_file(b)}])
        item = res.get(pid, {"error": "no result returned"})
        st, err = verdict(item)
        rec = {"id": pid, "bucket": b.get("bucket"), "status": st, "error": (err or "")[:300]}
        with lock:
            results[pid] = st
            with open(log, "a") as f: f.write(json.dumps(rec) + "\n")

    open(log, "w").close()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for n, _ in enumerate(ex.map(work, bank), 1):
            if n % 25 == 0 or n == len(bank): print(f"  {n}/{len(bank)}", flush=True)
    n_pass = sum(1 for s in results.values() if s == "PASS")
    n_fail = sum(1 for s in results.values() if s == "FAIL")
    lines = [f"bank={a.bank}  problems={len(bank)}  FAIL={n_fail}  PASS={n_pass}",
             "PASS problems (should be none): " + ", ".join(sorted(p for p, s in results.items() if s == "PASS"))]
    print("\n".join(lines))
    with open(txt, "w") as f: f.write("\n".join(lines) + "\n")

if __name__ == "__main__":
    main()
