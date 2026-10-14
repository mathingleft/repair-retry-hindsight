"""Re-verify every PASS row with the FIXED soundness screen (SORRY_RE) and patch statuses in place.

Background (HANDOFF 01:35 PT 2026-08-29): the screen matched "declaration uses 'sorry'" while Lean 4.29 emits
"declaration uses `sorry`", so proofs containing sorry/admit were recorded PASS. Only PASS rows can change (the fix
adds a screen; it cannot turn an error into a pass), so only PASS rows are re-run through Lean.

    python reverify_pass.py --kimina-url http://localhost:8050 --files out/verified.jsonl out/depth_f.jsonl ... [--apply]

Writes a sidecar log (out/sorry_reverify.jsonl: one line per PASS row with old/new verdict and the messages) and, with
--apply, rewrites each file with flipped rows set to FAIL (error = the Lean warning text, so a resumed depth trajectory
sees what the compiler said) after copying the original to <file>.bak-sorrybug. Idempotent: rows already FAIL are skipped.
"""
import argparse, json, os, shutil, sys, threading
from concurrent.futures import ThreadPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import verify_run
from verify_run import verify_codes, verdict, SORRY_RE

def key(r):
    return (r["id"], r["arm"], r.get("sample_idx", r.get("traj")), r.get("attempt"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kimina-url", required=True); ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--apply", action="store_true"); ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("LEAN_SERVER_MAX_REPLS", 8)))
    ap.add_argument("--log", default=os.path.join(HERE, "out", "sorry_reverify.jsonl"))
    a = ap.parse_args()
    lock = threading.Lock(); summary = {}
    for path in a.files:
        rows = [json.loads(l) for l in open(path) if l.strip()]
        todo = [(i, r) for i, r in enumerate(rows) if r.get("status") == "PASS"]
        print(f"{path}: {len(rows)} rows, {len(todo)} PASS rows to re-verify", flush=True)
        flips = {}
        def work(item):
            i, r = item
            code = r.get("lean_code") or r.get("code")
            cid = f"{i}"
            res = verify_codes(a, [{"custom_id": cid, "proof": code}])
            it = res.get(cid, {"error": "no result returned"})
            st, err = verdict(it)
            msgs = (it.get("response") or {}).get("messages") or it.get("messages") or []
            warn = [m.get("data") for m in msgs if SORRY_RE.search(m.get("data") or "")]
            crash = "verifier crash" in str(it.get("error") or "") or "no result returned" in str(it.get("error") or "")
            rec = {"file": path, "key": key(r), "old": "PASS", "new": st, "error": err, "sorry_warnings": warn, "crash": crash,
                   "static_token": bool(__import__("re").search(r"(?<![A-Za-z0-9_'.])(sorry|admit)(?![A-Za-z0-9_'])", code or ""))}
            with lock:
                with open(a.log, "a") as f: f.write(json.dumps(rec) + "\n")
                if st == "FAIL" and not crash: flips[i] = "declaration uses `sorry`" if warn else (err or "FAIL on re-verification")
                elif crash: summary.setdefault("crash", []).append((path, key(r)))
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            for n, _ in enumerate(ex.map(work, todo), 1):
                if n % 100 == 0 or n == len(todo): print(f"  {n}/{len(todo)}", flush=True)
        summary[path] = (len(todo), len(flips))
        print(f"  -> {len(flips)} PASS rows flip to FAIL", flush=True)
        if a.apply and flips:
            bak = path + ".bak-sorrybug"
            if not os.path.exists(bak): shutil.copy2(path, bak)
            for i, err in flips.items():
                rows[i]["status"] = "FAIL"; rows[i]["error"] = err; rows[i]["sorry_screen_fix"] = "2026-08-29"
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                for r in rows: f.write(json.dumps(r) + "\n")
            os.replace(tmp, path); print(f"  patched {path} (backup {bak})", flush=True)
    print("SUMMARY", json.dumps({k: v for k, v in summary.items()}, default=str))
    if summary.get("crash"): print(f"WARNING: {len(summary['crash'])} rows crashed on re-verification and were left PASS; re-run to settle them")

if __name__ == "__main__":
    main()
