"""R20: re-verify every retained PASS row in Lean, report-only.

Differs from experiments/decomp_rerun/reverify_pass.py in two ways:
  1. It handles BOTH retained row schemas. The adaptive/one-step files use
     (status, lean_code|code); the memoryless-I depth files use
     (verification_result, extracted_proof). The original sorry-screen
     re-verification only matched the first schema, so the memoryless
     baseline arms were never re-verified.
  2. It never rewrites the frozen data. Every verdict goes to a sidecar
     JSONL log; flips are reported, not applied.

Usage:
  python r20_reverify.py --kimina-url http://127.0.0.1:8123 \
      --files-from r20_files.txt --log out/r20_log.jsonl --workers 16
"""
import argparse
import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from common import ARTIFACT, DATA
DECOMP = str(DATA)
sys.path.insert(0, DECOMP)
from verify_run import verify_codes, verdict, SORRY_RE  # noqa: E402

STATIC_SORRY = re.compile(r"(?<![A-Za-z0-9_'.])(sorry|admit)(?![A-Za-z0-9_'])")


def rel(path):
    """Name a record file relative to the artifact root, so the log is portable."""
    try:
        return Path(path).resolve().relative_to(ARTIFACT).as_posix()
    except ValueError:
        return str(path)


def status_of(r):
    return r.get("status") if "status" in r else r.get("verification_result")


def code_of(r):
    return r.get("lean_code") or r.get("code") or r.get("extracted_proof")


def key_of(r):
    return [r.get("id", r.get("problem_id")), r.get("arm"),
            r.get("sample_idx", r.get("traj", r.get("trajectory_id"))),
            r.get("attempt", r.get("attempt_id"))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kimina-url", required=True)
    ap.add_argument("--files-from", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()

    files = [l.strip() for l in open(a.files_from) if l.strip()]
    os.makedirs(os.path.dirname(a.log), exist_ok=True)

    # Resume support: skip rows already logged.
    done = set()
    if os.path.exists(a.log):
        for line in open(a.log):
            if line.strip():
                rec = json.loads(line)
                done.add((rec["file"], json.dumps(rec["key"])))

    lock = threading.Lock()
    totals = {"checked": 0, "flips": 0, "crash": 0, "skipped": len(done)}

    for path in files:
        rows = [json.loads(l) for l in open(path) if l.strip()]
        todo = [(i, r) for i, r in enumerate(rows) if status_of(r) == "PASS"
                and (rel(path), json.dumps(key_of(r))) not in done]
        print(f"{path}: {len(rows)} rows, {len(todo)} PASS rows to re-verify", flush=True)
        if not todo:
            continue
        flips = 0

        def work(item):
            nonlocal flips
            i, r = item
            code = code_of(r)
            cid = str(i)
            res = verify_codes(a, [{"custom_id": cid, "proof": code}])
            it = res.get(cid, {"error": "no result returned"})
            st, err = verdict(it)
            msgs = (it.get("response") or {}).get("messages") or it.get("messages") or []
            warn = [m.get("data") for m in msgs if SORRY_RE.search(m.get("data") or "")]
            crash = ("verifier crash" in str(it.get("error") or "")
                     or "no result returned" in str(it.get("error") or ""))
            rec = {"file": rel(path), "key": key_of(r), "schema": "status" if "status" in r else "verification_result",
                   "old": "PASS", "new": st, "error": err, "sorry_warnings": warn,
                   "crash": crash, "static_token": bool(STATIC_SORRY.search(code or ""))}
            with lock:
                with open(a.log, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                totals["checked"] += 1
                if crash:
                    totals["crash"] += 1
                elif st != "PASS":
                    totals["flips"] += 1
                    flips += 1

        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            for n, _ in enumerate(ex.map(work, todo), 1):
                if n % 100 == 0 or n == len(todo):
                    print(f"  {n}/{len(todo)} (flips so far in file: {flips})", flush=True)
        print(f"  -> {flips} PASS rows would flip", flush=True)

    print(json.dumps(totals, indent=1), flush=True)


if __name__ == "__main__":
    main()
