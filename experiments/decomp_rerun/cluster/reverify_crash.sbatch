#!/bin/bash
#SBATCH -p normal -c 8 --mem=32G -t 1:00:00
#SBATCH -J decomp-reverify-crash -o logs/reverify-crash-%j.out
: "${VERICODE_SCRATCH:?set VERICODE_SCRATCH to the directory holding models, environments and caches (ENVIRONMENT.md)}"
# Re-verify only "verifier crash" rows (REPL died mid-proof -> FAIL by default). Removes them from
# OUTDIR/verified.jsonl (backup kept), reruns verify_run.py (which redoes exactly the missing keys), and
# re-checks the row-count guard. Generic: OUTDIR=out_m2_ext2 BANK=out_m2/bank2.jsonl ANALYZE="python ... | tee ..."
#   sbatch --export=ALL,OUTDIR=...,BANK=...,ANALYZE=... cluster/reverify_crash.sbatch
set -euo pipefail
: "${OUTDIR:?}"; BANK=${BANK:-bank.jsonl}
export LEAN_SERVER_LEAN_VERSION=v4.29.0 LEAN_SERVER_MAX_REPLS=${LEAN_SERVER_MAX_REPLS:-4} LEAN_SERVER_PORT=8072
source ${VERICODE_SCRATCH}/kimina-env/bin/activate
export HOME=${VERICODE_SCRATCH}/fakehome
export PATH=${VERICODE_SCRATCH}/fakehome/.elan/bin:$PATH
[ -x ${VERICODE_SCRATCH}/fakehome/.elan/toolchains/leanprover--lean4---v4.29.0/bin/lake ] || { echo "TOOLCHAIN BROKEN"; exit 1; }
cd ${VERICODE_SCRATCH}/kimina-lean-server
python -m server &
SRV=$!
cd "$SLURM_SUBMIT_DIR"
N=$(wc -l < "$OUTDIR/verified.jsonl")
cp "$OUTDIR/verified.jsonl" "$OUTDIR/verified.jsonl.bak-crash-$SLURM_JOB_ID"
grep -c "verifier crash" "$OUTDIR/verified.jsonl" || { echo "no crash rows; nothing to do"; kill $SRV; exit 0; }
grep -v "verifier crash" "$OUTDIR/verified.jsonl" > "$OUTDIR/verified.jsonl.tmp" && mv "$OUTDIR/verified.jsonl.tmp" "$OUTDIR/verified.jsonl"
for i in $(seq 120); do curl -sf localhost:8072/health >/dev/null && break; sleep 5; done
curl -sf localhost:8072/health >/dev/null || { echo "KIMINA NEVER CAME UP"; exit 1; }
python - <<'PY'
import requests
code = "import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 400000\n\nopen BigOperators Real Nat Topology Rat\n\nexample : (2:ℕ) + 2 = 4 := by norm_num\n"
r = requests.post("http://localhost:8072/verify", json={"codes": [{"custom_id": "probe", "proof": code}], "timeout": 300, "infotree_type": None}, timeout=400)
r.raise_for_status(); it = r.json()["results"][0]
errs = [m for m in ((it.get("response") or {}).get("messages") or []) if m.get("severity") == "error"]
assert not it.get("error") and not errs, it
print("kimina probe PASS")
PY
python verify_run.py --kimina-url http://localhost:8072 --outdir "$OUTDIR" --bank "$BANK"
M=$(wc -l < "$OUTDIR/verified.jsonl"); R=$(grep -c "verifier crash" "$OUTDIR/verified.jsonl" || true)
echo "rows $N -> $M, remaining crash rows: $R"; [ "$M" = "$N" ] || { echo "GUARD FAILED: row count changed"; exit 1; }
[ -n "${ANALYZE:-}" ] && eval "$ANALYZE"
kill $SRV
