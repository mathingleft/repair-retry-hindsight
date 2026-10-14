#!/bin/bash
#SBATCH -p normal -c 8 --mem=24G -t 0:20:00
#SBATCH -J probe-kimina -o logs/probe-%j.out
: "${VERICODE_SCRATCH:?set VERICODE_SCRATCH to the directory holding models, environments and caches (ENVIRONMENT.md)}"
# Smoke test: kimina server from SCRATCH with HOME redirected (works on nodes without /home).
set -uo pipefail
export HOME=${VERICODE_SCRATCH}/fakehome
export LEAN_SERVER_LEAN_VERSION=v4.29.0 LEAN_SERVER_MAX_REPLS=4
hostname; ls -d ${HOME} 2>&1 | head -1
source ${VERICODE_SCRATCH}/kimina-env/bin/activate
cd ${VERICODE_SCRATCH}/kimina-lean-server && python -m server > "$SLURM_SUBMIT_DIR/logs/kimina-probe-$SLURM_JOB_ID.log" 2>&1 &
KS=$!
for i in $(seq 120); do curl -sf localhost:8000/health >/dev/null && break; sleep 5; done
curl -sf localhost:8000/health || { echo "KIMINA NEVER CAME UP"; tail -20 "$SLURM_SUBMIT_DIR/logs/kimina-probe-$SLURM_JOB_ID.log"; kill $KS; exit 1; }
echo; echo "--- check (expect PASS then FAIL):"
python - <<'PY'
import requests, json, time
good = "import Mathlib\n\ntheorem t (a b : ℕ) : a + b = b + a := by\n  omega\n"
bad  = "import Mathlib\n\ntheorem t (a b : ℕ) : a + b = b + a := by\n  simp only []\n"
t=time.time()
r = requests.post("http://localhost:8000/verify", json={"codes":[{"custom_id":"good","proof":good},{"custom_id":"bad","proof":bad}],"timeout":120,"infotree_type":None}, timeout=600).json()
print(f"{time.time()-t:.0f}s")
for res in r.get("results", []):
    msgs = (res.get("response") or {}).get("messages", [])
    errs = [m for m in msgs if m.get("severity")=="error"]
    print(res.get("custom_id"), "error" if res.get("error") else "", "PASS" if not errs and not res.get("error") else "FAIL", [ (m.get("data") or "")[:80] for m in errs[:2]])
PY
kill $KS
