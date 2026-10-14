#!/bin/bash
#SBATCH -p normal -c 16 --mem=48G -t 4:00:00
#SBATCH -J decomp-verify-pred -o logs/verify-pred-%j.out
: "${VERICODE_SCRATCH:?set VERICODE_SCRATCH to the directory holding models, environments and caches (ENVIRONMENT.md)}"
# Re-verify every frozen predecessor in the bank on the current stack (verify_predecessors.py).
#   sbatch cluster/verify_predecessors.sbatch            # main bank
#   sbatch --export=ALL,BANK=out_m2/bank2.jsonl cluster/verify_predecessors.sbatch
# Runs its own kimina on $PORT (same recipe as reverify_pass.sbatch).
set -euo pipefail
PORT=${PORT:-8050}
export LEAN_SERVER_LEAN_VERSION=v4.29.0 LEAN_SERVER_MAX_REPLS=${LEAN_SERVER_MAX_REPLS:-8} LEAN_SERVER_PORT=$PORT
source ${VERICODE_SCRATCH}/kimina-env/bin/activate
export HOME=${VERICODE_SCRATCH}/fakehome
export PATH=${VERICODE_SCRATCH}/fakehome/.elan/bin:$PATH
[ -x ${VERICODE_SCRATCH}/fakehome/.elan/toolchains/leanprover--lean4---v4.29.0/bin/lake ] || { echo "TOOLCHAIN BROKEN"; exit 1; }
cd ${VERICODE_SCRATCH}/kimina-lean-server
python -m server &
SRV=$!
cd "$SLURM_SUBMIT_DIR"
for i in $(seq 120); do curl -sf localhost:$PORT/health >/dev/null && break; sleep 5; done
curl -sf localhost:$PORT/health >/dev/null || { echo "KIMINA NEVER CAME UP"; exit 1; }
python - <<PY
import requests
code = "import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 400000\n\nopen BigOperators Real Nat Topology Rat\n\n"
r = requests.post("http://localhost:$PORT/verify", json={"codes": [{"custom_id": "ok", "proof": code + "example : (2:ℕ) + 2 = 4 := by norm_num\n"}, {"custom_id": "sorry", "proof": code + "example : (2:ℕ) + 2 = 4 := by sorry\n"}], "timeout": 300, "infotree_type": None}, timeout=400)
r.raise_for_status(); res = {x["custom_id"]: x for x in r.json()["results"]}
import verify_run
assert verify_run.verdict(res["ok"])[0] == "PASS", res["ok"]
assert verify_run.verdict(res["sorry"])[0] == "FAIL", res["sorry"]
print("kimina probe PASS; sorry screen FIRES:", verify_run.verdict(res["sorry"]))
PY
python verify_predecessors.py --kimina-url http://localhost:$PORT ${BANK:+--bank "$BANK"}
echo "VERIFY-PRED DONE"
kill $SRV
