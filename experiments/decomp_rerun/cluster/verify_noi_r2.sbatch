#!/bin/bash
#SBATCH -p normal -c 16 --mem=48G -t 6:00:00
#SBATCH -J decomp-verify-noi-r2 -o logs/verify-noi-r2-%j.out
: "${VERICODE_SCRATCH:?set VERICODE_SCRATCH to the directory holding models, environments and caches (ENVIRONMENT.md)}"
# Streaming verify for out_noi_r2/ (no-instruction arms), then guarded final pass + analyze_noi.py.
# Own kimina on port 8020 so it can share a node with the main verify-stream's kimina (8000).
#   sbatch --export=ALL,GEN_JOB=<ext gen jobid> -d after:<ext gen jobid> cluster/verify_ext.sbatch
set -euo pipefail
: "${GEN_JOB:?set GEN_JOB=<generation job id>}"
export LEAN_SERVER_LEAN_VERSION=v4.29.0 LEAN_SERVER_MAX_REPLS=${LEAN_SERVER_MAX_REPLS:-8} LEAN_SERVER_PORT=8080
source ${VERICODE_SCRATCH}/kimina-env/bin/activate
export HOME=${VERICODE_SCRATCH}/fakehome
export PATH=${VERICODE_SCRATCH}/fakehome/.elan/bin:$PATH
[ -x ${VERICODE_SCRATCH}/fakehome/.elan/toolchains/leanprover--lean4---v4.29.0/bin/lake ] || { echo "TOOLCHAIN BROKEN"; exit 1; }
cd ${VERICODE_SCRATCH}/kimina-lean-server
python -m server &
SRV=$!
cd "$SLURM_SUBMIT_DIR"
EXPECT=668
for i in $(seq 120); do curl -sf localhost:8080/health >/dev/null && break; sleep 5; done
curl -sf localhost:8080/health >/dev/null || { echo "KIMINA NEVER CAME UP"; exit 1; }
# functional probe: /health passes even when REPLs cannot spawn
python - <<'PY'
import requests
code = "import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 400000\n\nopen BigOperators Real Nat Topology Rat\n\nexample : (2:ℕ) + 2 = 4 := by norm_num\n"
r = requests.post("http://localhost:8080/verify", json={"codes": [{"custom_id": "probe", "proof": code}], "timeout": 300, "infotree_type": None}, timeout=400)
r.raise_for_status(); it = r.json()["results"][0]
errs = [m for m in ((it.get("response") or {}).get("messages") or []) if m.get("severity") == "error"]
assert not it.get("error") and not errs, it
print("kimina probe PASS")
PY
mkdir -p out_noi_r2; touch out_noi_r2/rollouts.jsonl
while true; do
  python verify_run.py --kimina-url http://localhost:8080 --outdir out_noi_r2 || echo "verify pass failed; will retry"
  if [ -z "$(squeue -h -j "$GEN_JOB" 2>/dev/null)" ]; then echo "generation job $GEN_JOB finished"; break; fi
  sleep 120
done
n=$(wc -l < out_noi_r2/rollouts.jsonl); e=$(grep -c '"finish": "error:' out_noi_r2/rollouts.jsonl || true)
d=$(python -c 'import json; k=[(r["id"],r["arm"],r["sample_idx"]) for r in map(json.loads, open("out_noi_r2/rollouts.jsonl"))]; print(len(k)-len(set(k)))')
echo "FINAL rollouts=$n errors=$e dupes=$d"; [ "$n" = "$EXPECT" ] && [ "$e" = 0 ] && [ "$d" = 0 ] || { echo "GUARD FAILED: generation incomplete"; exit 1; }
python verify_run.py --kimina-url http://localhost:8080 --outdir out_noi_r2
v=$(wc -l < out_noi_r2/verified.jsonl); echo "verified=$v"; [ "$v" = "$EXPECT" ] || { echo "GUARD FAILED: verified != expected"; exit 1; }
python analyze_noi_r2.py | tee out_noi_r2/ANALYSIS_NOI_R2.txt
kill $SRV
