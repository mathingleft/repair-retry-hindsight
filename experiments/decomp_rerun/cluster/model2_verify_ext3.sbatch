#!/bin/bash
#SBATCH -p normal -c 32 --mem=96G -t 6:00:00
#SBATCH -J decomp-verify-m2ext3 -o logs/verify-m2ext3-%j.out
: "${VERICODE_SCRATCH:?set VERICODE_SCRATCH to the directory holding models, environments and caches (ENVIRONMENT.md)}"
# Streaming verify for out_m2_ext3/ (prover EXT3: F_C_other, F_E_match, F_CE_match), then guarded final pass +
# analyze_ext3.py on the prover dirs. Statements from bank2. 16 REPLs. Own kimina on port 8082.
#   sbatch --export=ALL,GEN_JOB=<m2 ext3 gen jobid> -d after:<m2 ext3 gen jobid> cluster/model2_verify_ext3.sbatch
set -euo pipefail
: "${GEN_JOB:?set GEN_JOB=<generation job id>}"
export LEAN_SERVER_LEAN_VERSION=v4.29.0 LEAN_SERVER_MAX_REPLS=${LEAN_SERVER_MAX_REPLS:-16} LEAN_SERVER_PORT=8082
source ${VERICODE_SCRATCH}/kimina-env/bin/activate
export HOME=${VERICODE_SCRATCH}/fakehome
export PATH=${VERICODE_SCRATCH}/fakehome/.elan/bin:$PATH
[ -x ${VERICODE_SCRATCH}/fakehome/.elan/toolchains/leanprover--lean4---v4.29.0/bin/lake ] || { echo "TOOLCHAIN BROKEN"; exit 1; }
cd ${VERICODE_SCRATCH}/kimina-lean-server
python -m server &
SRV=$!
cd "$SLURM_SUBMIT_DIR"
EXPECT=$(python -c 'print(3*4*sum(1 for _ in open("out_m2/bank2.jsonl")))')
for i in $(seq 120); do curl -sf localhost:8082/health >/dev/null && break; sleep 5; done
curl -sf localhost:8082/health >/dev/null || { echo "KIMINA NEVER CAME UP"; exit 1; }
# functional probe: /health passes even when REPLs cannot spawn
python - <<'PY'
import requests
code = "import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 400000\n\nopen BigOperators Real Nat Topology Rat\n\nexample : (2:ℕ) + 2 = 4 := by norm_num\n"
r = requests.post("http://localhost:8082/verify", json={"codes": [{"custom_id": "probe", "proof": code}], "timeout": 300, "infotree_type": None}, timeout=400)
r.raise_for_status(); it = r.json()["results"][0]
errs = [m for m in ((it.get("response") or {}).get("messages") or []) if m.get("severity") == "error"]
assert not it.get("error") and not errs, it
print("kimina probe PASS")
PY
mkdir -p out_m2_ext3; touch out_m2_ext3/rollouts.jsonl
while true; do
  python verify_run.py --kimina-url http://localhost:8082 --outdir out_m2_ext3 --bank out_m2/bank2.jsonl || echo "verify pass failed; will retry"
  if [ -z "$(squeue -h -j "$GEN_JOB" 2>/dev/null)" ]; then echo "generation job $GEN_JOB finished"; break; fi
  sleep 120
done
n=$(wc -l < out_m2_ext3/rollouts.jsonl); e=$(grep -c '"finish": "error:' out_m2_ext3/rollouts.jsonl || true)
d=$(python -c 'import json; k=[(r["id"],r["arm"],r["sample_idx"]) for r in map(json.loads, open("out_m2_ext3/rollouts.jsonl"))]; print(len(k)-len(set(k)))')
echo "FINAL rollouts=$n errors=$e dupes=$d"; [ "$n" = "$EXPECT" ] && [ "$e" = 0 ] && [ "$d" = 0 ] || { echo "GUARD FAILED: generation incomplete"; exit 1; }
python verify_run.py --kimina-url http://localhost:8082 --outdir out_m2_ext3 --bank out_m2/bank2.jsonl
v=$(wc -l < out_m2_ext3/verified.jsonl); echo "verified=$v"; [ "$v" = "$EXPECT" ] || { echo "GUARD FAILED: verified != expected"; exit 1; }
python analyze_ext3.py out_m2 out_m2_ext out_m2_ext2 out_m2_ext3 out_m2/bank2.jsonl | tee out_m2_ext3/ANALYSIS_M2_EXT3.txt
kill $SRV
