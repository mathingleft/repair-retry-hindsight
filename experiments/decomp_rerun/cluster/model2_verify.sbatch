#!/bin/bash
#SBATCH -p normal -c 16 --mem=48G -t 10:00:00
#SBATCH -J decomp-verify-m2 -o logs/verify-m2-%j.out
: "${VERICODE_SCRATCH:?set VERICODE_SCRATCH to the directory holding models, environments and caches (ENVIRONMENT.md)}"
# Streaming verify for out_m2/ (second model). Builds out_m2/bank2.jsonl as soon as all 668 bare rows are
# verified (the GPU job waits for it), then keeps verifying the stage-2 arms; final guard + analyze_m2.py.
#   sbatch --export=ALL,GEN_JOB=<m2 gen jobid> -d after:<m2 gen jobid> cluster/model2_verify.sbatch
set -euo pipefail
: "${GEN_JOB:?set GEN_JOB=<generation job id>}"
export LEAN_SERVER_LEAN_VERSION=v4.29.0 LEAN_SERVER_MAX_REPLS=${LEAN_SERVER_MAX_REPLS:-8} LEAN_SERVER_PORT=8030
source ${VERICODE_SCRATCH}/kimina-env/bin/activate
export HOME=${VERICODE_SCRATCH}/fakehome
export PATH=${VERICODE_SCRATCH}/fakehome/.elan/bin:$PATH
[ -x ${VERICODE_SCRATCH}/fakehome/.elan/toolchains/leanprover--lean4---v4.29.0/bin/lake ] || { echo "TOOLCHAIN BROKEN"; exit 1; }
cd ${VERICODE_SCRATCH}/kimina-lean-server
python -m server &
SRV=$!
cd "$SLURM_SUBMIT_DIR"
for i in $(seq 120); do curl -sf localhost:8030/health >/dev/null && break; sleep 5; done
curl -sf localhost:8030/health >/dev/null || { echo "KIMINA NEVER CAME UP"; exit 1; }
python - <<'PY'
import requests
code = "import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 400000\n\nopen BigOperators Real Nat Topology Rat\n\nexample : (2:ℕ) + 2 = 4 := by norm_num\n"
r = requests.post("http://localhost:8030/verify", json={"codes": [{"custom_id": "probe", "proof": code}], "timeout": 300, "infotree_type": None}, timeout=400)
r.raise_for_status(); it = r.json()["results"][0]
errs = [m for m in ((it.get("response") or {}).get("messages") or []) if m.get("severity") == "error"]
assert not it.get("error") and not errs, it
print("kimina probe PASS")
PY
mkdir -p out_m2; touch out_m2/rollouts.jsonl
nbare() { python -c 'import json,sys; print(sum(1 for l in open("out_m2/verified.jsonl") if json.loads(l)["arm"]=="bare"))' 2>/dev/null || echo 0; }
while true; do
  python verify_run.py --kimina-url http://localhost:8030 --outdir out_m2 || echo "verify pass failed; will retry"
  if [ ! -s out_m2/bank2.jsonl ] && [ "$(nbare)" = 668 ]; then python build_bank2.py out_m2 --model dsp-v2-7b || echo "bank2 build deferred"; fi
  if [ -z "$(squeue -h -j "$GEN_JOB" 2>/dev/null)" ]; then echo "generation job $GEN_JOB finished"; break; fi
  sleep 120
done
[ -s out_m2/bank2.jsonl ] || { echo "GUARD FAILED: no bank2"; exit 1; }
EXPECT=$(python -c 'import json; print(668 + 4*5*sum(1 for _ in open("out_m2/bank2.jsonl")))')
n=$(wc -l < out_m2/rollouts.jsonl); e=$(grep -c '"finish": "error:' out_m2/rollouts.jsonl || true)
d=$(python -c 'import json; k=[(r["id"],r["arm"],r["sample_idx"]) for r in map(json.loads, open("out_m2/rollouts.jsonl"))]; print(len(k)-len(set(k)))')
echo "FINAL rollouts=$n expect=$EXPECT errors=$e dupes=$d"; [ "$n" = "$EXPECT" ] && [ "$e" = 0 ] && [ "$d" = 0 ] || { echo "GUARD FAILED: generation incomplete"; exit 1; }
python verify_run.py --kimina-url http://localhost:8030 --outdir out_m2
v=$(wc -l < out_m2/verified.jsonl); echo "verified=$v"; [ "$v" = "$EXPECT" ] || { echo "GUARD FAILED: verified != expected"; exit 1; }
python analyze_m2.py out_m2 | tee out_m2/ANALYSIS_M2.txt
kill $SRV
