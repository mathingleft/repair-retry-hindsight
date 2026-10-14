#!/bin/bash
#SBATCH -p normal -c 32 --mem=96G -t 12:00:00
#SBATCH -J decomp-verify-memI -o logs/verify-memI-%j.out
: "${VERICODE_SCRATCH:?set VERICODE_SCRATCH to the directory holding models, environments and caches (ENVIRONMENT.md)}"
# Streaming verify for out_memI/ (memoryless_I, 7,040 rows): merges the shard files into out_memI/memI_rollouts.jsonl
# each pass, verifies with the final corrected pipeline (memoryless_I.py --phase verify: extract_lean_code splice,
# 120 s, heartbeats 400000, sorry/admit screen, infra retries), waits for BOTH gen jobs, guarded final pass, analysis.
# 16 REPLs. Own kimina on port 8041.
#   sbatch --export=ALL,GEN_JOBS=<j0>,<j1> -d after:<j0>:<j1> cluster/memI_verify.sbatch
set -euo pipefail
: "${GEN_JOBS:?set GEN_JOBS=<jobid>,<jobid>}"
export LEAN_SERVER_LEAN_VERSION=v4.29.0 LEAN_SERVER_MAX_REPLS=${LEAN_SERVER_MAX_REPLS:-16} LEAN_SERVER_PORT=8041
source ${VERICODE_SCRATCH}/kimina-env/bin/activate
export HOME=${VERICODE_SCRATCH}/fakehome
export PATH=${VERICODE_SCRATCH}/fakehome/.elan/bin:$PATH
[ -x ${VERICODE_SCRATCH}/fakehome/.elan/toolchains/leanprover--lean4---v4.29.0/bin/lake ] || { echo "TOOLCHAIN BROKEN"; exit 1; }
cd ${VERICODE_SCRATCH}/kimina-lean-server
python -m server &
SRV=$!
cd "$SLURM_SUBMIT_DIR"
EXPECT=7040
for i in $(seq 120); do curl -sf localhost:8041/health >/dev/null && break; sleep 5; done
curl -sf localhost:8041/health >/dev/null || { echo "KIMINA NEVER CAME UP"; exit 1; }
python - <<'PY'
import requests
code = "import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 400000\n\nopen BigOperators Real Nat Topology Rat\n\nexample : (2:ℕ) + 2 = 4 := by norm_num\n"
r = requests.post("http://localhost:8041/verify", json={"codes": [{"custom_id": "probe", "proof": code}], "timeout": 300, "infotree_type": None}, timeout=400)
r.raise_for_status(); it = r.json()["results"][0]
errs = [m for m in ((it.get("response") or {}).get("messages") or []) if m.get("severity") == "error"]
assert not it.get("error") and not errs, it
print("kimina probe PASS")
PY
mkdir -p out_memI
merge() { python - <<'PY'
import json, glob, os
seen = set(); out = []
for f in sorted(glob.glob("out_memI/shard*/memI_rollouts.jsonl")):
    for l in open(f):
        try: d = json.loads(l)
        except Exception: continue
        k = (d["problem_id"], d["trajectory_id"], d["attempt_id"])
        if k in seen: continue
        seen.add(k); out.append(l if l.endswith("\n") else l + "\n")
tmp = "out_memI/memI_rollouts.jsonl.tmp"; open(tmp, "w").writelines(out); os.replace(tmp, "out_memI/memI_rollouts.jsonl")
print(f"merged {len(out)} rollouts")
PY
}
gen_alive() { for j in ${GEN_JOBS//,/ }; do [ -n "$(squeue -h -j "$j" 2>/dev/null)" ] && return 0; done; return 1; }
while true; do
  merge
  python depth/memoryless_I.py --phase verify --kimina-url http://localhost:8041 --outdir out_memI || echo "verify pass failed; will retry"
  if ! gen_alive; then echo "generation jobs $GEN_JOBS finished"; break; fi
  sleep 180
done
merge
n=$(wc -l < out_memI/memI_rollouts.jsonl); e=$(grep -c '"gen_status": "error:' out_memI/memI_rollouts.jsonl || true)
d=$(python -c 'import json; k=[(r["problem_id"],r["trajectory_id"],r["attempt_id"]) for r in map(json.loads, open("out_memI/memI_rollouts.jsonl"))]; print(len(k)-len(set(k)))')
echo "FINAL rollouts=$n errors=$e dupes=$d"; [ "$n" = "$EXPECT" ] && [ "$e" = 0 ] && [ "$d" = 0 ] || { echo "GUARD FAILED: generation incomplete"; exit 1; }
python depth/memoryless_I.py --phase verify --kimina-url http://localhost:8041 --outdir out_memI
v=$(wc -l < out_memI/memI_verified.jsonl); echo "verified=$v"; [ "$v" = "$EXPECT" ] || { echo "GUARD FAILED: verified != expected"; exit 1; }
echo "infra-error rows: $(grep -c 'infra_error_after_retries' out_memI/memI_verified.jsonl || true)"
python depth/analyze_memI.py out_memI out | tee out_memI/ANALYSIS_MEMI.txt
kill $SRV
