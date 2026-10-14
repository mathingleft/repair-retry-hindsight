#!/bin/bash
#SBATCH -p normal -c 32 --mem=96G -t 6:00:00
#SBATCH -J decomp-verify-ext3 -o logs/verify-ext3-%j.out
: "${VERICODE_SCRATCH:?set VERICODE_SCRATCH to the directory holding models, environments and caches (ENVIRONMENT.md)}"
# Streaming verify for out_ext3/ (Qwen EXT3: F_C_other, F_E_match, F_CE_match = 2004 rollouts). Merges the two
# shard files into out_ext3/rollouts.jsonl each pass, waits for BOTH gen jobs, guarded final pass, analyze_ext3.py.
# 16 REPLs (3 arms, vs 8 for the one-arm EXT2). Own kimina on port 8031.
#   sbatch --export=ALL,GEN_JOBS=<j0>,<j1> -d after:<j0>:<j1> cluster/verify_ext3.sbatch
set -euo pipefail
: "${GEN_JOBS:?set GEN_JOBS=<jobid>,<jobid>}"
export LEAN_SERVER_LEAN_VERSION=v4.29.0 LEAN_SERVER_MAX_REPLS=${LEAN_SERVER_MAX_REPLS:-16} LEAN_SERVER_PORT=8031
source ${VERICODE_SCRATCH}/kimina-env/bin/activate
export HOME=${VERICODE_SCRATCH}/fakehome
export PATH=${VERICODE_SCRATCH}/fakehome/.elan/bin:$PATH
[ -x ${VERICODE_SCRATCH}/fakehome/.elan/toolchains/leanprover--lean4---v4.29.0/bin/lake ] || { echo "TOOLCHAIN BROKEN"; exit 1; }
cd ${VERICODE_SCRATCH}/kimina-lean-server
python -m server &
SRV=$!
cd "$SLURM_SUBMIT_DIR"
EXPECT=$(python -c "from arms import EXT3_ARMS; print(668*len(EXT3_ARMS))")
for i in $(seq 120); do curl -sf localhost:8031/health >/dev/null && break; sleep 5; done
curl -sf localhost:8031/health >/dev/null || { echo "KIMINA NEVER CAME UP"; exit 1; }
python - <<'PY'
import requests
code = "import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 400000\n\nopen BigOperators Real Nat Topology Rat\n\nexample : (2:ℕ) + 2 = 4 := by norm_num\n"
r = requests.post("http://localhost:8031/verify", json={"codes": [{"custom_id": "probe", "proof": code}], "timeout": 300, "infotree_type": None}, timeout=400)
r.raise_for_status(); it = r.json()["results"][0]
errs = [m for m in ((it.get("response") or {}).get("messages") or []) if m.get("severity") == "error"]
assert not it.get("error") and not errs, it
print("kimina probe PASS")
PY
mkdir -p out_ext3
merge() { python - <<'PY'
import json, glob, os
seen = set(); out = []
for f in sorted(glob.glob("out_ext3/shard*/rollouts.jsonl")):
    for l in open(f):
        try: d = json.loads(l)
        except Exception: continue          # partial trailing line
        k = (d["id"], d["arm"], d["sample_idx"])
        if k in seen: continue
        seen.add(k); out.append(l if l.endswith("\n") else l + "\n")
tmp = "out_ext3/rollouts.jsonl.tmp"
open(tmp, "w").writelines(out); os.replace(tmp, "out_ext3/rollouts.jsonl")
print(f"merged {len(out)} rollouts")
PY
}
gen_alive() { for j in ${GEN_JOBS//,/ }; do [ -n "$(squeue -h -j "$j" 2>/dev/null)" ] && return 0; done; return 1; }
while true; do
  merge
  python verify_run.py --kimina-url http://localhost:8031 --outdir out_ext3 || echo "verify pass failed; will retry"
  if ! gen_alive; then echo "generation jobs $GEN_JOBS finished"; break; fi
  sleep 120
done
merge
n=$(wc -l < out_ext3/rollouts.jsonl); e=$(grep -c '"finish": "error:' out_ext3/rollouts.jsonl || true)
d=$(python -c 'import json; k=[(r["id"],r["arm"],r["sample_idx"]) for r in map(json.loads, open("out_ext3/rollouts.jsonl"))]; print(len(k)-len(set(k)))')
echo "FINAL rollouts=$n errors=$e dupes=$d"; [ "$n" = "$EXPECT" ] && [ "$e" = 0 ] && [ "$d" = 0 ] || { echo "GUARD FAILED: generation incomplete"; exit 1; }
python verify_run.py --kimina-url http://localhost:8031 --outdir out_ext3
v=$(wc -l < out_ext3/verified.jsonl); echo "verified=$v"; [ "$v" = "$EXPECT" ] || { echo "GUARD FAILED: verified != expected"; exit 1; }
python analyze_ext3.py out out_ext out_ext2 out_ext3 | tee out_ext3/ANALYSIS_EXT3.txt
kill $SRV
