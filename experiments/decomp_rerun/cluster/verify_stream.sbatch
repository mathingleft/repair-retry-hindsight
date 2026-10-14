#!/bin/bash
#SBATCH -p normal -c 16 --mem=48G -t 8:00:00
#SBATCH -J decomp-verify-stream -o logs/verify-%j.out
: "${VERICODE_SCRATCH:?set VERICODE_SCRATCH to the directory holding models, environments and caches (ENVIRONMENT.md)}"
# Streaming verify: verifies rollouts WHILE the generation job (GEN_JOB) is still running, then does the
# guarded final pass + analyze. Submit from the scratch run dir:
#   sbatch --export=ALL,GEN_JOB=<gen jobid> cluster/verify_stream.sbatch
# Read-only on out/rollouts.jsonl (single-writer rule intact); verify_run.py is resumable and skips
# partial trailing lines and API-error rows (which gpu_run.sbatch strips and regenerates).
set -euo pipefail
: "${GEN_JOB:?set GEN_JOB=<generation job id>}"
export LEAN_SERVER_LEAN_VERSION=v4.29.0 LEAN_SERVER_MAX_REPLS=${LEAN_SERVER_MAX_REPLS:-8}
source ${VERICODE_SCRATCH}/kimina-env/bin/activate
export HOME=${VERICODE_SCRATCH}/fakehome
export PATH=${VERICODE_SCRATCH}/fakehome/.elan/bin:$PATH   # lake/lean on nodes without /home
# Guard: the scratch elan toolchain was found gutted (empty bin/) at 00:16 on 2026-08-28; every new REPL then dies.
[ -x ${VERICODE_SCRATCH}/fakehome/.elan/toolchains/leanprover--lean4---v4.29.0/bin/lake ] || { echo "TOOLCHAIN BROKEN: restore with rsync -a ${HOME}/.elan/toolchains/leanprover--lean4---v4.29.0/ <same path on scratch>/"; exit 1; }
cd ${VERICODE_SCRATCH}/kimina-lean-server
python -m server &
SRV=$!
cd "$SLURM_SUBMIT_DIR"
EXPECT=$(python -c "from arms import ARMS; print(668*len(ARMS))")   # 167 problems x k=4 per arm
for i in $(seq 120); do curl -sf localhost:8000/health >/dev/null && break; sleep 5; done
curl -sf localhost:8000/health >/dev/null || { echo "KIMINA NEVER CAME UP"; exit 1; }
while true; do
  python verify_run.py --kimina-url http://localhost:8000 --outdir out || echo "verify pass failed; will retry"
  if [ -z "$(squeue -h -j "$GEN_JOB" 2>/dev/null)" ]; then echo "generation job $GEN_JOB finished"; break; fi
  sleep 120
done
n=$(wc -l < out/rollouts.jsonl); e=$(grep -c '"finish": "error:' out/rollouts.jsonl || true)
d=$(python -c 'import json; k=[(r["id"],r["arm"],r["sample_idx"]) for r in map(json.loads, open("out/rollouts.jsonl"))]; print(len(k)-len(set(k)))')
echo "FINAL rollouts=$n errors=$e dupes=$d"; [ "$n" = "$EXPECT" ] && [ "$e" = 0 ] && [ "$d" = 0 ] || { echo "GUARD FAILED: generation incomplete"; exit 1; }
python verify_run.py --kimina-url http://localhost:8000 --outdir out
v=$(wc -l < out/verified.jsonl); echo "verified=$v"; [ "$v" = "$EXPECT" ] || { echo "GUARD FAILED: verified != expected"; exit 1; }
python analyze.py out | tee out/ANALYSIS.txt
kill $SRV
