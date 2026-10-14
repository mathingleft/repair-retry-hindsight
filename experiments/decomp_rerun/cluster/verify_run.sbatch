#!/bin/bash
#SBATCH -p normal -c 16 --mem=48G -t 8:00:00
#SBATCH -J decomp-verify -o logs/verify-%j.out
: "${VERICODE_SCRATCH:?set VERICODE_SCRATCH to the directory holding models, environments and caches (ENVIRONMENT.md)}"
# Submit from the dir holding out/ (the scratch run dir) AFTER generation finishes.
set -euo pipefail
export LEAN_SERVER_LEAN_VERSION=v4.29.0 LEAN_SERVER_MAX_REPLS=${LEAN_SERVER_MAX_REPLS:-8}   # cpu_count-1 default = 63 REPLs -> OOM in a cgroup
source ${VERICODE_SCRATCH}/kimina-env/bin/activate
export HOME=${VERICODE_SCRATCH}/fakehome
export PATH=${VERICODE_SCRATCH}/fakehome/.elan/bin:$PATH   # lake/lean on nodes without /home   # elan toolchain mirrored here; works on nodes without /home
# Guard: the scratch elan toolchain was found gutted (empty bin/) at 00:16 on 2026-08-28; every new REPL then dies.
[ -x ${VERICODE_SCRATCH}/fakehome/.elan/toolchains/leanprover--lean4---v4.29.0/bin/lake ] || { echo "TOOLCHAIN BROKEN: restore with rsync -a ${HOME}/.elan/toolchains/leanprover--lean4---v4.29.0/ <same path on scratch>/"; exit 1; }
cd ${VERICODE_SCRATCH}/kimina-lean-server
python -m server &
SRV=$!
cd "$SLURM_SUBMIT_DIR"
EXPECT=$(python -c "from arms import ARMS; print(668*len(ARMS))")   # 167 problems x k=4 per arm   # submit from the dir that holds out/ (scratch run dir)
# Completeness guard (HANDOFF "After generation"): 4008 rows, no API-error rows, no duplicate keys.
n=$(wc -l < out/rollouts.jsonl); e=$(grep -c '"finish": "error:' out/rollouts.jsonl || true)
d=$(python -c 'import json; k=[(r["id"],r["arm"],r["sample_idx"]) for r in map(json.loads, open("out/rollouts.jsonl"))]; print(len(k)-len(set(k)))')
echo "rollouts=$n errors=$e dupes=$d"; [ "$n" = "$EXPECT" ] && [ "$e" = 0 ] && [ "$d" = 0 ] || { echo "GUARD FAILED: not verifying"; exit 1; }
for i in $(seq 120); do curl -sf localhost:8000/health >/dev/null && break; sleep 5; done
curl -sf localhost:8000/health >/dev/null || { echo "KIMINA NEVER CAME UP"; exit 1; }
python verify_run.py --kimina-url http://localhost:8000 --outdir out
python analyze.py out | tee out/ANALYSIS.txt
kill $SRV
