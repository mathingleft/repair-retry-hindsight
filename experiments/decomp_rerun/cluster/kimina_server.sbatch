#!/bin/bash
#SBATCH -p normal -c 16 --mem=48G -t 8:00:00
#SBATCH -J kimina-server -o logs/kimina-server-%j.out
: "${VERICODE_SCRATCH:?set VERICODE_SCRATCH to the directory holding models, environments and caches (ENVIRONMENT.md)}"
# Standalone kimina server on a CPU node for in-loop verification over the network (plan B for GPU nodes
# where kimina cannot exec `lake`). Publishes its host in logs/kimina-host.<jobid>; runs until cancelled.
set -uo pipefail
export LEAN_SERVER_LEAN_VERSION=v4.29.0 LEAN_SERVER_MAX_REPLS=${LEAN_SERVER_MAX_REPLS:-8} LEAN_SERVER_PORT=8010   # 8000 may be taken by a verify-stream kimina on the same node
export HOME=${VERICODE_SCRATCH}/fakehome
export PATH=${VERICODE_SCRATCH}/fakehome/.elan/bin:$PATH
# Guard: the scratch elan toolchain was found gutted (empty bin/) at 00:16 on 2026-08-28; every new REPL then dies.
[ -x ${VERICODE_SCRATCH}/fakehome/.elan/toolchains/leanprover--lean4---v4.29.0/bin/lake ] || { echo "TOOLCHAIN BROKEN: restore with rsync -a ${HOME}/.elan/toolchains/leanprover--lean4---v4.29.0/ <same path on scratch>/"; exit 1; }
source ${VERICODE_SCRATCH}/kimina-env/bin/activate
cd ${VERICODE_SCRATCH}/kimina-lean-server
python -m server &
SRV=$!
for i in $(seq 120); do curl -sf localhost:8010/health >/dev/null && break; sleep 5; done
curl -sf localhost:8010/health >/dev/null || { echo "KIMINA NEVER CAME UP"; exit 1; }
echo "$(hostname):8010" > "$SLURM_SUBMIT_DIR/logs/kimina-host.$SLURM_JOB_ID"
echo "kimina up on $(hostname):8010"
wait $SRV
