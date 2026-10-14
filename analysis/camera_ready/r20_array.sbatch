#!/bin/bash
#SBATCH --job-name=r20-shard
#SBATCH --array=0-3
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=r20_a%A_%a.out
#SBATCH --error=r20_a%A_%a.err

set -uo pipefail

# Submit from a work directory holding shard_<i>.txt (one record file per line) and
# a copy of r20_reverify.py. KIMINA_DIR is a built Kimina service (ENVIRONMENT.md),
# KIMINA_PYTHON its Python, VERICODE_ARTIFACT the artifact root.
R20=${R20_DIR:-$SLURM_SUBMIT_DIR}
KIM=${KIMINA_DIR:?set KIMINA_DIR to the Kimina service directory}
PY=${KIMINA_PYTHON:?set KIMINA_PYTHON to the Kimina environment python}
ART=${VERICODE_ARTIFACT:?set VERICODE_ARTIFACT to the artifact root}
PORT=$((8200 + SLURM_ARRAY_TASK_ID))

echo "node: $SLURMD_NODENAME  job: $SLURM_JOB_ID  started: $(date)"

cd "$KIM" || exit 1
export LEAN_SERVER_LEAN_VERSION=v4.29.0
export LEAN_SERVER_MAX_REPLS=16
export LEAN_SERVER_PORT=$PORT
export LEAN_SERVER_HOST=127.0.0.1
export PATH="$HOME/.elan/bin:$PATH"

"$PY" -m server > "$R20/server_a$SLURM_ARRAY_TASK_ID.log" 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null' EXIT

echo "waiting for kimina server on port $PORT (pid $SERVER_PID)"
READY=0
for i in $(seq 1 180); do
  if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "server process died; tail of its log:"; tail -40 "$R20/server_a$SLURM_ARRAY_TASK_ID.log"; exit 1
  fi
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then READY=1; break; fi
  sleep 10
done
if [ "$READY" != "1" ]; then
  echo "server did not become healthy in 30 min; tail of its log:"; tail -40 "$R20/server_a$SLURM_ARRAY_TASK_ID.log"; exit 1
fi
echo "server healthy after ~$((i*10))s"

# Functional check: a true theorem must pass, a sorry'd one must not.
cd "$ART" || exit 1
"$PY" scripts/smoke_test.py --kimina-url "http://127.0.0.1:$PORT" || echo "WARNING: smoke test reported a problem; continuing"

cd "$R20" || exit 1
"$PY" "$R20/r20_reverify.py" \
  --kimina-url "http://127.0.0.1:$PORT" \
  --files-from "$R20/shard_$SLURM_ARRAY_TASK_ID.txt" \
  --log "$R20/out/r20_shard_$SLURM_ARRAY_TASK_ID.jsonl" \
  --workers 16

echo "finished: $(date)"
