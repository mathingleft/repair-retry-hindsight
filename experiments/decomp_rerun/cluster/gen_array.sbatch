#!/bin/bash
#SBATCH --job-name=decomp-gen
#SBATCH --partition=normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=28G
#SBATCH --time=12:00:00
#SBATCH --array=0-15
#SBATCH --output=logs/gen-%A_%a.out
# env: WORK (model+llama.cpp root), BUNDLE (this decomp_rerun dir), NSHARD (=array size)
set -euo pipefail
: "${WORK:?}"; : "${BUNDLE:?}"; NSHARD="${NSHARD:-16}"
MODEL=$(ls "$WORK"/models/*.gguf | head -1)
# stage to node-local disk if it fits — avoids 16 nodes hammering NFS for the same file
SZ=$(stat -c %s "$MODEL"); FREE=$(df --output=avail -B1 /tmp | tail -1)
if [ "$FREE" -gt $((SZ + 5*1024*1024*1024)) ]; then
  LOCAL="/tmp/$(basename "$MODEL")"
  [ -f "$LOCAL" ] && [ "$(stat -c %s "$LOCAL")" = "$SZ" ] || cp "$MODEL" "$LOCAL"
  MODEL="$LOCAL"
fi
PORT=$((10000 + SLURM_ARRAY_TASK_ID))
"$WORK/llama.cpp/build/bin/llama-server" -m "$MODEL" --port "$PORT" \
  -c 81920 --parallel 4 -t "$SLURM_CPUS_PER_TASK" --no-warmup &
SRV=$!
for i in $(seq 120); do curl -s "localhost:$PORT/health" | grep -q ok && break; sleep 5; done
python3 "$BUNDLE/run_gen.py" --phase main --base-url "http://localhost:$PORT" \
  --arms "${ARMS_OVERRIDE:-bare,F,F_E,F_C,F_CE,F_R}" --shard "${SLURM_ARRAY_TASK_ID}:${NSHARD}" --concurrency 4 --outdir "$BUNDLE/out"
kill $SRV
