#!/bin/bash
#SBATCH --job-name=decomp-refl
#SBATCH --partition=normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=28G
#SBATCH --time=4:00:00
#SBATCH --output=logs/refl-%j.out
set -euo pipefail
: "${WORK:?}"; : "${BUNDLE:?}"
MODEL=$(ls "$WORK"/models/*.gguf | head -1)
"$WORK/llama.cpp/build/bin/llama-server" -m "$MODEL" --port 10999 \
  -c 40960 --parallel 4 -t "$SLURM_CPUS_PER_TASK" --no-warmup &
SRV=$!
for i in $(seq 120); do curl -s localhost:10999/health | grep -q ok && break; sleep 5; done
python3 "$BUNDLE/run_gen.py" --phase reflections --base-url http://localhost:10999 \
  --concurrency 4 --outdir "$BUNDLE/out"
kill $SRV
