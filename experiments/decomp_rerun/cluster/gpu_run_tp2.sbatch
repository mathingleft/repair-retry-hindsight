#!/bin/bash
#SBATCH -p gpu --gres=gpu:2 -c 10 --mem=50G -t 12:00:00
#SBATCH -J decomp-gpu2 -o logs/gpu-%j.out
: "${VERICODE_SCRATCH:?set VERICODE_SCRATCH to the directory holding models, environments and caches (ENVIRONMENT.md)}"
set -euo pipefail
source cluster/cuda_env.sh
source ${VERICODE_SCRATCH}/vllm-env/bin/activate
export HF_HOME=${VERICODE_SCRATCH}/hf VLLM_CACHE_ROOT=${VERICODE_SCRATCH}/vllm_cache
MODEL=${VERICODE_SCRATCH}/models/Qwen3.6-27B
vllm serve "$MODEL" --served-model-name qwen3.6-27b -tp 2 --max-model-len 131072 --max-num-seqs 64 --disable-custom-all-reduce --port 8017 &
SRV=$!
for i in $(seq 180); do curl -sf localhost:8017/health >/dev/null && break; sleep 10; done
curl -sf localhost:8017/health >/dev/null || { echo "SERVER NEVER CAME UP"; exit 1; }
# strip_err FILE: drop rows the runner recorded as API errors so the (idempotent) runner retries them.
strip_err() { n=$(grep -c '"finish": "error:' "$1" 2>/dev/null || true); [ "${n:-0}" = 0 ] && return 1; echo "$1: retrying $n error rows"; cp "$1" "$1.bak-err-$SLURM_JOB_ID-$RANDOM"; grep -v '"finish": "error:' "$1" > "$1.tmp" && mv "$1.tmp" "$1"; return 0; }
python run_gen.py --phase reflections --base-url http://localhost:8017 --concurrency 16
strip_err out/reflections.jsonl && python run_gen.py --phase reflections --base-url http://localhost:8017 --concurrency 16
python run_gen.py --phase main --base-url http://localhost:8017 --concurrency 24
for pass in 1 2; do strip_err out/rollouts.jsonl || break; python run_gen.py --phase main --base-url http://localhost:8017 --concurrency 24; done
kill $SRV
