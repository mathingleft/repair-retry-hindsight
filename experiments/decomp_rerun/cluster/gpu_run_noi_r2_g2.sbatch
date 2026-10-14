#!/bin/bash
#SBATCH -p gpu --qos=gpu --gres=gpu:2 -c 10 --mem=48G -t 1:30:00
#SBATCH -J decomp-noi-r2-g2 -o logs/noi-r2-%j.out
: "${VERICODE_SCRATCH:?set VERICODE_SCRATCH to the directory holding models, environments and caches (ENVIRONMENT.md)}"
# gpu:2 / TP2 variant (same bf16 weights; only the tensor-parallel split differs). Fresh-seed replicate of F_noI: same prompt, seeds 22072701+k, own outdir out_noi_r2 (668 rows).
# gpu_run.sbatch (bf16 TP4, max-num-seqs 64, concurrency 48). Separate outdir: zero interference with out/.
#   sbatch --qos=gpu -d afterany:<gen job> -w gpu-node-03 cluster/gpu_run_ext.sbatch
set -euo pipefail
source cluster/cuda_env.sh
source ${VERICODE_SCRATCH}/vllm-env/bin/activate
export HF_HOME=${VERICODE_SCRATCH}/hf VLLM_CACHE_ROOT=${VERICODE_SCRATCH}/vllm_cache
MODEL=${VERICODE_SCRATCH}/models/Qwen3.6-27B
ARMS_NOI=F_noI
mkdir -p out_noi_r2
vllm serve "$MODEL" --served-model-name qwen3.6-27b -tp 2 --max-model-len 131072 --max-num-seqs 64 --disable-custom-all-reduce --port 8017 &
SRV=$!
for i in $(seq 180); do curl -sf localhost:8017/health >/dev/null && break; sleep 10; done
curl -sf localhost:8017/health >/dev/null || { echo "SERVER NEVER CAME UP"; exit 1; }
strip_err() { n=$(grep -c '"finish": "error:' "$1" 2>/dev/null || true); [ "${n:-0}" = 0 ] && return 1; echo "$1: retrying $n error rows"; cp "$1" "$1.bak-err-$SLURM_JOB_ID-$RANDOM"; grep -v '"finish": "error:' "$1" > "$1.tmp" && mv "$1.tmp" "$1"; return 0; }
python run_gen.py --phase main --arms "$ARMS_NOI" --outdir out_noi_r2 --seed-base 22072701 --base-url http://localhost:8017 --concurrency 48
for pass in 1 2; do strip_err out_noi_r2/rollouts.jsonl || break; python run_gen.py --phase main --arms "$ARMS_NOI" --outdir out_noi_r2 --seed-base 22072701 --base-url http://localhost:8017 --concurrency 48; done
kill $SRV
