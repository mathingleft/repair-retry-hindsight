#!/bin/bash
#SBATCH -p gpu --qos=gpu --gres=gpu:4 -c 20 --mem=80G -t 3:00:00
#SBATCH -J decomp-ext -o logs/ext-%j.out
: "${VERICODE_SCRATCH:?set VERICODE_SCRATCH to the directory holding models, environments and caches (ENVIRONMENT.md)}"
# Extension arms F_E_shuf + N_noI (arms.EXT_ARMS) -> out_ext/rollouts.jsonl. Same stack and flags as
# gpu_run.sbatch (bf16 TP4, max-num-seqs 64, concurrency 48). Separate outdir: zero interference with out/.
#   sbatch --qos=gpu -d afterany:<gen job> -w gpu-node-03 cluster/gpu_run_ext.sbatch
set -euo pipefail
source cluster/cuda_env.sh
source ${VERICODE_SCRATCH}/vllm-env/bin/activate
export HF_HOME=${VERICODE_SCRATCH}/hf VLLM_CACHE_ROOT=${VERICODE_SCRATCH}/vllm_cache
MODEL=${VERICODE_SCRATCH}/models/Qwen3.6-27B
ARMS_EXT=$(python -c "from arms import EXT_ARMS; print(','.join(EXT_ARMS))")
mkdir -p out_ext
vllm serve "$MODEL" --served-model-name qwen3.6-27b -tp 4 --max-model-len 131072 --max-num-seqs 64 --disable-custom-all-reduce --port 8017 &
SRV=$!
for i in $(seq 180); do curl -sf localhost:8017/health >/dev/null && break; sleep 10; done
curl -sf localhost:8017/health >/dev/null || { echo "SERVER NEVER CAME UP"; exit 1; }
strip_err() { n=$(grep -c '"finish": "error:' "$1" 2>/dev/null || true); [ "${n:-0}" = 0 ] && return 1; echo "$1: retrying $n error rows"; cp "$1" "$1.bak-err-$SLURM_JOB_ID-$RANDOM"; grep -v '"finish": "error:' "$1" > "$1.tmp" && mv "$1.tmp" "$1"; return 0; }
python run_gen.py --phase main --arms "$ARMS_EXT" --outdir out_ext --base-url http://localhost:8017 --concurrency 48
for pass in 1 2; do strip_err out_ext/rollouts.jsonl || break; python run_gen.py --phase main --arms "$ARMS_EXT" --outdir out_ext --base-url http://localhost:8017 --concurrency 48; done
kill $SRV
