#!/bin/bash
#SBATCH -p gpu --qos=gpu --gres=gpu:2 -c 10 --mem=48G -t 3:00:00
#SBATCH -J decomp-ext3-g2 -o logs/ext3-%j.out
: "${VERICODE_SCRATCH:?set VERICODE_SCRATCH to the directory holding models, environments and caches (ENVIRONMENT.md)}"
# EXT3 arms (arms.EXT3_ARMS = F_C_other, F_E_match, F_CE_match; 2026-09-06) on Qwen, split over TWO gpu:2 TP2
# jobs by run_gen's stable (id,arm,k) hash shard so both spare half-nodes work at once (4-GPU cap = 2 x gpu:2).
# Each shard writes its OWN file (out_ext3/shard$SHARD/rollouts.jsonl; no concurrent appends to one file);
# verify_ext3.sbatch merges them into out_ext3/rollouts.jsonl. Same stack/flags as gpu_run_ext2_g2.sbatch.
#   sbatch --export=ALL,SHARD=0 cluster/gpu_run_ext3_g2.sbatch
#   sbatch --export=ALL,SHARD=1 cluster/gpu_run_ext3_g2.sbatch
set -euo pipefail
: "${SHARD:?set SHARD=0 or 1}"; NSHARD=${NSHARD:-2}; PORT=$((8027 + SHARD))
source cluster/cuda_env.sh
source ${VERICODE_SCRATCH}/vllm-env/bin/activate
export HF_HOME=${VERICODE_SCRATCH}/hf VLLM_CACHE_ROOT=${VERICODE_SCRATCH}/vllm_cache
MODEL=${VERICODE_SCRATCH}/models/Qwen3.6-27B
ARMS_EXT3=$(python -c "from arms import EXT3_ARMS; print(','.join(EXT3_ARMS))")
OUT=out_ext3/shard$SHARD; mkdir -p "$OUT"
[ -s bank_match.json ] || { echo "NO bank_match.json (run build_match.py)"; exit 1; }
python - <<'PY'
import json, arms
b = json.loads(open("bank.jsonl").readline())
for a in arms.EXT3_ARMS:
    print(f"--- {a} prompt (first bank problem) ---"); print(arms.build_prompt(a, b)[:1200]); print()
PY
vllm serve "$MODEL" --served-model-name qwen3.6-27b -tp 2 --max-model-len 131072 --max-num-seqs 64 --disable-custom-all-reduce --port $PORT &
SRV=$!
for i in $(seq 180); do curl -sf localhost:$PORT/health >/dev/null && break; sleep 10; done
curl -sf localhost:$PORT/health >/dev/null || { echo "SERVER NEVER CAME UP"; exit 1; }
strip_err() { n=$(grep -c '"finish": "error:' "$1" 2>/dev/null || true); [ "${n:-0}" = 0 ] && return 1; echo "$1: retrying $n error rows"; cp "$1" "$1.bak-err-$SLURM_JOB_ID-$RANDOM"; grep -v '"finish": "error:' "$1" > "$1.tmp" && mv "$1.tmp" "$1"; return 0; }
GEN="python run_gen.py --phase main --arms $ARMS_EXT3 --outdir $OUT --base-url http://localhost:$PORT --concurrency 48 --shard $SHARD:$NSHARD"
$GEN
for pass in 1 2; do strip_err $OUT/rollouts.jsonl || break; $GEN; done
echo "shard $SHARD generated $(date)"
kill $SRV
