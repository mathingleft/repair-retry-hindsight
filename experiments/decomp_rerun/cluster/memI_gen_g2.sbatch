#!/bin/bash
#SBATCH -p gpu --qos=gpu --gres=gpu:2 -c 10 --mem=48G -t 8:00:00
#SBATCH -J decomp-memI-g2 -o logs/memI-%j.out
: "${VERICODE_SCRATCH:?set VERICODE_SCRATCH to the directory holding models, environments and caches (ENVIRONMENT.md)}"
# Memoryless instruction-only retry control (depth/memoryless_I.py, 2026-09-06): 11 problems x 10 traj x 64
# attempts = 7,040 fixed-prompt draws, no history, no early stop. Split over TWO gpu:2 TP2 jobs by the stable
# (problem, traj, attempt) hash (--shard). Same stack/flags as every other Qwen job here (bf16, TP2,
# max-num-seqs 64, concurrency 48). Each shard writes out_memI/shard$SHARD/memI_rollouts.jsonl.
#   sbatch --export=ALL,SHARD=0 -d afterany:<ext3 shard0 job> cluster/memI_gen_g2.sbatch
#   sbatch --export=ALL,SHARD=1 -d afterany:<ext3 shard1 job> cluster/memI_gen_g2.sbatch
set -euo pipefail
: "${SHARD:?set SHARD=0 or 1}"; NSHARD=${NSHARD:-2}; PORT=$((8037 + SHARD))
source cluster/cuda_env.sh
source ${VERICODE_SCRATCH}/vllm-env/bin/activate
export HF_HOME=${VERICODE_SCRATCH}/hf VLLM_CACHE_ROOT=${VERICODE_SCRATCH}/vllm_cache
MODEL=${VERICODE_SCRATCH}/models/Qwen3.6-27B
OUT=out_memI/shard$SHARD; mkdir -p "$OUT"
python -c "import vllm, torch, transformers; print('stack: vllm', vllm.__version__, 'torch', torch.__version__, 'transformers', transformers.__version__)"
python - <<'PY'
import json, sys; sys.path.insert(0, "depth"); import depth_run as D
p = json.loads(open("depth/depth_bank.jsonl").readline())
print("--- memoryless_I prompt as sent (first problem) ---"); print(D.chat(D.task(p))); print("seed(t=0,k=1) =", D.seed_for(p["id"], 0, 1))
PY
vllm serve "$MODEL" --served-model-name qwen3.6-27b -tp 2 --max-model-len 131072 --max-num-seqs 64 --disable-custom-all-reduce --port $PORT &
SRV=$!
for i in $(seq 180); do curl -sf localhost:$PORT/health >/dev/null && break; sleep 10; done
curl -sf localhost:$PORT/health >/dev/null || { echo "SERVER NEVER CAME UP"; exit 1; }
strip_err() { n=$(grep -c '"gen_status": "error:' "$1" 2>/dev/null || true); [ "${n:-0}" = 0 ] && return 1; echo "$1: retrying $n error rows"; cp "$1" "$1.bak-err-$SLURM_JOB_ID-$RANDOM"; grep -v '"gen_status": "error:' "$1" > "$1.tmp" && mv "$1.tmp" "$1"; return 0; }
GEN="python depth/memoryless_I.py --phase gen --outdir $OUT --base-url http://localhost:$PORT --concurrency 48 --shard $SHARD:$NSHARD"
$GEN
for pass in 1 2; do strip_err $OUT/memI_rollouts.jsonl || break; $GEN; done
echo "memI shard $SHARD generated $(date)"
kill $SRV
