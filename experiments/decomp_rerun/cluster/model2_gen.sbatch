#!/bin/bash
#SBATCH -p gpu --qos=gpu --gres=gpu:4 -c 20 --mem=80G -t 4:00:00
#SBATCH -J decomp-m2 -o logs/m2-%j.out
: "${VERICODE_SCRATCH:?set VERICODE_SCRATCH to the directory holding models, environments and caches (ENVIRONMENT.md)}"
# Second model (DeepSeek-Prover-V2-7B), own frozen predecessors. Stage 1: bare k=4 on the 167 problems.
# Stage 2 (after the verify job has written out_m2/bank2.jsonl): I,F,F_E,F_C,F_CE on bank2.
#   sbatch --qos=gpu -d afterany:<prev> -w gpu-node-03 cluster/model2_gen.sbatch
set -euo pipefail
source cluster/cuda_env.sh
source ${VERICODE_SCRATCH}/vllm-env/bin/activate
export HF_HOME=${VERICODE_SCRATCH}/hf VLLM_CACHE_ROOT=${VERICODE_SCRATCH}/vllm_cache
M2=${VERICODE_SCRATCH}/models/DeepSeek-Prover-V2-7B
[ -f "$M2/config.json" ] && [ -f "$M2/model-00002-of-000002.safetensors" ] && [ -f "$M2/tokenizer.json" ] || { echo "MODEL MISSING/INCOMPLETE at $M2"; exit 1; }
export DECOMP_MODEL_DIR="$M2" DECOMP_TASK_HEADER=1
python cluster/m2_tok_guard.py || exit 1     # see the guard file: garbled-tokenizer incident, 04:41     # prover-native prompt: Mathlib header inside the code block
SERVED=dsp-v2-7b
mkdir -p out_m2
python -c "from arms import build_prompt; import json; b=json.loads(open('bank.jsonl').readline()); print('--- bare prompt ---'); print(build_prompt('bare', b)); print('--- F_E prompt ---'); print(build_prompt('F_E', b)[:1200])"
vllm serve "$M2" --served-model-name $SERVED -tp 4 --max-model-len 32768 --max-num-seqs 64 --disable-custom-all-reduce --port 8017 &
SRV=$!
for i in $(seq 180); do curl -sf localhost:8017/health >/dev/null && break; sleep 10; done
curl -sf localhost:8017/health >/dev/null || { echo "SERVER NEVER CAME UP"; exit 1; }
strip_err() { n=$(grep -c '"finish": "error:' "$1" 2>/dev/null || true); [ "${n:-0}" = 0 ] && return 1; echo "$1: retrying $n error rows"; cp "$1" "$1.bak-err-$SLURM_JOB_ID-$RANDOM"; grep -v '"finish": "error:' "$1" > "$1.tmp" && mv "$1.tmp" "$1"; return 0; }
GEN="python run_gen.py --phase main --outdir out_m2 --model $SERVED --base-url http://localhost:8017 --concurrency 48"
$GEN --arms bare
for pass in 1 2; do strip_err out_m2/rollouts.jsonl || break; $GEN --arms bare; done
echo "stage 1 (bare) generated $(date); waiting for out_m2/bank2.jsonl from the verify job"
for i in $(seq 150); do [ -s out_m2/bank2.jsonl ] && break; sleep 60; done     # up to 2.5 h
[ -s out_m2/bank2.jsonl ] || { echo "BANK2 NEVER APPEARED"; kill $SRV; exit 1; }
echo "bank2: $(wc -l < out_m2/bank2.jsonl) problems"
$GEN --arms I,F,F_E,F_C,F_CE --bank out_m2/bank2.jsonl
for pass in 1 2; do strip_err out_m2/rollouts.jsonl || break; $GEN --arms bare; $GEN --arms I,F,F_E,F_C,F_CE --bank out_m2/bank2.jsonl; done
echo "stage 2 generated $(date)"
kill $SRV
