#!/bin/bash
#SBATCH -p normal --gres=gpu:1 -c 10 --mem=40G -t 2:00:00
#SBATCH -J decomp-m2ext2 -o logs/m2ext2-%j.out
: "${VERICODE_SCRATCH:?set VERICODE_SCRATCH to the directory holding models, environments and caches (ENVIRONMENT.md)}"
# Prover EXT2 arm (2026-08-31): F_CE_shuf on the 137 bank2 problems -> out_m2_ext2/. Same derangement seed
# as the prover F_E_shuf (arms.SHUF_BANK = bank2 via run_gen --bank). gpu:1 on QOS normal, tp=1.
# Verify: cluster/model2_verify_ext2.sbatch.
set -euo pipefail
source cluster/cuda_env.sh
source ${VERICODE_SCRATCH}/vllm-env/bin/activate
export HF_HOME=${VERICODE_SCRATCH}/hf VLLM_CACHE_ROOT=${VERICODE_SCRATCH}/vllm_cache
M2=${VERICODE_SCRATCH}/models/DeepSeek-Prover-V2-7B
[ -f "$M2/config.json" ] && [ -f "$M2/model-00002-of-000002.safetensors" ] && [ -f "$M2/tokenizer.json" ] || { echo "MODEL MISSING/INCOMPLETE at $M2"; exit 1; }
export DECOMP_MODEL_DIR="$M2" DECOMP_TASK_HEADER=1
python cluster/m2_tok_guard.py || exit 1     # see the guard file: garbled-tokenizer incident, 04:41
SERVED=dsp-v2-7b
mkdir -p out_m2_ext2
[ -s out_m2/bank2.jsonl ] || { echo "NO bank2"; exit 1; }
python - <<'PY'
import json, arms
arms.SHUF_BANK = "out_m2/bank2.jsonl"; b = json.loads(open("out_m2/bank2.jsonl").readline())
print("--- prover F_CE_shuf prompt ---"); print(arms.build_prompt("F_CE_shuf", b)[:1500])
PY
vllm serve "$M2" --served-model-name $SERVED --max-model-len 32768 --max-num-seqs 64 --disable-custom-all-reduce --port 8018 &
SRV=$!
for i in $(seq 180); do curl -sf localhost:8018/health >/dev/null && break; sleep 10; done
curl -sf localhost:8018/health >/dev/null || { echo "SERVER NEVER CAME UP"; exit 1; }
strip_err() { n=$(grep -c '"finish": "error:' "$1" 2>/dev/null || true); [ "${n:-0}" = 0 ] && return 1; echo "$1: retrying $n error rows"; cp "$1" "$1.bak-err-$SLURM_JOB_ID-$RANDOM"; grep -v '"finish": "error:' "$1" > "$1.tmp" && mv "$1.tmp" "$1"; return 0; }
GEN="python run_gen.py --phase main --outdir out_m2_ext2 --model $SERVED --base-url http://localhost:8018 --concurrency 48 --bank out_m2/bank2.jsonl"
$GEN --arms F_CE_shuf
for pass in 1 2; do strip_err out_m2_ext2/rollouts.jsonl || break; $GEN --arms F_CE_shuf; done
echo "prover F_CE_shuf generated $(date)"
kill $SRV
