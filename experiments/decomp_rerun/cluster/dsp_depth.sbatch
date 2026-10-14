#!/bin/bash
#SBATCH -p gpu --qos=gpu --gres=gpu:1 -c 8 --mem=40G -t 8:00:00
#SBATCH -J decomp-dsp-depth -o logs/dsp-%j.out
: "${VERICODE_SCRATCH:?set VERICODE_SCRATCH to the directory holding models, environments and caches (ENVIRONMENT.md)}"
# DeepSeek-Prover-V2-7B depth arms (2026-09-11): screen | memI | f | C | CE | CE_s | E_s, one arm per gpu:1 job,
# in-loop kimina (8 REPLs; gpu:1, 10 CPUs, 48G so a job fits a fragmented node), per-arm ports so several jobs can share a node. Same serving as the prover's one-step
# jobs (model2_gen*.sbatch: vLLM bf16 gpu:1, max-model-len 32768, tokenizer guard, DECOMP_TASK_HEADER=1) and the
# same decoding as every arm (GEN: max_tokens 16384, temperature 0.7; the prover ships no generation_config.json,
# so vLLM's top_p 1 / top_k off apply, as in the one-step prover runs). Bank = depth/depth_bank_dsp.jsonl (written by
# the screen job). Submit from the scratch run dir:
#   S=$(sbatch --parsable --export=ALL,ARM=screen -J dsp-screen cluster/dsp_depth.sbatch)
#   for A in memI f C CE; do sbatch --export=ALL,ARM=$A -J dsp-$A -d afterok:$S cluster/dsp_depth.sbatch; done
#   sbatch --export=ALL,ARM=CE_s -J dsp-CEs -d afterok:<CE job>  cluster/dsp_depth.sbatch
#   sbatch --export=ALL,ARM=E_s  -J dsp-Es  -d afterok:<f job>   cluster/dsp_depth.sbatch
# Resumable: resubmit the same line with -d afterany:<jobid> on a timeout.
set -euo pipefail
: "${ARM:?set ARM=screen|memI|f|C|CE|CE_s|E_s|CE_m|all}"
case "$ARM" in
  screen) VPORT=8031; KPORT=8011;; memI) VPORT=8032; KPORT=8012;; f) VPORT=8033; KPORT=8013;;
  C) VPORT=8034; KPORT=8014;; CE) VPORT=8035; KPORT=8015;; CE_s) VPORT=8036; KPORT=8016;; E_s) VPORT=8037; KPORT=8018;; CE_m) VPORT=8039; KPORT=8021;;
  all) VPORT=8038; KPORT=8019;;
  *) echo "bad ARM $ARM"; exit 1;;
esac
VPORT=${VPORT_ENV:-$VPORT}; KPORT=${KPORT_ENV:-$KPORT}      # override when two ARM=all jobs share a node (18:39 collision)
TP=${TP:-1}                                                # TP=2 with --gres=gpu:2: one KV pool ~2.5x two gpu:1 servers
source cluster/cuda_env.sh
source ${VERICODE_SCRATCH}/vllm-env/bin/activate
export HF_HOME=${VERICODE_SCRATCH}/hf VLLM_CACHE_ROOT=${VERICODE_SCRATCH}/vllm_cache
export LEAN_SERVER_LEAN_VERSION=v4.29.0 LEAN_SERVER_MAX_REPLS=${LEAN_SERVER_MAX_REPLS:-10} LEAN_SERVER_PORT=$KPORT
M2=${VERICODE_SCRATCH}/models/DeepSeek-Prover-V2-7B
[ -f "$M2/config.json" ] && [ -f "$M2/model-00002-of-000002.safetensors" ] && [ -f "$M2/tokenizer.json" ] || { echo "MODEL MISSING/INCOMPLETE at $M2"; exit 1; }
export DECOMP_MODEL_DIR="$M2" DECOMP_TASK_HEADER=1 DECOMP_SERVED_MODEL=dsp-v2-7b
export DECOMP_DEPTH_BANK="$PWD/depth/depth_bank_dsp.jsonl"
[ "$ARM" = screen ] || [ "$ARM" = all ] || [ -s "$DECOMP_DEPTH_BANK" ] || { echo "NO PROVER DEPTH BANK $DECOMP_DEPTH_BANK (run ARM=screen first)"; exit 1; }
# Arm lock (2026-09-12 23:45): two differently-shaped chains (gpu:2 and gpu:1) may be queued for the same arm so that
# either shape of free GPU is taken; whichever starts first owns the arm, any other copy exits at once.
LOCK=out_dsp_codearms/$ARM.lock
if [ -f "$LOCK" ]; then OTHER=$(cat "$LOCK"); if [ "$OTHER" != "$SLURM_JOB_ID" ] && squeue -h -j "$OTHER" -o %t 2>/dev/null | grep -q .; then echo "ARM $ARM is owned by running job $OTHER; exiting"; exit 0; fi; fi
echo $SLURM_JOB_ID > "$LOCK"
python cluster/m2_tok_guard.py || exit 1
python -c "import vllm, torch, transformers; print('stack: vllm', vllm.__version__, 'torch', torch.__version__, 'transformers', transformers.__version__)"
KIMINA=${VERICODE_SCRATCH}/kimina-lean-server
RUNDIR=$PWD
mkdir -p out_dsp out_dsp_memI out_dsp_codearms
# KV_FP8=1 (user decision 18:45 PT 2026-09-11, recorded in PREREG_dsp_depth): fp8 KV cache doubles the 60k-token KV budget of this
# full-MHA 7B on one L40S, so ~2x requests decode at once. Numerics deviation from the one-step prover serving (bf16 KV).
KVFLAG=""; [ "${KV_FP8:-0}" = 1 ] && KVFLAG="--kv-cache-dtype fp8"
echo "serving: vllm serve dsp-v2-7b tp=$TP max-model-len 32768 max-num-seqs 64 ${KVFLAG:-bf16-kv}"
vllm serve "$M2" --served-model-name dsp-v2-7b -tp $TP --max-model-len 32768 --max-num-seqs 64 --disable-custom-all-reduce $KVFLAG --port $VPORT &
VS=$!
export PATH=${VERICODE_SCRATCH}/fakehome/.elan/bin:$PATH
[ -x ${VERICODE_SCRATCH}/fakehome/.elan/toolchains/leanprover--lean4---v4.29.0/bin/lake ] || { echo "TOOLCHAIN BROKEN: restore with rsync -a ${HOME}/.elan/toolchains/leanprover--lean4---v4.29.0/ <same path on scratch>/"; exit 1; }
KS=""
if [ -n "${KIMINA_JOB:-}" ]; then
  # 18:10 PT 2026-09-11: kimina on a CPU node (cluster/kimina_server.sbatch, 14 REPLs) so the GPU job's cores go to
  # vLLM's engine loop; 10 local REPLs + 4 runners on 12 cores starved it (decode 15-20 tok/s per sequence).
  for i in $(seq 360); do [ -s "logs/kimina-host.$KIMINA_JOB" ] && break; sleep 10; done
  [ -s "logs/kimina-host.$KIMINA_JOB" ] || { echo "KIMINA JOB $KIMINA_JOB NEVER PUBLISHED A HOST"; exit 1; }
  KPORT_URL="http://$(cat logs/kimina-host.$KIMINA_JOB)"
else
  ( source ${VERICODE_SCRATCH}/kimina-env/bin/activate
    export PATH=${VERICODE_SCRATCH}/fakehome/.elan/bin:$PATH
    echo "kimina subshell: which lake=$(which lake)" | cut -c1-200
    cd "$KIMINA" && python -m server ) &
  KS=$!
  for i in $(seq 120); do curl -sf localhost:$KPORT/health >/dev/null && break; sleep 5; done
  curl -sf localhost:$KPORT/health >/dev/null || { echo "KIMINA NEVER CAME UP"; exit 1; }
  KPORT_URL=http://localhost:$KPORT
fi
probe_kimina() {
  KURL="$1" ${VERICODE_SCRATCH}/kimina-env/bin/python - <<'PY'
import os, requests
code = ("import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 400000\n\n"
        "open BigOperators Real Nat Topology Rat\n\nexample : (2:ℕ) + 2 = 4 := by norm_num\n")
r = requests.post(os.environ["KURL"] + "/verify", json={"codes": [{"custom_id": "probe", "proof": code}],
                  "timeout": 300, "infotree_type": None}, timeout=400)
print("kimina probe", os.environ["KURL"], "HTTP", r.status_code, flush=True); r.raise_for_status()
it = r.json()["results"][0]
errs = [m for m in ((it.get("response") or {}).get("messages") or []) if m.get("severity") == "error"]
assert not it.get("error") and not errs, it
print("kimina probe PASS", flush=True)
PY
}
KURL=$KPORT_URL; KJ=""
if ! probe_kimina "$KURL"; then
  echo "LOCAL KIMINA PROBE FAILED -> plan B: kimina on a CPU node"
  [ -n "$KS" ] && kill $KS 2>/dev/null || true
  KJ=$(sbatch --parsable cluster/kimina_server.sbatch)
  for i in $(seq 360); do [ -s "logs/kimina-host.$KJ" ] && break; sleep 10; done
  [ -s "logs/kimina-host.$KJ" ] || { echo "KIMINA CPU JOB $KJ NEVER PUBLISHED A HOST"; scancel $KJ; kill $VS; exit 1; }
  KURL="http://$(cat logs/kimina-host.$KJ)"
  probe_kimina "$KURL" || { echo "REMOTE KIMINA PROBE FAILED ($KURL)"; scancel $KJ; kill $VS; exit 1; }
fi
echo "using kimina at $KURL"
for i in $(seq 180); do curl -sf localhost:$VPORT/health >/dev/null && break; sleep 10; done
curl -sf localhost:$VPORT/health >/dev/null || { echo "VLLM NEVER CAME UP"; exit 1; }
cd "$RUNDIR"
BASE=http://localhost:$VPORT
python - <<'PY'
import json, os, sys; sys.path.insert(0, "depth"); import depth_run as D
bank = os.environ["DECOMP_DEPTH_BANK"] if os.path.exists(os.environ["DECOMP_DEPTH_BANK"]) else "out_m2/bank2.jsonl"
p = json.loads(open(bank).readline())
print("--- bare prompt as sent (first problem of", bank, ") ---"); print(D.chat(D._bare(p))[:700]); print("--- task line ---"); print(D.task(p)[:200])
PY
case "$ARM" in
  screen)
    python depth/cold_screen.py --phase screen --base-url $BASE --model dsp-v2-7b --kimina-url "$KURL" --outdir out_dsp --n ${SCREEN_N:-32} --concurrency 48
    python depth/cold_screen.py --phase select --outdir out_dsp --out-bank "$DECOMP_DEPTH_BANK";;
  memI)
    strip_err() { n=$(grep -c '"gen_status": "error:' "$1" 2>/dev/null || true); [ "${n:-0}" = 0 ] && return 1; echo "$1: retrying $n error rows"; cp "$1" "$1.bak-err-$SLURM_JOB_ID-$RANDOM"; grep -v '"gen_status": "error:' "$1" > "$1.tmp" && mv "$1.tmp" "$1"; return 0; }
    GEN="python depth/memoryless_I.py --phase gen --outdir out_dsp_memI --base-url $BASE --model dsp-v2-7b --concurrency 48"
    $GEN; for pass in 1 2; do strip_err out_dsp_memI/memI_rollouts.jsonl || break; $GEN; done
    python depth/memoryless_I.py --phase verify --outdir out_dsp_memI --kimina-url "$KURL" --workers ${LEAN_SERVER_MAX_REPLS};;
  f)
    python depth/depth_run.py --arm f --base-url $BASE --kimina-url "$KURL" --concurrency 48 --outdir out_dsp;;
  C|CE|CE_s|E_s|CE_m)
    python depth/code_arms_run.py --arm "$ARM" --model dsp-v2-7b --base-url $BASE --kimina-url "$KURL" --concurrency 48 --outdir out_dsp_codearms --depth-outdir out_dsp;;
  all)
    # Single-GPU plan (2026-09-11: the cluster had no free GPU for hours): one prover server + one kimina serve every
    # arm. Screen first; then memI (low concurrency so the 7,040 independent draws do not starve the chains), f, C, CE
    # in parallel; then CE_s (needs CE) and E_s (needs f) in parallel. Each runner is resumable on its own files.
    [ -s "$DECOMP_DEPTH_BANK" ] || {
      python depth/cold_screen.py --phase screen --base-url $BASE --model dsp-v2-7b --kimina-url "$KURL" --outdir out_dsp --n ${SCREEN_N:-32} --concurrency 48
      python depth/cold_screen.py --phase select --outdir out_dsp --out-bank "$DECOMP_DEPTH_BANK"; }
    cat "$DECOMP_DEPTH_BANK" | python -c "import sys,json; print('bank:', [json.loads(l)['id'] for l in sys.stdin])"
    strip_err() { n=$(grep -c '"gen_status": "error:' "$1" 2>/dev/null || true); [ "${n:-0}" = 0 ] && return 1; echo "$1: retrying $n error rows"; cp "$1" "$1.bak-err-$SLURM_JOB_ID-$RANDOM"; grep -v '"gen_status": "error:' "$1" > "$1.tmp" && mv "$1.tmp" "$1"; return 0; }
    SUB=${ARMS_SUBSET:-AB}
    run_memI() { GEN="python depth/memoryless_I.py --phase gen --outdir out_dsp_memI --base-url $BASE --model dsp-v2-7b --concurrency 16"
      $GEN; for pass in 1 2; do strip_err out_dsp_memI/memI_rollouts.jsonl || break; $GEN; done
      python depth/memoryless_I.py --phase verify --outdir out_dsp_memI --kimina-url "$KURL" --workers 6; echo "memI done $(date)"; }
    code_arm() { python depth/code_arms_run.py --arm "$1" --model dsp-v2-7b --base-url $BASE --kimina-url "$KURL" --concurrency 24 --outdir out_dsp_codearms --depth-outdir out_dsp > logs/dsp-$SLURM_JOB_ID-$1.out 2>&1; echo "$1 done $(date)"; }
    PA=""; PB=""
    if [[ "$SUB" == *A* ]]; then
      run_memI > logs/dsp-$SLURM_JOB_ID-memI.out 2>&1 &  PM=$!
      ( python depth/depth_run.py --arm f --base-url $BASE --kimina-url "$KURL" --concurrency 24 --outdir out_dsp > logs/dsp-$SLURM_JOB_ID-f.out 2>&1; echo "f done $(date)"; code_arm E_s ) &  PF=$!
      PA="$PM $PF"
    fi
    if [[ "$SUB" == *B* ]]; then
      ( code_arm C ) &  PC=$!
      ( code_arm CE; code_arm CE_s ) &  PCE=$!
      PB="$PC $PCE"
    fi
    wait $PA $PB; echo "all arms done ($SUB) $(date)";;
esac
echo "dsp $ARM done $(date)"; rm -f out_dsp_codearms/$ARM.lock
kill $VS $KS 2>/dev/null || true
[ -n "$KJ" ] && scancel $KJ
true
