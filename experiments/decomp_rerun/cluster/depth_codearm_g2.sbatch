#!/bin/bash
#SBATCH -p gpu --qos=gpu --gres=gpu:2 -c 10 --mem=48G -t 08:00:00
#SBATCH -J decomp-depth-codearm-g2 -o logs/depth-%j.out
: "${VERICODE_SCRATCH:?set VERICODE_SCRATCH to the directory holding models, environments and caches (ENVIRONMENT.md)}"
# Code-bearing sequential depth arms C / CE / CE_s (depth/code_arms_run.py, 2026-09-07). Same lane, stack and
# flags as the blind_F / memI jobs (bf16, TP2, max-num-seqs 64, concurrency 48), in-loop kimina like every depth arm.
# One arm per job; CE_s needs CE complete (reads out_codearms/CE_rows.jsonl):
#   sbatch --export=ALL,ARM=C  -J decomp-depth-C-g2  cluster/depth_codearm_g2.sbatch
#   sbatch --export=ALL,ARM=CE -J decomp-depth-CE-g2 cluster/depth_codearm_g2.sbatch
#   sbatch --export=ALL,ARM=CE_s -J decomp-depth-CEs-g2 -d afterok:<CE job> cluster/depth_codearm_g2.sbatch
# Resumable; on timeout resubmit the same line with -d afterany:<jobid>. Submit from the scratch run dir.
set -euo pipefail
: "${ARM:?set ARM=C, CE, CE_s, E_s or CE_m}"
# Per-arm ports: two of these jobs can share a node (19:41 PT: C and CE both on gpu-node-06 -> C's kimina failed to bind
# 8000 and both runners verified through CE's kimina). vLLM on $VPORT, kimina on $KPORT (LEAN_SERVER_PORT).
case "$ARM" in C) VPORT=8021; KPORT=8001;; CE) VPORT=8022; KPORT=8002;; CE_s) VPORT=8023; KPORT=8003;; E_s) VPORT=8024; KPORT=8004;; CE_m) VPORT=8025; KPORT=8005;; *) echo "bad ARM $ARM"; exit 1;; esac
export LEAN_SERVER_PORT=$KPORT
source cluster/cuda_env.sh                 # venv CUDA13 toolchain + HOME=/scratch/.../fakehome
source ${VERICODE_SCRATCH}/vllm-env/bin/activate
export LEAN_SERVER_LEAN_VERSION=v4.29.0 LEAN_SERVER_MAX_REPLS=${LEAN_SERVER_MAX_REPLS:-8}   # cpu_count-1 default = 63 REPLs -> OOM in a cgroup
MODEL=${VERICODE_SCRATCH}/models/Qwen3.6-27B
KIMINA=${VERICODE_SCRATCH}/kimina-lean-server
RUNDIR=$PWD
vllm serve "$MODEL" --served-model-name qwen3.6-27b -tp 2 --max-model-len 131072 --max-num-seqs 64 --disable-custom-all-reduce --port $VPORT &
VS=$!
# oat nodes have no /home: kimina spawns `lake`/`lean` from PATH, so use the elan mirror on scratch.
export PATH=${VERICODE_SCRATCH}/fakehome/.elan/bin:$PATH
# Guard: the scratch elan toolchain was found gutted (empty bin/) at 00:16 on 2026-08-28; every new REPL then dies.
[ -x ${VERICODE_SCRATCH}/fakehome/.elan/toolchains/leanprover--lean4---v4.29.0/bin/lake ] || { echo "TOOLCHAIN BROKEN: restore with rsync -a ${HOME}/.elan/toolchains/leanprover--lean4---v4.29.0/ <same path on scratch>/"; exit 1; }
( source ${VERICODE_SCRATCH}/kimina-env/bin/activate
  # venv activate runs `deactivate nondestructive` first, which RESTORES the pre-vllm-env PATH and drops
  # the elan export above (root cause of jobs 1701161/1701316/1701383). So re-export AFTER activating.
  export PATH=${VERICODE_SCRATCH}/fakehome/.elan/bin:$PATH
  echo "kimina subshell: which lake=$(which lake)  PATH=$PATH" | cut -c1-300
  cd "$KIMINA" && python -m server ) &
KS=$!
for i in $(seq 120); do curl -sf localhost:$KPORT/health >/dev/null && break; sleep 5; done
curl -sf localhost:$KPORT/health >/dev/null || { echo "KIMINA NEVER CAME UP"; exit 1; }
# Functional probe: /health passes even when REPLs cannot spawn (job 1701161 burned 782 attempts on 500s).
probe_kimina() {   # $1 = base url; returns 0 iff a norm_num example verifies with no error
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
KURL=http://localhost:$KPORT; KJ=""
if ! probe_kimina "$KURL"; then
  echo "LOCAL KIMINA PROBE FAILED -> plan B: kimina on a CPU node, verify over the network"
  kill $KS 2>/dev/null || true
  KJ=$(sbatch --parsable cluster/kimina_server.sbatch)
  for i in $(seq 360); do [ -s "logs/kimina-host.$KJ" ] && break; sleep 10; done     # up to 60 min of queue wait
  [ -s "logs/kimina-host.$KJ" ] || { echo "KIMINA CPU JOB $KJ NEVER PUBLISHED A HOST"; scancel $KJ; kill $VS; exit 1; }
  KURL="http://$(cat logs/kimina-host.$KJ)"   # file holds host:port
  probe_kimina "$KURL" || { echo "REMOTE KIMINA PROBE FAILED ($KURL)"; scancel $KJ; kill $VS; exit 1; }
fi
echo "using kimina at $KURL"
for i in $(seq 180); do curl -sf localhost:$VPORT/health >/dev/null && break; sleep 10; done
curl -sf localhost:$VPORT/health >/dev/null || { echo "VLLM NEVER CAME UP"; exit 1; }
cd "$RUNDIR"
python -c "import vllm, torch, transformers; print('stack: vllm', vllm.__version__, 'torch', torch.__version__, 'transformers', transformers.__version__)"
python depth/code_arms_run.py --arm "$ARM" --base-url http://localhost:$VPORT --kimina-url "$KURL" --concurrency 48 --outdir out_codearms --depth-outdir out
kill $VS $KS 2>/dev/null || true
[ -n "$KJ" ] && scancel $KJ
true
