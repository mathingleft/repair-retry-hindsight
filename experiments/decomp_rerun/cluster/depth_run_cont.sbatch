#!/bin/bash
#SBATCH -p gpu --gres=gpu:4 -c 18 --mem=72G -t 08:00:00
# Continuation of depth_run.sbatch (2026-08-28): resumable assert -> f (no-ops if already complete) -> NEW noise arm.
#SBATCH -J decomp-depth-cont -o logs/depth-%j.out
: "${VERICODE_SCRATCH:?set VERICODE_SCRATCH to the directory holding models, environments and caches (ENVIRONMENT.md)}"
# Submit from the SCRATCH run dir (${VERICODE_SCRATCH}/decomp_rerun_run), chained
# AFTER the one-step job:  sbatch -d afterany:<jobid> cluster/depth_run.sbatch
# Runs BOTH depth arms sequentially (assert, then f anchor) with in-loop kimina.
# Node-independent: nothing here touches /home (gpu-node-03 has no /home mount);
# kimina + elan live on scratch, HOME is redirected by cuda_env.sh.
set -euo pipefail
source cluster/cuda_env.sh                 # venv CUDA13 toolchain + HOME=/scratch/.../fakehome
source ${VERICODE_SCRATCH}/vllm-env/bin/activate
export LEAN_SERVER_LEAN_VERSION=v4.29.0 LEAN_SERVER_MAX_REPLS=${LEAN_SERVER_MAX_REPLS:-8}   # cpu_count-1 default = 63 REPLs -> OOM in a cgroup
MODEL=${VERICODE_SCRATCH}/models/Qwen3.6-27B
KIMINA=${VERICODE_SCRATCH}/kimina-lean-server
RUNDIR=$PWD
vllm serve "$MODEL" --served-model-name qwen3.6-27b -tp 4 --max-model-len 131072 --max-num-seqs 64 --disable-custom-all-reduce --port 8017 &
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
for i in $(seq 120); do curl -sf localhost:8000/health >/dev/null && break; sleep 5; done
curl -sf localhost:8000/health >/dev/null || { echo "KIMINA NEVER CAME UP"; exit 1; }
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
KURL=http://localhost:8000; KJ=""
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
for i in $(seq 180); do curl -sf localhost:8017/health >/dev/null && break; sleep 10; done
curl -sf localhost:8017/health >/dev/null || { echo "VLLM NEVER CAME UP"; exit 1; }
cd "$RUNDIR"
python depth/depth_run.py --arm assert --base-url http://localhost:8017 --kimina-url "$KURL" --concurrency 48 --outdir out
python depth/depth_run.py --arm f       --base-url http://localhost:8017 --kimina-url "$KURL" --concurrency 48 --outdir out
python depth/depth_run.py --arm noise   --base-url http://localhost:8017 --kimina-url "$KURL" --concurrency 48 --outdir out
kill $VS $KS 2>/dev/null || true
[ -n "$KJ" ] && scancel $KJ
true
