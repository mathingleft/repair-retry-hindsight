#!/bin/bash
#SBATCH -p normal --gres=gpu:1 -c 4 --mem=12G -t 0:10:00 -x gpu-node-03
#SBATCH -J probe-fi -o logs/probe-%j.out
: "${VERICODE_SCRATCH:?set VERICODE_SCRATCH to the directory holding models, environments and caches (ENVIRONMENT.md)}"
set -euo pipefail
source cluster/cuda_env.sh
source ${VERICODE_SCRATCH}/vllm-env/bin/activate
hostname; which nvcc; nvcc --version | tail -1
python - <<'PY'
import torch, flashinfer
print("torch", torch.__version__, "flashinfer", flashinfer.__version__, "dev", torch.cuda.get_device_name(0))
logits = torch.randn(4, 1000, device="cuda")
out = flashinfer.sampling.top_k_top_p_sampling_from_logits(logits, top_k=50, top_p=0.9)
print("SAMPLING OK", out.shape)
PY
