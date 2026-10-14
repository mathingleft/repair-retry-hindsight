#!/bin/bash
#SBATCH -p normal --gres=gpu:1 -c 2 --mem=4G -t 0:05:00
#SBATCH -J probe-gpu1 -o logs/probe-%j.out
: "${VERICODE_SCRATCH:?set VERICODE_SCRATCH to the directory holding models, environments and caches (ENVIRONMENT.md)}"
set -x
hostname; date
nvidia-smi -L
ls -d ${HOME}/research-project ${VERICODE_SCRATCH}/models/Qwen3.6-27B
module load cuda/12.9.0 && echo "module ok"
source ${VERICODE_SCRATCH}/vllm-env/bin/activate && python -c "import vllm,torch;print(vllm.__version__, torch.cuda.is_available())"
