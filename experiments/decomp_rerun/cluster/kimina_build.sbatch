#!/bin/bash
#SBATCH -p normal -c 16 --mem=32G -t 4:00:00
#SBATCH -J kimina-build -o build.log
: "${VERICODE_SCRATCH:?set VERICODE_SCRATCH to the directory holding models, environments and caches (ENVIRONMENT.md)}"
set -euo pipefail
export LEAN_SERVER_LEAN_VERSION=v4.29.0
python3 -m venv ${VERICODE_SCRATCH}/kimina-env
${VERICODE_SCRATCH}/kimina-env/bin/pip install -U pip
${VERICODE_SCRATCH}/kimina-env/bin/pip install -e .
bash setup.sh
