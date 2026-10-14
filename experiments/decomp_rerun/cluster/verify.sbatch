#!/bin/bash
#SBATCH --job-name=decomp-verify
#SBATCH --partition=normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=logs/verify-%j.out
# Prereq (one-time, ~1-2h): cd <repo>/archive/kimina-lean-server && pip install -e . && bash setup.sh
set -euo pipefail
: "${BUNDLE:?}"; : "${KIMINA_DIR:?path to kimina-lean-server checkout}"
cd "$KIMINA_DIR" && python3 -m server & SRV=$!
for i in $(seq 60); do curl -s localhost:8000/health | grep -q -i ok && break; sleep 5; done
python3 "$BUNDLE/verify_run.py" --kimina-url http://localhost:8000 --outdir "$BUNDLE/out"
kill $SRV
