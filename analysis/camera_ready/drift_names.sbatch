#!/bin/bash
# Build the 2024 library (no prebuilt cache exists for the DeepSeek fork) and dump the
# constant names of both environments for drift.py.  sbatch drift_names.sbatch [CACHE]
#SBATCH --job-name=drift-names
#SBATCH --partition=normal
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=drift_names_%j.log
set -euo pipefail
CACHE=${1:-${DRIFT_CACHE:-mathlib-drift-cache}}
HERE=${SLURM_SUBMIT_DIR:-$(pwd)}
export PATH="$HOME/.elan/bin:$PATH"
for v in old new; do
  cd "$CACHE/$v"
  [ "$v" = old ] && lake build Mathlib   # no-op once built
  DRIFT_OUT="$CACHE/$v.names.tsv" lake env lean "$HERE/dump_names.lean"
  wc -l "$CACHE/$v.names.tsv"
done
