#!/bin/bash
#SBATCH -p normal -c 16 --mem=32G -t 3:00:00
#SBATCH -J kimina-rebuild -o logs/kimina-rebuild-%j.out
: "${VERICODE_SCRATCH:?set VERICODE_SCRATCH to the directory holding models, environments and caches (ENVIRONMENT.md)}"
# Rebuild the Mathlib dependency packages after the 2026-08-28 00:16 file deletion (batteries oleans gone,
# toolchain bin/ gutted). Uses the scratch elan mirror so the result is valid on /home-less oat nodes.
set -uo pipefail
export HOME=${VERICODE_SCRATCH}/fakehome
export PATH=${VERICODE_SCRATCH}/fakehome/.elan/bin:$PATH
export LEAN_SERVER_LEAN_VERSION=v4.29.0
K=${VERICODE_SCRATCH}/kimina-lean-server
[ -x $HOME/.elan/toolchains/leanprover--lean4---v4.29.0/bin/lake ] || { echo "toolchain bin missing: restore from /home first"; exit 1; }
cd $K/mathlib4
echo "== before: batteries oleans $(find .lake/packages/batteries/.lake/build -name '*.olean' | wc -l), mathlib oleans $(find .lake/build/lib/lean -name '*.olean' | wc -l)"
# Prebuilt dependency archives survive the purge (fresh mtimes); their EXTRACTED files do not. Re-extract first.
PW=.lake/packages/proofwidgets; BT=.lake/packages/batteries
[ -f $PW/.lake/ProofWidgets4.tar.gz ] && tar -xzf $PW/.lake/ProofWidgets4.tar.gz -C $PW/.lake/build && echo "proofwidgets extracted: js $(ls $PW/.lake/build/js | wc -l)"
[ -f $BT/.lake/build.barrel ] && tar -xzf $BT/.lake/build.barrel -C $BT/.lake/build && echo "batteries extracted: oleans $(find $BT/.lake/build -name '*.olean' | wc -l)"
lake exe cache get 2>&1 | tail -3
lake build 2>&1 | tail -8; rc=${PIPESTATUS[0]}; echo "lake build exit=$rc"; [ "$rc" = 0 ] || { echo "LAKE BUILD FAILED"; exit 1; }
echo "== after: batteries oleans $(find .lake/packages/batteries/.lake/build -name '*.olean' | wc -l), mathlib oleans $(find .lake/build/lib/lean -name '*.olean' | wc -l)"
cd $K/repl && lake build 2>&1 | tail -2 && cd $K/mathlib4
printf 'import Mathlib\nopen BigOperators Real Nat Topology Rat\n#check Finset.sum\nexample : (2:ℕ) + 2 = 4 := by norm_num\n' > /tmp/probe_$$.lean
if lake env lean /tmp/probe_$$.lean; then echo "REBUILD PROBE PASS"; else echo "REBUILD PROBE FAIL"; exit 1; fi
rm -f /tmp/probe_$$.lean
# Purge protection: anything unpacked with preserved timestamps must be touched or it vanishes at ~00:16.
find ${VERICODE_SCRATCH}/kimina-lean-server ${VERICODE_SCRATCH}/fakehome/.elan -type f ! -newermt '2026-06-01' -exec touch {} + ; echo "touched old-mtime files"
# snapshot for fast restore next time (toolchain + dependency builds; Mathlib's own oleans come from cache get)
tar -cf ${VERICODE_SCRATCH}/kimina-deps-snapshot.tar -C ${VERICODE_SCRATCH} fakehome/.elan/toolchains -C $K/mathlib4/.lake packages 2>/dev/null && echo "snapshot: ${VERICODE_SCRATCH}/kimina-deps-snapshot.tar ($(du -sh ${VERICODE_SCRATCH}/kimina-deps-snapshot.tar | cut -f1))"
