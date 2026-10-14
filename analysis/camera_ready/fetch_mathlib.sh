#!/usr/bin/env bash
# Fetch the two Lean libraries that drift.py classifies unknown identifiers against,
# plus the intermediate Mathlib releases whose deprecation records name the renames.
#
#   bash fetch_mathlib.sh [CACHE]        (default CACHE=$DRIFT_CACHE, else ./mathlib-drift-cache)
#
# new/  the verifier's library: Mathlib 8a178386 on Lean v4.29.0 (artifact
#       environments/revisions.json). A built copy is reused if present.
# old/  DeepSeek-Prover-V2's evaluation-era library: the mathlib4 submodule pinned by
#       deepseek-ai/DeepSeek-Prover-V1.5 (xinhjBrant/mathlib4 @ 2f65ba7f, 2024-08-07,
#       Lean v4.9.0-rc1), the public Lean 4.9 environment DeepSeek's provers are evaluated in.
# releases/  Mathlib/ sources of upstream tags v4.10.0 ... v4.28.0, for deprecated aliases.
#
# Then dump each environment's constant names (dump_names.lean, run by drift.py's
# sbatch helper or by hand):  lake env lean --run dump_names.lean > names.txt
set -euo pipefail
CACHE=${1:-${DRIFT_CACHE:-mathlib-drift-cache}}
mkdir -p "$CACHE/releases"
export PATH="$HOME/.elan/bin:$PATH"

NEW_REV=8a178386ffc0f5fef0b77738bb5449d50efeea95
OLD_REPO=https://github.com/xinhjBrant/mathlib4.git
OLD_REV=2f65ba7f1a9144b20c8e7358513548e317d26de1

# new: reuse the verifier build if it is at the pinned revision, else clone and fetch the cache
LOCAL=${VERICODE_MATHLIB:-}   # optional: an existing build of the verifier's Mathlib to reuse
if [ ! -e "$CACHE/new" ]; then
  if [ -d "$LOCAL/.lake/build" ] && [ "$(git -C "$LOCAL" rev-parse HEAD)" = "$NEW_REV" ]; then
    ln -s "$LOCAL" "$CACHE/new"
  else
    git clone --filter=blob:none https://github.com/leanprover-community/mathlib4.git "$CACHE/new"
    git -C "$CACHE/new" checkout -q "$NEW_REV"
    (cd "$CACHE/new" && lake exe cache get)
  fi
fi

# old: pinned DeepSeek environment; prebuilt oleans from the Mathlib cache
if [ ! -d "$CACHE/old/.git" ]; then
  git init -q "$CACHE/old"
  git -C "$CACHE/old" remote add origin "$OLD_REPO"
  git -C "$CACHE/old" fetch -q --depth 1 origin "$OLD_REV"
  git -C "$CACHE/old" checkout -q FETCH_HEAD
fi
(cd "$CACHE/old" && lake exe cache get)

# intermediate releases: Mathlib/ sources only
for v in $(seq 10 28); do
  t=v4.$v.0; d="$CACHE/releases/$t"
  [ -d "$d" ] && continue
  mkdir -p "$d"
  curl -sfL "https://codeload.github.com/leanprover-community/mathlib4/tar.gz/refs/tags/$t" \
    | tar -xz -C "$d" --strip-components=1 --wildcards '*/Mathlib/*' \
    || { echo "missing tag $t" >&2; rmdir "$d" 2>/dev/null || true; }
done
echo "cache ready: $CACHE"
