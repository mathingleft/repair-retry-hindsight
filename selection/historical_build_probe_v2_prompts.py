#!/usr/bin/env python3
"""Materialize the teacher-probe v2 datasets: V1 vs V4 repair prompts, one failure
per problem (plans/sprightly-splashing-haven.md).

Probe v1 (experiments/teacher_probe.py) was underpowered at the unit that matters:
its 60 rollouts per substrate were repeated failures of the same 12 FIFO problems,
so the problem-level N was 12 and qwen4b's apparent V4 win decomposed into one
problem. v2 is the confirmatory run — wider, not longer: ONE attempt-failure per
problem across the whole adaptive-bench kept set (~180 problems/substrate), single
preregistered hypothesis V1 (incumbent header) vs V4 (imperative + failed proof).

Per the materializer pattern (an ablation arm = a {id, statement, prompt} dataset +
the EXISTING eval), this is a pure data step — no new eval loop. Each arm's rows
carry a baked per-row prompt; the sweep runs them with `prompts: {baked: null}` so
eval.py honors the row prompt (same mechanism as the playbook replay).

What it reads
-------------
$SCRATCH/eval/<bench>/kept.jsonl              the bench's problem set (id, statement,
                                              pass_rate measured at up-to-1024 samples)
$SCRATCH/eval/<bench>/round0_k16_verified.jsonl  16 verified rollouts/problem; FAIL rows
                                              carry the compiler `error` + `lean_code`

Selection: per kept problem, the lowest-rollout_idx round-0 row with status==FAIL,
nonempty error, truncated==false, a REAL extracted proof, AND a materialized prompt
that fits the context window. Two kinds of synthetic-fallback row are skipped, because
there the "failed proof" / "error" is OUR placeholder, not a compiler signal about
anything the model actually wrote (memory: synthesized fallbacks must not look like
real model output):
  - lean_code is the `extract_lean_code_no_body*` undefined-identifier marker;
  - lean_code is the commented-out-theorem fallback (starts with `-- Error:`, error
    "unexpected end of input") — the whole proof is a comment block.
Length guard: the chosen failure must materialize under MAX_PROMPT_TOKENS on BOTH
arms (a handful of pathological errors — recursion-depth dumps — run to ~90k tokens
and would 400 the whole eval at the 40960 context; one such request sank the first
submission). Prompt length is the SAME error text fed to V1 and V4, so selecting on
it cannot bias the V1-vs-V4 contrast. We take the lowest-rollout_idx failure that
fits rather than dropping the problem. Exactly one row per problem — problems are the
independent units, which kills the clustering that sank v1. Zero-pass problems are
kept (a header unlocking one would be the most valuable possible signal); the
manifest records pass_rate so the report can stratify.

What it writes
--------------
$SCRATCH/data/probe_v2_<substrate>_v1.jsonl       arm rows {id, statement, prompt}
$SCRATCH/data/probe_v2_<substrate>_v4.jsonl
$SCRATCH/data/probe_v2_<substrate>_manifest.jsonl {id, bucket, pass_rate, rollout_idx,
                                                   error, lean_code} for the report
                                                   (copy-rate + bucket breakdown).

Usage
-----
    python3 experiments/data-local/build_probe_v2_prompts.py --substrate qwen4b
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "experiments"))
sys.path.insert(0, str(PROJECT_DIR / "experiments" / "analysis"))

from tokenizers import Tokenizer                                  # noqa: E402
from configs.teacher_probe_variants import VARIANTS                # noqa: E402
from failure_taxonomy import classify                              # noqa: E402
from globals.utils import (                                        # noqa: E402
    build_student_prompt, prune_errors, read_jsonl, write_jsonl, log,
)

SCRATCH = os.environ.get("SCRATCH", ".")

# substrate → (adaptive-bench dir, local tokenizer.json). The corrected nonthinking
# benches; the tokenizer is the model's real BPE so the length guard counts the
# exact tokens the server will see, not a char approximation.
BENCHES = {
    "qwen4b":  "adaptive-qwen35-4b-1024-nonthinking",
    "qwen27b": "adaptive-qwen36-27b-1024-nonthinking",
}
TOKENIZER = {
    "qwen4b":  os.environ.get("VERICODE_SCRATCH", ".") + "/models/Qwen/Qwen3.5-4B/base/tokenizer.json",
    "qwen27b": os.environ.get("VERICODE_SCRATCH", ".") + "/models/Qwen/Qwen3.6-27B/base/tokenizer.json",
}
ARMS = ("v1_incumbent", "v4_imperative_proof")

# Length guard: the eval serves at 40960 context and asks for 16384 output tokens.
# Keep BOTH arms' prompts under (context - output) so every repair attempt gets the
# full untruncated 16384 output budget — that keeps the V1/V4 pair symmetric (no
# arm-dependent output clamp) and avoids the 400-BadRequest that sank the first
# submission. The single user message is wrapped by the Qwen chat template
# (<|im_start|>user\n … <|im_end|>\n<|im_start|>assistant\n — vLLM applies it, no
# system prompt; see vllm_server._chat_one_async); we tokenize that exact wrapped
# string. SCAFFOLD_RESERVE covers the disable_thinking empty-think scaffold the
# template may prepend to the assistant turn (a few tokens), so the cap stays a true
# upper bound on the server-side prompt length.
CHAT_WRAP = "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n"
SCAFFOLD_RESERVE = 64
MAX_PROMPT_TOKENS = 40960 - 16384 - SCAFFOLD_RESERVE


def materialize(statement: str, error: str, lean_code: str, variant: str) -> str:
    """Header + bare student prompt — byte-identical assembly to probe v1
    (teacher_probe.materialize): concatenation-built (no .format over proof text —
    brace hazard), error text through the same prune_errors dedup training uses."""
    err = prune_errors([{"status": "FAIL", "error": error}])
    header = VARIANTS[variant](err, lean_code)
    return header + build_student_prompt(statement)


def build(substrate: str) -> None:
    bench = Path(SCRATCH) / "eval" / BENCHES[substrate]
    kept = read_jsonl(bench / "kept.jsonl")
    tok = Tokenizer.from_file(TOKENIZER[substrate])
    log(f"[probe-v2] {substrate}: {len(kept)} kept problems from {bench.name}")

    # Gather eligible failures per kept problem, then pick the lowest-rollout_idx one
    # that materializes under the length cap on BOTH arms (see module docstring).
    kept_ids = {r["id"] for r in kept}
    candidates: dict[str, list[dict]] = {}
    for r in read_jsonl(bench / "round0_k16_verified.jsonl"):
        if (r["id"] not in kept_ids or r["status"] != "FAIL"
                or not (r.get("error") or "").strip() or r.get("truncated")):
            continue
        code = r.get("lean_code") or ""
        # Synthetic-fallback rows — the "proof"/"error" is our placeholder, not the
        # model's (see module docstring; memory: synthesized-fallbacks constraint).
        if "extract_lean_code_no_body" in code or code.lstrip().startswith("-- Error:"):
            continue
        candidates.setdefault(r["id"], []).append(r)

    def prompt_tokens(content: str) -> int:
        return len(tok.encode(CHAT_WRAP % content).ids)

    def fits(r: dict) -> bool:
        return all(prompt_tokens(materialize(r["statement"], r["error"],
                                             r["lean_code"], a)) <= MAX_PROMPT_TOKENS
                   for a in ARMS)

    chosen: dict[str, dict] = {}
    n_oversized = 0
    for pid, rows in candidates.items():
        for r in sorted(rows, key=lambda x: x["rollout_idx"]):
            if fits(r):
                chosen[pid] = r
                break
        else:
            n_oversized += 1   # every eligible failure for this problem was too long

    pass_rate = {r["id"]: r["pass_rate"] for r in kept}
    arm_rows: dict[str, list[dict]] = {a: [] for a in ARMS}
    manifest: list[dict] = []
    for pid in sorted(chosen):
        r = chosen[pid]
        bucket, _ = classify(r["status"], r["error"])
        for arm in ARMS:
            arm_rows[arm].append({
                "id":        pid,
                "statement": r["statement"],
                "prompt":    materialize(r["statement"], r["error"],
                                         r["lean_code"], arm),
            })
        manifest.append({
            "id":          pid,
            "bucket":      bucket,
            "pass_rate":   pass_rate[pid],
            "rollout_idx": r["rollout_idx"],
            "error":       r["error"],
            "lean_code":   r["lean_code"],
        })

    n_skipped = len(kept) - len(chosen)
    if not manifest:
        sys.exit(f"[probe-v2] FATAL: {substrate} yielded no eligible failures")

    out_root = Path(SCRATCH) / "data"
    short = {a: a.split("_")[0] for a in ARMS}                      # v1 / v4
    for arm in ARMS:
        write_jsonl(out_root / f"probe_v2_{substrate}_{short[arm]}.jsonl", arm_rows[arm])
    write_jsonl(out_root / f"probe_v2_{substrate}_manifest.jsonl", manifest)

    by_bucket = Counter(m["bucket"] for m in manifest)
    n_zero = sum(1 for m in manifest if m["pass_rate"] == 0)
    log(f"[probe-v2] {substrate}: {len(manifest)} problems materialized "
        f"({n_skipped} kept problems skipped: no eligible round-0 failure, or all "
        f"eligible failures exceeded the {MAX_PROMPT_TOKENS}-token cap — "
        f"{n_oversized} of the latter)")
    log(f"[probe-v2]   buckets: {dict(by_bucket.most_common())}")
    log(f"[probe-v2]   zero-pass@1024: {n_zero}  nonzero: {len(manifest) - n_zero}")
    log(f"[probe-v2]   wrote probe_v2_{substrate}_{{v1,v4,manifest}}.jsonl -> {out_root}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--substrate", required=True, choices=sorted(BENCHES))
    build(p.parse_args().substrate)


if __name__ == "__main__":
    main()
