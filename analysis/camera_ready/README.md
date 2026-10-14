# Camera-ready analyses (VERICODEGEN 2026, submission 149)

Every number added to the manuscript for the camera-ready version is produced by a
script here. All of them read the frozen records of the released artifact and write
to `out/`, which holds the committed reference outputs; none regenerates samples or
re-runs the verifier, except `r20_reverify.py`, which only re-reads verdicts and
never rewrites a record file.

The artifact is found automatically (`artifact_path.py`): `paper/artifact/` in the
paper repository, the repository root in the public release, or `$VERICODE_ARTIFACT`.
Scripts that use the main reproduction's outputs read them from the artifact's
`generated/`, or from `$GENERATED`. Run everything and compare with the reference:

```bash
python scripts/reproduce.py                          # from the artifact root, once
python analysis/camera_ready/run_all.py --check      # add --skip-slow to skip r6/token_matched
```

## Environment

The artifact's `requirements-analysis.txt` plus `tokenizers` is enough:

```bash
python3 -m venv .venv && ./.venv/bin/pip install -r requirements-analysis.txt tokenizers
```

`r6.py` additionally needs the vendored model tokenizers in the artifact's
`tokenizers/{qwen,deepseek}/` (restored by `scripts/unpack_data.py`).

## Scripts

| Script | Tracker row | Produces |
|---|---|---|
| `as_ce.py` | A1, C3, C4 | `out/sequential_ce.json` — the artifact's `analyze_sequential.py` with three contrasts added to `effects()`: `f-I` (manuscript `E-I`), `CE_m-I`, and `C-f` (manuscript `C-E`). Needs `common.py` beside it. |
| `ce.py` | C4 | Prints `C-E` and `CE-E` for both models from the generic `Random(0)` matrix. Kept as the record of a wrong turn: for Qwen these contrasts are assigned from a *replayed* stream (`Random(22072606)`), not this one, so use `as_ce.py` for the reported value. |
| `r1.py` | R1 | `out/r1_deepseek_identical.json` — DeepSeek one-step matched contrasts excluding the 37 byte-identical matched donors. |
| `r4.py` | R4, D4, L3 | `out/r4_replacement.json` — characterizes the `F_C_other` replacement predecessor (donor class agreement, length ratios) and the replacement effect on matched subsets. |
| `r6.py` | R6, W1, L2 | `out/r6_tokens.json` — input-token cost of sequential repair vs memoryless retrying, and the equal-input-token comparison. Re-renders adaptive prompts with the runner's own `render()`/FIFO rule. Slow (~10 min). |
| `r8.py` | R8, W2 | `out/r8_conditional.json` — first-failure-conditional endpoints for both models. |
| `r9.py` | R9, R11, L4 | `out/r9_loo.json` — leave-one-problem-out ranges and achieved interval half-widths for every `k=64` endpoint. |
| `r10.py` | R10 | `out/r10_signflip.json` — exact paired randomization tests over all 2^11 sign assignments. |
| `r2.py` | R2, D5 | `out/r2_match_quality.json` — one-step diagnostic match quality (token-length ratio, lexical Jaccard, byte-identity) and the correct-vs-matched contrasts re-estimated on the best-matched subsets. |
| `r13.py` | R13 | `out/r13_multipredecessor.json` — attempt-2 success for `C` vs `E` across freshly sampled predecessors. Floor-limited on these banks; kept because that is the honest answer. |
| `r13b.py` | R13, L1 | `out/r13_multipredecessor_behavior.json` — the same design with a copying outcome, which does resolve. Hierarchical bootstrap: theorems, then trajectories. |
| `r16.py` | R16 | `out/r16_nearcopy.json` — near-copy measures over every retained one-step sample on the full bank, not conditioned on success. |
| `r16_seq.py` | R16 (sequential) | `out/r16_sequential.json` — **not used in the paper.** It measures recurrence against *normalized* proof bodies, whereas the released sequential repetition figures (`generated/sequential_repetition.json`, via `artifact/analysis/supporting.py`) compare *raw* saved code. Its 81.8% is therefore not comparable with the paper's 75.5% for the same arm. To publish it, switch to the raw-string convention first. |
| `r18.py` | R18 | `out/r18_shared_subset.json` — both models restricted to the 137 shared one-step problems. |
| `r20_reverify.py` | R20 | `out/r20/r20_shard_*.jsonl` — one line per re-verified row. Report-only. |
| `r20_array.sbatch` | R20 | Launches the above as a 4-way array, each shard starting its own Kimina server. Configured by `KIMINA_DIR`, `KIMINA_PYTHON`, `VERICODE_ARTIFACT`. |
| `parseable.py` | `tab:parseable`, App. parseable | `out/parseable.json` — the main one-step contrasts, pooled precision estimate, and benefit bounds on the problems whose statements the verifier can parse (rule: no `∑ x in` / `∏ x in` binder). |
| `per_sample.py` | per-sample paragraph | `out/per_sample.json` — per-sample pass rates against instruction-only retrying, paired by problem. |
| `lean3.py` | `tab:lean3`, drift paragraph 2 | `out/lean3.json` — Lean 3 predecessor split, `begin` rates, copying, and the sequential Lean 3 shares. |
| `loo_table.py` | `tab:loo` | `out/loo_table.json` — full leave-one-out table including the sign and interval-exclusion counts (reduced banks: `Random(1)`, 10,000 resamples). |
| `multiplicity.py` | multiplicity paragraph, `tab:claims` | `out/multiplicity.json` — Holm and BH within model over the 15 one-step comparisons (bootstrap p) and the sequential tables (exact randomization p). Needs `generated/`. |
| `build_attempt_tokens.py` | `tab:token_matched` | `data/sequential_attempt_tokens.csv.xz` — one row per sequential attempt and memoryless draw with prompt/response token counts. Built from the ten full traces the release omits (`FULL_TRACES=<dir>`; they are on the authors' working branch), so it is not rerun by `run_all.py`; it checks every row against the released compact records. |
| `token_matched.py` | `tab:token_matched`, token paragraphs | `out/token_matched.json` — fixed per-trajectory token budgets with censoring limits, paired problem bootstrap, exact randomization p, from the table above. |
| `fetch_mathlib.sh`, `drift_names.sbatch`, `dump_names.lean` | `tab:drift` | Fetch the verifier's Mathlib (`8a17838`), DeepSeek-Prover-V1.5's pinned Mathlib (`xinhjBrant/mathlib4@2f65ba7f`, Lean v4.9.0-rc1) and the intermediate releases for deprecation records into `$DRIFT_CACHE`, then dump both environments' constant names. Building the old library takes a CPU node. |
| `drift.py` | `tab:drift`, drift paragraph 1 | `out/drift.json` — classifies the first unknown identifier of each predecessor and failed-sample diagnostic (rules in its docstring). Run only when `$DRIFT_CACHE` exists. |
| `run_all.py` | — | Runs every script above in order; `--check` compares with the committed `out/`. |

## Conventions these scripts preserve

The released analysis assigns bootstrap intervals from several different index
matrices, and they are **not** interchangeable:

- one-step contrasts: percentile problem bootstrap, 10,000 resamples, seed 20260906
- Qwen sequential: `python_indices(random.Random(0), 10000, 11)`, sorted endpoints
- DeepSeek sequential: `default_rng(20260913).integers(0, 11, (200000, 11))`
- the registered `CE_m` family, both models: 10,000 resamples, sorted endpoints
- Qwen cross-arm pairs among `{C, CE, CE_s, blind_I, f}`: a *replayed* stream,
  `random.Random(22072606)`, consumed in the order `analyze_memI.py` used

`as_ce.py` adds contrasts without disturbing any of this: the replay loops consume
their indices unconditionally, so adding a key to `effects()` shifts nothing. The
check that this holds is that all 39 pre-existing contrasts in
`artifact/generated/sequential.json` reproduce bit-identically — verify before
trusting any new number:

```bash
./.venv/bin/python as_ce.py
python3 - <<'EOF'
import json
rel = json.load(open('<artifact>/generated/sequential.json'))
new = json.load(open('out/sequential_ce.json'))
bad = [(m, k) for m in rel for k, v in rel[m]['endpoint_contrasts'].items()
       if [round(x, 6) for x in v['ci95_pp']] != [round(x, 6) for x in new[m]['endpoint_contrasts'][k]['ci95_pp']]]
print('unchanged:', not bad, bad)
EOF
```

## R20 re-verification

`out/r20/` holds the verdict for all 7,217 PASS rows retained in the artifact:
7,217 confirmed, 0 verdict changes, 0 verifier failures, 33 files.

This extends the original correction pass, which covered 3,197 rows. The gap was
a schema one: `experiments/decomp_rerun/reverify_pass.py` selects rows by
`r["status"] == "PASS"`, but the memoryless-`I` depth records store their verdict
under `verification_result` and their proof under `extracted_proof`, so those arms
were skipped. `r20_reverify.py` handles both conventions.

To re-run, from a node with the Kimina stack (not a login node):

```bash
KIMINA_DIR=... KIMINA_PYTHON=... VERICODE_ARTIFACT=... sbatch r20_array.sbatch
```

A static grep for `sorry` over the retained proofs flags one row
(`out_confirm_error/b2`, arm `F_E`, `mathd_numbertheory_765`, sample 3, line 123).
Lean passes it correctly: the token sits inside a `try` block that Lean's
unreachable-tactic linter reports as never executed, so it is never elaborated and
contributes no `sorryAx`. A proof whose body is only `sorry` is rejected by the
same screen. This is recorded here rather than in the paper because nothing in the
artifact runs such a grep.
