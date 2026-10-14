# Pre-registration: code-bearing class- and length-matched wrong-diagnostic arm `CE_m` (Qwen3.6-27B and DeepSeek-Prover-V2-7B)

Written 2026-09-12 21:35 PT, before any attempt of this arm was generated (jobs submitted 21:32 PT: Qwen 1720195
with timeout-only continuation 1720196; prover 1720197 with continuation 1720198; all four queued, no GPU free at
submission). Motivation: `CE_s` (PREREG_code_arms_depth_2026-09-07) draws its wrong diagnostic from a fixed partner
problem, position-matched, with no control of the error's class or length, while `E_s` (PREREG_E_s_seq_2026-09-08)
is class- and length-matched. The interaction `(CE - CE_s) - (f - E_s)` therefore mixes two matching policies. A
an earlier depth `CE_shuf` campaign is class/length-matched but uses the one-step block
wording and an exact 7,000-token cap, so it cannot sit beside the same-stack arms here. `CE_m` closes that gap on this
config, on both models.

## What is run
- **Arm `CE_m`** (`depth/code_arms_run.py --arm CE_m`): identical to `CE` in every respect (same bank, 10 trajectories
  x <= 64 attempts, early stop at first PASS, attempt 1 = bare prompt, `depth_run.HEADER` + `## Previous attempt N`
  blocks with the archived HPC `Proof attempt:` / `Lean compiler feedback:` lines + `depth_run.task(p)`, FIFO eviction beyond
  `PROMPT_TOKEN_CAP` (chars/3.5 > 7000), `seed_for` grid, GEN decoding, `depth_run.verify`) except that the text after
  `Lean compiler feedback:` is a WRONG diagnostic from another problem, matched to the trajectory's own error at that
  attempt by the `E_s` policy (`MatchedErrors`, the one-step `F_E_match` rule): same coarse class
  (`build_match.classify`), token length as close as possible (cost = |log(tok_src / tok_own)|, model tokenizer,
  identical text penalised +50), never the same problem, one use per source row per trajectory, ties broken by
  `random.Random(seed_for(pid, t, j))`, pooled non-syntax family when a class has < 5 cross-problem candidates.
  The matched error for own attempt j is drawn once and stays fixed in later prompts of that trajectory.
- **Pool**: the CE arm's own FAIL errors on the same model (`out/depth_CE.jsonl`, 4,361 rows for Qwen;
  `out_dsp/depth_CE.jsonl`, 5,817 rows for the prover): errors produced under the same code-bearing regime, stack and
  verifier, the exact counterpart of `E_s` drawing from the `f` pool. Per block the source row, classes, pool, token
  lengths, cost and tie count are logged (`shuf_sources`).
- **Rendered prompt check**: given the same (code, error) pairs, the `CE_m` prompt is byte-identical to the `CE`
  prompt (tested before submission).
- **Serving**: Qwen as every Qwen depth arm (bf16, TP2, `cluster/depth_codearm_g2.sbatch`, in-loop kimina); prover
  as the completed prover arms (fp8 KV, TP2, `cluster/dsp_depth.sbatch`), the deviation already recorded in
  PREREG_dsp_depth. Job id per row; timeouts resume from the rows file with the same seeds.
- **Logged per attempt**: as for C / CE / CE_s / E_s. Depth-format mirror `depth_CE_m.jsonl` at the end.

## Estimators and endpoints
Same estimators and paired problem bootstrap (10,000 resamples, percentile 95% CI, k in {1, 2, 4, 8, 16, 32, 64},
paired sign test alongside) as the earlier preregs; each model on its own 11 problems, no pooling across models.
- **Primary** (per model): `CE - CE_m` at k = 64 — the value of the error's problem-specific content beside the
  model's own failed code, under matched class and length.
- **Secondary** at k = 64: `CE_m - memI`; `CE_m - C` (does a matched wrong error add anything to bare code);
  `CE_m - CE_s` (matching policy sensitivity); the symmetric interaction `(CE - CE_m) - (f - E_s)`
  (`analyze_interaction.py` with `CE_m` as the code-side wrong-diagnostic arm); every k in the grid.
- **Exploratory**: proof-repeat rate; match quality; token cost; infrastructure-error counts.

## Result language (fixed in advance)
For `CE - CE_m` at k = 64: CI entirely above 0 -> "the error's content helps beside the own failed code"; spans 0 ->
"not established"; entirely below 0 -> "the matched wrong error does better than the true one". For the interaction:
entirely above 0 -> "diagnostic content is worth more beside the own failed proof"; spans 0 -> "not established";
below 0 -> "worth less".

## Deviations stated in advance
- `CE_m` errors come from the CE pool, `E_s` errors from the `f` pool; each is the own-regime pool for its side.
- The rows of the completed arms were generated before this arm; the pool is therefore fixed and public before any
  `CE_m` draw. Online matching cannot enforce balanced global reuse; per-trajectory no-reuse is the substitute.

## Addendum 2026-09-12 23:50 PT (scheduling only)
No GPU was free at submission and the 8-hour jobs (1720195-1720198) could not backfill; cancelled unstarted at 23:37 and
replaced by chains of short singleton-named jobs that resume from the rows file: Qwen `qwen-CEm` 1720308-1720311
(gpu:2 TP2, 3 h each); prover `dsp-CEm` 1720312/1720313 (gpu:2 TP2 fp8, 3 h) plus a separately named `dsp-CEm1`
1720316/1720317 (gpu:1 TP1 fp8, 5 h) so either shape of free GPU is taken; the prover launcher gained an arm lock so
only one copy ever owns the arm. Rows carry the job id; serving shape (TP1 vs TP2) is recoverable from the job log.
