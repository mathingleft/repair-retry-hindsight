# Pre-registration: code-bearing sequential arms C / CE / CE_s on the shared cluster stack

Written 2026-09-07 19:40 PT, before any attempt of these arms was generated (submitted 19:41 PT as jobs 1712685 C, 1712686 CE, 1712687 CE_s. At 19:46 PT, 73 C rows in, C and CE were found sharing one node with a fixed kimina port, C's kimina having failed to bind; 1712685 and 1712687 were cancelled and resubmitted with per-arm ports as 1712690 C (resuming its 73 rows, same seeds) and 1712691 CE_s afterok:1712686. At 19:50 PT CE was found to have lost the vLLM bind on 8017 the same way (C's server had it), so after the cancel every CE request failed: 1712686 and 1712691 were cancelled, the 48 CE rows with gen_status error were stripped (backup CE_rows.jsonl.bak-porterr-1712686; 64 ok rows kept, trajectories kept contiguous from attempt 1), and CE / CE_s were resubmitted with per-arm ports as 1712695 and 1712696 afterok:1712695. Rows carry the job id that produced them.). Motivation: the preregistered
`CE - I` endpoint (`PREREG_memoryless_I_2026-09-06.md`) cannot be computed on one stack. The archived HPC `gf` (= CE)
trajectories are intact but on a different serving stack and not mounted here; `g` (= C) was purged on
2026-08-14; a shuffled-error sequential arm was never run. These three arms are therefore re-run here on the exact
stack, problems, seeds, verifier and scaffold of `memoryless_I` and the local depth arms (`assert`, `f`, `noise`,
`blind*`), so every contrast below is same-stack.

## What is run
- **Runner**: `experiments/decomp_rerun/depth/code_arms_run.py`, one arm per gpu:2 / TP2 job
  (`cluster/depth_codearm_g2.sbatch`), in-loop kimina verification, resumable.
- **Problems**: the same 11 theorems (`depth/depth_bank.jsonl`). All analysed, none dropped.
- **Protocol**: 10 trajectories x <= 64 attempts, early stop at the first PASS (the depth-arm protocol; the
  `blind*` and `memoryless_I` arms are the no-early-stop nulls). Attempt 1 = the bare task prompt
  (`Complete the following Lean 4 code:` ...), as in every sequential arm here and on archived HPC.
- **Seeds**: `depth_run.seed_for(problem, trajectory, attempt)`, the grid shared by every depth arm and memI.
- **Decoding / serving**: `max_tokens 16384, temperature 0.7`, `top_k 20 / top_p 0.95` from
  `generation_config.json`, thinking off; Qwen3.6-27B bf16, vLLM 0.28.0, torch 2.13.0+cu130, transformers 5.16.1,
  L40S TP2, `--max-num-seqs 64`, concurrency 48. The stack string is printed in every job log.
- **Verification**: `depth_run.verify` (kimina, Lean v4.29.0, 120 s timeout, standard preamble incl.
  `maxHeartbeats 400000`, sorry/admit screen `SORRY_RE`, transient errors retried 3x). Identical to the
  local depth arms. Infrastructure failures are recorded in `infrastructure_status`, separately from the verdict,
  and count as FAIL for the trajectory (as in the depth arms).
- **Prompt scaffold** (attempts > 1): `depth_run.HEADER` + one `## Previous attempt N` block per prior failed
  attempt, oldest first, FIFO-evicted beyond `PROMPT_TOKEN_CAP` (chars/3.5 > 7000), + `depth_run.task(p)`.
  Byte-identical to the local `f` arm except for block content:
  - `f`    (existing): `Lean compiler feedback:\n{own error}`
  - `C`    : `Proof attempt:\n```lean4\n{own failed code}\n````
  - `CE`   : `Proof attempt:\n```lean4\n{own failed code}\n````  +  `Lean compiler feedback:\n{own error}`
  - `CE_s` : the CE block with the error text replaced by another problem's error (below)
  The proof/feedback lines are the archived HPC `gf` lines verbatim (`solvers/fifo_refiner.render_fifo_prompt`); the
  header sentence is the depth arms' HEADER rather than the archived HPC `REFINE_HEADER`, so `C/CE/CE_s - f` isolate
  block content. `{own failed code}` = the verifier's input for that attempt (`lean_extract.extract_lean_code`
  splice), what the archived HPC runner stored as `lean_code`. Error text capped at 1200 chars (depth_run rule).
- **CE_s error source**: partner problem = seeded draw with `random.Random(22072605)` (the `arms.shuffled_partner`
  constant) over the 11 sorted ids, never itself. Fixed table (problem -> partner):
  algebra_amgm -> mathd_numbertheory_234; amc12_2000_p20 -> imo_1983_p6; amc12a_2009_p7 -> mathd_numbertheory_234;
  amc12a_2013_p4 -> imo_1962_p2; imo_1962_p2 -> imo_1983_p6; imo_1963_p5 -> amc12a_2009_p7;
  imo_1983_p6 -> amc12a_2009_p7; mathd_algebra_276 -> mathd_numbertheory_234; mathd_algebra_332 ->
  mathd_numbertheory_314; mathd_numbertheory_234 -> amc12a_2013_p4; mathd_numbertheory_314 -> algebra_amgm.
  For the block describing prior attempt j of trajectory t: the CE arm's error at (partner, t, j) if that row
  exists and failed; else a `random.Random(seed_for(pid, t, j))` draw from the partner's CE errors; else from the
  partner's local `f`-arm errors. The source is logged per block. CE_s is therefore launched after CE completes.
- **Logged per attempt** (`out_codearms/<arm>_rows.jsonl`): problem_id, trajectory_id, attempt_id, seed,
  user_prompt, full_prompt (chat-templated string sent), history (rendered blocks), window_len, prompt/response
  token counts (server usage + tokenizer recount), raw_response, finish_reason, gen_status, extracted_proof,
  verification_result, compiler_output, infrastructure_status, timings, job id. Depth-format mirror
  `out/depth_<arm>.jsonl` written at the end for the existing analysis scripts.

## Estimators and uncertainty
As in the memI prereg: `p_arm,i(k)` = fraction of the 10 trajectories of problem i with a PASS by attempt k;
memI `p_I,i(k)` = pooled without-replacement estimator over 640 draws; `Delta_k` = mean over the 11 problems of
the difference; paired problem bootstrap, 10,000 resamples, percentile 95% CI; k in {1, 2, 4, 8, 16, 32, 64}.
Between two sequential arms the same problem-bootstrap on `p_A,i(k) - p_B,i(k)`; the paired sign test with
problem as unit is reported alongside for continuity with the one-step analyses.

## Endpoints
- **Primary**: `CE - I` at k = 64 (now same-stack; supersedes the cross-stack version in the memI prereg).
- **Secondary** at k = 64: `C - I`, `CE_s - I`, `CE - C` (own error's value beside own code), `CE - CE_s`
  (error content beside code, the depth twin of `F_CE` vs `F_CE_shuf`), `CE - f` (code presence beside the error),
  `C - f`.
- **Exploratory**: every k in the grid; conditional-on-attempt-1-failure sensitivity; per-problem
  heterogeneity; byte-identical repeat rate of an earlier proof body (the archived HPC `f` vs `gf` 0.7% vs 73.7%
  mechanism, now recomputable here); window_len / eviction profile; token cost; infrastructure-error counts.

## Result language (fixed in advance)
For `Delta_64` of arm A minus memI: CI entirely below 0 -> "A underperforms retrying"; spans 0 -> "A does not
establish an advantage over retrying" (no equivalence claim); entirely above 0 -> "A beats retrying" and the
universal retry claim is withdrawn for that arm. Between arms the same three-way wording on the CI.

## Deviations stated in advance
- Header sentence differs from the archived HPC `gf` prompt (see above); the local `f` arm is the matched reference.
- Attempt 1 is the bare prompt in every sequential arm while memI uses the instruction line on every draw
  (unchanged from the memI prereg; the conditional-on-attempt-1-failure analysis is the sensitivity check).
- CE_s's error is position-matched to the partner's CE trajectory where possible, not a single fixed error per
  problem as in the one-step `F_CE_shuf`; the fallback hierarchy and per-block source are logged.
- Any job that times out is resumed from its rows file with the same seeds; resumed attempts are identifiable by
  job id. Rows whose `gen_status` or `infrastructure_status` is not ok are counted and reported, never dropped.
