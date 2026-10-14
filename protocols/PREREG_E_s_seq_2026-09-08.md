# Pre-registration: code-free wrong-diagnostic sequential arm `E_s` (= f_s, E_s^seq)

Written 2026-09-08 19:33 PT, before any attempt of this arm was generated (job 1713705 submitted 19:34 PT; timeout-only continuation 1713706 afternotok). Motivation: the post-hoc interaction
analysis of 2026-09-08 (`out_codearms/ANALYSIS_INTERACTION.txt`) had to use the content-free `noise` arm as the
no-code contrast, because no code-free wrong-diagnostic arm existed at depth. `E_s` is the direct no-code analogue
of `CE_s`, so `(CE - CE_s) - (f - E_s)` becomes the clean test of whether problem-specific diagnostic content is
worth more when the model is conditioned on its own failed proof.

## What is run
- **Arm `E_s`**: identical to the local `f` arm in every respect (same 11 problems `depth/depth_bank.jsonl`, 10
  trajectories x <= 64 attempts, early stop at the first PASS, attempt 1 = the bare task prompt, `depth_run.HEADER`
  + `## Previous attempt N` blocks + `depth_run.task(p)`, FIFO eviction beyond `PROMPT_TOKEN_CAP`, seed grid
  `depth_run.seed_for`, decoding `max_tokens 16384 / temperature 0.7 / top_k 20 / top_p 0.95`, thinking off,
  `depth_run.verify` with kimina Lean v4.29.0 / 120 s / sorry screen, Qwen3.6-27B bf16 on vLLM 0.28.0 TP2) except
  that the text after `Lean compiler feedback:` in every block is a WRONG diagnostic from another problem instead
  of the trajectory's own error. The block itself is produced by `depth_run.block("f", n, error)`, byte-identical
  to `f`. Runner: `depth/code_arms_run.py --arm E_s` (`MatchedErrors`), launcher `cluster/depth_codearm_g2.sbatch`
  with per-arm ports (vLLM 8024, kimina 8004).
- **Matching policy** (the one-step `F_E_match` policy of `build_match.py`, applied online):
  for the block describing the trajectory's own failed attempt j with own error e_j (capped at 1200 chars as in
  `f`), the shown error is drawn from the pool of the local `f` arm's own FAIL errors (`out/depth_f.jsonl`, same
  stack and verifier, 3,783 rows) such that
  1. same coarse class: `build_match.classify` on the first message (syntax / unsolved / linarith / omega /
     no_goals / rewrite / simp / type / unknown / resource / crash / other);
  2. token length as close as possible: cost = |log(tok_src / tok_own)| with the model tokenizer; an identical
     text to e_j carries the builder's +50 penalty;
  3. never the same problem;
  4. reuse: the builder's perfect matching is not possible online, so a source row is used at most once per
     trajectory; ties broken by `random.Random(seed_for(pid, t, j))`;
  5. fallback: if fewer than 5 cross-problem candidates exist in the class, the pooled non-syntax family is used
     (the builder's `tactic_pool` rule); if none, any other problem's error.
  The matched error for own attempt j is drawn once, when attempt j fails, and stays fixed in every later prompt
  of that trajectory (as a real FIFO block would). Per block the source row, own/source class, pool, token lengths,
  cost and tie count are logged (`shuf_sources`). Resume replays the draws deterministically from the logged rows.
- **Logged per attempt**: as for C / CE / CE_s (prompt, history, raw text, proof, verdict, diagnostic, gen/infra
  status, both token counts, seed, job id); depth-format mirror `out/depth_E_s.jsonl` at the end.

## Estimators and uncertainty
As in the memI and code-arm prereg: `p_arm,i(k)` = fraction of the 10 trajectories of problem i with a PASS by
attempt k; paired problem bootstrap over the 11 problems, 10,000 resamples, percentile 95% CI, k in
{1, 2, 4, 8, 16, 32, 64}; paired sign test (problem as unit) alongside.

## Endpoints
- **Primary**: `Delta_interaction(64) = (CE - CE_s) - (f - E_s)` at k = 64, computed within each bootstrap
  replicate (`depth/analyze_interaction.py` with `E_s` in place of `noise`).
- **Secondary** at k = 64: `f - E_s` (diagnostic content without code), `E_s - noise` (a wrong real diagnostic vs
  content-free filler), `E_s - memI`, `CE_s - E_s` (code presence under a wrong diagnostic), and the interaction at
  every k in the grid.
- **Exploratory**: byte-identical proof-repeat rate; window / eviction profile; token cost; match quality
  (same-class fraction, token-ratio distribution, pool fallbacks); infrastructure-error counts.

## Result language (fixed in advance)
For the primary: CI entirely above 0 -> "diagnostic content is worth more beside the model's own failed proof";
spans 0 -> "the interaction is not established"; entirely below 0 -> "diagnostic content is worth less beside the
own failed proof". Between arms the usual three-way wording on the CI.

## Deviations stated in advance
- `CE_s` drew its wrong error position-matched from the partner problem's CE trajectory with a fixed partner per
  problem; `E_s` draws class- and length-matched errors from the whole cross-problem `f` pool, per the user's
  specification. The two wrong-error arms therefore differ in how the wrong error was chosen; the match-quality
  statistics are reported so the difference is visible.
- Online matching cannot enforce the builder's balanced (perfect-matching) reuse across the whole run; the
  per-trajectory no-reuse rule is the substitute.
- Any timeout is resumed from the rows file with the same seeds; rows with non-ok gen or infra status are counted
  and reported, never dropped.
