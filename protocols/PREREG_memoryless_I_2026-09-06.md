# Pre-registration: memoryless instruction-only retry control for the 11-theorem sequential experiment

Written 2026-09-06 21:55 PT, before any draw of this run was verified (generation jobs 1711361/1711362 were
queued behind the EXT3 jobs; verify job 1711363). Committed before data.

## What is run
- **Arm `memoryless_I`**: every draw is the fixed prompt `Now produce a correct, complete proof.` followed by the
  theorem in the exact serialization the sequential runners end their prompts with (`depth_run.task`, byte-identical
  to `fifo_refiner.render_fifo_prompt`'s tail). No history, no failure marker, no predecessor proof, no diagnostic.
- **Problems**: the same 11 theorems as the sequential experiment (`depth/depth_bank.jsonl` = `headtohead_qwen27b.jsonl`
  minus `amc12b_2021_p1`). All 11 analysed, none dropped for being trivially solved or never solved.
- **Draws**: 640 per theorem = 10 trajectories x 64 attempts, all generated (no early stop), 7,040 total.
- **Seeds**: `depth_run.seed_for(problem, trajectory, attempt)`, the deterministic per-tuple grid every depth arm in
  this directory shares. The archived HPC code-bearing runner (`run_ttt_solve.py` / `fifo_refiner.py`) set no per-request
  seed (only a config-level `seed: 42`), so there is no request-seed grid to inherit from it. Seed pairing is not
  claimed to make outputs paired.
- **Decoding**: the actual runner config: `max_tokens 16384, temperature 0.7`; `top_k 20, top_p 0.95` from the
  model's `generation_config.json` via vLLM; thinking off (`enable_thinking=False` in the chat template).
- **Serving stack — deviation stated up front**: Qwen3.6-27B bf16 on shared cluster (vLLM 0.28.0, L40S, TP2), the stack
  every same-repository depth arm (`assert`, `f`, `noise`, `blind_I`) was generated on. The code-bearing sequential
  run was served from the archived HPC image (vLLM 0.21 + transformers @ 2026-05-21, GH200); that image and cluster are not
  reachable from shared cluster, so an exact-stack replication is not possible here. The runner is stack-agnostic
  (`--base-url`, `--model`) and can be re-run unchanged on the archived HPC stack; the stack actually used is logged in every
  job log (`stack: vllm ... torch ... transformers ...`).
- **Verification**: the final corrected pipeline (`lean_extract.extract_lean_code` splice with the standard
  preamble incl. `maxHeartbeats 400000`, Lean v4.29.0 via kimina, 120 s timeout, sorry/admit screen with the corrected
  warning regex, transient infrastructure errors retried; infrastructure status logged separately from the verdict).
- **Logged per draw**: problem_id, trajectory_id, attempt_id, seed, user_prompt, full_prompt (chat-templated string
  sent), prompt_token_count (server usage + tokenizer recount), raw_response (retained), response_token_count
  (both), finish_reason, gen_status, extracted_proof, verification_result, compiler_output, compiler_messages,
  infrastructure_status, job ids.

## Estimators
- Memoryless I, problem i with n_i = 640 draws and c_i passes: `p_I,i(k) = 1 - C(n_i - c_i, k) / C(n_i, k)`.
- Repair arm, problem i: `p_repair,i(k) = #{trajectories with a PASS by attempt k} / #trajectories`.
- `Delta_k = (1/11) sum_i [p_repair,i(k) - p_I,i(k)]`, k in {1, 2, 4, 8, 16, 32, 64}.
- Uncertainty: paired problem bootstrap — resample the 11 problems with replacement, carry every trajectory and all
  640 draws of each sampled problem, recompute Delta_k; 10,000 resamples; percentile 95% CI. Problems, not
  trajectories, are the unit.

## Endpoints
- **Primary**: `CE - I` at k = 64 (CE = failed proofs + correct diagnostics, the code-bearing sequential arm).
- **Secondary**: `C - I`, `CE_s - I`, `CE - C`, `CE - CE_s` at k = 64; `CE - I` at every k in the grid.
- **Exploratory**: breadth/depth allocation, per-problem heterogeneity, token cost, conditional-on-attempt-1-failure
  sensitivity (repair: trajectories whose attempt 1 failed; I: pooled estimator over k - 1 draws).

## Result language (fixed in advance)
- CI for `Delta_64` entirely below 0: "repair underperforms retrying".
- CI spans 0: "repair does not establish an advantage over retrying". No equivalence claim (no margin defined).
- CI entirely above 0: "repair beats retrying"; the paper's universal retry claim is then withdrawn.

## Attempt 1
The sequential arms' first attempt is the bare task prompt (`Complete the following Lean 4 code:` ...), while
`memoryless_I` uses the instruction line from attempt 1. The existing arms are not modified. `CE - I` is therefore a
comparison of complete deployed protocols; the conditional-on-attempt-1-failure analysis above is the sensitivity
check.

## Data dependencies
The C / CE / CE_s trajectories live on the archived HPC store, not in this repository. `depth/analyze_memI.py` takes any
depth-format attempt log (`arm, id, traj, attempt, status`) as a repair arm (`--repair CE=path`); until those files
are exported here, the same-stack local depth arms (`out/depth_{assert,f,noise}.jsonl`) are reported as a
same-stack reference, labelled as such, and no CE contrast is claimed.
