# Pre-registration: the sequential / depth arms on DeepSeek-Prover-V2-7B (memI, f, C, CE, CE_s, E_s)

Written 2026-09-11 09:40 PT, before any prover depth attempt was generated (screen job 1716934 submitted 09:35 PT;
arm jobs chained behind it: 1716935 memI, 1716936 f, 1716937 C, 1716938 CE (afterok:screen); 1716939 CE_s afterok:CE; 1716940 E_s afterok:f. At 09:39 PT, with no free GPU on the cluster and the earliest release forecast at 14:23, those seven jobs were cancelled unstarted and replaced by ONE gpu:1 job, 1716941 (`ARM=all`): screen, then memI + f + C + CE concurrently on one prover server, then E_s (after f) + CE_s (after CE). Same runners, files, seeds and per-arm outputs; only the scheduling differs.) User request: run the Qwen sequential arms on the prover,
with the experimental setup and the problem selection made correct for the prover, within 8 hours.

## Model, serving, decoding (the prover's one-step setup, unchanged)
- DeepSeek-Prover-V2-7B bf16, vLLM 0.28.0, one L40S (gpu:1), `--max-model-len 32768 --max-num-seqs 64`, served as
  `dsp-v2-7b`; tokenizer guard `cluster/m2_tok_guard.py` at job start (the LlamaTokenizerFast bug of 08-28).
- Prompts through the prover's chat template (`depth_run.chat`, user turn only). The Mathlib header is placed
  inside the code block (`DECOMP_TASK_HEADER=1`, `depth_run._stmt`), as in every one-step prover arm
  (`arms._task`). Qwen prompts are byte-identical to before when the switch is unset (checked).
- Decoding `max_tokens 16384, temperature 0.7` (GEN); the prover ships no `generation_config.json`, so vLLM's
  defaults (top_p 1, top_k off) apply, as in the one-step prover runs.
- Verification: `depth_run.verify` / `verify_run.verify_codes` on kimina, Lean v4.29.0, 120 s, standard preamble,
  sorry screen; in-loop, 8 REPLs per job. Infra failures logged separately and counted as FAIL.

## Problem selection (the part that must be right for the prover)
The Qwen depth bank is the 11-problem head-to-head set chosen as hard-but-not-hopeless for Qwen3.6-27B, cold
pass counts /256 = (0,0,0,0,1,2,3,4,9,41,45). The prover needs its own such set, so:
1. **Cold screen** (`depth/cold_screen.py --phase screen`): 32 bare-prompt draws per problem over the prover's own
   bank (`out_m2/bank2.jsonl`, its 137 frozen-failure problems), same prompt/decoding/verifier as attempt 1 of every
   sequential arm, seeds `700000 + md5(pid|screen|k) % 100000`.
2. **Selection** (`--phase select`): the Qwen targets sorted descending; for each target the unused prover problem
   with the closest cold rate; ties broken by (a) at least one PASS anywhere in the prover's one-step arms
   (`out_dsp/prover_onestep_passes.json`, 68 samples per problem) because Qwen's 0/256 problems were still
   solvable at depth, then (b) a seeded name-blind order `random.Random(22072606)`. Written to
   `depth/depth_bank_dsp.jsonl` with the cold counts, one-step counts and the Qwen target per problem, and
   `out_dsp/SELECTION.txt`. No problem is dropped afterwards for being trivially solved or never solved.
   Overlap with the Qwen 11 is allowed and reported.
   Stated limits: 32 draws resolve cold rates to 3.1%, so the four Qwen targets between 0.4% and 1.6% map to
   0/32 problems along with the four zero targets; the one-step tie-break is what separates "hard" from
   "hopeless" there.

## Arms (each identical to its Qwen twin except for model, header and bank)
- `memoryless_I` (`depth/memoryless_I.py`): 11 x 10 x 64 = 7,040 fixed-instruction draws, no history, no early
  stop, seeds `depth_run.seed_for`.
- `f` (`depth/depth_run.py --arm f`): sequential, own compiler error per block, early stop at first PASS.
- `C`, `CE`, `CE_s`, `E_s` (`depth/code_arms_run.py`): as in `PREREG_code_arms_depth_2026-09-07.md` and
  `PREREG_E_s_seq_2026-09-08.md`. `CE_s` partners = the same seeded derangement over the prover's 11 sorted ids;
  position-matched CE errors with the seeded pool fallback. `E_s` = class + length matched wrong errors drawn from
  the prover's own `f` arm (`out_dsp/depth_f.jsonl`), the prover tokenizer for lengths.
- Protocol constants shared with Qwen: 10 trajectories, 64 attempts, `seed_for` grid, HEADER + `## Previous
  attempt N` blocks + task line, FIFO eviction beyond `PROMPT_TOKEN_CAP` (chars/3.5 > 7000), attempt 1 = bare
  prompt, error text capped at 1200 chars.
- Dependencies: CE_s after CE, E_s after f. One gpu:1 job per arm, per-arm ports, resumable.
- Output: `out_dsp/depth_{f,C,CE,CE_s,E_s}.jsonl` (depth format), `out_dsp_codearms/<arm>_rows.jsonl` (full rows),
  `out_dsp_memI/memI_verified.jsonl`, `out_dsp/cold_screen.jsonl`.

## Estimators and endpoints (as for Qwen)
`p_arm,i(k)` = fraction of the 10 trajectories with a PASS by attempt k; memI pooled without-replacement
estimator over 640 draws; paired problem bootstrap over the prover's 11 problems, 10,000 resamples, percentile
95% CI, k in {1, 2, 4, 8, 16, 32, 64}; paired sign test alongside.
- **Primary**: `CE - memI` at k = 64 on the prover.
- **Secondary** at k = 64: `C - memI`, `CE_s - memI`, `E_s - memI`, `f - memI`; `CE - C`, `CE - CE_s`, `CE - f`,
  `C - f`, `f - E_s`, `CE_s - E_s`; the interaction `(CE - CE_s) - (f - E_s)`.
- **Cross-model**: the same contrasts side by side with the Qwen values; no pooling across models.
- **Exploratory**: every k; per-problem heterogeneity; proof-repeat rate; match quality for E_s; token cost;
  infrastructure-error counts; the cold-screen distribution itself.

## Result language (fixed in advance)
For arm A minus memI at k = 64: CI entirely below 0 -> "A underperforms retrying"; spans 0 -> "A does not
establish an advantage over retrying"; entirely above 0 -> "A beats retrying". Between arms the same three-way
wording. For the interaction: entirely above 0 -> "diagnostic content is worth more beside the own failed
proof"; spans 0 -> "not established"; below 0 -> "worth less".

## Deviations stated in advance
- Problem set differs from Qwen's by construction (prover-specific selection above); cross-model comparisons are
  of protocol effects, not of problems.
- 8 REPLs and gpu:1 per job because the cluster had one free GPU at submission; jobs may queue behind other
  users. Any timeout resumes from the rows file with the same seeds; job id logged per row.
- The prover's response lengths and pass rates differ from Qwen's, so the FIFO window holds a different number of
  blocks; window_len is logged per attempt.

## Amendment 2026-09-11 12:20 PT (after the cold screen, before any arm data was analysed)
The screen put 92 of 137 problems at 0/32, and only 2 of those 92 had a one-step PASS, so the preregistered
tie-break could not separate "hard" from "hopeless": the six zero/near-zero targets were filled with problems that
had never been solved anywhere (`out_dsp_bank1_partial/SELECTION.bank1.txt`). Job 1717051 had started the arms on
that bank (about 50 attempts per arm); it was cancelled at 12:20 and its partial rows parked in
`out_dsp_bank1_partial/`, unused. Rule change: candidates must show solvability evidence (>= 1 cold PASS or >= 1
one-step PASS); the quantile matching and tie-breaks are otherwise unchanged (`cold_screen.py --require-evidence`).
Resulting bank (cold /32, one-step /68): amc12b_2021_p4 6, 28; mathd_numbertheory_100 5, 2; mathd_algebra_362 1, 3;
algebra_2varlineareq_... 0, 1; mathd_numbertheory_150 0, 1; imo_1962_p2 1, 10; mathd_algebra_288 1, 3;
mathd_numbertheory_314 1, 4; mathd_numbertheory_296 1, 1; mathd_numbertheory_5 1, 0; aime_1983_p1 2, 0.
Five problems carry over from the first bank; imo_1962_p2 and mathd_numbertheory_314 are also in the Qwen bank.
The arms run as job 1717063 (started 12:21 on gpu-node-02, screen skipped, same servers and seeds). Everything else in
this document stands.

## Addendum 2026-09-11 18:10 PT (scheduling only; no change to arms, seeds, prompts, bank or verifier)
Job 1717063 ran 5h48 on one GPU and reached ~2,200 attempts per sequential arm (f 62/110 solved, C 18, CE 17) and
1,182/7,040 memI draws, far below plan. Diagnosis from the vLLM log: the prover's full multi-head attention leaves a
60,352-token KV cache, so only 5-8 requests decode at once with 100+ waiting, and decode ran at 15-20 tok/s per
sequence because 10 Lean REPLs, four runners and the engine loop shared 12 cores. It was cancelled at 18:09 (rows kept;
every runner resumes by key with the same seeds) and replaced by: kimina on a CPU node (1719068, cpu-node-03, 14 REPLs)
and two gpu:1 servers, A = 1719069 (memI, f, then E_s) and B = 1719070 (C, CE, then CE_s), `ARMS_SUBSET`, 8 cores
each, runner concurrency 24. The first-bank partial rows remain parked and unused. The 8-hour target was missed.

## Addendum 2026-09-11 18:45 PT — serving deviation accepted by the user: fp8 KV cache
At 18:35 every GPU on the cluster was allocated and the bf16-KV servers were delivering ~25 attempts/min/GPU
(KV-bound: 60,352 tokens, 5-13 concurrent requests), a ~14 GPU-hour remainder. The user chose to accept an fp8 KV cache
(`vllm serve ... --kv-cache-dtype fp8`, `KV_FP8=1` in `cluster/dsp_depth.sbatch`), which doubles the KV budget and so
the number of requests decoding at once. Numerics: keys/values are stored in 8-bit float, so attention reads slightly
rounded values; seeds still fix each draw, but the sampled trajectories are not the ones bf16 serving would have
produced. Every prover depth row carries the job id; rows from 1717063 / 1719069 (bf16 KV, ~2,300 f, 2,244 C, 2,205 CE,
1,250 memI attempts) are kept and resumed from, so the arms are a bf16/fp8 mixture by attempt, identifiable per row.
Jobs 1719069/1719070 cancelled unstarted-or-resumed at 18:39; A = 1719082 (memI, f, E_s) and B = 1719083 (C, CE, CE_s)
started 18:39 on gpu-node-02 with kimina 1719068. Prompts, seeds, bank, decoding parameters and verifier unchanged.

## Addendum 2026-09-11 19:00 PT — one TP2 server (scheduling/serving only)
A (1719082) and B (1719083) landed on the same node with the all-in-one mode's fixed port pair; A's API server never
bound, its runners queued on B's server, its GPU idled. Under the prover tokenizer the window prompts are ~11k tokens
(same chars/3.5 cap as Qwen), so even with fp8 KV a gpu:1 server decodes ~10 requests and prefill dominates
(prompt 3-4k tok/s vs generation ~140 tok/s). Both cancelled at 18:59 and replaced by one gpu:2 tensor-parallel job
1719089 (`TP=2`, fp8 KV, pinned to gpu-node-02, all six arms in the wave order, kimina 1719068): one ~290k-token KV pool,
~2.5x the two separate servers, and prefix-cache headroom. Rows resume by key; job id per row. Launcher gained
VPORT_ENV/KPORT_ENV overrides so two all-in-one jobs can never collide again.

## Completion 2026-09-12 03:20 PT — all six prover depth arms finished (data committed, analysis not yet run)
Job 1719089 (gpu:2 TP2, fp8 KV, gpu-node-02, kimina 1719068 then 1719425) COMPLETED at 03:20:24 after 8h20m; the chained
continuation 1719424 was cancelled unstarted. Every arm has 110 trajectories, each ended by a PASS or attempt 64;
no duplicate (problem, trajectory, attempt) keys; one infra-error row in CE_s counted as FAIL. Rows are a bf16/fp8-KV
mixture by attempt as recorded above (job id per row).

| arm | rows | trajectories solved /110 |
|---|---|---|
| f | 3457 | 74 |
| C | 5824 | 27 |
| CE | 5844 | 27 |
| CE_s | 5844 | 25 |
| E_s | 2857 | 90 |
| memoryless_I | 7040 draws, 573 PASS | 80 |

Per problem (best arm, trajectories solved /10): amc12b_2021_p4 10, mathd_numbertheory_100 10, mathd_algebra_362 10,
algebra_2varlineareq_… 8, mathd_numbertheory_150 10, imo_1962_p2 10, mathd_algebra_288 10, mathd_numbertheory_314 10,
mathd_numbertheory_296 0 (0/640 memI, 0/10 in every arm), mathd_numbertheory_5 8, aime_1983_p1 6.
Files (repo mirror of the scratch run dir): `experiments/decomp_rerun/out_dsp/depth_{f,C,CE,CE_s,E_s,memI}.jsonl`
(depth-format mirrors; `depth_memI.jsonl` = `out_dsp_memI/memI_verified.jsonl`), `out_dsp/cold_screen.jsonl` (the
32-draw screen), `out_dsp/prover_onestep_passes.json`, and the full per-attempt rows gzipped in
`out_dsp_codearms/{C,CE,CE_s,E_s}_rows.jsonl.gz`. The preregistered contrasts (§Analysis) have not been computed yet.
