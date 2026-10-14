# Preregistration — confirmatory precision run for the error-CONTENT contrast

**Written and committed 2026-08-30 (PT) by anonymous, BEFORE any new generation was launched.**
**Branch:** `precision_campaign`, forked from `baseline_snapshot` @ `baseline_snapshot`.
**Status at writing:** no new rollout has been generated. No new outcome has been observed.

The purpose of this run is **precision, not significance**. It is designed to narrow the confidence
interval on a single prespecified contrast under a fixed sampling and analysis plan. There is no
interim look, no adaptive stopping, and no sample-size change after seeing results. A result that is
"not significant" is an expected and acceptable outcome; so is a result that is significant. Neither
changes the analysis below.

---

## 0. Why this run exists

At `baseline_snapshot` the direct error-content contrast on the 167-problem bank is:

| regime | correct error `F_E` | shuffled error `F_E_shuf` | difference | 95% CI |
|---|---|---|---|---|
| Qwen3.6-27B, instruction | 41/167 | 39/167 | **+1.2 pp** | [-4.1, +6.5] pp |

The interval is ~10.5 pp wide — wide enough to be compatible with a meaningful benefit *and* with a
meaningful harm. The `+4.8 pp` `F_E` vs `F` result is **not** this quantity: it compares a correct
error block against *no error block*, so it mixes the error's information with its mere presence,
formatting, length and token count. The target here is **correct vs shuffled**, which holds presence,
formatting and block structure fixed and varies only the text.

### An audit finding that qualifies the estimand (see §5)

`arms.shuffled_error()` is documented in `arms.py` as "a fixed **derangement** of the 171 bank ids".
**It is not a derangement.** It draws, independently for each problem, a uniform id among those whose
error *text* differs from that problem's own:

```python
rng = random.Random(22072605)
for pid in sorted(bank):
    others = [q for q in sorted(bank) if bank[q] != bank[pid]]
    m[pid] = bank[rng.choice(others)]          # <-- with replacement, per pid
```

Consequences, measured over the 167 analysis problems:

- It is a **draw with replacement**, not a permutation: 70 distinct error texts are used out of 95
  available; **25 source texts never appear**; the most-reused text is assigned to **27** problems.
- Because the draw is uniform over *ids* and 171 ids share only 95 distinct texts, common error texts
  are over-represented relative to the `F_E` arm's marginal.
- **No length matching and no failure-type matching.** Realised mean error length is 4,831 chars in
  the shuffled arm vs 2,903 in the correct arm (**1.66x**), though medians are close (199 vs 179).
  The gap is almost entirely one 337,195-char error (`mathd_numbertheory_277`) drawn twice in the
  shuffled arm versus once in the correct arm.

So `F_E` and `F_E_shuf` do **not** differ *only* in whether the diagnostic belongs to the current
problem; they also differ slightly in the marginal distribution of diagnostic texts. Measured impact
is small — excluding both monster-error problems moves the estimate from +1.20 pp to +1.21 pp — but
the defect is real and is carried forward as a documented limitation plus preregistered sensitivity
analyses (§10). **This run deliberately reuses the existing map unchanged** so that it is genuinely
confirmatory of the reported number; building a corrected, length-stratified derangement would answer
a different question and is recommended as separate future work (§13).

---

## 1. Primary estimand

**Estimand A.** The difference in the **expected number of problems solved at least once in k = 4
samples**, on the **fixed 167-problem bank**, between the correct-diagnostic arm and the
shuffled-diagnostic arm, expressed in percentage points of the 167 problems:

$$\theta = \frac{1}{167}\sum_{i=1}^{167}\Big[\Pr(\text{solved}_i \mid F\_E,\,k{=}4) - \Pr(\text{solved}_i \mid F\_E\_shuf,\,k{=}4)\Big]$$

This is the quantity the paper reports ("problems solved at k=4"). It is held **primary** and is not
replaced by either alternative below.

**Explicitly NOT the primary estimand:**

- **B — per-generation success probability.** A different scale (per rollout, not per problem).
  Reported as a prespecified secondary (§9) because it is a more sensitive detector of any content
  effect, but it is not what the paper reports.
- **C — pass@K for K > 4.** `pass@K` is K-dependent *by construction*: both arms saturate as K grows,
  so the contrast itself moves with K. Raising k is **not** a way to buy sample size for the k = 4
  estimand; it substitutes a different estimand. Not reported as a confirmation of anything.

Because the bank is **fixed**, the only source of randomness in Estimand A is **rollout sampling**.
That is what this design attacks.

## 2. Primary contrast

`F_E` minus `F_E_shuf`, in percentage points of 167 problems. Sign convention: **positive means the
correct diagnostic solves more problems.**

## 3. Population / problem bank

The frozen 167-problem bank `experiments/decomp_rerun/bank.jsonl` — all 171 rows minus the 4 rows
with `is_framing_holdout = true` (`algebra_amgm_sumasqdivbgeqsuma`, `amc12a_2013_p4`, `imo_1963_p5`,
`mathd_algebra_276`). Identical to the bank used at `baseline_snapshot`. `bank.jsonl` is unchanged by this run;
its md5 is recorded in the run metadata.

**No new problems are added.** Estimand A is defined on this bank, so adding problems would change
the estimand rather than sharpen it.

## 4. Fixed predecessor rule

Frozen, taken verbatim from `bank.jsonl`: each problem carries a fixed `predecessor_code`,
`predecessor_error` and `predecessor_rollout_idx` chosen at bank-construction time. **No predecessor
is re-sampled, re-selected or re-verified for this run.** Both arms see the same frozen failure.

## 5. Exact arms and prompts

Built by `arms.build_prompt()` at `baseline_snapshot`, unmodified. Both arms are:

```
You previously attempted to prove this theorem and it did not verify.

Lean compiler feedback from the failed attempt:
{ERROR}

Produce a correct, complete proof.

Complete the following Lean 4 code:

```lean4
{statement with trailing ":= sorry" stripped}
```
```

then wrapped in the Qwen chat template with `enable_thinking=False`, which renders the assistant turn
as `<think>\n\n</think>\n\n`.

- **Arm `F_E`** — `{ERROR}` = `problem["predecessor_error"]` (this problem's own diagnostic).
- **Arm `F_E_shuf`** — `{ERROR}` = `arms.shuffled_error(problem["id"])` (another problem's
  diagnostic, from the fixed `random.Random(22072605)` map over `bank.jsonl`, reused **byte-identically**
  from `baseline_snapshot`).

The two prompts are **byte-identical except for the diagnostic text**: same failure sentence, same
`Lean compiler feedback from the failed attempt:` header, same instruction line, same task block,
same template. Verified programmatically as a launch gate (§7, GATE 3).

`DECOMP_TASK_HEADER` is **unset** (Qwen's prompts do not carry the Mathlib header inline).

Prompt length, for the record (Qwen3.6 tokenizer, full templated prompt): `F_E` median 189 / max
88,297 tokens; `F_E_shuf` median 215 / max 88,320. With `max_tokens = 16384` and
`--max-model-len 131072`, the worst case is ~104.7k < 131.07k, so **no prompt is truncated and no
request overflows context.**

## 6. Number of problems, samples, replicate blocks

- Problems: **167** (all of them; no subsampling).
- Samples per problem per arm per block: **k = 4** — unchanged, because k defines the estimand.
- **New independent replicate blocks: R_new = 4**, per arm.
- New rollouts: 2 arms x 167 problems x 4 samples x 4 blocks = **5,344**.

A "replicate block" is a complete fresh k = 4 sweep of both arms over all 167 problems, using a fresh
seed base. Blocks are independent by construction (disjoint seeds).

**Why replicate blocks rather than higher k or more problems** — the decisive audit result:

The between-problem *effect* heterogeneity in this contrast is **estimated at zero**, so replicate
blocks reduce the interval as 1/sqrt(R) with **no variance floor**. Evidence:

1. Observed `Var_i(d_i) = 0.1203` for the 167 per-problem differences `d_i` (20 discordant problems:
   11 `F_E`-only, 9 `F_E_shuf`-only).
2. A **same-arm test-retest** already exists in the repo: `F_noI` was re-run with fresh seeds
   (`out_noi_r2`, seeds 22072701+k). There the true effect is **exactly zero**, and it produced
   **19 discordant problems** with `Var_i(d_i) = 0.1144`.
3. **20 discordances (content contrast) vs 19 discordances (same arm, reseeded).** The per-problem
   discordance in the content contrast is indistinguishable from pure Monte-Carlo noise.
4. A nonparametric-MLE mixture over per-problem success rates, **validated on that same test-retest to
   within 0.8%** (predicted `Var_i(d_i)` 0.1135 vs observed 0.1144), attributes essentially all of the
   contrast's variance to rollout noise. Point estimate of the between-problem component: **0**;
   bootstrap 95% upper bound **0.0332**.

A byproduct worth recording: the independent-arms prediction (0.1398) **exceeds** the observed 0.1203.
`run_gen.py` sets `seed = seed_base + k`, independent of arm and problem, so both arms share per-sample
seeds — **common random numbers**. The implied per-problem covariance is +0.0097. This is a genuine
(if modest) variance reduction and **this run preserves it**: within a block, both arms use the same
seed for a given `k`.

### Projected precision (Estimand A; sigma^2_within = 0.1203, n = 167)

R counts blocks **per arm**; R = 1 is the existing data. Cost basis: 1,336 rollouts/block; generation
~50 min/1,336 rollouts on `gpu:4` TP4 (job 1701499); verification ~50 rows/min (job 1701029).

| R | rollouts | gen h | verify h | SE (pp) | 95% CI width (pp) | 90% half-width (pp) |
|---|---|---|---|---|---|---|
| 1 (existing) | 1,336 | 0.8 | 0.4 | 2.68 | **10.52** | 4.42 |
| 2 | 2,672 | 1.7 | 0.9 | 1.90 | 7.44 | 3.12 |
| 3 | 4,008 | 2.5 | 1.3 | 1.55 | 6.08 | 2.55 |
| **4 (this run, new data alone)** | **5,344** | **3.3** | **1.8** | **1.34** | **5.26** | **2.21** |
| 5 (= this run + existing block) | 6,680 | — | — | 1.20 | 4.71 | 1.97 |
| 8 | 10,688 | 6.7 | 3.6 | 0.95 | 3.72 | 1.56 |
| 16 | 21,376 | 13.3 | 7.1 | 0.67 | 2.63 | 1.10 |

**R_new = 4 is the chosen budget**: it halves the interval (10.5 -> 5.3 pp) on new data alone for
~5 h of one 4-GPU node, and is sufficient to address a +/-5 pp equivalence margin (§11). R = 8 would
buy a further 1.5 pp of width for double the compute; that is not judged worth it in advance.

### The alternatives, and why they were rejected

- **Higher k, same problems (design 2).** Changes the estimand (§1C). Extrapolating the fitted mixture
  gives a difference of +1.20 pp at k = 4 but +0.33 pp at k = 8 and negative beyond — the contrast is
  k-dependent, so k = 8 cannot confirm a k = 4 number. (Those extrapolations are themselves unreliable
  beyond k~8, being fitted on k = 4 counts; they are cited only to show the estimand moves.)
- **More problems (design 3).** Would sharpen a *population* estimand, not Estimand A, and the
  resulting bank is not the bank the paper reports. Cost is also dominated by re-running the
  `base_pass_rate_k256` predecessor-selection sweep per new problem. Recorded for completeness:
  +167 problems -> 7.44 pp width, +501 -> 5.26 pp.
- **Per-rollout clustered analysis (design 4).** Free, uses the same data, and reported as a secondary
  (§9) — but on a different scale. Rollouts are **never** treated as independent observations:
  problem is always the cluster / pairing unit.

## 7. Seeds

Prospectively fixed, disjoint from every seed used at `baseline_snapshot` (which used 22072600 for reflections,
22072601-604 for the main sweep, 22072701-704 for the `F_noI` replicate; 22072605 is the shuffle-map
RNG and is **not** a generation seed):

| new block | `--seed-base` | per-sample seeds (k = 0..3) |
|---|---|---|
| 1 | 22073001 | 22073001, 22073002, 22073003, 22073004 |
| 2 | 22073101 | 22073101, 22073102, 22073103, 22073104 |
| 3 | 22073201 | 22073201, 22073202, 22073203, 22073204 |
| 4 | 22073301 | 22073301, 22073302, 22073303, 22073304 |

Within a block, **both arms use the same seed for a given k** (common random numbers, as at
`baseline_snapshot`). Blocks use disjoint seed bases and are therefore independent.

## 8. Model and decoding configuration

| item | value |
|---|---|
| model | Qwen3.6-27B, bf16 |
| weights | `/scratch/anonymous/hf/hub/models--Qwen--Qwen3.6-27B`, HF revision **`6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`** (52 GB, 15 shards, 0 incomplete blobs) |
| tokenizer / chat template | that snapshot's; `apply_chat_template(..., add_generation_prompt=True, enable_thinking=False)` |
| thinking | **off** (renders `<think>\n\n</think>\n\n`) |
| temperature | 0.7 |
| top_p | 0.95 — **sent explicitly** (see below) |
| top_k | 20 — **sent explicitly** (see below) |
| max_tokens | 16384 |
| serving | vLLM **0.21.0**, TP 4, `--max-model-len 131072 --max-num-seqs 64 --disable-custom-all-reduce --skip-mm-profiling` |
| concurrency | 48 |
| env | `VLLM_USE_FLASHINFER_SAMPLER=0`; caches redirected to `/scratch` (see below) |

### Serving-stack deviation — declared in advance

**New generation CANNOT be performed on the original serving stack.** The original runs used
`/scratch/anonymous/vllm-env` (vLLM **0.28**) and `/scratch/anonymous/models/Qwen3.6-27B`.
Both are mode `drwx------` under another user and are **unreadable to anonymous**. The only accessible
stack is `/home/anonymous/vllmvenv` (vLLM **0.21.0**, torch 2.11.0+cu130, transformers 5.8.0.dev0)
with anonymous's own HF snapshot of the same public repo.

Two consequences, both handled prospectively:

1. **Sampling parameters are pinned explicitly.** At `baseline_snapshot` the request sent only
   `temperature=0.7, max_tokens=16384`, and vLLM 0.28 then applied the model's
   `generation_config.json` (`temperature 1.0, top_k 20, top_p 0.95`), yielding *effective*
   temperature 0.7 / top_k 20 / top_p 0.95. To make the sampling distribution identical regardless of
   version behaviour, this run **sends `top_k=20` and `top_p=0.95` explicitly** via `--gen-extra`,
   alongside `temperature=0.7, max_tokens=16384`.
   **Measured (GATE 1 probe, job 1704046): vLLM 0.21 DOES apply the file, and logs the same line
   quoted in `run_gen.py`** —
   `Default vLLM sampling parameters have been overridden by the model's generation_config.json:
   {'temperature': 1.0, 'top_k': 20, 'top_p': 0.95}`.
   So the effective sampling on this stack is identical to `baseline_snapshot`; the explicit values make it
   part of the record rather than an implicit dependency on the serving version.
2. **Model revision cannot be byte-compared** to the original weights, because the original directory
   is unreadable. Same public repo (`Qwen/Qwen3.6-27B`), revision recorded above. This is an
   irreducible provenance gap and is stated as such in the limitations.

### Three further environment deviations, all measured and declared

3. **`--skip-mm-profiling`.** Qwen3.6-27B registers as `Qwen3_5ForConditionalGeneration`, so vLLM 0.21
   runs a dummy **vision** profiling pass at startup; with transformers 5.8.0.dev0 that pass dies
   (`StopIteration` in `Qwen3VLProcessor.get_text_with_replacements` — vision placeholder tokens with
   no image attached; jobs 1704025/1704026). The flag skips *memory profiling of the vision encoder
   only*. Our workload is text-only and never sends an image, so it cannot affect any generation.
4. **`VLLM_USE_FLASHINFER_SAMPLER=0`.** FlashInfer's top-k/top-p sampler JIT-compiles at startup and
   its bundled CCCL headers are version-skewed against the nvcc wheel (job 1704043). **This does not
   change our sampling**: `vllm/v1/sample/ops/topk_topp_sampler.py` falls back to the PyTorch-native
   path whenever per-request generators are present (`if (k is None and p is None) or generators`),
   and **every request in this run sets `seed`**. FlashInfer was therefore only ever reachable from
   the startup `_dummy_sampler_run`, never from a real rollout — on this stack or on `baseline_snapshot`'s.
5. **Filesystem/venv relocation.** The GPU compute nodes mount only `/scratch/users`; `/home` is
   absent (the repo's own `cluster/cuda_env.sh` records the same fact). The venv was therefore
   copied to `/scratch/anonymous/vllmvenv` and all `~`-relative caches (Triton, Inductor, FlashInfer,
   HF) redirected to `/scratch/anonymous/confirm-home` via `cluster/cuda_env_user.sh`. Two wheels
   were added to the venv so CUDA JIT has a toolchain matching torch's CUDA 13:
   `nvidia-cuda-nvcc 13.3.73`, `nvidia-cuda-cccl 13.3.3.4.1`. No existing package was changed or
   downgraded.

Because both arms are generated **in the same job, on the same stack, interleaved**, the stack change
cannot bias the *contrast* — it is a batch effect common to both arms. It does affect
**combinability** with the old block, which is why the primary analysis is **new data only** (§12).

## 9. Verifier configuration

| item | value |
|---|---|
| verifier | kimina at `/scratch/anonymous/kimina-src`, anonymous's own build |
| Lean | v4.29.0 (`LEAN_SERVER_LEAN_VERSION=v4.29.0`) |
| Mathlib | the repaired stack (upstream `lake-manifest.json` restored, ProofWidgets4 v0.0.95 unpacked; all 8,247 `lake build` jobs complete) |
| preamble spliced | `import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 400000\n\nopen BigOperators Real Nat Topology Rat\n\n` via `lean_extract.extract_lean_code` |
| per-proof timeout | **120 s** (`verify_run.py` default — the value the original pass used) |
| REPL memory cap | `LEAN_SERVER_MAX_REPL_MEM=12G` (as original) |
| REPLs | `LEAN_SERVER_MAX_REPLS=8` |
| readline patch | **applied** (`limit=REPL_STREAM_LIMIT`, 256 MiB) — fixes the kimina 64 KiB `readline` bug that turned large-output verdicts into fake HTTP 500 "REPL died" errors |
| `sorry`/`admit` screen | `SORRY_RE = r"declaration uses\s+[\`'‘\"]?sorry"` — PASS requires zero `error`-severity messages **and** no `sorry` warning |

## 10. Treatment of truncations, missing generations, verifier crashes

Fixed in advance:

1. **Truncation** (`finish_reason == "length"`, i.e. the 16,384-token cap). Kept, and **counted as
   FAIL** unless the extracted proof verifies. This matches `baseline_snapshot` (there, truncation ran
   1.2-1.6% in these arms: `F_E` 10/668, `F_E_shuf` 9/668). Truncation rate is reported per arm as an
   integrity check; a >3x change from ~1.5% is flagged in the write-up but does **not** alter the
   analysis.
2. **API-error rows** (`finish` starts with `error:`). Stripped and **regenerated with the same seed**
   by the job's retry loop, up to 2 extra passes, exactly as `gpu_run_ext.sbatch` did. The final row
   count must be exactly 5,344 with **zero** `error:` rows; the job fails loudly otherwise.
3. **Missing generations.** None tolerated. Every (problem, arm, block, k) cell must be present.
   Row-count gates are hard failures.
4. **Verifier crashes / infra failures.** Retried from **stored proof text** (never regenerated),
   first through kimina, then — for rows still crashing — directly against the Lean REPL with
   unbounded blocking reads (`repl_direct.py`), at the same 120 s budget. Recovered verdicts are
   applied with a `crash_provenance` field.
   - **Primary analysis rule:** any row whose status is still not a genuine Lean verdict after that
     recovery is **counted as FAIL** (conservative, and identical to the `baseline_snapshot` convention).
   - **Preregistered sensitivity (§14.7):** repeat the primary analysis (a) excluding unresolved rows,
     and (b) counting them all as PASS, to bound their influence.
   - Precedent: at `baseline_snapshot`, all 179 infra-FAIL rows were resolved with **0 flips** and the
     instruction-regime arms have **`infra_fail_remaining = 0`** for both `F_E` and `F_E_shuf`. This
     rule is therefore expected to be vacuous, and is fixed now so it cannot be chosen later.
5. **`sorry`/`admit`.** Screened by the fixed matcher above; such rows are FAIL. No exceptions.

## 11. Primary confidence interval

Let `s_{i,r}^{A}` be 1 if arm A solved problem i in block r within its k = 4 samples, and
`d_{i,r} = s_{i,r}^{F\_E} - s_{i,r}^{F\_E\_shuf}`. With R = 4 new blocks:

- Point estimate: `D = mean over i and r of d_{i,r}`, in pp.
- **Primary 95% CI: problem-clustered, block-averaged.** Variance estimated by treating each of the
  167 problems as an independent cluster of R block-level differences:
  `Var(D) = (1/n^2) * sum_i Var_hat(d_{i,.})/R`, with `Var_hat(d_{i,.})` the sample variance of that
  problem's R block differences. Interval: `D +/- t_{0.975, n(R-1)} * sqrt(Var(D))`.
  This relies only on independence **across problems** and **across blocks**, both true by
  construction. It is the correct interval for Estimand A (bank fixed, rollouts random).
- **Reported alongside (not primary): block-level t-interval.** `D_r` = the block-level difference for
  block r; `D +/- t_{0.975,R-1} * sd(D_r)/sqrt(R)`. Exactly valid with R-1 = 3 df, assumption-light,
  and a check on the clustered interval.
- **Reported alongside (not primary): population interval.** The paired problem bootstrap already in
  `analyze_ext.contrast()` (20,000 resamples of problems, percentile method), which additionally
  treats the 167 problems as a sample from a population of problems. This is the interval the paper
  currently quotes; it is **wider** and does **not** shrink to zero as R grows — its floor is set by
  between-problem heterogeneity (95% upper bound on the floor: **5.53 pp** width). Both are reported,
  clearly labelled, and never conflated.

Rollouts are never treated as independent observations. Problem is always the cluster.

## 12. Primary hypothesis / equivalence test

Two-sided, alpha = 0.05, on the primary contrast:

1. **Primary — equivalence (TOST).** H0: `|theta| >= delta`. Equivalence at margin `delta` is declared
   iff the **90% clustered CI** lies entirely inside `(-delta, +delta)`.
2. **Companion — superiority.** Exact paired sign test on per-problem pass counts summed over blocks
   (the statistic `analyze_ext.sign()` already implements), plus McNemar on the block-averaged solved
   indicator. Reported for continuity with the existing tables.

### The margin, recommended and separated from the choice

**Recommendation: delta = 5 pp** (about 8.4 of 167 problems). Justification, set out before seeing
new data:

- It is the scale at which the paper's other effects are read as real: `F_E` vs `F` is +4.8 pp, the
  instruction line is +18.0 pp, a nonsense sentence is +9.0 pp. An error-content effect smaller than
  the *presence* effect it is nested inside would not change any claim in the paper.
- It is reachable with this budget: at R = 4 the 90% half-width is 2.21 pp, so +/-5 pp equivalence is
  established provided the observed `|D| < 2.79 pp` — comfortably clear of the current +1.2 pp
  estimate and of the +/-1.2 pp scale of same-arm reseeding noise.

**A tighter margin is NOT promised.** delta = 3 pp would need `|D| < 0.79 pp` at R = 4, which this
budget cannot guarantee in advance; establishing +/-3 pp robustly needs R ~ 6-8 (§6 table). **If
equivalence at the chosen margin is not established, the interval is reported as-is together with the
largest benefit still compatible with its upper limit. A non-significant result is never described as
proof of no effect.**

**Final choice of delta is recorded here as 5 pp and is fixed by the commit of this file.**

## 13. Multiplicity policy

- **One primary contrast, one primary test.** `F_E` vs `F_E_shuf`, Qwen3.6-27B, instruction regime,
  Estimand A. No alpha adjustment applies to it.
- Everything in §9 (secondary), §14 (sensitivity) and §15 (heterogeneity) is **explicitly secondary
  and unadjusted**, reported with intervals rather than as tests, and **cannot** be promoted to
  primary. In particular the no-instruction regime, the per-generation scale, and the DeepSeek bank
  are **not** substitute primaries.
- The three regimes already reported (Qwen no-instruction; DeepSeek with and without instruction) are
  **not re-run** and no new test is performed on them. They enter only the descriptive heterogeneity
  table (§15).

## 14. Fixed stopping rule

**Generate all 5,344 rollouts, verify all of them, then analyse once.**

- No interim analysis. No look at arm-level outcomes during generation or verification.
- The sample size is fixed at R_new = 4 blocks by this commit and will not be increased or decreased
  after seeing any new outcome — including if the result is "nearly" significant, or if the interval
  "nearly" clears the margin.
- The only permitted early termination is **infrastructure failure** (job crash, node failure, server
  never coming up, a launch gate failing). Under it, the run is restarted from its resumable row files
  toward the same 5,344-row target; a partial file is never analysed.
- Job-level progress monitoring is limited to row counts and error-row counts. **Pass counts, per-arm
  solve counts and any contrast are not computed until all 5,344 rows are verified.** The analysis
  script is written and committed before launch (§16) and is run once.

### Launch gates (all must pass before generation starts)

- **GATE 1 — serving stack. PASSED (job 1704046, 2026-08-30).** vLLM 0.21.0 booted the snapshot on
  4x GPU TP4, `/health` responded, and one completion (`"OK"`) was returned for a fixed prompt **not
  in the bank**. No experimental outcome was produced. The `generation_config.json` override line was
  captured (see section 8).
- **GATE 2 — prompt fidelity. PASSED** (`gates_confirm.py`). For all 167 problems,
  `build_prompt("F_E", p)` and `build_prompt("F_E_shuf", p)` are byte-identical after deleting the
  diagnostic text (167/167), every prompt carries the failure sentence, the error header and the
  instruction line, and no code block leaks into these error-only arms.
  **`bank.jsonl` md5 = `23fb60e57d95c7528ec0d330aa65f215`** (171 rows, 167 analysis problems) — pinned
  here and re-checked by the generation job.
- **GATE 3 — shuffle-map fidelity. PASSED** (`gates_confirm.py`). The `random.Random(22072605)` map is
  reproduced over all 171 bank ids; 0 problems keep their own error text.
  **shuffle-map md5 = `90025b483557736dda2ea7d9cb2586b0`** — pinned here and re-checked by the
  generation job. (The gate also prints the known non-derangement of section 0 as a NOTE; it is
  documented, not a failure.)
- **GATE 4 — verifier stack equivalence.** Before any new row is verified, the verifier must reproduce
  `baseline_snapshot` verdicts on stored `lean_code` from rows **not** being re-verified: recorded-PASS
  reproduced within tolerance 2, recorded-FAIL within tolerance 0. (This is the gate that caught a
  silently-broken Mathlib on 2026-08-30; it is mandatory.)

## 15. Planned sensitivity analyses

All prespecified; all secondary; none can replace the primary.

1. **Unresolved infra rows** — primary rule (FAIL) vs excluded vs all-PASS.
2. **Truncated rows** (`finish == "length"`) — excluded rather than FAIL.
3. **Length-mismatch strata** (addresses §0): restrict to problems whose shuffled-error token length
   is within 4x of their own; separately exclude the 2 problems assigned a >20k-char error.
4. **Shuffle-reuse strata**: problems assigned a heavily reused error text (>=5 of 167) vs rarely
   reused.
5. **Estimand B** — per-generation success probability, problem-clustered paired CI.
6. **Block-level t-interval** vs the primary clustered interval.
7. **Population (problem-bootstrap) interval** vs the fixed-bank interval.
8. **Old-vs-new batch check** — the old block's contrast vs the new blocks' contrast, as a test of the
   vLLM 0.28 -> 0.21 batch effect.

## 16. Rules for combining old and new data

Fixed in advance:

- **Primary = new data only** (R_new = 4). Self-contained, one serving stack, one job.
- **Prespecified combined estimate = fixed-effect inverse-variance meta-analysis of two batches**:
  batch 1 = the `baseline_snapshot` block (vLLM 0.28), batch 2 = the 4 new blocks (vLLM 0.21). Weights
  `1/SE^2`, each batch's SE computed within batch. Reported with Cochran's Q / I^2 across the two
  batches as a heterogeneity diagnostic.
- **Raw rollouts are NEVER pooled across stacks**, and old and new blocks are never averaged as if
  exchangeable. If the batch check (§15.8) shows the two batches' contrasts differ by more than the
  combined interval admits, the combined estimate is reported but **the new-only estimate stands as
  primary** and the discrepancy is reported as a stack effect.
- Old data is **not re-verified** for this run beyond what `baseline_snapshot` + the 2026-08-30 crash-row
  recovery already settled.

## 17. Heterogeneity reporting (descriptive only)

The four existing correct-vs-shuffled cells are tabulated side by side, **never averaged**:

| model | regime | bank | n | correct | shuffled |
|---|---|---|---|---|---|
| Qwen3.6-27B | instruction | `bank.jsonl` | 167 | 41 | 39 |
| Qwen3.6-27B | no instruction | `bank.jsonl` | 167 | 32 | 29 |
| DeepSeek-Prover-V2-7B | instruction | `out_m2/bank2.jsonl` | **137** | 30 | 32 |
| DeepSeek-Prover-V2-7B | no instruction | `out_m2/bank2.jsonl` | **137** | 32 | 32 |

The DeepSeek rows use a **different bank of 137 problems**, built from that model's *own* failures
(`build_bank2.py`). The two banks are **not** interchangeable and their percentage points are not
commensurable; no pooled average across banks will be reported.

## 18. Presence vs content — the distinction this run must preserve

Reported separately and never merged:

- **Presence of an error block**: `F_E` vs `F` = +4.8 pp [-1.2, +10.8]. Confounds the diagnostic's
  information with its presence, formatting, length and token count.
- **Problem-specific correctness of the content**: `F_E` vs `F_E_shuf` = the primary contrast here.
  Holds presence, formatting and structure fixed.

A finding that content adds little does **not** imply the error block adds little, and the write-up
will state that explicitly.

## 19. Deviations

Any deviation from this file will be recorded in
`out_confirm_error/DEVIATIONS.md` with its date, reason, and whether it was made before or after any
new outcome was observed.
