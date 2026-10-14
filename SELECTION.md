# Problem selection and what can be reconstructed

Run `python analysis/selection.py`. It writes selected banks and CSV
manifests under `generated/`, checking against the retained experiment inputs.

## Qwen one-step: 171 predecessors to 167 problems

The included `bank.jsonl` contains 171 fixed predecessor records. Exclude exactly
`algebra_amgm_sumasqdivbgeqsuma`, `amc12a_2013_p4`, `imo_1963_p5`, and
`mathd_algebra_276`, marked `is_framing_holdout`. All 167 remaining records have
screening estimates below 0.25, with mean 0.0235191616766467 and 132 zero estimates.
`selection/reconstructed_source_manifest.jsonl` independently agrees on each
selected problem's recorded rate and predecessor sample index.

The inherited field `base_pass_rate_k256` is a historical name. It is not evidence
that every problem was screened with exactly 256 draws. The recovered builder
`selection/historical_build_probe_v2_prompts.py` describes an adaptive screening
pool measured with up to 1,024 draws and failures from `round0_k16_verified.jsonl`.
Its candidate rule is: retain `FAIL` rows with nonempty diagnostics and
`truncated == false`; skip synthetic `extract_lean_code_no_body` and `-- Error:`
fallbacks; choose the lowest `rollout_idx` whose rendered V1 and V4 prompts both
fit 24,512 tokens (40,960 context minus 16,384 output minus 64 reserved tokens).
Diagnostics are deduplicated/truncated by the historical prompt renderer before
this length test. This screening guard is not a cap on the later one-step
decomposition prompts, which can have much longer unpruned diagnostics.

The full `kept.jsonl`, raw screening outcomes, and all dependencies of that older
prompt renderer were not recovered. The archived builder is evidence of the
historical method, not a standalone executable entry point. Recreating those raw
screening estimates from the selected bank would be circular; no replacement
screen or invented screening data are supplied. Repeating the submitted
experiment starts with the explicitly frozen predecessor bank.

## DeepSeek one-step: 167 to 137

Read the original `bare` arm in `out_m2/verified.jsonl` before filtering to
`bank2.jsonl`. There are four samples on each of the 167 Qwen problems. Drop the
30 problems where all four passed. Sort failing samples by `sample_idx`, discard
the runner's two synthetic no-body error forms, and choose the first remaining
failure. No additional problem is lost for degenerate failures. This reproduces
all 137 IDs and every predecessor proof, diagnostic, and index byte for byte.
The 30 excluded problems must not be included in DeepSeek's reported bare count:
the filtered count is 29/137, not 59/167.

## Qwen depth: random selection from the 171-problem pool

`depth/depth_bank.jsonl` freezes a random selection from the full
171-problem screened pool, before the four framing holdouts define the later
167-problem one-step analysis bank. After excluding `amc12b_2021_p1`, the final
depth bank contains 11 problems. Recorded baseline counts out of 256 are
0, 0, 0, 0, 1, 2, 3, 4, 9, 41, 45. The selected IDs and counts are retained and
can be checked. The exact historical random seed and draw script are unavailable,
so reproduction uses the frozen IDs rather than redrawing a new bank.

## DeepSeek depth: approximately matched to Qwen screening rates

`out_dsp/cold_screen.jsonl` contains 32 draws for each of 137 problems (4,384
rows). Non-OK infrastructure/generation rows remain unsuccessful (61 rows).
Aggregate evidence of one-step solvability from all 17 retained arms: 68 draws
per problem. The stored `prover_onestep_passes.json` also contains the 30 excluded
bare-only IDs; restrict it to the 137 candidates.

DeepSeek's sequential bank was constructed by approximately matching Qwen's
screening success rates within its own failure bank, subject to observed
solvability and with seeded tie-breaking. Operationally, the amendment requires
at least one success in the cold screen or the one-step arms. Sort Qwen's eleven
target rates in descending order. For each target,
choose an unused eligible DeepSeek problem minimizing, lexicographically:

1. Absolute distance between cold rate and target rate.
2. Zero if any one-step draw passed, otherwise one.
3. Position in `random.Random(22072606).sample(sorted(all_137_ids), 137)`.

This reproduces all eleven selected IDs, counts, and targets exactly. The
original and amended selection notes are both retained in `out_dsp/`; the
amended bank is the one used in all seven final sequential arms.

## Wrong-context selection

`arms.shuffled_partner` and `depth/code_arms_run.partners` iterate sorted bank
IDs and independently choose donors with `random.Random(22072605)`. One-step
candidates must have different diagnostic text from the recipient (which also
excludes the recipient itself); sequential candidates must have a different ID.
Donors may repeat: this is a fixed self-excluding mapping, not a permutation.
The one-step Qwen mapping uses all 171 bank records, including holdouts as possible
donors. DeepSeek uses its 137-record bank. Matched one-step donors are frozen in
`bank_match.json` and `out_m2/bank2_match.json`.
Sequential `CE_s` takes the donor's same trajectory/attempt diagnostic when
available and failed; otherwise it samples the sorted donor error pool with
`seed_for(pid, trajectory, attempt)`. `E_s` instead matches class and token length
online from the completed `f` pool, without reusing a donor row within a trajectory
when the primary pool permits. The submission retains the exact newly selected donor at every Qwen `E_s` attempt after the first in `out_codearms/E_s_donors.jsonl`; its source IDs resolve to the retained failed `f` records.
