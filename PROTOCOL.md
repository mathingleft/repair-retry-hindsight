# Recorded protocol and release decisions

The retained proofs and verdicts are frozen. This release repairs analysis and
packaging; it does not replace scientific outcomes with new samples.

| Item | Release decision and evidence |
|---|---|
| Qwen one-step bank | 167 included IDs from the 171-record predecessor bank; four explicit framing exclusions. Historical adaptive screening, not a verified uniform 256-draw procedure. See `SELECTION.md`. |
| Qwen sequential bank | Randomly sampled from the full 171-problem screened pool, not the later 167-problem one-step bank. See `SELECTION.md` for the frozen selection record. |
| DeepSeek banks | Reconstruct 137 one-step predecessors exactly. DeepSeek's sequential bank was constructed by approximately matching Qwen's screening success rates within its own failure bank, subject to observed solvability and with seeded tie-breaking; that procedure reconstructs exactly from retained files. |
| Sequential arms | Seven reported arms: manuscript `I,C,CE,CE_m,CE_s,E,E_m`; raw records `memI,C,CE,CE_m,CE_s,f,E_s`. `CE_s` is the genuine ordinary cross-problem shuffle and is not replaced by `CE_m`. |
| Final memoryless baseline | `out/depth_memI.jsonl` and `out_dsp/depth_memI.jsonl`, 7,040 draws each. Earlier Qwen `depth_blind_I.jsonl` is a separate experiment and is not the final baseline. |
| History budget | Both final runners evict oldest complete blocks while `sum(len(block))/3.5 > 7000`. This is an approximate character-based history cap. It does not enforce 7,000 tokenizer-measured tokens. Error displays are capped at 1,200 characters. |
| Qwen serving | Recorded main generation environment: vLLM 0.28.0, torch 2.13, Transformers 5.16.1; bf16 weights, TP4 or TP2 launch profiles, L40S GPUs. See `figs/env_freeze.txt` and archived launch settings. |
| Precision serving | Recorded separately: vLLM 0.21.0, torch 2.11.0+cu130, Transformers 5.8.0.dev0, TP4. Explicit temperature .7, top-k 20, top-p .95, max output 16,384. Source `out_confirm_error/RUN_METADATA.json`. |
| DeepSeek depth serving | Retained bf16-KV rows continued under fp8 KV; later TP2. Protocol addenda document the changes, which can affect sampling even with the same seeds. Compact adaptive rows do not retain per-attempt jobs or serving profiles, so the export cannot reconstruct arm-specific mixtures or isolate serving effects. |
| Wrong-context mapping | `CE_s` uses a fixed self-excluding cross-problem partner mapping. `CE_m` and `E_m` use class/length matching from the corresponding `CE` and raw `f` failure pools. Compact recipient records and matching donor ledgers are retained; ten full prompt/response traces are omitted. These are redundant for outcome recounts but not serving provenance. The `CE_m` import came from source commit `c2c361656670037d0d8ebb45ce08bfd489e5500a`. |
| Missing final analyses | Recovered original helper analyses; added portable exporters, final two-model sequential analysis, and final figure drivers. |
| Uncertainty | Original pooled/batch estimators and fixed seeds are retained. Where final plot source was missing, the documented problem-bootstrap estimator was reconstructed. No new scientific runs were used to resolve provenance uncertainty. |
| Soundness correction | Keep corrected outcomes and the 3,197-row recheck ledger, including 224 flipped passes. Never pool the pre-correction verdicts. |
| Token costs | Retokenize original Qwen one-step visible responses and reconstructed prompts with the bundled tokenizer. Default analysis reads the newly frozen per-row token cache. Unlogged token IDs, invisible stop tokens, and incomplete campaign output cannot be reconstructed as billed costs. |
| Compute | Archived allocation ledger sums to 219.8967 GPU-hours. Subtracting one known 48-hour timeout yields 171.8967; other failed/cancelled allocations remain. This is a partial ledger and excludes later campaigns. |

## Statistical definitions

One-step pass@4 is the fraction of included problems with at least one retained
PASS among four samples. Paired tests carry all samples within a problem;
sign tests and problem-bootstrap contrasts follow the retained analysis sources.
The precision pooled estimate is `1 - choose(16-c,4)/choose(16,4)`, averaged over
problems. Its fixed-bank independent-sampling normal interval, problem-bootstrap
interval, original four-block analysis, and batch sensitivity are distinct outputs.

Adaptive sequential pass@k is the mean over problems of the fraction of ten
trajectories passing by attempt k. The memoryless baseline uses
`1 - choose(640-c,k)/choose(640,k)`. It is not the fraction of arbitrary groups
of 64 containing a pass. Final paired intervals resample the eleven problems.
The Qwen sensitivity conditions adaptive trajectories on failure at attempt 1
and compares with the pooled memoryless estimator at k-1.

One-step copying normalizes proof bodies using the retained helper. Sequential
repetition instead counts byte-identical saved proof bodies appearing earlier in
the same trajectory, among attempts after the first. These definitions must not
be interchanged. All-sample predecessor similarity and success-conditioned
unknown-identifier similarity are also separate statistics.

## Anonymization and provenance

The records are the filesystem export prepared for anonymous review. In it, personal
account paths were replaced with `/scratch/anonymous` and `/home/anonymous`
placeholders and internal host names were normalized; the public release keeps
those bytes unchanged so that the hashes match.
Scientific IDs, proofs, diagnostics, and verdicts are preserved. Sampling seeds
and job identifiers survive where present in the exported records; compact
adaptive rows omit them. Do not infer complete per-attempt serving provenance
from historical job notes. Public upstream library/model attribution and licenses remain.
Hashes refer to exported bytes, after operational identifier redaction. Historical
MD5 values in campaign notes refer to the original records and may differ from
the public SHA-256 manifest. They are not used to validate the export.

The original private source-to-export ledger is intentionally outside this
artifact. Reproduction uses only the public manifest and relative paths.
Preserved historical notes may describe superseded experiments; the mapping in
`RESULTS.md` defines which data feed the supplied paper results.

## Behavioral example checks

The saved reflection/diagnostic example has these diagnostic-only outcomes:
sample 0 retains `begin` and fails, samples 1 and 3 pass without it, and sample 2
avoids it but fails. The copying example preserves the actual saved proof tokens.
`analysis/export_examples.py` exports and checks these exact sample records.

The clean CPU environment pins every installed analysis dependency. The fresh
verifier/GPU setup remains an optional protocol recipe; no new model sampling
or fresh full Lean dependency build was performed for submission preparation.

Private author-repository branch names are removed from the review export. Public
model/library revisions and the retained `CE_m` source commit remain exact for
scientific provenance. The release manifest hashes the exported analysis code and
data.
