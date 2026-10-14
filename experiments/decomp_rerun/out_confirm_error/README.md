# out_confirm_error/ — confirmatory error-content precision run (2026-08-30)

Outputs of the run preregistered in `../PREREG_CONFIRM_ERROR_CONTENT_2026-08-30.md`.

Kept strictly separate from every pre-existing result directory (`out/`, `out_ext/`, `out_noi/`,
`out_noi_r2/`, `out_m2*`). Nothing here overwrites or re-verifies earlier data.

    b1/ b2/ b3/ b4/   one independent k=4 replicate block each (2 arms x 167 problems x 4 samples
                      = 1,336 rollouts per block), seed bases 22073001 / 22073101 / 22073201 / 22073301
      rollouts.jsonl  raw generations
      verified.jsonl  Lean 4.29 / Mathlib verdicts (120 s budget, sorry/admit screened)
    STACK_EQUIV.jsonl GATE 4 log: this verifier vs the baseline_snapshot verdicts
    ANALYSIS_CONFIRM.txt  output of ../analyze_confirm_error.py (run once)
    RESULT_CONFIRM.json   machine-readable primary result
    RUN_METADATA.json     commit hashes, job ids, environment versions, checksums
    DEVIATIONS.md         any departure from the preregistration (created only if one occurs)
