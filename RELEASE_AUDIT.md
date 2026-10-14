# Release audit

This checklist applies the NeurIPS Code and Data Submission Guidelines and the
Papers with Code research-code checklist to the public release. The public
release is the anonymous review artifact (submission 149) with the changes listed
under "Changes from the review artifact".

## Release

- **Public:** hosted at <https://github.com/mathingleft/repair-retry-hindsight>.
  The version used for the camera-ready paper is tagged `v1.0.0`.
- **Licensed:** code under MIT (`LICENSE`); records, expected outputs, figures,
  and documents under CC BY 4.0 (`LICENSE-DATA`); third-party assets keep their
  own licenses (`ASSETS.md`).
- **Offline principal-result reproduction:** after installing the pinned CPU
  dependencies, `python scripts/reproduce.py` is the single entry point. It
  restores and hashes frozen inputs, validates grids and donors, recounts results,
  and regenerates tables and figures without model inference or a verifier.
- **Limits are explicit:** fresh GPU generation and Lean reverification are
  optional workflows because they require model weights, GPU resources, and a
  separately built verifier. `ENVIRONMENT.md` gives pins and entry points; frozen
  outcomes support the numerical reproduction.
- **Data included:** compact sample-level outcomes, exact banks, selection
  evidence, and donor ledgers needed for every reported analysis are included.
  Ten full prompt/response traces are listed as omitted in
  `manifests/package.json`. Their omission preserves reported outcome recounts
  but loses adaptive per-attempt job/serving provenance. Arm-specific serving
  mixtures and serving-stratified sensitivities are not reconstructible from
  this export; recovering a source-backed ledger remains an open audit item.
- **Reverification coverage:** the optional workflow enumerates all seven
  sequential arms and the separate Qwen fresh failure-notice replicate.
  `analysis/check_reverification_coverage.py` tests coverage without invoking
  Lean. This is not evidence of an independently completed reverification run.


## Changes from the review artifact

- Authors, license, and citation metadata added; review-use terms
  (`REVIEW_USE.md`) withdrawn in favour of the licenses above.
- Launch scripts under `experiments/decomp_rerun/cluster/` and four Python
  entry points read model, environment, and cache locations from
  `$VERICODE_SCRATCH` (or an explicit argument) instead of hard-coded
  anonymized paths. No command, flag, or setting was otherwise changed.
- Frozen records, expected outputs, historical notes, and the registration
  documents (`protocols/`, `experiments/decomp_rerun/PREREG_*.md`) are
  unchanged byte-for-byte, including their anonymized operational paths, so
  their hashes still match the review artifact.
- `analysis/plot_onestep.py` labels the appendix one-step figures in the
  paper's notation (estimators and values unchanged).
- `analysis/camera_ready/` adds the scripts and reference outputs for every
  analysis added to the camera-ready paper.
- `manifests/release.json` is regenerated over the new file set.

## Code-completeness checklist

| Recommendation | Artifact status |
|---|---|
| Dependency specification | `requirements-analysis.txt` pins the CPU stack; `requirements-runners.txt`, `ENVIRONMENT.md`, and `environments/revisions.json` document optional generation/verifier requirements. |
| Training code | Not applicable: the study evaluates public pretrained models and performs no training or fine-tuning. |
| Evaluation code | Included for frozen-outcome analysis, optional fresh sampling, and optional Lean reverification. |
| Pretrained models | Weights are not redistributed; public model identifiers and revisions are pinned, and the tokenizer/config assets used by the analyses are bundled with licenses. |
| README, expected results, exact commands | `README.md` gives setup and reproduction commands; `RESULTS.md` maps each result to frozen inputs and generated outputs; `analysis/check_results.py` and `analysis/check_reported_results.py` fail on numerical disagreement. |


Sources consulted: NeurIPS, [“Code and Data Submission Guidelines”](https://neurips.cc/public/guides/CodeSubmissionPolicy);
Papers with Code, [“Tips for Publishing Research Code”](https://github.com/paperswithcode/releasing-research-code).
