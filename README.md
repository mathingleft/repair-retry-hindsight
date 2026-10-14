# Repair, Retry, and the Cost of Hindsight: code and data

Code, frozen experiment records, and analyses for

> Jack Li\* and Sophia Li\*. **Repair, Retry, and the Cost of Hindsight: Verifier
> Feedback in Formal Theorem Proving.** NeurIPS 2026 Workshop on AI for Verifiable
> Coding (VeriCodeGen). (\*Equal contribution; order determined by coin flip.)

Every number, table, and figure in the paper is recomputed from the records in
this repository by `scripts/reproduce.py` (main analyses) and
`analysis/camera_ready/` (analyses added for the camera-ready version). No model
inference or Lean verifier is needed to reproduce them.

This repository contains experiment code, frozen data, exact problem sets,
result analyses, and eight figures. CPU analysis regenerates the numerical
results and figures without model inference, a verifier service, or external data.
Fresh model sampling is a separate, optional workflow.

The repository contains every outcome and input needed for the reported
analyses and figures. It omits ten full sequential traces that
repeat prompts, response text, and histories; proofs, diagnostics, and verdicts
remain in the compact sample-level records. Donor ledgers retain all 4,055 Qwen
`E_m` selections and all 5,755 Qwen plus 5,793 DeepSeek `CE_m` selections, with
source identifiers, classes, lengths, and matching metadata. These were exported
from the full traces after checking their proofs and outcomes against the compact
records. Two unused historical baseline runs are also omitted.
`manifests/package.json` lists the omissions. No additional download is needed.
The omitted traces are redundant for the reported outcome recounts, not for
serving-provenance auditing. Compact adaptive rows lack per-attempt generation
job IDs and serving profiles; this package cannot reconstruct each arm's mixture
of numerical serving configurations or a serving-stratified sensitivity analysis.
Historical protocol addenda describe changes but do not replace a per-row ledger.

Large text records are stored losslessly as compressed `.xz` files.
The reproduction command automatically restores their original filenames and
verifies every restored byte against SHA-256 hashes. This needs only Python's
standard library and no download. Allow about 700 MB for the unpacked records.
To inspect the records in your editor before running analyses, first run:

```bash
python3 scripts/unpack_data.py
```

## 1. Reproduce the results on CPU

From this directory, using Python 3.12:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-analysis.txt
.venv/bin/python scripts/reproduce.py
```

After dependency installation the command is offline. It verifies SHA-256 hashes,
checks sample and trajectory completeness, reconstructs bank selection from the
retained evidence, recounts all conditions, recomputes uncertainty and behavioral
checks, and writes tables and PNG/PDF figures under `generated/`. It finishes with
`PASS` against frozen reported results and recovered historical reference analyses.
Allow several minutes and a few GB of RAM. `--no-figures` skips plotting.
`make reproduce PYTHON=.venv/bin/python` is equivalent.

Start with `generated/checks.json`, `reported_result_checks.json`, `behavioral_examples.json`, `onestep_counts.csv`,
`sequential_horizons.csv`, `sequential_contrasts.csv`, and `selection.json`.
The [result-to-input map](RESULTS.md) locates each family of runs and outputs.
All eight figures in `figures/` are built by the supplied scripts.
Some plotting layouts were reconstructed from the original figure definitions;
the submitted version uses these reproducible exports. Numerical tables and
qualitative examples are checked directly against the retained sample records.

## 2. Inspect the frozen data and selection

All experiment paths are under `experiments/decomp_rerun/`.
`manifests/inputs.json` records hashes of every frozen analysis input, tokenizer,
and environment pin. No credentials or other checkout is required.
`analysis/validate.py` fails on altered inputs, missing/duplicate samples,
incomplete trajectories, or disagreements between included full and compact records.

The selected problems and fixed predecessors are part of the experimental input.
DeepSeek one-step and sequential selection reconstruct exactly from included
screening outcomes. Qwen's retained 171 predecessors minus four fixed exclusions
reproduce the 167-problem one-step bank. Its sequential bank was randomly sampled
from the full 171-problem screened pool. The [selection account](SELECTION.md)
records the frozen IDs, counts, and replay limits.

Historical launch scripts and notes document the experiments; use the portable
entry points below for fresh experiments.

## 3. Optionally recount tokens or reverify proofs

Install `requirements-runners.txt` alongside the analysis requirements (a
separate environment may be used with both requirement files installed).
The default analysis uses the frozen per-row cache `data/token_counts.csv`.
To independently rebuild that cache from bundled tokenizers and visible responses:

```bash
python analysis/recount_tokens.py
python analysis/supporting.py
python analysis/check_results.py
```

This uses CPU tokenization only. It counts reconstructed prompts and visible text,
not billing records. Original output token IDs and complete output text are not
available for every campaign; see [PROTOCOL.md](PROTOCOL.md).

For independent Lean checking, build the pinned service as described in
[ENVIRONMENT.md](ENVIRONMENT.md), then, for example:

```bash
python scripts/reverify.py --model qwen --family onestep \
  --kimina-url http://localhost:8000 --output fresh/qwen-reverified.jsonl
```

Both model names (`qwen`, `deepseek`) and families (`onestep`, `sequential`,
Qwen-only `precision` and `fresh-noi`) are supported. Sequential verification
covers all seven reported arms, including `CE_m`. Qwen `onestep` includes the
fresh failure-notice replicate; `fresh-noi` selects that replicate alone.
`--limit 8` gives a small service smoke test.
Reverification writes a new file and never overwrites frozen outcomes.
To inspect coverage without a service or runner dependencies, after unpacking:

```bash
python3 scripts/reverify.py --model qwen --family sequential --dry-run
python3 analysis/check_reverification_coverage.py
```

The CPU reproduction includes this coverage regression check and writes
`generated/reverification_coverage.json`. Enumeration does not constitute Lean
reverification; a full fresh verifier run remains unperformed during packaging.

## 4. Optionally repeat model generation

Follow [ENVIRONMENT.md](ENVIRONMENT.md) to download pinned weights, start vLLM,
and start Kimina. This step requires GPU resources. No new generation was used to
produce this artifact. Review the command list before launching with `--dry-run`:

```bash
python scripts/rerun.py --model qwen --family onestep \
  --base-url http://localhost:8017 --served-model qwen3.6-27b \
  --kimina-url http://localhost:8000 --output fresh/qwen-onestep --dry-run
```

Remove `--dry-run` to execute. Change the family to `precision` or `sequential`;
for DeepSeek use `--model deepseek --served-model dsp-v2-7b`.
The wrapper preserves bank, prompt builders, sample seeds, and verification rules.
It runs the complete arm family into the chosen output directory. Original
campaigns distributed arms across jobs; fresh runs use the supplied server.
For a second Qwen failure-notice replicate use `--family fresh-noi`.

Seeds do not make fresh vLLM outputs identical across hardware, batching, or
software versions. In particular, DeepSeek sequential sampling historically
continued across KV-cache and tensor-parallel configurations. Exact reproduction
of reported counts uses the frozen outcomes; fresh sampling repeats the protocol
under the explicitly chosen serving profile.

## 5. Camera-ready analyses

`analysis/camera_ready/` holds the analyses added after review, each reading only
the frozen records and writing JSON to `analysis/camera_ready/out/` (the committed
copies are the reference outputs). See its README for the script-to-result map.
After `scripts/reproduce.py` has unpacked the records:

```bash
.venv/bin/python analysis/camera_ready/run_all.py
```

`r6.py` (token-matched comparison) takes about ten minutes; `drift.py` needs the
two Mathlib source trees fetched by `analysis/camera_ready/fetch_mathlib.sh`.
`r20_reverify.py` re-verifies every retained passing proof and needs the Lean
service from section 3.

## Scope

The scientific claims can be recounted from retained sample-level outcomes.
The release does not claim that missing upstream Qwen screening draws, deleted
launch attempts, or historical server scheduling can be recovered. Old analyses
under `expected/` include superseded sequential variants; only the one-step token
section of that historical token audit is used as a regression reference. The
reported sequential family has seven arms: manuscript `I,C,CE,CE_m,CE_s,E,E_m`,
stored as raw `memI,C,CE,CE_m,CE_s,f,E_s`.

Upstream model/tokenizer and Kimina license notices remain with their files.
The artifact contains no model weights; model downloads use the public revisions
listed in `environments/revisions.json`.

See [ASSETS.md](ASSETS.md) for upstream attribution and
[RELEASE_AUDIT.md](RELEASE_AUDIT.md) for the release and code-completeness audit.

Paths of the form `/scratch/anonymous/...` inside frozen records and historical
notes are the operational paths as exported for anonymous review. They are kept
byte-for-byte so that every record matches the hashes in `manifests/`; the
launch scripts read the equivalent location from `$VERICODE_SCRATCH`
(see [ENVIRONMENT.md](ENVIRONMENT.md)).

## Contents

| Path | Contents |
|---|---|
| `experiments/` | Experimental runners, prompt builders, frozen outcomes, banks, screening records, and historical launch records |
| `analysis/` | Recounts, statistical and behavioral analyses, selection validation, and figure generation |
| `analysis/camera_ready/` | Analyses added for the camera-ready paper, with reference outputs |
| `scripts/` | CPU reproduction and optional generation, model serving, and proof verification entry points |
| `figures/` | Eight reference PDF result figures; reproduction writes new PDFs and PNGs to `generated/figures/` |
| `expected/` | Frozen numerical reference tables and recovered historical analysis outputs |
| `data/`, `selection/` | Cached token counts and exact problem-selection evidence |
| `environments/`, `requirements-*.txt` | Dependency and model revision pins, setup configuration |
| `manifests/` | Input and release hashes, documented optional trace omissions |
| `protocols/` | Recorded experimental protocols |
| `tokenizers/`, `third_party/` | Tokenizer assets, verifier client/setup files, upstream licenses |
| Root Markdown files | Reproduction instructions, result-to-input map, selection rules, experimental details, release audit, and attribution |
| `LICENSE`, `LICENSE-DATA`, `CITATION.cff` | MIT license for code, CC BY 4.0 for data and documents, citation metadata |

## Citation

```bibtex
@inproceedings{li2026repair,
  title     = {Repair, Retry, and the Cost of Hindsight: Verifier Feedback in Formal Theorem Proving},
  author    = {Li, Jack and Li, Sophia},
  booktitle = {NeurIPS 2026 Workshop on AI for Verifiable Coding},
  year      = {2026}
}
```

## License

Code is released under the [MIT License](LICENSE). Experiment records, expected
outputs, figures, and documents are released under
[CC BY 4.0](LICENSE-DATA). Bundled third-party material (MiniF2F statements,
tokenizer files, the Kimina snapshot) keeps its own license; see
[ASSETS.md](ASSETS.md).
