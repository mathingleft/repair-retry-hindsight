# Environments and optional fresh experiments

CPU reproduction needs only `requirements-analysis.txt` under Python 3.12.
The frozen experiment data include a 194-package historical generation freeze
at `experiments/decomp_rerun/figs/env_freeze.txt`. It documents that environment;
it is not the installation recipe for the small CPU analysis environment.

## Verifier

The exact public revisions are in `environments/revisions.json`:

| Component | Pin |
|---|---|
| Lean | v4.29.0 |
| Mathlib | `8a178386ffc0f5fef0b77738bb5449d50efeea95` |
| Lean REPL | `495777293cb9cc3c62787a8c11393da1a9dd9505` |
| Kimina server | `fb2393de3461db35eda4c714e3fd21187e92ec90` |

Install elan using its upstream installation instructions. With `elan`, `lake`,
`git`, Python 3.12, and normal build tools on PATH, create a new service directory:

```bash
python scripts/setup_verifier.py --directory services/kimina
cd services/kimina
LEAN_SERVER_LEAN_VERSION=v4.29.0 LEAN_SERVER_MAX_REPLS=8 \
  LEAN_SERVER_PORT=8000 .venv/bin/python -m server
```

The setup command downloads and builds public dependencies. It copies the frozen
Lake manifests to preserve transitive revisions. It never reuses or overwrites an
existing service directory. This build recipe was assembled from the retained
working stack; a full fresh dependency download/build was not executed for this
release. The small original Kimina source snapshot is also in `third_party/kimina`.

Use 16 CPU cores and 48 GB RAM as the historical verification-job profile, with
8 REPL workers as the conservative startup default. Kimina health alone is not a
functional proof check: run `scripts/smoke_test.py --kimina-url
http://localhost:8000` from the artifact root. It verifies a true theorem and
checks that both a false proof and an admitted proof are unsuccessful.

Verification fixes the dataset theorem statement, prepends imports of Mathlib
and Aesop and `maxHeartbeats 400000`, and extracts the model's proof body.
Requests use a 120-second limit. The corrected warning rule rejects declaration
use of `sorry` (including `admit`), in addition to errors and infrastructure
failures. The original harness is retained, with its limitations.

## Historical launch scripts

The SLURM scripts under `experiments/decomp_rerun/cluster/` are the scripts that
launched the original campaigns, kept for provenance; use the portable entry
points in `scripts/` for new runs. They expect one directory holding model
weights (`models/`), Python environments (`vllm-env/`, `kimina-env/`), the Kimina
service (`kimina-lean-server/`), caches, and a relocated home (`fakehome/`), and
read its location from `VERICODE_SCRATCH`:

```bash
export VERICODE_SCRATCH=/path/to/scratch
```

They stop with a message if it is unset.

## GPU generation

Use a separate CUDA-capable Python environment with vLLM 0.28.0 for the main
campaign profile and install `requirements-runners.txt` for the client runners.
The precision profile instead recorded vLLM 0.21.0; its complete settings and
package versions are in `out_confirm_error/RUN_METADATA.json`. These historical
versions are reported from retained files rather than inferred from today's
package defaults. Do not install conflicting campaign profiles into one venv.

The public weight revisions are:

| Model | Public repository and revision |
|---|---|
| Qwen | `Qwen/Qwen3.6-27B` at `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9` |
| DeepSeek | `deepseek-ai/DeepSeek-Prover-V2-7B` at `a8d9e14432b2e8dd9df2a4d4e70f1ba9bc8d9b7b` |

The actual local tokenizer/config snapshots are bundled. In particular, retain
the DeepSeek tokenizer configuration used after the historical tokenizer-class
repair. The model's originally published tokenizer configuration alone is not a
replacement for this frozen configuration. Weight revisions are recovered from
the local model caches; the original precision metadata records that the two
users' Qwen weight copies could not then be byte-compared.

Example launches (first inspect with `--dry-run`):

```bash
python scripts/serve_model.py --model qwen --tp 2 --dry-run
python scripts/serve_model.py --model deepseek --tp 1 --dry-run
```

Remove `--dry-run` to download/serve weights. Main Qwen used TP4 and later TP2
on L40S GPUs with approximately 46 GB visible memory each. DeepSeek one-step used
one GPU; depth continued from one-GPU bf16 KV to fp8 KV and later TP2. To use the
final DeepSeek depth serving profile choose `--tp 2 --kv-cache fp8`. This is the
practical fresh-run default, not a claim that every historical row used it.

The launcher explicitly supplies effective sampling defaults: Qwen top-k 20,
top-p .95; DeepSeek top-k off, top-p 1. Both runners request temperature .7 and
16,384 output tokens, with a separate 1,024-token reflection generation.
Qwen chat rendering disables thinking. Explicit defaults avoid depending on a
changed public model generation-config file; historical requests sometimes relied
on server defaults instead. Batching and backend differences can change samples.

`scripts/rerun.py` uses supplied model and verifier endpoints. It does not launch
services or reserve GPUs. Sequential order is memoryless I, f, C, CE, CE_s,
CE_m, E_s; the completed CE/f pools supply matched wrong diagnostics to CE_m/E_s.
Outputs are resumable by row
key. A fresh run must use a new directory outside the frozen data tree. Failed
requests are recorded according to the retained runners; examine them before
interpreting a fresh run, rather than silently deleting failed observations.

For fresh one-step outcomes, the original `analyze.py` accepts the new output
directory. For fresh sequential outcomes, use:

```bash
python experiments/decomp_rerun/depth/analyze_memI.py \
  fresh/qwen-sequential/memoryless \
  fresh/qwen-sequential/depth_C.jsonl fresh/qwen-sequential/depth_CE.jsonl \
  fresh/qwen-sequential/depth_CE_m.jsonl fresh/qwen-sequential/depth_CE_s.jsonl \
  fresh/qwen-sequential/depth_f.jsonl \
  fresh/qwen-sequential/depth_E_s.jsonl
```

Generation and complete reverification were deliberately not rerun during this
CPU-only packaging session. A clean fresh server build and a GPU rerun therefore
remain untested paths; the offline numerical reproduction is tested separately.
