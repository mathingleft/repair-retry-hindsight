# Asset provenance and terms

The release contains no model weights. Frozen theorem statements, predecessors,
proof samples and diagnostics are the actual experimental inputs and outputs.
Anonymization for review changed operational account/host metadata only, not
scientific IDs, proof text, sample identities or outcomes.

| Asset | Creator / source | Terms and retained notice |
|---|---|---|
| MiniF2F Lean statements | Kunhao Zheng, Jesse Michael Han, Stanislas Polu; [MiniF2F](https://github.com/openai/miniF2F) | Apache-2.0 for Lean; `third_party/minif2f/LICENSE`. The upstream checkout revision of the inherited Lean 4 conversion was not retained; reproduce with the hashed statements in the four banks. |
| Qwen tokenizer/config | Qwen team; public revision in `environments/revisions.json` | Apache-2.0; `tokenizers/qwen/LICENSE` |
| DeepSeek tokenizer/config | DeepSeek team; public revision in `environments/revisions.json` | MIT; `tokenizers/deepseek/LICENSE` (the `LICENSE` file of the model repository at the pinned revision; its README also refers to a model agreement that is not present at that revision and that in any case permits redistribution with notices) |
| Kimina source | Project Numina | MIT; `third_party/kimina/LICENSE` |
| Lean | [Lean contributors](https://github.com/leanprover/lean4) | Apache-2.0; downloaded separately for optional reverification |
| Mathlib | [Mathlib community](https://github.com/leanprover-community/mathlib4/blob/master/LICENSE) | Apache-2.0; downloaded separately at the frozen revision |
| Lean REPL | [Lean community](https://github.com/leanprover-community/repl/blob/master/LICENSE) | Apache-2.0; downloaded separately at the frozen revision |

The bundled model and third-party licenses apply to their respective assets.
Research code is licensed under MIT (`LICENSE`) and derived experiment records
under CC BY 4.0 (`LICENSE-DATA`).
The artifact does not grant new rights over upstream material. No participant or
private personal dataset was used in these mathematical benchmark experiments.

LLM coding assistance was used during release preparation to recover analysis and
plotting entry points and check the package against frozen observations.
No missing scientific outcome was replaced with an LLM-generated estimate.
