"""Export one row per attempt, in a single schema, from any of the three attempt-log formats in this project.

    python depth/export_attempt_rows.py --memI  <depth_memI.jsonl>                      --out rows.jsonl
    python depth/export_attempt_rows.py --cscs  <attempts.jsonl> [<raw_outputs.jsonl>] --arm CE --out rows.jsonl
    python depth/export_attempt_rows.py --depth <depth_<arm>.jsonl>                     --out rows.jsonl
    (any number of sources; --problems <bank.jsonl> restricts to the bank's ids; --coverage prints per-field fill)

Output schema (one JSON object per line; a field that the source never logged is null and named in `missing`):
  arm, problem_id, trajectory_id, attempt_id,
  prompt              full prompt sent (chat-templated when the source kept it)
  history             prior-attempt window rendered into the prompt, when the source kept it
  generated_proof     Lean code extracted from the response
  raw_response        full model output
  verification_result PASS | FAIL | TIMEOUT | ERROR ...
  compiler_diagnostic compiler output / error string
  generation_status   stop | length | <runner gen_status>
  infrastructure_status ok | error | null
  input_tokens, output_tokens
  seed
  source, missing

Sources:
  --memI   depth/memoryless_I.py output. Every field is logged.
  --cscs   experiments/run_ttt_solve.py output (the archived HPC code-bearing sequential run): attempts.jsonl
           (AttemptRecord: problem_id, arm, trajectory_idx, attempt_idx, status, error, n_gen_tokens,
           prompt_tokens, truncated, repetitive, ...) joined on (problem_id, trajectory_idx, attempt_idx)
           with raw_outputs.jsonl (generator, lean_code, window_len, reflector). That runner never wrote the
           prompt, a per-request seed, or a separate infrastructure status: prompt/history are null (the
           FIFO prompt is a deterministic function of the prior attempts' lean_code+error and window_len,
           see solvers/fifo_refiner.render_fifo_prompt), seed is null, infrastructure_status is derived
           (ERROR status = verifier exception -> "error", else "ok"), generation_status is derived from
           `truncated`. Token counts are tokenizer recounts, not server usage.
  --depth  depth/depth_run.py output (arm,id,traj,attempt,status,code,error): no prompt, raw text, tokens
           or seed were kept in the merged file.
"""
import argparse, collections, json, sys

FIELDS = ["arm", "problem_id", "trajectory_id", "attempt_id", "prompt", "history", "generated_proof", "raw_response",
          "verification_result", "compiler_diagnostic", "generation_status", "infrastructure_status",
          "input_tokens", "output_tokens", "seed", "source", "missing"]
CONTENT = [f for f in FIELDS if f not in ("arm", "problem_id", "trajectory_id", "attempt_id", "source", "missing")]


def _row(**kw):
    r = {f: kw.get(f) for f in FIELDS}
    r["missing"] = [f for f in CONTENT if r[f] is None]
    return r


def _jsonl(path):
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def from_memI(path):
    for d in _jsonl(path):
        cm = d.get("compiler_output")
        if not cm and d.get("compiler_messages"):
            cm = "\n".join(m if isinstance(m, str) else json.dumps(m) for m in d["compiler_messages"])
        yield _row(arm=d.get("arm", "memoryless_I"), problem_id=d["problem_id"], trajectory_id=d["trajectory_id"],
                   attempt_id=d["attempt_id"], prompt=d.get("full_prompt") or d.get("user_prompt"),
                   history="",  # fixed prompt, no prior-attempt window by construction
                   generated_proof=d.get("extracted_proof"), raw_response=d.get("raw_response"),
                   verification_result=d.get("verification_result"), compiler_diagnostic=cm if cm is not None else "",
                   generation_status=(d.get("finish_reason") or d.get("gen_status")),
                   infrastructure_status=d.get("infrastructure_status"),
                   input_tokens=d.get("prompt_token_count", d.get("prompt_token_count_tokenizer")),
                   output_tokens=d.get("response_token_count", d.get("response_token_count_tokenizer")),
                   seed=d.get("seed"), source=path)


def from_cscs(attempts_path, raw_path=None, arm=None):
    raw = {}
    if raw_path:
        for d in _jsonl(raw_path):
            raw[(d["problem_id"], d["trajectory_idx"], d["attempt_idx"])] = d
    for d in _jsonl(attempts_path):
        key = (d["problem_id"], d["trajectory_idx"], d["attempt_idx"])
        r = raw.get(key, {})
        st = d.get("status")
        yield _row(arm=arm or d.get("arm"), problem_id=d["problem_id"], trajectory_id=d["trajectory_idx"],
                   attempt_id=d["attempt_idx"], prompt=None, history=None,
                   generated_proof=r.get("lean_code"), raw_response=r.get("generator"),
                   verification_result=st, compiler_diagnostic=d.get("error"),
                   generation_status=("length" if d.get("truncated") else "stop") if "truncated" in d else None,
                   infrastructure_status=("error" if st == "ERROR" else "ok") if st else None,
                   input_tokens=d.get("prompt_tokens"), output_tokens=d.get("n_gen_tokens"),
                   seed=None, source=attempts_path)


def from_depth(path):
    for d in _jsonl(path):
        yield _row(arm=d.get("arm"), problem_id=d.get("id") or d.get("problem_id"),
                   trajectory_id=d.get("traj", d.get("trajectory_id")), attempt_id=d.get("attempt", d.get("attempt_id")),
                   generated_proof=d.get("code"), compiler_diagnostic=d.get("error"), verification_result=d.get("status"),
                   source=path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--memI", action="append", default=[])
    ap.add_argument("--cscs", action="append", nargs="+", default=[], metavar=("ATTEMPTS", "RAW_OUTPUTS"))
    ap.add_argument("--depth", action="append", default=[])
    ap.add_argument("--arm", help="arm label for --cscs sources (e.g. CE for gf, C for g)")
    ap.add_argument("--problems", help="bank jsonl; keep only these ids")
    ap.add_argument("--out", required=True)
    ap.add_argument("--coverage", action="store_true")
    a = ap.parse_args()

    keep = None
    if a.problems:
        keep = {d["id"] for d in _jsonl(a.problems)}

    gens = [from_memI(p) for p in a.memI] + [from_cscs(*c, arm=a.arm) for c in a.cscs] + [from_depth(p) for p in a.depth]
    n = 0; filled = collections.Counter(); per_arm = collections.Counter(); seen = set(); dupes = 0
    with open(a.out, "w") as out:
        for g in gens:
            for r in g:
                if keep is not None and r["problem_id"] not in keep:
                    continue
                k = (r["arm"], r["problem_id"], r["trajectory_id"], r["attempt_id"])
                if k in seen: dupes += 1
                seen.add(k)
                out.write(json.dumps(r, ensure_ascii=False) + "\n"); n += 1; per_arm[r["arm"]] += 1
                for f in CONTENT:
                    if r[f] is not None: filled[f] += 1
    print(f"wrote {n} rows -> {a.out}  (dupe keys: {dupes})", file=sys.stderr)
    print("rows per arm:", dict(per_arm), file=sys.stderr)
    if a.coverage and n:
        print("field coverage (non-null / rows):", file=sys.stderr)
        for f in CONTENT:
            print(f"  {f:22s} {filled[f]:6d} / {n}  {'FULL' if filled[f] == n else 'PARTIAL' if filled[f] else 'ABSENT'}", file=sys.stderr)


if __name__ == "__main__":
    main()
