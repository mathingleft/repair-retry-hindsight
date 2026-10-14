"""Code-bearing sequential depth arms C / CE / CE_s on the shared cluster stack (2026-09-07, user request).

Same protocol, problems, seeds, decoding, verifier and scaffold as the local depth arms (depth_run.py: assert, f,
noise) and the memoryless_I control: 11 problems (depth_bank.jsonl) x 10 trajectories x <=64 attempts, early stop
at the first PASS, attempt 1 = the bare task prompt, attempts >1 = depth_run.HEADER + one "## Previous attempt N"
block per prior failed attempt (oldest first, FIFO-evicted beyond depth_run.PROMPT_TOKEN_CAP) + depth_run.task(p).
seed = depth_run.seed_for(pid, traj, attempt). Rendered through depth_run.chat() (thinking off).

Only the BLOCK CONTENT differs from the local `f` arm, whose block is
    ## Previous attempt N
    Lean compiler feedback:
    {error}
  C    : ## Previous attempt N / Proof attempt: ```lean4 {own failed code} ```                 (code, no error)
  CE   : ## Previous attempt N / Proof attempt: ```lean4 {own failed code} ``` / Lean compiler feedback: {own error}
  CE_s : the CE block with the error text replaced by ANOTHER problem's error (see below)
The proof/feedback lines are the archived HPC `gf` block lines verbatim (solvers/fifo_refiner.render_fifo_prompt); the
header sentence is the depth arms' HEADER, not the archived HPC REFINE_HEADER, so that C/CE/CE_s differ from the local `f`
arm in block content only. {own failed code} = the verifier's input for that attempt (lean_extract.extract_lean_code
splice: preamble + statement + model body), exactly what the archived HPC runner put in its window (`lean_code`).
Error text is capped at 1200 chars as in depth_run.

CE_s error source. Partner problem = seeded derangement over the 11 sorted ids with random.Random(22072605), the
same rng constant as arms.shuffled_partner (which draws over the 167-problem bank, so its partners would fall
outside the 11). For the block describing prior attempt j of trajectory t, the error shown is the CE arm's error
for (partner, t, j) when that row exists and failed; otherwise a random.Random(seed_for(pid, t, j)) draw from
the partner's pool of CE errors; otherwise from the partner's local `f`-arm errors. The source used is logged per
block (shuf_sources). CE must therefore be complete before CE_s runs (--ce-log).

Logging (one row per attempt, <outdir>/<arm>_rows.jsonl, resumable by (problem_id, trajectory_id, attempt_id)):
problem_id, trajectory_id, attempt_id, seed, user_prompt, full_prompt (chat-templated string sent), history
(the rendered blocks), window_len, prompt/response token counts (server usage + tokenizer recount), raw_response,
finish_reason, gen_status, extracted_proof, verification_result, compiler_output, infrastructure_status, timings,
job id. At the end the depth-format mirror out/depth_<arm>.jsonl (arm,id,traj,attempt,status,error,code) is
rewritten from the rows file for analyze_depth.py / analyze_memI.py.
"""
import argparse, asyncio, json, os, random, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import depth_run
from depth_run import chat, task, seed_for, GEN, M_TRAJ, B_ATT, HEADER, PROMPT_TOKEN_CAP, verify
from lean_extract import extract_lean_code

BANK = os.environ.get("DECOMP_DEPTH_BANK") or os.path.join(HERE, "depth_bank.jsonl")
ARMS = ["C", "CE", "CE_s", "E_s", "CE_m"]
# CE_m (2026-09-12, user request): the code-bearing twin of E_s. The CE block (own failed code + "Lean compiler
# feedback:" line) with the error text replaced by a class + length MATCHED wrong error (MatchedErrors, the E_s /
# one-step F_E_match policy) drawn from the CE arm's own error pool (<depth-outdir>/depth_CE.jsonl: errors produced
# under the same code-bearing regime, same stack and verifier), never the same problem, one use per trajectory,
# seeded ties, per-block provenance logged. Unlike CE_s (fixed partner problem, position-matched errors, no class or
# length control), CE_m controls the wrong error's class and length, so CE - CE_m isolates the error's problem-specific
# content beside the own failed code exactly as f - E_s does without it.
ERR_CAP = 1200
DERANGE_SEED = 22072605

# E_s (= f_s, E_s^seq; 2026-09-08, user request): the code-free analogue of CE_s. Byte-identical to the local `f` arm
# (depth_run.block("f", n, error)) except that every block's error is a WRONG diagnostic from another problem,
# matched to the trajectory's own error at that attempt by the one-step F_E_match policy (build_match.py):
# same coarse class (build_match.classify on the first message), token length as close as possible
# (cost = |log(tok_src / tok_own)|, model tokenizer), never the same problem, identical text penalised. The pool is
# the local `f` arm's own errors (out/depth_f.jsonl, same stack and verifier). Online, so instead of a perfect
# matching the reuse rule is: a source row is used at most once per trajectory; ties broken by
# random.Random(seed_for(pid, t, j)). Classes with < MIN_CANDS cross-problem candidates fall back to the pooled
# non-syntax family (build_match's tactic_pool rule). Per block the source row, class, lengths, cost are logged.
MIN_CANDS = 5


class MatchedErrors:
    def __init__(self, f_log):
        import math
        from build_match import classify
        self.classify, self.log = classify, math.log
        self.rows = []                                   # (pid, traj, attempt, error, class, ntok)
        for l in open(f_log):
            try: d = json.loads(l)
            except Exception: continue
            if d.get("status") == "FAIL" and d.get("error"):
                e = d["error"][:ERR_CAP]
                self.rows.append((d["id"], d["traj"], d["attempt"], e, classify(e), ntok(e)))
        self.by_class = {}
        for i, r in enumerate(self.rows): self.by_class.setdefault(r[4], []).append(i)
        self.nonsyntax = [i for i, r in enumerate(self.rows) if r[4] != "syntax"]
        assert self.rows, f"empty error pool {f_log}"

    def get(self, pid, t, j, own_error, used):
        """-> (error_text, info) for the block describing own attempt j; `used` = source indices already shown
        in this trajectory (mutated)."""
        own = (own_error or "")[:ERR_CAP]; c = self.classify(own) if own else "other"; n = max(1, ntok(own))
        cands = [i for i in self.by_class.get(c, []) if self.rows[i][0] != pid and i not in used]
        pool = c
        if len(cands) < MIN_CANDS:
            cands = [i for i in self.nonsyntax if self.rows[i][0] != pid and i not in used]; pool = "tactic_pool"
        if not cands:
            cands = [i for i in range(len(self.rows)) if self.rows[i][0] != pid]; pool = "any"
        def cost(i):
            r = self.rows[i]
            return (50.0 if r[3] == own else 0.0) + abs(self.log(max(r[5], 1) / n))
        best = min(cost(i) for i in cands)
        ties = [i for i in cands if cost(i) == best]
        i = random.Random(seed_for(pid, t, j)).choice(ties); used.add(i)
        r = self.rows[i]
        return r[3], {"src": f"{r[0]}|{r[1]}|{r[2]}", "own_class": c, "src_class": r[4], "pool": pool,
                      "own_tok": n, "src_tok": r[5], "cost": round(best, 4), "ties": len(ties)}


def partners(ids):
    rng = random.Random(DERANGE_SEED); m = {}
    for pid in sorted(ids):
        m[pid] = rng.choice([q for q in sorted(ids) if q != pid])
    return m


def block(arm, n, code, error):
    lines = [f"## Previous attempt {n}", "Proof attempt:\n```lean4\n" + (code or "").rstrip() + "\n```"]
    if arm in ("CE", "CE_s", "CE_m"):
        lines.append("Lean compiler feedback:\n" + ((error or "(no output)").rstrip()))
    return "\n".join(lines)


def render(arm, p, hist):
    """hist: list of (code, error) per prior failed attempt, oldest first. Returns (user_prompt, history_text)."""
    if not hist:
        return depth_run._bare(p), ""
    if arm == "E_s":                                   # the f arm's block, byte for byte (error only)
        blocks = [depth_run.block("f", i + 1, e) for i, (c, e) in enumerate(hist)]
    else:
        blocks = [block(arm, i + 1, c, e) for i, (c, e) in enumerate(hist)]
    while blocks and sum(len(b) for b in blocks) / 3.5 > PROMPT_TOKEN_CAP:
        blocks.pop(0)                                  # FIFO eviction, oldest first (depth_run rule)
    history = "\n\n".join(blocks)
    return "\n\n".join([HEADER, history, task(p)]), history


def key(d): return (d["problem_id"], d["trajectory_id"], d["attempt_id"])


def load_rows(path):
    if not os.path.exists(path): return []
    out = []
    for l in open(path):
        try: out.append(json.loads(l))
        except Exception: pass
    return out


def ntok(text):
    if not text: return 0
    depth_run.chat("x")
    return len(depth_run._tok(text, add_special_tokens=False)["input_ids"])


class ShufErrors:
    """Partner-problem error lookup for CE_s (see module docstring)."""
    def __init__(self, ids, ce_log, f_log):
        self.partner = partners(ids)
        self.at = {}; self.pool = {}; self.fpool = {}
        for d in load_rows(ce_log) if ce_log else []:
            if d.get("verification_result") == "FAIL" and d.get("compiler_output"):
                self.at[(d["problem_id"], d["trajectory_id"], d["attempt_id"])] = d["compiler_output"][:ERR_CAP]
                self.pool.setdefault(d["problem_id"], []).append(d["compiler_output"][:ERR_CAP])
        if f_log and os.path.exists(f_log):
            for l in open(f_log):
                try: d = json.loads(l)
                except Exception: continue
                if d.get("status") == "FAIL" and d.get("error"):
                    self.fpool.setdefault(d["id"], []).append(d["error"][:ERR_CAP])
        for pid in ids:
            q = self.partner[pid]
            assert self.pool.get(q) or self.fpool.get(q), f"no partner errors for {pid} -> {q}"

    def get(self, pid, t, j):
        q = self.partner[pid]
        if (q, t, j) in self.at: return self.at[(q, t, j)], f"CE:{q}|{t}|{j}"
        rng = random.Random(seed_for(pid, t, j))
        if self.pool.get(q): return rng.choice(sorted(self.pool[q])), f"CE_pool:{q}"
        return rng.choice(sorted(self.fpool[q])), f"f_pool:{q}"


async def gen(session, a, full_prompt, seed, sem):
    body = dict(model=a.model, prompt=full_prompt, seed=seed, **GEN)
    async with sem:
        for att in range(5):
            try:
                async with session.post(f"{a.base_url}/v1/completions", json=body,
                                        timeout=aiohttp_timeout()) as r:
                    j = await r.json(); c = j["choices"][0]
                    return c["text"], c.get("finish_reason"), j.get("usage") or {}, None
            except Exception as e:
                err = f"error:{e}"
                if att < 4: await asyncio.sleep(10 * (att + 1))
    return None, None, {}, err


def aiohttp_timeout():
    import aiohttp
    return aiohttp.ClientTimeout(total=7200)


async def trajectory(a, p, t, hist, start, session, sem, out, shuf):
    """hist: list of (code, error) already logged for this trajectory (resume)."""
    matched, used = [], set()                        # E_s: one matched wrong error per own failed attempt, fixed once drawn
    for k in range(start, B_ATT + 1):
        srcs = []
        if a.arm == "CE_s":
            shown = []
            for j, (c, e) in enumerate(hist, 1):
                se, src = shuf.get(p["id"], t, j); shown.append((c, se)); srcs.append(src)
        elif a.arm in ("E_s", "CE_m"):
            while len(matched) < len(hist):          # replay in order (also rebuilds the resume state deterministically)
                j = len(matched) + 1
                matched.append(shuf.get(p["id"], t, j, hist[j - 1][1], used))
            shown = [(c, m[0]) for (c, e), m in zip(hist, matched)]
            srcs = [m[1] for m in matched]
        else:
            shown = hist
        user_prompt, history = render(a.arm, p, shown)
        full_prompt = chat(user_prompt); seed = seed_for(p["id"], t, k)
        t0 = time.time()
        txt, fin, usage, gerr = await gen(session, a, full_prompt, seed, sem)
        t1 = time.time()
        code = extract_lean_code(txt, p["statement"]) if txt else None
        st, verr = await asyncio.get_event_loop().run_in_executor(None, verify, a.kimina_url, code)
        t2 = time.time()
        infra = "ok"
        if gerr: infra = "gen_error_after_retries: " + gerr[:200]
        elif verr and ("verifier crash" in verr or "header command timed out" in verr):
            infra = "infra_error_after_retries: " + verr[:200]
        rec = {"arm": a.arm, "problem_id": p["id"], "trajectory_id": t, "attempt_id": k, "seed": seed,
               "user_prompt": user_prompt, "full_prompt": full_prompt, "history": history, "window_len": len(shown),
               "shuf_sources": srcs if a.arm in ("CE_s", "E_s", "CE_m") else None,
               "prompt_token_count": usage.get("prompt_tokens", ntok(full_prompt)),
               "prompt_token_count_tokenizer": ntok(full_prompt),
               "raw_response": txt, "response_token_count": usage.get("completion_tokens", ntok(txt)),
               "response_token_count_tokenizer": ntok(txt),
               "finish_reason": fin, "gen_status": "ok" if gerr is None else gerr,
               "extracted_proof": code, "verification_result": st, "compiler_output": verr,
               "infrastructure_status": infra,
               "gen_seconds": round(t1 - t0, 2), "verify_seconds": round(t2 - t1, 2),
               "job": os.environ.get("SLURM_JOB_ID"),
               "sampling": dict(GEN, seed=seed, note="top_k 20 / top_p 0.95 from generation_config.json via vLLM; enable_thinking=False")}
        with open(out, "a") as f: f.write(json.dumps(rec) + "\n")
        if st == "PASS": return
        hist.append((code, (verr or "")[:ERR_CAP]))


def mirror(rows_path, depth_path):
    rows = sorted(load_rows(rows_path), key=key)
    with open(depth_path, "w") as f:
        for d in rows:
            f.write(json.dumps({"arm": d["arm"], "id": d["problem_id"], "traj": d["trajectory_id"],
                                "attempt": d["attempt_id"], "status": d["verification_result"],
                                "error": d["compiler_output"], "code": d["extracted_proof"]}) + "\n")
    return len(rows)


async def main():
    import aiohttp
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=ARMS, required=True)
    ap.add_argument("--base-url", required=True); ap.add_argument("--model", default="qwen3.6-27b")
    ap.add_argument("--kimina-url", required=True)
    ap.add_argument("--concurrency", type=int, default=48)
    ap.add_argument("--outdir", default=os.path.join(os.path.dirname(HERE), "out_codearms"))
    ap.add_argument("--depth-outdir", default=os.path.join(os.path.dirname(HERE), "out"))
    ap.add_argument("--ce-log", default=None, help="CE_s: CE rows file (default <outdir>/CE_rows.jsonl)")
    ap.add_argument("--f-log", default=None, help="CE_s fallback pool / E_s pool (default <depth-outdir>/depth_f.jsonl)")
    ap.add_argument("--pool-log", default=None, help="E_s / CE_m matched-error pool (depth-format jsonl; CE_m default <depth-outdir>/depth_CE.jsonl)")
    a = ap.parse_args(); os.makedirs(a.outdir, exist_ok=True)
    bank = [json.loads(l) for l in open(BANK) if l.strip()]
    ids = [p["id"] for p in bank]
    out = os.path.join(a.outdir, f"{a.arm}_rows.jsonl")
    shuf = None
    if a.arm == "CE_s":
        ce_log = a.ce_log or os.path.join(a.outdir, "CE_rows.jsonl")
        f_log = a.f_log or os.path.join(a.depth_outdir, "depth_f.jsonl")
        shuf = ShufErrors(ids, ce_log, f_log)
        print("CE_s partners:", json.dumps(shuf.partner, indent=1), flush=True)
    elif a.arm in ("E_s", "CE_m"):
        pool = a.pool_log or a.f_log or os.path.join(a.depth_outdir, "depth_f.jsonl" if a.arm == "E_s" else "depth_CE.jsonl")
        shuf = MatchedErrors(pool)
        import collections
        print(f"{a.arm} pool: {len(shuf.rows)} errors from {pool}; classes",
              dict(collections.Counter(r[4] for r in shuf.rows)), flush=True)
    # resume state from the rows file
    state = {}
    for d in sorted(load_rows(out), key=key):
        e = state.setdefault((d["problem_id"], d["trajectory_id"]), {"hist": [], "next": 1, "solved": False})
        if d["attempt_id"] != e["next"]: continue          # ignore out-of-order/duplicate partial rows
        e["next"] = d["attempt_id"] + 1
        if d["verification_result"] == "PASS": e["solved"] = True
        else: e["hist"].append((d["extracted_proof"], (d["compiler_output"] or "")[:ERR_CAP]))
    todo = [(p, t) for p in bank for t in range(M_TRAJ)
            if not state.get((p["id"], t), {}).get("solved") and state.get((p["id"], t), {"next": 1})["next"] <= B_ATT]
    print(f"{a.arm}: {len(todo)} trajectories to run/resume ({len(state)} with logged rows)", flush=True)
    sem = asyncio.Semaphore(a.concurrency)
    async with aiohttp.ClientSession() as s:
        await asyncio.gather(*[trajectory(a, p, t, list(state.get((p["id"], t), {}).get("hist", [])),
                                          state.get((p["id"], t), {}).get("next", 1), s, sem, out, shuf)
                               for p, t in todo])
    n = mirror(out, os.path.join(a.depth_outdir, f"depth_{a.arm}.jsonl"))
    print(f"done -> {out} ; depth mirror {n} rows", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
