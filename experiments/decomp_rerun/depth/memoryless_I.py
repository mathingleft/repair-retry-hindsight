"""Memoryless instruction-only retry control for the 11-theorem sequential experiment (2026-09-06, user spec).

Every draw gets the SAME fixed prompt with no history of any kind:

    Now produce a correct, complete proof.

    ```lean4
    {statement}
    ```

i.e. depth_run.task(p): the instruction line + theorem in exactly the serialization the FIFO / depth runners
end their prompts with (experiments/solvers/fifo_refiner.py render_fifo_prompt, depth_run.render). Rendered
through depth_run.chat() (Qwen chat template, enable_thinking=False), the same code path as the depth arms.

Grid: 11 problems x 10 trajectories x 64 attempts = 7,040 draws; seed = depth_run.seed_for(pid, traj, attempt),
the deterministic per-(problem, trajectory, attempt) grid shared by every depth arm in this directory. No early
stop: all 640 draws per problem are generated so pass@k is estimable at every k.

Phases (resumable, keyed by (id, traj, attempt)):
  gen    : --phase gen  --base-url http://localhost:PORT [--shard i:N]  -> <outdir>/memI_rollouts.jsonl
           keeps full_prompt (chat-templated string actually sent), raw_response, token counts (server usage +
           tokenizer recount), finish_reason, seed, request timing.
  verify : --phase verify --kimina-url http://localhost:PORT           -> <outdir>/memI_verified.jsonl
           final corrected verifier: lean_extract.extract_lean_code splice, 120 s timeout, heartbeats 400000
           in the preamble, sorry/admit screen (verify_run.SORRY_RE), transient infra errors retried
           (verify_run.verify_codes), infrastructure_status recorded separately from the mathematical verdict.
           Raw text is retained in the verified rows too.
"""
import argparse, asyncio, hashlib, json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import depth_run                                     # chat(), task(), seed_for(), GEN, M_TRAJ, B_ATT
from depth_run import chat, task, seed_for, GEN, M_TRAJ, B_ATT

BANK = os.environ.get("DECOMP_DEPTH_BANK") or os.path.join(HERE, "depth_bank.jsonl")
ARM = "memoryless_I"

def key(d): return (d["problem_id"], d["trajectory_id"], d["attempt_id"])

def load_rows(path):
    if not os.path.exists(path): return []
    out = []
    for l in open(path):
        try: out.append(json.loads(l))
        except Exception: pass                   # partial trailing line
    return out

def mine(shard, pid, t, k):
    if shard is None: return True
    h = int(hashlib.md5(f"{pid}|{t}|{k}".encode()).hexdigest(), 16)
    return h % shard[1] == shard[0]

def ntok(text):
    if not text: return 0
    depth_run.chat("x")                          # ensure tokenizer loaded
    return len(depth_run._tok(text, add_special_tokens=False)["input_ids"])

async def gen_phase(a):
    import aiohttp
    shard = tuple(int(x) for x in a.shard.split(":")) if a.shard else None
    bank = [json.loads(l) for l in open(BANK) if l.strip()]
    out = os.path.join(a.outdir, "memI_rollouts.jsonl")
    done = {key(d) for d in load_rows(out)}
    jobs = [(p, t, k) for p in bank for t in range(M_TRAJ) for k in range(1, B_ATT + 1)
            if (p["id"], t, k) not in done and mine(shard, p["id"], t, k)]
    print(f"{ARM} gen: {len(jobs)} draws to run ({len(done)} done)", flush=True)
    sem = asyncio.Semaphore(a.concurrency)
    async with aiohttp.ClientSession() as s:
        async def one(p, t, k):
            user_prompt = task(p); full_prompt = chat(user_prompt); seed = seed_for(p["id"], t, k)
            body = dict(model=a.model, prompt=full_prompt, seed=seed, **GEN)
            txt = fin = None; usage = {}; err = None; t0 = time.time()
            async with sem:
                for att in range(5):
                    try:
                        async with s.post(f"{a.base_url}/v1/completions", json=body,
                                          timeout=aiohttp.ClientTimeout(total=7200)) as r:
                            j = await r.json(); c = j["choices"][0]
                            txt, fin, usage, err = c["text"], c.get("finish_reason"), j.get("usage") or {}, None
                            break
                    except Exception as e:
                        err = f"error:{e}"
                        if att < 4: await asyncio.sleep(10 * (att + 1))
            rec = {"arm": ARM, "problem_id": p["id"], "trajectory_id": t, "attempt_id": k, "seed": seed,
                   "user_prompt": user_prompt, "full_prompt": full_prompt,
                   "prompt_token_count": usage.get("prompt_tokens", ntok(full_prompt)),
                   "prompt_token_count_tokenizer": ntok(full_prompt),
                   "raw_response": txt, "response_token_count": usage.get("completion_tokens", ntok(txt)),
                   "response_token_count_tokenizer": ntok(txt),
                   "finish_reason": fin, "gen_status": "ok" if err is None else err,
                   "gen_seconds": round(time.time() - t0, 2), "gen_job": os.environ.get("SLURM_JOB_ID"),
                   "sampling": dict(GEN, seed=seed, note="top_k 20 / top_p 0.95 from generation_config.json via vLLM; enable_thinking=False")}
            with open(out, "a") as f: f.write(json.dumps(rec) + "\n")
        await asyncio.gather(*[one(*j) for j in jobs])
    print("done ->", out, flush=True)

def verify_phase(a):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    import verify_run                                   # final corrected verifier pieces
    from lean_extract import extract_lean_code
    bank = {json.loads(l)["id"]: json.loads(l) for l in open(BANK) if l.strip()}
    src = os.path.join(a.outdir, "memI_rollouts.jsonl"); dst = os.path.join(a.outdir, "memI_verified.jsonl")
    done = {key(d) for d in load_rows(dst)}
    rows = [r for r in load_rows(src) if key(r) not in done and r.get("gen_status") == "ok"]
    print(f"{ARM} verify: {len(rows)} rollouts to verify ({len(done)} done)", flush=True)
    va = argparse.Namespace(kimina_url=a.kimina_url, timeout=a.timeout)
    lock = threading.Lock()
    def work(r):
        cid = f"{r['problem_id']}|{r['trajectory_id']}|{r['attempt_id']}"
        txt = r.get("raw_response")
        if not txt:
            rec = dict(r, extracted_proof=None, verification_result="FAIL", compiler_output="no code extracted",
                       infrastructure_status="ok")
        else:
            code = extract_lean_code(txt, bank[r["problem_id"]]["statement"])
            res = verify_run.verify_codes(va, [{"custom_id": cid, "proof": code}])
            item = res.get(cid, {"error": "no result returned"})
            st, err = verify_run.verdict(item)
            infra = "ok"
            e = str(item.get("error") or "")
            if "verifier crash" in e or "no result returned" in e or any(t in e for t in verify_run.TRANSIENT):
                infra = "infra_error_after_retries: " + e[:200]
            if st == "FAIL" and code and "extract_lean_code_no_body_parsed" in code:
                err = "no proof body parsed (unparseable output); " + (err or "")
            msgs = ((item.get("response") or {}).get("messages") or item.get("messages") or [])
            rec = dict(r, extracted_proof=code, verification_result=st, compiler_output=err,
                       compiler_messages=[{"severity": m.get("severity"), "data": (m.get("data") or "")[:2000]} for m in msgs][:20],
                       infrastructure_status=infra, verify_job=os.environ.get("SLURM_JOB_ID"))
        with lock:
            with open(dst, "a") as f: f.write(json.dumps(rec) + "\n")
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for i, _ in enumerate(ex.map(work, rows), 1):
            if i % 100 == 0 or i == len(rows): print(f"  {i}/{len(rows)}", flush=True)
    print("done ->", dst, flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["gen", "verify"], required=True)
    ap.add_argument("--base-url"); ap.add_argument("--model", default="qwen3.6-27b")
    ap.add_argument("--kimina-url"); ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("LEAN_SERVER_MAX_REPLS", 8)))
    ap.add_argument("--concurrency", type=int, default=48)
    ap.add_argument("--shard", default=None, help="i:N over md5(pid|traj|attempt)")
    ap.add_argument("--outdir", default=os.path.join(os.path.dirname(HERE), "out_memI"))
    a = ap.parse_args(); os.makedirs(a.outdir, exist_ok=True)
    if a.phase == "gen": asyncio.run(gen_phase(a))
    else: verify_phase(a)

if __name__ == "__main__":
    main()
