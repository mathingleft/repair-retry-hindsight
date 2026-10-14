"""Cold screen + depth-bank selection for a second model (DeepSeek-Prover-V2-7B, 2026-09-11).

The Qwen depth bank (depth_bank.jsonl) is the 11-problem head-to-head set chosen as hard-but-not-hopeless for
Qwen3.6-27B, with cold pass counts out of 256 (0,0,0,0,1,2,3,4,9,41,45). For the prover we have no such set, so:

  --phase screen : n cold draws per problem of the BARE task prompt (depth_run._bare, header per
                   DECOMP_TASK_HEADER) over the prover's own bank (out_m2/bank2.jsonl, its 137 frozen-failure
                   problems), same decoding as every arm (GEN), seed = 700000 + md5(pid|screen|k) % 100000,
                   verified in-loop (depth_run.verify). Resumable. -> <outdir>/cold_screen.jsonl
  --phase select : per-problem cold rate -> pick 11 problems whose cold-rate profile matches the Qwen bank's,
                   target by target (Qwen targets sorted descending; for each, the unused prover problem with the
                   closest cold rate; ties: prefer problems with >= 1 PASS anywhere in the prover's one-step arms
                   (evidence of solvability, since Qwen's 0/256 problems were still solvable at depth), then a seeded
                   name-blind order (random.Random(22072606)).
                   -> <out-bank> (id, statement, base_pass, base_n, onestep_pass, onestep_n, qwen_target) and
                   <outdir>/SELECTION.txt
"""
import argparse, asyncio, collections, hashlib, json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import depth_run
from depth_run import chat, GEN, verify
from lean_extract import extract_lean_code


def seed_for(pid, k):
    return 700000 + int(hashlib.md5(f"{pid}|screen|{k}".encode()).hexdigest(), 16) % 100000


def load_rows(path):
    if not os.path.exists(path): return []
    out = []
    for l in open(path):
        try: out.append(json.loads(l))
        except Exception: pass
    return out


async def screen(a):
    import aiohttp
    bank = [json.loads(l) for l in open(a.bank) if l.strip()]
    out = os.path.join(a.outdir, "cold_screen.jsonl")
    done = {(d["id"], d["k"]) for d in load_rows(out)}
    jobs = [(p, k) for p in bank for k in range(1, a.n + 1) if (p["id"], k) not in done]
    print(f"screen: {len(bank)} problems x {a.n} draws; {len(jobs)} to run ({len(done)} done)", flush=True)
    sem = asyncio.Semaphore(a.concurrency); lock = asyncio.Lock()
    async with aiohttp.ClientSession() as s:
        async def one(p, k):
            prompt = depth_run._bare(p); full = chat(prompt); seed = seed_for(p["id"], k)
            body = dict(model=a.model, prompt=full, seed=seed, **GEN)
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
            code = extract_lean_code(txt, p["statement"]) if txt else None
            st, verr = await asyncio.get_event_loop().run_in_executor(None, verify, a.kimina_url, code)
            infra = "ok"
            if err: infra = "gen_error_after_retries: " + err[:200]
            elif verr and ("verifier crash" in verr or "header command timed out" in verr):
                infra = "infra_error_after_retries: " + verr[:200]
            rec = {"id": p["id"], "k": k, "seed": seed, "status": st, "error": verr, "code": code, "raw": txt,
                   "finish_reason": fin, "gen_status": "ok" if err is None else err, "infrastructure_status": infra,
                   "prompt_tokens": usage.get("prompt_tokens"), "response_tokens": usage.get("completion_tokens"),
                   "seconds": round(time.time() - t0, 2), "job": os.environ.get("SLURM_JOB_ID")}
            async with lock:
                with open(out, "a") as f: f.write(json.dumps(rec) + "\n")
        await asyncio.gather(*[one(p, k) for p, k in jobs])
    print("done ->", out, flush=True)


def select(a):
    bank = {json.loads(l)["id"]: json.loads(l) for l in open(a.bank) if l.strip()}
    rows = load_rows(os.path.join(a.outdir, "cold_screen.jsonl"))
    n = collections.Counter(r["id"] for r in rows); c = collections.Counter(r["id"] for r in rows if r["status"] == "PASS")
    bad = sum(1 for r in rows if r["gen_status"] != "ok" or r["infrastructure_status"] != "ok")
    rate = {pid: c[pid] / n[pid] for pid in n}
    one = json.load(open(a.onestep)) if a.onestep and os.path.exists(a.onestep) else {}
    qwen = sorted((d["base_pass"] / d["base_n"] for d in map(json.loads, open(a.qwen_bank))), reverse=True)
    import random
    order = {pid: i for i, pid in enumerate(random.Random(22072606).sample(sorted(rate), len(rate)))}   # seeded, name-blind tie-break
    # --require-evidence (amendment 2026-09-11 12:20 PT, before any arm data was analysed): candidates must show
    # solvability somewhere (>= 1 cold PASS or >= 1 one-step PASS). The prover's screen put 92/137 problems at 0/32 and
    # only 2 of those had a one-step pass, so the preregistered tie-break alone filled the zero-target slots with
    # problems no arm could ever solve; the depth contrasts need problems that can move.
    pool = [pid for pid in rate if pid in bank and (not a.require_evidence or c[pid] > 0 or one.get(pid, [0, 0])[0] > 0)]
    used = set(); picks = []
    for tgt in qwen:
        cands = [pid for pid in pool if pid not in used]
        key = lambda pid: (abs(rate[pid] - tgt), 0 if one.get(pid, [0, 0])[0] > 0 else 1, order[pid])
        best = min(cands, key=key); used.add(best); picks.append((tgt, best))
    lines = [("selection rule: solvability evidence required (amendment 12:20 PT 2026-09-11)" if a.require_evidence else "selection rule: preregistered tie-break only"),
             f"cold screen: {len(n)} problems, {sum(n.values())} draws ({bad} non-ok rows), n per problem = {sorted(set(n.values()))}",
             f"cold-rate distribution: zero {sum(1 for p in rate if rate[p] == 0)}, (0, .05] {sum(1 for p in rate if 0 < rate[p] <= .05)}, "
             f"(.05, .2] {sum(1 for p in rate if .05 < rate[p] <= .2)}, > .2 {sum(1 for p in rate if rate[p] > .2)}",
             "", "selected (qwen target -> prover problem: cold pass / n, one-step passes / samples):"]
    with open(a.out_bank, "w") as f:
        for tgt, pid in picks:
            d = bank[pid]; op = one.get(pid, [0, 0])
            f.write(json.dumps({"id": pid, "statement": d["statement"], "base_pass": c[pid], "base_n": n[pid],
                                "onestep_pass": op[0], "onestep_n": op[1], "qwen_target": round(tgt, 4)}) + "\n")
            lines.append(f"  {tgt:.4f} -> {pid:32s} {c[pid]:3d}/{n[pid]}  one-step {op[0]}/{op[1]}")
    open(os.path.join(a.outdir, "SELECTION.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines)); print("->", a.out_bank)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["screen", "select"], required=True)
    ap.add_argument("--bank", default=os.path.join(os.path.dirname(HERE), "out_m2", "bank2.jsonl"))
    ap.add_argument("--qwen-bank", default=os.path.join(HERE, "depth_bank.jsonl"))
    ap.add_argument("--onestep", default=os.path.join(os.path.dirname(HERE), "out_dsp", "prover_onestep_passes.json"))
    ap.add_argument("--out-bank", default=os.path.join(HERE, "depth_bank_dsp.jsonl"))
    ap.add_argument("--outdir", default=os.path.join(os.path.dirname(HERE), "out_dsp"))
    ap.add_argument("--require-evidence", action="store_true")
    ap.add_argument("--n", type=int, default=32); ap.add_argument("--concurrency", type=int, default=48)
    ap.add_argument("--base-url"); ap.add_argument("--model", default="dsp-v2-7b"); ap.add_argument("--kimina-url")
    a = ap.parse_args(); os.makedirs(a.outdir, exist_ok=True)
    if a.phase == "screen": asyncio.run(screen(a))
    else: select(a)


if __name__ == "__main__":
    main()
