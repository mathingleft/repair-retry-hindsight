"""Generation runner for the decomposition rerun. OpenAI-compatible API
(llama.cpp `llama-server`, vLLM, anything with /v1/completions).

Phases:
  python run_gen.py --phase reflections --base-url http://localhost:8080
  python run_gen.py --phase main        --base-url http://localhost:8080

Resumable: rows already in the output file are skipped. K=4 samples/arm/problem.
Config mirrors the original design: 16k budget, Qwen non-thinking sampling.
ACTUAL DECODING OF THE COMPLETED RUNS (established from the vLLM log, 2026-08-28):
  the request sets temperature=0.7 and max_tokens=16384 only; vLLM 0.28 then applies
  the model's generation_config.json defaults, logged at startup as
  "Default vLLM sampling parameters have been overridden by the model's
  generation_config.json: {temperature: 1.0, top_k: 20, top_p: 0.95}" — so the
  effective sampling was temperature 0.7, top_k 20, top_p 0.95. (An earlier version
  of this docstring said top_p 0.8; that was never what the request sent.)
  Thinking is disabled in chat() below (enable_thinking=False), not in the sbatch
  files; 0 of the 4008 prereg rollouts contain a <think> tag.
"""
import argparse, asyncio, json, os, sys
import aiohttp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arms import ARMS, build_prompt, build_reflector_prompt

# The original pipeline (globals/utils.py) wraps the user message in the model's chat
# template, and the paper spec is "thinking off" (results/PAPER_OUTLINE.md section 3,
# truncation 1.3-1.8%). Raw /v1/completions text let the model think spontaneously
# (60% of thinking rows hit the 16k cap). enable_thinking=False renders the Qwen
# assistant turn as "<think>\n\n</think>\n\n".
MODEL_DIR = os.environ.get("DECOMP_MODEL_DIR", os.path.join(os.environ.get("VERICODE_SCRATCH", "."), "models/Qwen3.6-27B"))
_tok = None
def chat(user_msg):
    global _tok
    if _tok is None:
        from transformers import AutoTokenizer
        _tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    return _tok.apply_chat_template([{"role": "user", "content": user_msg}], tokenize=False,
                                    add_generation_prompt=True, enable_thinking=False)

K = 4
GEN = dict(max_tokens=16384, temperature=0.7)          # prereg: temp .7, 16k cap
REFL_GEN = dict(max_tokens=1024, temperature=0.7, seed=22072600)  # prereg frozen reflection
BASE_SEED = 22072601                                   # prereg children 22072601..604

def load_bank(path, include_holdouts=False):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    return [r for r in rows if include_holdouts or not r["is_framing_holdout"]]

def done_keys(path):
    if not os.path.exists(path): return set()
    out = set()
    for l in open(path):
        try: d = json.loads(l)
        except Exception: continue
        out.add((d["id"], d.get("arm", "reflection"), d.get("sample_idx", 0)))
    return out

async def complete(session, base_url, model, prompt, gen, sem):
    async with sem:
        for attempt in range(5):
            try:
                async with session.post(f"{base_url}/v1/completions", json=dict(
                        model=model, prompt=chat(prompt), **gen), timeout=aiohttp.ClientTimeout(total=7200)) as r:
                    j = await r.json()
                    c = j["choices"][0]
                    return c["text"], c.get("finish_reason")
            except Exception as e:
                if attempt == 4: return None, f"error:{e}"
                await asyncio.sleep(10 * (attempt + 1))

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["reflections", "main"], required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", default="qwen3.6-27b")
    ap.add_argument("--bank", default=os.path.join(os.path.dirname(__file__), "bank.jsonl"))
    ap.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__), "out"))
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--seed-base", type=int, default=BASE_SEED, help="rollout seeds = seed-base + k (prereg 22072601); a fresh-seed replicate uses e.g. 22072701")
    ap.add_argument("--include-holdouts", action="store_true",
                    help="also generate the 4 is_framing_holdout problems (bank backfill; 2026-08-28)")
    ap.add_argument("--shard", default=None,
                    help="i:N — run only jobs with stable_hash(id,arm,k) %% N == i")
    a = ap.parse_args()
    shard = tuple(int(x) for x in a.shard.split(":")) if a.shard else None
    os.makedirs(a.outdir, exist_ok=True)
    import arms as _arms; _arms.SHUF_BANK = a.bank      # F_E_shuf draws from the bank actually in use (prover: bank2)
    bank = load_bank(a.bank, include_holdouts=a.include_holdouts)
    sem = asyncio.Semaphore(a.concurrency)

    if a.phase == "reflections":
        out = os.path.join(a.outdir, "reflections.jsonl")
        done = done_keys(out)
        todo = [p for p in bank if (p["id"], "reflection", 0) not in done]
        print(f"reflections: {len(todo)} to generate ({len(done)} done)")
        async with aiohttp.ClientSession() as s:
            async def one(p):
                txt, fin = await complete(s, a.base_url, a.model, build_reflector_prompt(p), REFL_GEN, sem)
                with open(out, "a") as f:
                    f.write(json.dumps({"id": p["id"], "reflection": txt, "finish": fin}) + "\n")
            await asyncio.gather(*[one(p) for p in todo])
        return

    refl = {}
    rp = os.path.join(a.outdir, "reflections.jsonl")
    if os.path.exists(rp):
        for l in open(rp):
            d = json.loads(l); refl[d["id"]] = d["reflection"]
    arms = a.arms.split(",")
    if any(x.startswith("F_R") for x in arms):
        missing = [p["id"] for p in bank if p["id"] not in refl]
        assert not missing, f"run --phase reflections first ({len(missing)} missing)"
    out = os.path.join(a.outdir, "rollouts.jsonl")
    done = done_keys(out)
    import hashlib
    def _mine(pid, arm, k):
        if shard is None: return True
        h = int(hashlib.md5(f"{pid}|{arm}|{k}".encode()).hexdigest(), 16)
        return h % shard[1] == shard[0]
    jobs = [(p, arm, k) for p in bank for arm in arms for k in range(K)
            if (p["id"], arm, k) not in done and _mine(p["id"], arm, k)]
    print(f"main: {len(jobs)} generations to run ({len(done)} done)")
    async with aiohttp.ClientSession() as s:
        async def one(p, arm, k):
            prompt = build_prompt(arm, p, reflection=refl.get(p["id"]))
            gen = dict(GEN, seed=a.seed_base + k)
            txt, fin = await complete(s, a.base_url, a.model, prompt, gen, sem)
            with open(out, "a") as f:
                f.write(json.dumps({"id": p["id"], "arm": arm, "sample_idx": k,
                                    "text": txt, "finish": fin}) + "\n")
        await asyncio.gather(*[one(*j) for j in jobs])
    print("done ->", out)

if __name__ == "__main__":
    asyncio.run(main())
