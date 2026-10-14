"""Assertion-only depth arm + same-stack f anchor.

Arms (window = one block per PRIOR failed attempt, oldest first, FIFO-evicted to
fit PROMPT_TOKEN_CAP; scaffold identical across arms, only block CONTENT differs):
  assert : "## Previous attempt N: failed."            (nothing else)
  f      : "## Previous attempt N\nLean compiler feedback:\n{error}"

Protocol matches the original depth runs: 11 problems x 10 trajectories x <=64 attempts,
early-stop on first PASS. Verification in-loop via kimina /verify with the
admit screen. Resumable: (arm, problem, traj) trajectories re-load from the log.

  python depth_run.py --arm assert --base-url http://localhost:8017 \
      --kimina-url http://localhost:8000

Decoding: temperature 0.7 + max_tokens 16384 in the request; top_k 20 / top_p 0.95 come from the
model's generation_config.json via vLLM (see run_gen.py). Thinking off via chat(). Seeds: seed_for()
is deterministic and shared by both arms per (problem, trajectory, attempt) — corrected 2026-08-28;
the FP8 run before that fix (out_depth_fp8.hashseed-1701420) used per-process hash() seeds.
"""
import argparse, asyncio, hashlib, json, os, re, sys
# Soundness screen. Lean 4.29 emits "declaration uses `sorry`" (backticks); older builds used ASCII quotes.
# The screen committed before 2026-08-29 01:30 PT matched only 'sorry' and therefore NEVER fired on this stack
# (bug found by the user; see HANDOFF). Match any quoting, and `sorry` without quotes.
import re as _re
SORRY_RE = _re.compile(r"declaration uses\s+[`'‘\"]?sorry")
import aiohttp, requests

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from lean_extract import extract_lean_code   # pipeline splice: preamble + statement + model body
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

M_TRAJ, B_ATT = 10, 64
GEN = dict(max_tokens=16384, temperature=0.7)
PROMPT_TOKEN_CAP = 7000          # ~chars/3.5; FIFO-evict oldest blocks beyond this
HEADER = ("You previously attempted to prove this theorem and it did not verify. "
          "Your prior attempts are summarized below.")
TASK = "Now produce a correct, complete proof.\n\n```lean4\n{statement}\n```"
FENCE = re.compile(r"```(?:lean4?|lean)\n(.*?)```", re.S)

# Second-model runs (DeepSeek-Prover-V2-7B, 2026-09-11): the prover's one-step arms put the Mathlib header inside the
# code block (arms._task under DECOMP_TASK_HEADER=1). Same switch here; unset (default) leaves every Qwen prompt
# byte-identical. DECOMP_DEPTH_BANK selects the problem bank (default depth_bank.jsonl); DECOMP_SERVED_MODEL the
# vLLM served-model name (default qwen3.6-27b).
_HEADER = ("import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 400000\n\n"
           "open BigOperators Real Nat Topology Rat\n\n")
def _stmt(p):
    s = p["statement"].rstrip()
    if s.endswith("sorry"): s = s[:-5].rstrip()
    if os.environ.get("DECOMP_TASK_HEADER") == "1": s = _HEADER + s
    return s
BANK_PATH = os.environ.get("DECOMP_DEPTH_BANK") or os.path.join(HERE, "depth_bank.jsonl")
SERVED_MODEL = os.environ.get("DECOMP_SERVED_MODEL", "qwen3.6-27b")

def task(p):
    return TASK.format(statement=_stmt(p))

# Depth twin of the one-step nonsense arms (2026-08-28, post-hoc; the five 13-token random-word sentences of
# arms.py N, N2..N5, copied verbatim so this file stays import-free). Arm `noise`: each prior attempt's block
# is "## Previous attempt n: <sentence>", cycling the five sentences by attempt index, so block content is
# disjoint from anything the verifier said. noise vs f = disjoint-content comparison at depth with recomputable
# data (replaces the purged f-vs-r ledger).
NOISE_SENTS = ["Ennui inward evacuates catchier sequel tubing lubbers.",
               "Rosaries fanfares wanton rice dale dirtballs.",
               "Agreeable halting metaphor stunned obsidian minuend chose.",
               "Freelance fossil plenary vocatives erode cinched.",
               "Blue retriever workloads abortions panties mavens whittled."]

# Empirical blind-resampling arm (2026-08-28, post-hoc): every attempt is the bare task prompt, no window, and the
# trajectory does NOT stop at the first PASS, so each (problem, trajectory) is 64 independent cold draws on the
# current stack (640 per problem). discovery@k for this arm = first PASS by attempt k; passes/640 = measured base rate.
# Replaces the synthetic base-iid curve computed from the recovered ~256-draw pool, whose two surviving
# transcriptions disagree and which does not anchor to this stack (cold_anchor.py).
# Second null (2026-08-28 evening): `blind_I` = 64 independent draws of the depth arms' own instruction line
# ("Now produce a correct, complete proof." + statement), no HEADER, no blocks, no early stop. assert - blind_I then
# isolates what the header + accumulating failure markers add over resampling the best content-free prompt
# (the thesis's null; cf. Gu et al. on sequential vs parallel sampling). Differs from the one-step `I` prompt only by
# the word "Now" and the absent "Complete the following Lean 4 code:" line.
def block(arm, n, error):
    if arm == "assert":
        return f"## Previous attempt {n}: failed."
    if arm == "noise":
        return f"## Previous attempt {n}: {NOISE_SENTS[(n - 1) % len(NOISE_SENTS)]}"
    return f"## Previous attempt {n}\nLean compiler feedback:\n{error or '(no output)'}"

def render(arm, p, hist):
    if arm == "blind_I":
        return task(p)           # instruction line only, every attempt
    if arm == "blind_F":
        return HEADER.split(". ")[0] + ".\n\n" + _bare(p)   # sentence-only null (2026-08-29): the one-step F_noI prompt, every attempt
    if arm == "blind" or not hist:
        return _bare(p)          # attempt 1 = bare cold draw, same as FIFO protocol
    blocks = [block(arm, i + 1, e) for i, e in enumerate(hist)]
    while blocks and sum(len(b) for b in blocks) / 3.5 > PROMPT_TOKEN_CAP:
        blocks.pop(0)                                  # FIFO eviction, oldest first
    return "\n\n".join([HEADER] + blocks + [task(p)])

def _bare(p):
    return f"Complete the following Lean 4 code:\n\n```lean4\n{_stmt(p)}\n```"

def extract(text):
    if not text: return None
    m = FENCE.findall(text)
    if m: return m[-1].strip()
    return text.strip() if ("theorem" in text or ":=" in text) else None


def verify(kurl, code, timeout=120):
    if code is None: return "FAIL", "no code extracted"
    item = {}
    for t in range(3):   # a REPL that dies mid-proof makes kimina 500; header timeouts are load noise
        try:
            r = requests.post(f"{kurl}/verify", json={"codes": [{"custom_id": "x", "proof": code}],
                                                         "timeout": timeout, "infotree_type": None},
                              timeout=timeout + 60)
            r.raise_for_status()
            item = (r.json().get("results") or [{}])[0]
        except (requests.RequestException, ValueError) as e:
            item = {"error": f"verifier crash (REPL died / HTTP error): {str(e)[:200]}"}
        e = str(item.get("error") or "")
        if "header command timed out" not in e and "verifier crash" not in e: break
        import time; time.sleep(10)
    if item.get("error"): return "FAIL", str(item["error"])[:300]
    msgs = (item.get("response") or {}).get("messages") or item.get("messages") or []
    errs = [m for m in msgs if m.get("severity") == "error"]
    if errs: return "FAIL", "; ".join((m.get("data") or "")[:160] for m in errs[:4])
    if any(SORRY_RE.search(m.get("data") or "") for m in msgs):
        return "FAIL", "sorryAx screen"
    return "PASS", None

async def gen(session, url, prompt, seed, sem):
    async with sem:
        for a in range(5):
            try:
                async with session.post(f"{url}/v1/completions", json=dict(
                        model=SERVED_MODEL, prompt=chat(prompt), seed=seed, **GEN),
                        timeout=aiohttp.ClientTimeout(total=7200)) as r:
                    j = await r.json(); return j["choices"][0]["text"]
            except Exception as e:
                if a == 4: return None
                await asyncio.sleep(10 * (a + 1))

async def trajectory(arm, p, t, args, sem, log, done):
    hist = list(done.get((p["id"], t), {}).get("hist", []))
    start = done.get((p["id"], t), {}).get("next", 1)
    if done.get((p["id"], t), {}).get("solved") and not arm.startswith("blind"): return
    for k in range(start, B_ATT + 1):
        prompt = render(arm, p, hist)
        txt = await gen(session_holder[0], args.base_url, prompt, seed_for(p["id"], t, k), sem)
        code = extract_lean_code(txt, p["statement"]) if txt else None
        st, err = await asyncio.get_event_loop().run_in_executor(
            None, verify, args.kimina_url, code)
        rec = {"arm": arm, "id": p["id"], "traj": t, "attempt": k,
               "status": st, "error": err, "code": code}
        log.write(json.dumps(rec) + "\n"); log.flush()
        if st == "PASS" and not arm.startswith("blind"): return
        hist.append((err or "")[:1200])

session_holder = [None]

def seed_for(pid, t, k):
    """Deterministic, process-independent seed shared by BOTH arms for the same (problem, trajectory,
    attempt). (Corrected 2026-08-28: the original used Python's per-process-salted hash(), so the arms
    and any resumed run drew different seeds; the FP8 depth run before this fix used those seeds.)"""
    h = int(hashlib.md5(f"{pid}|{t}|{k}".encode()).hexdigest(), 16)
    return 900000 + (h % 100000)

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["assert", "f", "noise", "blind", "blind_I", "blind_F"], required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--kimina-url", required=True)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--outdir", default=os.path.join(HERE, "out"))
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    bank = [json.loads(l) for l in open(BANK_PATH) if l.strip()]
    path = os.path.join(a.outdir, f"depth_{a.arm}.jsonl")
    done = {}
    if os.path.exists(path):
        for l in open(path):
            try: d = json.loads(l)
            except Exception: continue
            key = (d["id"], d["traj"])
            e = done.setdefault(key, {"hist": [], "next": 1, "solved": False})
            e["hist"].append((d.get("error") or "")[:1200])
            e["next"] = d["attempt"] + 1
            if d["status"] == "PASS": e["solved"] = True
    sem = asyncio.Semaphore(a.concurrency)
    async with aiohttp.ClientSession() as s:
        session_holder[0] = s
        log = open(path, "a")
        await asyncio.gather(*[trajectory(a.arm, p, t, a, sem, log, done)
                               for p in bank for t in range(M_TRAJ)])
    print("done ->", path)

if __name__ == "__main__":
    asyncio.run(main())
