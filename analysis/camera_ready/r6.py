"""R6 (+R5 sequential half): input-token cost of sequential repair vs memoryless
retrying, and an equal-input-token comparison.

PC Q1/W1, owTS #4 and the meta-review all ask for a budget-normalized
comparison: attempts are matched, but a repair prompt carries the whole history
while a memoryless prompt is one short instruction.

Exactness:
  - memoryless I: the retained rows carry `user_prompt` verbatim, so its token
    counts are exact (no reconstruction).
  - C, CE, E: the runner's prompt is a deterministic function of the retained
    per-attempt (code, error) history, so prompts are re-rendered with the
    runner's own render()/block() and the same FIFO rule, then tokenized with
    the model's own tokenizer. This reconstructs input volume, not billing.
  - CE_s/CE_m are omitted: their displayed diagnostic comes from a donor ledger
    that is not fully retained, and they are not needed for the cost claim.
  - Output tokens are NOT recoverable for adaptive arms (only extracted proof
    bodies were retained), so this is an input-token comparison.
"""
import collections, json, os, sys
from math import comb
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer

from common import ARTIFACT
ART = ARTIFACT
ROOT = ART/'experiments/decomp_rerun'
DEPTH = ROOT/'depth'
sys.path.insert(0, str(DEPTH)); sys.path.insert(0, str(ROOT))
import types
# depth_run/code_arms_run import HTTP clients at module scope; this analysis only
# uses their pure prompt-rendering functions, so stub the clients out.
for _m in ('aiohttp','requests','openai'):
    if _m not in sys.modules:
        _mod = types.ModuleType(_m)
        _mod.__getattr__ = lambda name: None
        sys.modules[_m] = _mod

SEED, B = 20260906, 10000
ERR_CAP = 1200

def read(p): return [json.loads(l) for l in open(p) if l.strip()]

MODELS = [('qwen','out','depth_bank.jsonl','qwen',None),
          ('deepseek','out_dsp','depth_bank_dsp.jsonl','deepseek','1')]
out = {}
for model, sub, bankfile, tokname, taskhdr in MODELS:
    # the prover puts the Mathlib header inside the code block
    if taskhdr: os.environ['DECOMP_TASK_HEADER'] = taskhdr
    else: os.environ.pop('DECOMP_TASK_HEADER', None)
    os.environ['DECOMP_DEPTH_BANK'] = str(DEPTH/bankfile)
    for m in ('depth_run','code_arms_run'):
        sys.modules.pop(m, None)
    import depth_run, code_arms_run
    tok = Tokenizer.from_file(str(ART/f'tokenizers/{tokname}/tokenizer.json'))
    def ntok(s): return len(tok.encode(s).ids)

    bank = {r['id']: r for r in read(DEPTH/bankfile)}
    ids = sorted(bank)
    assert len(ids) == 11, (model, len(ids))

    # ---- memoryless I: exact, from retained user_prompt ----
    mem = collections.defaultdict(list)
    for r in read(ROOT/sub/'depth_memI.jsonl'):
        mem[r['problem_id']].append(r)
    mem_tok, mem_pass = {}, {}
    for pid in ids:
        rs = mem[pid]
        assert len(rs) == 640, (pid, len(rs))
        mem_tok[pid] = float(np.mean([ntok(r['user_prompt']) for r in rs]))
        mem_pass[pid] = sum(r['verification_result'] == 'PASS' for r in rs)

    # ---- adaptive arms: reconstruct prompts from retained history ----
    def arm_cost(arm):
        rows = read(ROOT/sub/f'depth_{arm}.jsonl')
        byt = collections.defaultdict(dict)
        for r in rows:
            byt[(r['id'], r['traj'])][r['attempt']] = r
        per_problem_tokens, per_problem_solved = {}, {}
        for pid in ids:
            trajs = [k for k in byt if k[0] == pid]
            tot, solved = [], 0
            for key in sorted(trajs):
                atts = byt[key]
                hist, cum, done = [], 0, False
                for a in sorted(atts):
                    r = atts[a]
                    prompt = code_arms_run.render(arm if arm != 'f' else 'E_s', bank[pid], hist)[0] \
                             if arm != 'f' else depth_run.render('f', bank[pid], [e for _, e in hist])
                    cum += ntok(prompt)
                    if r['status'] == 'PASS':
                        solved += 1; done = True; break
                    hist.append((r.get('code'), (r.get('error') or '')[:ERR_CAP]))
                tot.append(cum)
            per_problem_tokens[pid] = float(np.mean(tot))
            per_problem_solved[pid] = solved/len(trajs)
        return per_problem_tokens, per_problem_solved

    arms_cost = {}
    for arm in ['C','CE','f']:
        arms_cost[arm] = arm_cost(arm)

    # ---- equal-input-token comparison ----
    def pooled(n, c, k):
        k = int(min(k, n))
        return 1-(comb(n-c, k)/comb(n, k) if n-c >= k else 0.)

    def boot(d):
        d = np.asarray(d, dtype=float)
        ix = np.random.default_rng(SEED).integers(0, len(d), (B, len(d)))
        q = np.quantile(d[ix].mean(axis=1), [.025, .975])
        return 100*d.mean(), 100*q[0], 100*q[1]

    block = {'per_attempt_input_tokens': {}, 'through_64_attempts': {}, 'equal_token_budget': {}}
    mem_mean = float(np.mean([mem_tok[p] for p in ids]))
    block['per_attempt_input_tokens']['I'] = mem_mean
    print(f"===== {model} =====")
    print(f"  memoryless I: mean input tokens per draw = {mem_mean:,.0f} (exact, retained prompts)")
    for arm, label in [('C','C'),('CE','CE'),('f','E')]:
        t, s = arms_cost[arm]
        cum = float(np.mean([t[p] for p in ids]))
        per = cum/64
        block['per_attempt_input_tokens'][label] = None
        block['through_64_attempts'][label] = dict(mean_cumulative_input_tokens=cum,
                                                   mean_success=float(np.mean([s[p] for p in ids]))*100)
        # equal-token: how many memoryless draws the same spend buys, per problem
        eq = []
        for p in ids:
            m = t[p]/mem_tok[p]
            eq.append(pooled(640, mem_pass[p], m) - s[p])
        d_pp, lo, hi = boot(eq)
        n_draws = float(np.mean([t[p]/mem_tok[p] for p in ids]))
        block['equal_token_budget'][label] = dict(
            memoryless_draws_afforded=n_draws,
            I_minus_repair_pp=d_pp, ci95_pp=[lo, hi])
        print(f"  {label:2s}: cumulative input tokens through 64 attempts = {cum:,.0f} "
              f"({cum/mem_mean:,.0f}x the per-draw I cost)")
        print(f"      success {block['through_64_attempts'][label]['mean_success']:.1f}% vs "
              f"I at the same input-token spend ({n_draws:,.0f} draws): "
              f"I - {label} = {d_pp:+.1f} pp [{lo:+.1f},{hi:+.1f}]")
    block['I_at_64_attempts_percent'] = float(np.mean([pooled(640, mem_pass[p], 64) for p in ids]))*100
    block['I_cumulative_input_tokens_64_draws'] = mem_mean*64
    print(f"  (for reference: I at 64 attempts = {block['I_at_64_attempts_percent']:.1f}% "
          f"for {mem_mean*64:,.0f} input tokens)")
    print()
    out[model] = block
json.dump(out, open(Path(__file__).parent/'out/r6_tokens.json','w'), indent=1)
