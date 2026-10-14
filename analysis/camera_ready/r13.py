"""R13: does the failed-proof penalty generalize across DIFFERENT predecessors?

owTS #1, PC W5/Q2, and the meta-review's most-emphasized weakness: the
one-step design fixes one failed predecessor per theorem, so every one-step
effect is conditional on that single failure realization.

The sequential runs already contain the missing variation, without any new
generation. In each adaptive arm, attempt 1 is a bare draw, so a trajectory
that fails at attempt 1 and continues to attempt 2 is exactly a one-step
repair on a FRESHLY SAMPLED predecessor. Ten trajectories per problem give up
to ten independent predecessors per theorem.

Comparison at attempt 2, conditional on attempt 1 failing:
    C   = failure notice + own failed proof
    E   = failure notice + own diagnostic, no proof   (raw arm `f`)
so C - E at attempt 2 is a predecessor-exposure contrast under ten different
predecessors per theorem, the one-step F_C vs F_E contrast in miniature.

Units are nested: predecessors within theorems. Uncertainty resamples
problems first, then trajectories within each resampled problem (a
hierarchical bootstrap), which is the structure owTS asked for.
"""
import collections, json
from pathlib import Path
import numpy as np
from common import DATA

ROOT = DATA
SEED, B = 20260906, 10000

def read(p): return [json.loads(l) for l in open(p) if l.strip()]

def attempt2(sub, raw):
    """{problem: [outcome at attempt 2, for each trajectory whose attempt 1 failed]}"""
    byt = collections.defaultdict(dict)
    for r in read(ROOT/sub/f'depth_{raw}.jsonl'):
        byt[(r['id'], r['traj'])][r['attempt']] = r
    out = collections.defaultdict(list)
    for (pid, traj), atts in byt.items():
        a1 = atts.get(1)
        if a1 is None or a1['status'] == 'PASS':
            continue                      # no second attempt / no fresh predecessor
        a2 = atts.get(2)
        if a2 is None:
            continue
        out[pid].append(1 if a2['status'] == 'PASS' else 0)
    return out

res = {}
for model, sub in [('qwen','out'), ('deepseek','out_dsp')]:
    C, E = attempt2(sub,'C'), attempt2(sub,'f')
    ids = sorted(set(C) & set(E))
    rng = np.random.default_rng(SEED)
    # point estimate: problem-level mean of (per-problem mean C - per-problem mean E)
    pc = np.array([np.mean(C[p]) for p in ids])
    pe = np.array([np.mean(E[p]) for p in ids])
    diff = 100*float((pc-pe).mean())
    draws = np.empty(B)
    for b in range(B):
        pick = rng.integers(0, len(ids), len(ids))
        vals = []
        for j in pick:
            p = ids[j]
            c = C[p]; e = E[p]
            cs = [c[k] for k in rng.integers(0, len(c), len(c))]
            es = [e[k] for k in rng.integers(0, len(e), len(e))]
            vals.append(np.mean(cs) - np.mean(es))
        draws[b] = np.mean(vals)
    lo, hi = 100*np.quantile(draws, [.025, .975])
    npred_c = sum(len(C[p]) for p in ids); npred_e = sum(len(E[p]) for p in ids)
    res[model] = dict(problems=len(ids), predecessors_C=npred_c, predecessors_E=npred_e,
                      per_problem_C_percent=[round(100*x,1) for x in pc],
                      per_problem_E_percent=[round(100*x,1) for x in pe],
                      C_mean_percent=100*float(pc.mean()), E_mean_percent=100*float(pe.mean()),
                      C_minus_E_pp=diff, ci95_pp=[float(lo), float(hi)],
                      problems_with_negative_sign=int(((pc-pe) < 0).sum()),
                      problems_with_zero=int(((pc-pe) == 0).sum()))
    print(f'===== {model} =====')
    print(f'  {len(ids)} theorems; {npred_c} freshly sampled predecessors in C, {npred_e} in E')
    print(f'  attempt-2 success: C {100*pc.mean():.1f}%  vs  E {100*pe.mean():.1f}%')
    print(f'  C - E = {diff:+.1f} pp  95% hierarchical CI [{lo:+.1f},{hi:+.1f}]')
    print(f'  per-problem differences negative on {int(((pc-pe)<0).sum())}/{len(ids)} theorems, '
          f'zero on {int(((pc-pe)==0).sum())}')
    print()
json.dump(res, open(Path(__file__).parent/'out/r13_multipredecessor.json','w'), indent=1)
