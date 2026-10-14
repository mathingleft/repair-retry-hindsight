"""R13b: does predecessor exposure reproduce its BEHAVIORAL signature across
many independently sampled predecessors?

r13.py shows the attempt-2 success contrast is floor-limited on these banks
(2.7-4.6% success), so it cannot resolve the penalty across predecessors. The
copying mechanism is measurable there, so we test that instead: at attempt 2,
conditional on attempt 1 failing, does the model reproduce its own fresh
attempt-1 proof when that proof is shown (C) and not when it is withheld (E)?

Each trajectory contributes a different attempt-1 predecessor, so this asks
whether the exposure-to-copying link holds across ~100 predecessors per model
rather than the single frozen one the one-step design fixes.

Hierarchical bootstrap: problems, then trajectories within problems.
"""
import collections, difflib, json, re
from pathlib import Path
import numpy as np
from common import DATA

ROOT = DATA
SEED, B = 20260906, 10000

def read(p): return [json.loads(l) for l in open(p) if l.strip()]
def body(code):
    if code is None: return None
    b = code.split(':=', 1)[-1]
    b = re.sub(r'--[^\n]*', '', b)
    b = re.sub(r'/-.*?-/', '', b, flags=re.S)
    return re.sub(r'\s+', ' ', b).strip()

def pairs(sub, raw):
    """{problem: [(exact, near) at attempt 2 vs its own attempt-1 proof]}"""
    byt = collections.defaultdict(dict)
    for r in read(ROOT/sub/f'depth_{raw}.jsonl'):
        byt[(r['id'], r['traj'])][r['attempt']] = r
    out = collections.defaultdict(list)
    for (pid, traj), atts in byt.items():
        a1, a2 = atts.get(1), atts.get(2)
        if a1 is None or a2 is None or a1['status'] == 'PASS': continue
        b1, b2 = body(a1.get('code')) or '', body(a2.get('code')) or ''
        if not b1 or not b2: continue
        ratio = difflib.SequenceMatcher(None, b2[:3000], b1[:3000]).ratio()
        out[pid].append((1 if b2 == b1 else 0, 1 if ratio >= 0.9 else 0, ratio))
    return out

res = {}
for model, sub in [('qwen','out'), ('deepseek','out_dsp')]:
    C, E = pairs(sub,'C'), pairs(sub,'f')
    ids = sorted(set(C) & set(E))
    rng = np.random.default_rng(SEED)
    block = {'problems': len(ids),
             'predecessors_C': sum(len(C[p]) for p in ids),
             'predecessors_E': sum(len(E[p]) for p in ids)}
    print(f'===== {model} =====')
    print(f"  {len(ids)} theorems; {block['predecessors_C']} fresh predecessors in C, "
          f"{block['predecessors_E']} in E")
    for k, name in [(0,'exact'), (1,'near_0.9'), (2,'similarity')]:
        pc = np.array([np.mean([x[k] for x in C[p]]) for p in ids])
        pe = np.array([np.mean([x[k] for x in E[p]]) for p in ids])
        draws = np.empty(B)
        for b in range(B):
            pick = rng.integers(0, len(ids), len(ids)); vals = []
            for j in pick:
                p = ids[j]; c = C[p]; e = E[p]
                cs = [c[i][k] for i in rng.integers(0, len(c), len(c))]
                es = [e[i][k] for i in rng.integers(0, len(e), len(e))]
                vals.append(np.mean(cs) - np.mean(es))
            draws[b] = np.mean(vals)
        sc = 100 if k < 2 else 1
        lo, hi = sc*np.quantile(draws, [.025, .975])
        d = sc*float((pc-pe).mean())
        block[name] = dict(C=sc*float(pc.mean()), E=sc*float(pe.mean()), difference=d,
                           ci95=[float(lo), float(hi)],
                           problems_positive=int(((pc-pe) > 0).sum()))
        unit = 'pp' if k < 2 else '  '
        print(f"  {name:11s} C {sc*pc.mean():7.3f}  E {sc*pe.mean():7.3f}  "
              f"C-E {d:+7.3f} {unit} [{lo:+.3f},{hi:+.3f}]  positive on "
              f"{int(((pc-pe)>0).sum())}/{len(ids)} theorems")
    res[model] = block
    print()
json.dump(res, open(Path(__file__).parent/'out/r13_multipredecessor_behavior.json','w'), indent=1)
