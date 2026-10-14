"""R16, sequential half: attempt-to-attempt near-copying within trajectories.

The paper reports byte-identical recurrence of a prior proof body within each
sequential arm. This adds the approximate analogue: for every post-initial
attempt, the maximum SequenceMatcher ratio against ANY earlier proof body in
the same trajectory, so "near-repeat" is measured the same way "exact repeat"
is. Survivor-weighted and exploratory, as the released repetition numbers are.
"""
import collections, difflib, json, re
from pathlib import Path
import numpy as np
from common import DATA

ROOT = DATA
ARMS = [('C','C'),('CE','CE'),('CE_s','CE_s'),('CE_m','CE_m'),('f','E'),('E_s','E_m')]

def read(p): return [json.loads(l) for l in open(p) if l.strip()]
def body(code):
    if code is None: return None
    b = code.split(':=', 1)[-1]
    b = re.sub(r'--[^\n]*', '', b)
    b = re.sub(r'/-.*?-/', '', b, flags=re.S)
    return re.sub(r'\s+', ' ', b).strip()

out = {}
for model, sub in [('qwen','out'), ('deepseek','out_dsp')]:
    print(f'===== {model} (post-initial attempts within a trajectory) =====')
    print(f"  {'arm':5s} {'n':>7s} {'exact%':>8s} {'near.9%':>8s} {'max-sim':>8s}")
    block = {}
    for raw, label in ARMS:
        f = ROOT/sub/f'depth_{raw}.jsonl'
        if not f.exists(): continue
        byt = collections.defaultdict(dict)
        for r in read(f): byt[(r['id'], r['traj'])][r['attempt']] = r
        ex = nr = tot = 0; sims = []
        for key, atts in byt.items():
            seen = []
            for a in sorted(atts):
                b = body(atts[a].get('code')) or ''
                if seen and b:
                    tot += 1
                    best = max(difflib.SequenceMatcher(None, b[:3000], s[:3000]).ratio() for s in seen)
                    ex += any(b == s for s in seen)
                    nr += best >= 0.9
                    sims.append(best)
                if b: seen.append(b)
        if not tot: continue
        block[label] = dict(post_initial_attempts=tot, exact_percent=100*ex/tot,
                            near_0_9_percent=100*nr/tot, mean_max_similarity=float(np.mean(sims)))
        print(f"  {label:5s} {tot:7d} {100*ex/tot:8.1f} {100*nr/tot:8.1f} {np.mean(sims):8.3f}")
    out[model] = block
    print()
json.dump(out, open(Path(__file__).parent/'out/r16_sequential.json','w'), indent=1)
