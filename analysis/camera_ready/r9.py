"""R9 + R11: leave-one-problem-out sensitivity and achieved precision for the
k=64 sequential endpoints, both models.

owTS #3 and the meta-review: with 11 problem clusters, readers cannot tell
whether a few tasks dominate. For each reported endpoint this gives the range
of the point estimate across the 11 leave-one-out banks, flags any sign change,
and reports the achieved 95% interval half-width (R11).
"""
import json, random
import numpy as np
from as_ce import load_model, python_indices, interval, effects, ARMS
from common import GENERATED
from pathlib import Path

NAMES = [('CE-I','CE - I'),('C-I','C - I'),('CE-C','CE - C'),('CE-CE_s','CE - CE_s'),
         ('CE-CE_m','CE - CE_m'),('f-E_s','E - E_m'),('interaction','(CE-CE_s)-(E-E_m)'),
         ('matched_interaction','(CE-CE_m)-(E-E_m)'),('f-I','E - I'),('E_s-I','E_m - I'),
         ('CE_m-I','CE_m - I'),('CE_s-E_s','CE_s - E_m'),('CE_m-E_s','CE_m - E_m')]

released = json.load(open(GENERATED/'sequential.json'))
mine = json.load(open(Path(__file__).parent/'out/sequential_ce.json'))  # written by as_ce.py
out = {}
for model, sub in [('qwen','out'), ('deepseek','out_dsp')]:
    arms = ARMS+['noise'] if model == 'qwen' else ARMS
    ids, v, cond, details = load_model(sub, arms)
    eff = effects(v)
    ep = mine[model]['endpoint_contrasts']
    block = {}
    print(f"===== {model}  (11 problems: leave-one-out at k=64) =====")
    print(f"  {'contrast':20s} {'full':>8s} {'LOO min':>9s} {'LOO max':>9s} {'half-width':>11s}  flips?")
    for key, label in NAMES:
        d = eff[key][:, -1]                      # per-problem differences at k=64
        full = 100*d.mean()
        loo = np.array([100*np.delete(d, i).mean() for i in range(len(d))])
        lo, hi = ep[key]['ci95_pp']
        hw = (hi-lo)/2
        signflip = bool((loo.min() < 0) != (loo.max() < 0))
        # does any LOO bank move the point estimate across zero relative to the full bank?
        crosses = bool((full < 0) != (loo < 0).all() and (full < 0) != (loo < 0).any())
        worst = ids[int(np.argmax(np.abs(loo-full)))]
        block[key] = dict(label=label, full_pp=full, loo_min_pp=float(loo.min()), loo_max_pp=float(loo.max()),
                          ci95_pp=[lo, hi], half_width_pp=hw, loo_sign_change=signflip,
                          most_influential_problem=worst,
                          max_abs_shift_pp=float(np.max(np.abs(loo-full))))
        print(f"  {label:20s} {full:+8.2f} {loo.min():+9.2f} {loo.max():+9.2f} {hw:11.1f}  {'YES' if signflip else 'no'}")
    out[model] = dict(problem_ids=ids, endpoints=block)
    hws = [b['half_width_pp'] for b in block.values()]
    print(f"  achieved half-widths: median {np.median(hws):.1f} pp, min {min(hws):.1f}, max {max(hws):.1f}")
    print()
json.dump(out, open(Path(__file__).parent/'out/r9_loo.json','w'), indent=1)
