"""R10: exact paired randomization (sign-flip) tests for the k=64 endpoints.

owTS #3 notes that bootstrap inference with 11 clusters is coarse. With 11
problems the paired randomization distribution is exact: enumerate all 2^11 =
2048 sign assignments of the per-problem differences and compute the two-sided
p-value. This needs no resampling assumption and is reported alongside the
registered bootstrap intervals, not in place of them.
"""
import itertools, json, random
import numpy as np
from as_ce import load_model, effects, ARMS
from pathlib import Path

NAMES = [('CE-I','CE - I'),('C-I','C - I'),('CE-C','CE - C'),('CE-CE_s','CE - CE_s'),
         ('CE-CE_m','CE - CE_m'),('f-E_s','E - E_m'),('interaction','(CE-CE_s)-(E-E_m)'),
         ('matched_interaction','(CE-CE_m)-(E-E_m)'),('f-I','E - I'),('E_s-I','E_m - I'),
         ('CE_m-I','CE_m - I'),('CE_s-E_s','CE_s - E_m'),('CE_m-E_s','CE_m - E_m')]

SIGNS = np.array(list(itertools.product([1,-1], repeat=11)))   # 2048 x 11, exact
out = {}
for model, sub in [('qwen','out'), ('deepseek','out_dsp')]:
    arms = ARMS+['noise'] if model == 'qwen' else ARMS
    ids, v, cond, details = load_model(sub, arms)
    eff = effects(v)
    print(f"===== {model}  (exact paired randomization, 2048 sign assignments) =====")
    print(f"  {'contrast':20s} {'mean (pp)':>10s} {'exact p':>9s}  {'nonzero':>8s}")
    block = {}
    for key, label in NAMES:
        d = eff[key][:, -1]
        obs = abs(d.mean())
        null = np.abs((SIGNS*d).mean(axis=1))
        p = float((null >= obs - 1e-12).mean())
        nz = int((np.abs(d) > 1e-12).sum())
        block[key] = dict(label=label, mean_pp=float(100*d.mean()), exact_p=p, nonzero_problems=nz)
        print(f"  {label:20s} {100*d.mean():+10.2f} {p:9.4f}  {nz:5d}/11")
    out[model] = block
    print()
json.dump(out, open(Path(__file__).parent/'out/r10_signflip.json','w'), indent=1)
