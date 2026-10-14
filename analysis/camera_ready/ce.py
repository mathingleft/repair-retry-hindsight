"""C-E (prereg 'C-f'): a registered secondary the released script never emitted."""
import random
import numpy as np
from analyze_sequential import load_model, python_indices, interval, ARMS
for model, sub in [('qwen','out'), ('deepseek','out_dsp')]:
    arms = ARMS+['noise'] if model == 'qwen' else ARMS
    ids, v, cond, details = load_model(sub, arms)
    if model == 'deepseek':
        ix = np.random.default_rng(20260913).integers(0,11,(200000,11)); orig=False
    else:
        ix = python_indices(random.Random(0),10000,11); orig=True
    for name, a, b in [('C-E','C','f'), ('CE-E','CE','f')]:
        d = v[a][:,63]-v[b][:,63]
        lo, hi = interval(d, ix, orig)
        print(f'  {model:9s} {name:6s} {100*d.mean():+7.2f} pp  [{lo:+7.2f},{hi:+7.2f}]')
