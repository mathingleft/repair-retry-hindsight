"""Leave-one-problem-out table (manuscript Table tab:loo) at k=64, both models.

For each reported sequential endpoint: the full-bank difference, the range of the
point estimate over the 11 banks that each drop one problem, the number of banks
whose estimate has the opposite sign from the full bank ("Sign"), and the number of
banks on which the 95% interval's exclusion of zero differs from the full bank's
("Excl.").

Intervals, as stated in the table caption:
  reduced banks  Python random.Random(1), 10,000 problem resamples, drawn afresh
                 for each bank and shared by that bank's contrasts; sorted endpoints
  full bank      Qwen: random.Random(0), 10,000 resamples, sorted endpoints
                 DeepSeek: the registered NumPy procedure,
                 default_rng(20260913).integers(0, 11, (200000, 11)), np.quantile
The Qwen full-bank stream is not the registered analysis's (which replays other
streams), so its bounds can differ slightly from Table tab:qwen-seq.

Run from this directory: python loo_table.py   -> out/loo_table.json
"""
import json
import random
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import as_ce  # the artifact's analyze_sequential with the added E-I, CE_m-I, C-E contrasts
from as_ce import load_model, effects, python_indices, interval, ARMS

B = 10000
CONTRASTS = {
    'qwen': [('CE-I', 'CE - I'), ('C-I', 'C - I'), ('CE-C', 'CE - C'), ('CE-CE_s', 'CE - CE_s'),
             ('CE-CE_m', 'CE - CE_m'), ('CE_m-I', 'CE_m - I'), ('f-E_s', 'E - E_m'), ('f-I', 'E - I'),
             ('E_s-I', 'E_m - I'), ('interaction', '(CE-CE_s)-(E-E_m)'),
             ('matched_interaction', '(CE-CE_m)-(E-E_m)')],
    'deepseek': [('CE-I', 'CE - I'), ('C-I', 'C - I'), ('CE-C', 'CE - C'), ('CE-CE_s', 'CE - CE_s'),
                 ('CE-CE_m', 'CE - CE_m'), ('f-E_s', 'E - E_m'), ('CE_m-E_s', 'CE_m - E_m'),
                 ('interaction', '(CE-CE_s)-(E-E_m)'), ('matched_interaction', '(CE-CE_m)-(E-E_m)')],
}


def excludes_zero(ci):
    return bool(ci[0] > 0 or ci[1] < 0)


def main():
    out = {}
    for model, sub in [('qwen', 'out'), ('deepseek', 'out_dsp')]:
        ids, v, _, _ = load_model(sub, ARMS)
        eff = effects(v)
        n = len(ids)
        if model == 'qwen':
            full_ix, full_sorted = python_indices(random.Random(0), B, n), True
        else:
            full_ix, full_sorted = np.random.default_rng(20260913).integers(0, n, (200000, n)), False
        reduced_ix = [python_indices(random.Random(1), B, n - 1) for _ in range(n)]
        rows = {}
        for key, label in CONTRASTS[model]:
            d = eff[key][:, -1]
            full = float(100*d.mean())
            full_ci = [float(x) for x in interval(d, full_ix, full_sorted)]
            banks = []
            for i in range(n):
                di = np.delete(d, i)
                ci = [float(x) for x in interval(di, reduced_ix[i], True)]
                banks.append(dict(dropped=ids[i], delta_pp=float(100*di.mean()), ci95_pp=ci))
            est = [b['delta_pp'] for b in banks]
            rows[key] = dict(
                label=label, full_pp=full, full_ci95_pp=full_ci,
                range_pp=[min(est), max(est)],
                # banks strictly on the other side of zero; a bank at exactly zero is not a
                # sign change, and a zero full-bank estimate counts as non-negative
                # (DeepSeek CE-C: 2 banks at -1.0, 2 at +1.0, 7 at 0.0 -> 2)
                sign_changes=sum((e < -1e-9) if full > -1e-9 else (e > 1e-9) for e in est),
                exclusion_changes=sum(excludes_zero(b['ci95_pp']) != excludes_zero(full_ci) for b in banks),
                banks=banks)
        out[model] = dict(problem_ids=ids, contrasts=rows)
        print(f'===== {model}')
        for key, r in rows.items():
            print(f"  {r['label']:20s} {r['full_pp']:+7.1f}  [{r['range_pp'][0]:+.1f},{r['range_pp'][1]:+.1f}]"
                  f"  sign {r['sign_changes']}  excl {r['exclusion_changes']}"
                  f"  full CI [{r['full_ci95_pp'][0]:+.1f},{r['full_ci95_pp'][1]:+.1f}]")
    (HERE/'out').mkdir(exist_ok=True)
    json.dump(out, open(HERE/'out/loo_table.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
