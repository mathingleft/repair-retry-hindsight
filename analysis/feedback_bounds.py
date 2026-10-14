"""Exploratory paired problem-bootstrap bounds; never modifies source artifacts.

Run with sdvenv/bin/python analysis/feedback_bounds.py.
Outputs JSON to stdout. Family: all 16 diagnostic/reflection conditions in the
listed one-step directories, relative to each model's instruction-only arm.
Percentile bootstrap coverage is approximate, not a finite-sample guarantee.
"""
import collections
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1] / 'experiments/decomp_rerun'
SEED, B = 20260905, 100000
CONDITIONS = ['F_E', 'F_E_shuf', 'F_CE', 'F_CE_shuf',
              'F_E_noI', 'F_E_shuf_noI', 'F_CE_noI']

def main():
    rng = np.random.default_rng(SEED)
    results = []
    for model, bank, dirs, arms in [
        ('Qwen3.6-27B', 'bank.jsonl', ['out', 'out_ext', 'out_ext2', 'out_noi'],
         CONDITIONS + ['F_R', 'F_R_noI']),
        ('DeepSeek-Prover-V2-7B', 'out_m2/bank2.jsonl',
         ['out_m2', 'out_m2_ext', 'out_m2_ext2', 'out_m2_noi'], CONDITIONS),
    ]:
        ids = sorted(r['id'] for r in map(json.loads, (ROOT / bank).open())
                     if not r.get('is_framing_holdout'))
        rows = collections.defaultdict(dict)
        for directory in dirs:
            for r in map(json.loads, (ROOT / directory / 'verified.jsonl').open()):
                if r['id'] not in ids or r['arm'] not in arms + ['I']:
                    continue
                key = (r['id'], r['sample_idx'])
                assert key not in rows[r['arm']], (directory, r['arm'], key)
                rows[r['arm']][key] = r['status'] == 'PASS'
        solved = {}
        for arm in arms + ['I']:
            assert len(rows[arm]) == 4 * len(ids), (model, arm)
            counts = collections.Counter(pid for pid, _ in rows[arm])
            assert all(counts[pid] == 4 for pid in ids)
            solved[arm] = np.array([any(v for (p, _), v in rows[arm].items()
                                       if p == pid) for pid in ids], dtype=int)
        differences = np.stack([solved[a] - solved['I'] for a in arms], axis=1)
        draws = np.empty((B, len(arms)))
        for start in range(0, B, 1000):
            ix = rng.integers(0, len(ids), size=(min(1000, B-start), len(ids)))
            draws[start:start+len(ix)] = differences[ix].mean(axis=1) * 100
        for j, arm in enumerate(arms):
            results.append(dict(model=model, arm=arm, n=len(ids),
                difference_pp=float(differences[:, j].mean()*100),
                individual_95_ci_pp=np.quantile(draws[:, j], [.025, .975]).tolist(),
                family_95_upper_pp=float(np.quantile(draws[:, j], 1-.05/16))))
    assert len(results) == 16
    print(json.dumps(dict(seed=SEED, resamples=B, family_size=16,
        method='paired problem percentile bootstrap; Bonferroni one-sided family bounds',
        scope='one-step pass@4 versus I; exploratory approximate coverage',
        max_individual_upper_pp=max(r['individual_95_ci_pp'][1] for r in results),
        max_family_upper_pp=max(r['family_95_upper_pp'] for r in results),
        contrasts=results), indent=2))

if __name__ == '__main__':
    main()
