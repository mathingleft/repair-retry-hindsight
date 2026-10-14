"""Pooled pass@4 from the 16 fresh diagnostic-comparison draws per cell.

Reports paired problem-bootstrap intervals and, separately, an approximate
fixed-bank normal interval assuming iid draws within each cell AND independent
sampling across arms/problems. The latter uses an unbiased U-statistic variance
estimate; it is not the original block-based interval and does not account for
shared batch effects. Original block analysis is retained as a sensitivity.
"""
import collections
import json
from math import comb, sqrt
from pathlib import Path
from statistics import NormalDist
import numpy as np

ROOT = Path(__file__).resolve().parents[1] / 'experiments/decomp_rerun'

def failure_u(n, c, k):
    return comb(n-c, k)/comb(n, k) if n-c >= k else 0.0

def variance_u(n, c, k):
    # Two uniform k-subsets intersect in j positions with weight below.
    # E[U_m]=q^m; substituting U_m yields an unbiased estimate of Var(U_k).
    return sum(comb(k, j)*comb(n-k, k-j)/comb(n, k)
               * (failure_u(n, c, 2*k-j)-failure_u(n, c, 2*k))
               for j in range(k+1))

def main():
    ids = sorted(r['id'] for r in map(json.loads, (ROOT/'bank.jsonl').open())
                 if not r['is_framing_holdout'])
    arms = ('F_E', 'F_E_shuf')
    counts = collections.Counter()
    sizes = collections.Counter()
    seen = set()
    blocks = collections.defaultdict(set)
    for b in range(1, 5):
        for r in map(json.loads, (ROOT/f'out_confirm_error/b{b}/verified.jsonl').open()):
            key = (b, r['arm'], r['id'], r['sample_idx'])
            assert key not in seen and r['id'] in ids and r['arm'] in arms
            seen.add(key)
            sizes[(r['arm'], r['id'])] += 1
            if r['status']=='PASS':
                counts[(r['arm'], r['id'])] += 1
                blocks[(b, r['arm'])].add(r['id'])
    assert len(seen) == len(ids)*2*16
    assert all(sizes[(a,i)]==16 for a in arms for i in ids)
    values = np.array([[1-failure_u(16, counts[(a,i)], 4) for a in arms] for i in ids])
    diff = values[:,0]-values[:,1]
    rng = np.random.default_rng(20260906)
    draws = np.concatenate([diff[rng.integers(0,len(ids),(1000,len(ids)))].mean(axis=1)
                            for _ in range(100)])
    variance = sum(variance_u(16,counts[(a,i)],4) for a in arms for i in ids)/len(ids)**2
    intervals = {}
    for conf in (.90,.95):
        z = NormalDist().inv_cdf((1+conf)/2)
        intervals[str(conf)] = [100*(diff.mean()-z*sqrt(variance)),
                                100*(diff.mean()+z*sqrt(variance))]
    print(json.dumps(dict(n_problems=len(ids), n_draws_per_cell=16, target_k=4,
        pooled_success_percent=dict(zip(arms,(values.mean(axis=0)*100).tolist())),
        pooled_difference_pp=float(diff.mean()*100),
        problem_bootstrap_95_pp=(np.quantile(draws,[.025,.975])*100).tolist(),
        problem_bootstrap_90_pp=(np.quantile(draws,[.05,.95])*100).tolist(),
        fixed_bank_independent_sampling_normal_intervals_pp=intervals,
        fixed_bank_standard_error_pp=100*sqrt(variance),
        original_block_difference_pp=100*sum(len(blocks[(b,arms[0])])-len(blocks[(b,arms[1])])
                                            for b in range(1,5))/(4*len(ids)),
        bootstrap_seed=20260906, bootstrap_resamples=100000,
        caveats=['post hoc pooled estimator; iid within each problem/condition assumed',
                 'fixed-bank normal CI additionally assumes independent arms/problems; ignores shared batch shocks',
                 'problem bootstrap is a different uncertainty target from fixed-bank intervals',
                 'two unresolved infrastructure rows remain unsuccessful, matching source analysis']), indent=2))

if __name__ == '__main__':
    main()
