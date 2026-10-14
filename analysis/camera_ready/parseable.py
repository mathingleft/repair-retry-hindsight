"""Parseable-problem sensitivity (Appendix app:parseable, Table tab:parseable).

Thirty one-step problems, in both banks, state sums or products as `∑ x in s` /
`∏ x in s`, which the verifier's Mathlib (v4.29.0) no longer parses. Because
verification rebuilds every proof from the dataset statement, no attempt on
them can pass. This script

  1. identifies them from the frozen bank statements (rule below) and checks
     that every retained attempt on them fails;
  2. repeats the main one-step contrasts on all problems and on the parseable
     problems (137 Qwen, 107 DeepSeek-Prover);
  3. re-runs the artifact's own pooled correct-minus-shuffled estimator
     (analysis/pooled_diagnostics.py) and benefit bounds
     (analysis/feedback_bounds.py) on the parseable Qwen problems.

Rule: a problem is unparseable when its bank `statement` contains a big
operator written with the Lean 3 binder `∑ x in` / `∏ x in` (regex BIG).

Interval procedures, stated because the endpoints depend on them:
  - simple contrasts: per-problem pass@4 differences, percentile problem
    bootstrap, numpy default_rng(20260906), 10,000 resamples, np.quantile
    (the camera-ready one-step convention, see README);
  - the two interactions: per-problem pass@4 interaction, random.Random(1),
    10,000 resamples of randrange draws, sorted endpoints [250], [9750];
  - pooled estimator and benefit bounds: the artifact scripts' own seeds and
    resample counts, unchanged, applied to the subset.
Reads only frozen records under paper/artifact/; writes out/parseable.json.
"""
import collections, json, random, re, sys
from math import sqrt
from pathlib import Path
from statistics import NormalDist
import numpy as np

from artifact_path import ARTIFACT
ROOT = ARTIFACT / 'experiments/decomp_rerun'
sys.path.insert(0, str(ARTIFACT / 'analysis'))
from pooled_diagnostics import failure_u, variance_u  # artifact's own estimator pieces

OUT = Path(__file__).resolve().parent / 'out'
QDIRS = ['out', 'out_ext', 'out_ext2', 'out_ext3', 'out_noi']
DDIRS = ['out_m2', 'out_m2_ext', 'out_m2_ext2', 'out_m2_ext3', 'out_m2_noi']
EXTRA = ['out_noi_r2'] + [f'out_confirm_error/b{b}' for b in range(1, 5)]   # Qwen-only one-step records
BIG = re.compile(r'[∑∏]\s*\S+\s+in\s')
SEED, B = 20260906, 10000

CONTRASTS = {
    'F_CE-I': {'F_CE': 1, 'I': -1}, 'F_C-F': {'F_C': 1, 'F': -1}, 'F_CE-F_C': {'F_CE': 1, 'F_C': -1},
    'F_CE-F_CE_m': {'F_CE': 1, 'F_CE_match': -1}, 'F_E-F_E_m': {'F_E': 1, 'F_E_match': -1},
    'F_C_o-F_C': {'F_C_other': 1, 'F_C': -1}}
INTERACTIONS = {
    'matched interaction': {'F_CE': 1, 'F_CE_match': -1, 'F_E': -1, 'F_E_match': 1},
    'instruction interaction': {'F_CE': 1, 'F_C': -1, 'F_CE_noI': -1, 'F_C_noI': 1}}
TABLE = {'qwen': list(CONTRASTS) + list(INTERACTIONS),
         'deepseek': ['F_CE-I', 'F_C-F', 'F_CE-F_C', 'F_CE-F_CE_m', 'F_C_o-F_C']}


def read(p): return [json.loads(l) for l in open(p) if l.strip()]


def bank(model):
    f = 'bank.jsonl' if model == 'qwen' else 'out_m2/bank2.jsonl'
    return [r for r in read(ROOT / f) if not r.get('is_framing_holdout')]


def solved_sets(dirs, ids):
    solved, n = collections.defaultdict(set), collections.Counter()
    for d in dirs:
        for r in read(ROOT / d / 'verified.jsonl'):
            if r['id'] in ids:
                n[r['arm'], r['id']] += 1
                if r['status'] == 'PASS': solved[r['arm']].add(r['id'])
    return solved, n


def boot_np(d):
    ix = np.random.default_rng(SEED).integers(0, len(d), (B, len(d)))
    q = np.quantile(d[ix].mean(axis=1), [.025, .975])
    return [100 * float(d.mean()), 100 * float(q[0]), 100 * float(q[1])]


def boot_random1(d):
    rng, n = random.Random(1), len(d)
    s = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(B))
    return [100 * float(d.mean()), 100 * s[int(.025 * B)], 100 * s[int(.975 * B)]]


def contrasts(model, ids):
    solved, n = solved_sets(QDIRS if model == 'qwen' else DDIRS, set(ids))
    res = {}
    for name in TABLE[model]:
        spec = CONTRASTS.get(name) or INTERACTIONS[name]
        assert all(n[a, p] == 4 for a in spec for p in ids), (model, name)
        d = np.array([sum(s * (p in solved[a]) for a, s in spec.items()) for p in ids], dtype=float)
        res[name] = boot_random1(d) if name in INTERACTIONS else boot_np(d)
    return res


def pooled(ids):
    """analysis/pooled_diagnostics.py main(), restricted to `ids` (same seed and resamples)."""
    arms = ('F_E', 'F_E_shuf'); counts = collections.Counter(); sizes = collections.Counter()
    for b in range(1, 5):
        for r in read(ROOT / f'out_confirm_error/b{b}/verified.jsonl'):
            if r['id'] in ids:
                sizes[r['arm'], r['id']] += 1; counts[r['arm'], r['id']] += r['status'] == 'PASS'
    assert all(sizes[a, i] == 16 for a in arms for i in ids)
    values = np.array([[1 - failure_u(16, counts[a, i], 4) for a in arms] for i in ids])
    diff = values[:, 0] - values[:, 1]
    rng = np.random.default_rng(20260906)
    draws = np.concatenate([diff[rng.integers(0, len(ids), (1000, len(ids)))].mean(axis=1) for _ in range(100)])
    var = sum(variance_u(16, counts[a, i], 4) for a in arms for i in ids) / len(ids) ** 2
    z90 = NormalDist().inv_cdf(.95)
    return dict(n=len(ids), pooled_difference_pp=100 * float(diff.mean()),
                problem_bootstrap_95_pp=(100 * np.quantile(draws, [.025, .975])).tolist(),
                fixed_bank_90_pp=[100 * (diff.mean() - z90 * sqrt(var)), 100 * (diff.mean() + z90 * sqrt(var))])


def bounds(keep):
    """analysis/feedback_bounds.py main(), restricted to the problems in `keep` (same seed, 100,000 resamples)."""
    conds = ['F_E', 'F_E_shuf', 'F_CE', 'F_CE_shuf', 'F_E_noI', 'F_E_shuf_noI', 'F_CE_noI']
    rng = np.random.default_rng(20260905); B2 = 100000; out = []
    for model, dirs, arms in [('qwen', ['out', 'out_ext', 'out_ext2', 'out_noi'], conds + ['F_R', 'F_R_noI']),
                              ('deepseek', ['out_m2', 'out_m2_ext', 'out_m2_ext2', 'out_m2_noi'], conds)]:
        ids = sorted(keep[model]); solved, n = solved_sets(dirs, set(ids))
        diffs = np.stack([np.array([(p in solved[a]) - (p in solved['I']) for p in ids], dtype=float) for a in arms], 1)
        draws = np.empty((B2, len(arms)))
        for s in range(0, B2, 1000):
            draws[s:s + 1000] = diffs[rng.integers(0, len(ids), (1000, len(ids)))].mean(axis=1) * 100
        for j, a in enumerate(arms):
            out.append(dict(model=model, arm=a, diff=100 * float(diffs[:, j].mean()),
                            upper95=float(np.quantile(draws[:, j], .975)),
                            family_upper=float(np.quantile(draws[:, j], 1 - .05 / 16))))
    return dict(max_individual_upper_pp=max(r['upper95'] for r in out),
                max_family_upper_pp=max(r['family_upper'] for r in out), contrasts=out)


def main():
    result = {'rule': 'bank statement matches ' + BIG.pattern, 'models': {}}
    keep = {}
    for model in ('qwen', 'deepseek'):
        rows = bank(model); ids = [r['id'] for r in rows]   # frozen bank order
        bad = sorted(r['id'] for r in rows if BIG.search(r['statement']))
        dirs = (QDIRS + EXTRA) if model == 'qwen' else DDIRS
        att = [r for d in dirs for r in read(ROOT / d / 'verified.jsonl') if r['id'] in bad]
        assert not any(r['status'] == 'PASS' for r in att), 'an unparseable problem has a passing attempt'
        keep[model] = [p for p in ids if p not in bad]
        result['models'][model] = dict(
            problems=len(ids), unparseable=len(bad), parseable=len(keep[model]), unparseable_ids=bad,
            attempts_on_unparseable=len(att), passes_on_unparseable=0,
            contrasts=dict(all=contrasts(model, ids), parseable=contrasts(model, keep[model])))
    q, d = (set(result['models'][m]['unparseable_ids']) for m in ('qwen', 'deepseek'))
    result['same_30_in_both_banks'] = q == d
    result['total_attempts_on_unparseable'] = sum(result['models'][m]['attempts_on_unparseable'] for m in ('qwen', 'deepseek'))
    result['pooled_correct_minus_shuffled'] = dict(all=pooled(sorted(r['id'] for r in bank('qwen'))),
                                                   parseable=pooled(sorted(keep['qwen'])))
    full = {m: [r['id'] for r in bank(m)] for m in keep}
    result['benefit_bounds'] = dict(all=bounds(full), parseable=bounds(keep))
    for k in ('all', 'parseable'): result['benefit_bounds'][k].pop('contrasts')
    OUT.mkdir(exist_ok=True)
    (OUT / 'parseable.json').write_text(json.dumps(result, indent=1) + '\n')
    for m, blk in result['models'].items():
        print(f"== {m}: {blk['problems']} problems, {blk['unparseable']} unparseable, {blk['parseable']} parseable; "
              f"{blk['attempts_on_unparseable']} attempts on them, none pass")
        for name in TABLE[m]:
            a, p = blk['contrasts']['all'][name], blk['contrasts']['parseable'][name]
            print(f"  {name:24s} all {a[0]:+5.1f} [{a[1]:+5.1f},{a[2]:+5.1f}]   parseable {p[0]:+5.1f} [{p[1]:+5.1f},{p[2]:+5.1f}]")
    print('same 30 in both banks:', result['same_30_in_both_banks'], '; total attempts', result['total_attempts_on_unparseable'])
    for k, v in result['pooled_correct_minus_shuffled'].items():
        print(f"pooled F_E-F_E_s ({k}, n={v['n']}): {v['pooled_difference_pp']:+.2f} boot95 "
              f"[{v['problem_bootstrap_95_pp'][0]:+.2f},{v['problem_bootstrap_95_pp'][1]:+.2f}] fixed90 "
              f"[{v['fixed_bank_90_pp'][0]:+.2f},{v['fixed_bank_90_pp'][1]:+.2f}]")
    for k, v in result['benefit_bounds'].items():
        print(f"benefit bounds ({k}): largest individual upper {v['max_individual_upper_pp']:+.2f}, family {v['max_family_upper_pp']:+.2f}")


if __name__ == '__main__':
    main()
