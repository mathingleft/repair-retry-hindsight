"""R1: DeepSeek one-step matched-diagnostic contrasts excluding the problems whose
matched wrong diagnostic is byte-identical to the correct diagnostic.

Reviewers (PC W4/Q3, vnkR, owTS #5) all flagged that 37/137 matched donors carry
the same text as the correct diagnostic, which makes F_CE vs F_CE_m a null
intervention on those problems. This recomputes the contrasts on the complement
and, separately, on the 37 as a noise check.

Bootstrap matches the released one-step procedure: percentile problem bootstrap,
10,000 resamples, seed 20260906 (results_figures.py).
"""
import collections, json, sys
from pathlib import Path
import numpy as np
from common import DATA

ROOT = DATA
DDIRS = ['out_m2','out_m2_ext','out_m2_ext2','out_m2_ext3','out_m2_noi']
SEED, B = 20260906, 10000

def read(p): return [json.loads(l) for l in open(p) if l.strip()]

bank = {r['id']: r for r in read(ROOT/'out_m2/bank2.jsonl') if not r.get('is_framing_holdout')}
ids = sorted(bank)
assert len(ids) == 137, len(ids)

# --- identify byte-identical matched donors -------------------------------
mp = json.load(open(ROOT/'out_m2/bank2_match.json'))
summary, mmap = mp['summary'], mp['map']
identical = []
for pid in ids:
    src = mmap[pid]['src']
    if bank[pid]['predecessor_error'] == bank[src]['predecessor_error']:
        identical.append(pid)
print(f"byte-identical matched donors: {len(identical)} of {len(ids)}"
      f"  (match-map summary reports {summary['identical_text_pairs']})")
assert len(identical) == summary['identical_text_pairs'], 'identical-pair count disagrees with the match map'
clean = [p for p in ids if p not in set(identical)]

# --- pass@4 outcomes per arm ---------------------------------------------
rows = collections.defaultdict(list)
for d in DDIRS:
    for r in read(ROOT/d/'verified.jsonl'):
        if r['id'] in bank:
            rows[(r['arm'], r['id'])].append(r)
ARMS = {'F_CE':'F_CE','F_CE_m':'F_CE_match','F_E':'F_E','F_E_m':'F_E_match','F_C':'F_C','F':'F'}
pass4 = {}
for label, arm in ARMS.items():
    v = {}
    for pid in ids:
        rs = rows[(arm, pid)]
        assert len(rs) == 4, (arm, pid, len(rs))
        v[pid] = int(any(r['status'] == 'PASS' for r in rs))
    pass4[label] = v

def contrast(a, b, subset):
    d = np.array([pass4[a][p] - pass4[b][p] for p in subset], dtype=float)
    ix = np.random.default_rng(SEED).integers(0, len(d), (B, len(d)))
    q = np.quantile(d[ix].mean(axis=1), [.025, .975])
    return 100*d.mean(), 100*q[0], 100*q[1]

def interaction(subset):
    d = np.array([(pass4['F_CE'][p]-pass4['F_CE_m'][p]) - (pass4['F_E'][p]-pass4['F_E_m'][p])
                  for p in subset], dtype=float)
    ix = np.random.default_rng(SEED).integers(0, len(d), (B, len(d)))
    q = np.quantile(d[ix].mean(axis=1), [.025, .975])
    return 100*d.mean(), 100*q[0], 100*q[1]

out = {'n_identical': len(identical), 'identical_problem_ids': identical,
       'n_clean': len(clean), 'subsets': {}}
for name, subset in [('all_137', ids), ('clean_%d'%len(clean), clean), ('identical_%d'%len(identical), identical)]:
    print(f"\n===== {name}  (n={len(subset)}) =====")
    block = {'n': len(subset), 'solved_counts': {k: int(sum(pass4[k][p] for p in subset)) for k in pass4}}
    print('  solved:', block['solved_counts'])
    for a, b in [('F_CE','F_CE_m'), ('F_E','F_E_m'), ('F_CE','F_C'), ('F_CE','F_E')]:
        m, lo, hi = contrast(a, b, subset)
        block[f'{a}-{b}'] = dict(difference_pp=m, ci95_pp=[lo, hi])
        print(f'  {a}-{b:8s} {m:+7.2f} pp  [{lo:+7.2f},{hi:+7.2f}]')
    m, lo, hi = interaction(subset)
    block['matched_interaction'] = dict(difference_pp=m, ci95_pp=[lo, hi])
    print(f'  matched interaction {m:+7.2f} pp  [{lo:+7.2f},{hi:+7.2f}]')
    out['subsets'][name] = block
json.dump(out, open(Path(__file__).parent/'out/r1_deepseek_identical.json','w'), indent=1)
