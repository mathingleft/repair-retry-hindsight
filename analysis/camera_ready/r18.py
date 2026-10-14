"""R18: both models on the SHARED 137-problem subset.

PC W3/Q4: "The cross-model narrative is difficult to interpret because the
banks are model-specific and only partially overlapping... Can the authors
report exploratory results on a shared subset of problems across the two
models to reduce bank-selection confounding?"

DeepSeek's 137 one-step problems are a subset of Qwen's 167, and Qwen ran
every condition on all 167, so the shared-subset comparison needs no new
generation. Note what it does and does not fix: the problems are now shared,
but each model still conditions on its OWN failed predecessor for that
problem, so this removes bank-selection confounding and not predecessor
confounding. It stays exploratory and descriptive.
"""
import collections, json
from pathlib import Path
import numpy as np
from common import DATA

ROOT = DATA
SEED, B = 20260906, 10000
QDIRS = ['out','out_ext','out_ext2','out_ext3','out_noi']
DDIRS = ['out_m2','out_m2_ext','out_m2_ext2','out_m2_ext3','out_m2_noi']

def read(p): return [json.loads(l) for l in open(p) if l.strip()]
def boot(d):
    d = np.asarray(d, dtype=float)
    ix = np.random.default_rng(SEED).integers(0, len(d), (B, len(d)))
    q = np.quantile(d[ix].mean(axis=1), [.025, .975])
    return 100*float(d.mean()), 100*float(q[0]), 100*float(q[1])

def load(bankfile, dirs):
    bank = {r['id']: r for r in read(ROOT/bankfile) if not r.get('is_framing_holdout')}
    rows = collections.defaultdict(list)
    for d in dirs:
        for r in read(ROOT/d/'verified.jsonl'):
            if r['id'] in bank: rows[(r['arm'], r['id'])].append(r)
    return bank, rows

qb, qr = load('bank.jsonl', QDIRS)
db, dr = load('out_m2/bank2.jsonl', DDIRS)
shared = sorted(set(qb) & set(db))
print(f'Qwen bank {len(qb)}; DeepSeek bank {len(db)}; shared {len(shared)}\n')

ARMS = ['I','F','F_C','F_CE','F_E','F_CE_shuf','F_CE_match','F_E_match','F_C_other']
def p4(rows, arm, pid):
    rs = rows[(arm, pid)]
    return int(any(r['status']=='PASS' for r in rs)) if len(rs)==4 else None

out = {'shared_n': len(shared), 'models': {}}
for name, rows in [('qwen', qr), ('deepseek', dr)]:
    block = {'solved_on_shared': {}, 'contrasts': {}}
    for a in ARMS:
        v = [p4(rows,a,p) for p in shared]
        if any(x is None for x in v): continue
        block['solved_on_shared'][a] = int(sum(v))
    for a, b in [('F_C','F'), ('F_CE','F_C'), ('F_CE','F_CE_match'), ('F_C_other','F_C'), ('F_CE','I')]:
        va = [p4(rows,a,p) for p in shared]; vb = [p4(rows,b,p) for p in shared]
        if any(x is None for x in va+vb): continue
        d = [x-y for x,y in zip(va,vb)]
        m, lo, hi = boot(d)
        block['contrasts'][f'{a}-{b}'] = dict(difference_pp=m, ci95_pp=[lo,hi])
    out['models'][name] = block
    print(f'===== {name} on the shared {len(shared)} problems =====')
    print('  solved:', block['solved_on_shared'])
    for k,v in block['contrasts'].items():
        print(f"    {k:18s} {v['difference_pp']:+6.1f} pp [{v['ci95_pp'][0]:+6.1f},{v['ci95_pp'][1]:+6.1f}]")
    print()

# do the penalties land on the same problems?
qp = np.array([p4(qr,'F',p) - p4(qr,'F_C',p) for p in shared], dtype=float)
dp = np.array([p4(dr,'F',p) - p4(dr,'F_C',p) for p in shared], dtype=float)
both = int(((qp>0)&(dp>0)).sum()); either = int(((qp>0)|(dp>0)).sum())
out['penalty_colocation'] = dict(qwen_hurt=int((qp>0).sum()), deepseek_hurt=int((dp>0).sum()),
                                 both=both, either=either,
                                 jaccard=both/either if either else None,
                                 phi=float(np.corrcoef(qp>0, dp>0)[0,1]))
print('Problems where exposure costs a solve (F solved, F_C not):')
print(f"  Qwen {int((qp>0).sum())}, DeepSeek {int((dp>0).sum())}, both {both}, either {either}, "
      f"Jaccard {both/either if either else float('nan'):.2f}, phi {np.corrcoef(qp>0, dp>0)[0,1]:+.3f}")
json.dump(out, open(Path(__file__).parent/'out/r18_shared_subset.json','w'), indent=1)
