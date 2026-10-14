"""R16: near-copy measures over ALL retained outputs, both models.

PC Q7 ("Does the copying story persist under approximate similarity metrics
rather than exact proof-body equality?"), owTS #7 and the meta-review.

The paper's copying measure is exact equality of normalized proof bodies, and
its one continuous similarity analysis is restricted to passing samples on the
unknown-identifier subgroup. This computes, over every retained sample on the
full bank and without conditioning on success:

  exact      fraction of samples whose normalized body equals the predecessor's
  near(.9)   fraction with SequenceMatcher ratio >= 0.9 to the predecessor
  sim        mean ratio (continuous)
  tactic     mean Jaccard overlap of tactic-head multisets with the predecessor

Normalization and the 3,000-character truncation follow the released
results_figures.py so the numbers sit beside the existing figure. Uncertainty
is a paired problem bootstrap, 10,000 resamples, seed 20260906.
"""
import collections, difflib, json, re
from pathlib import Path
import numpy as np
from common import ARTIFACT

ART = ARTIFACT
ROOT = ART/'experiments/decomp_rerun'
SEED, B = 20260906, 10000
QDIRS = ['out','out_ext','out_ext2','out_ext3','out_noi']
DDIRS = ['out_m2','out_m2_ext','out_m2_ext2','out_m2_ext3','out_m2_noi']
ARMS = ['bare','I','F','F_E','F_C','F_CE','F_CE_shuf','F_CE_match','F_C_other']

def read(p): return [json.loads(l) for l in open(p) if l.strip()]

def body(code):
    if code is None: return None
    b = code.split(':=', 1)[-1]
    b = re.sub(r'--[^\n]*', '', b)
    b = re.sub(r'/-.*?-/', '', b, flags=re.S)
    return re.sub(r'\s+', ' ', b).strip()

TAC = re.compile(r"[A-Za-z_][A-Za-z0-9_'.]*")
def tactics(b):
    if not b: return collections.Counter()
    out = collections.Counter()
    for seg in re.split(r'<;>|;|\n', b):
        seg = seg.strip()
        if not seg: continue
        m = TAC.match(seg)
        if m: out[m.group(0)] += 1
    return out

def jaccard(a, c):
    if not a and not c: return 1.0
    inter = sum((a & c).values()); union = sum((a | c).values())
    return inter/union if union else 0.0

def boot(d):
    d = np.asarray(d, dtype=float)
    ix = np.random.default_rng(SEED).integers(0, len(d), (B, len(d)))
    q = np.quantile(d[ix].mean(axis=1), [.025, .975])
    return float(d.mean()), float(q[0]), float(q[1])

out = {}
for model, bankfile, dirs in [('qwen','bank.jsonl',QDIRS), ('deepseek','out_m2/bank2.jsonl',DDIRS)]:
    bank = {r['id']: r for r in read(ROOT/bankfile) if not r.get('is_framing_holdout')}
    ids = sorted(bank)
    rows = collections.defaultdict(list)
    for d in dirs:
        for r in read(ROOT/d/'verified.jsonl'):
            if r['id'] in bank: rows[(r['arm'], r['id'])].append(r)
    pred = {p: body(bank[p]['predecessor_code']) or '' for p in ids}
    ptac = {p: tactics(pred[p]) for p in ids}

    block = {}
    per_problem = {}
    for arm in ARMS:
        if not any((arm, p) in rows for p in ids): continue
        ex, nr, si, tj = [], [], [], []
        for p in ids:
            rs = rows[(arm, p)]
            if not rs: break
            e = n = 0; s = []; t = []
            for r in rs:
                b = body(r.get('lean_code')) or ''
                ratio = difflib.SequenceMatcher(None, b[:3000], pred[p][:3000]).ratio()
                e += (b == pred[p] and b != '')
                n += (ratio >= 0.9)
                s.append(ratio); t.append(jaccard(tactics(b), ptac[p]))
            ex.append(e/len(rs)); nr.append(n/len(rs)); si.append(np.mean(s)); tj.append(np.mean(t))
        if len(ex) != len(ids): continue
        per_problem[arm] = dict(exact=ex, near=nr, sim=si, tac=tj)
        block[arm] = {}
        for key, vals in [('exact',ex),('near_0.9',nr),('similarity',si),('tactic_jaccard',tj)]:
            m, lo, hi = boot(vals)
            block[arm][key] = dict(mean=100*m if key!='similarity' and key!='tactic_jaccard' else m,
                                   ci95=[100*lo, 100*hi] if key not in ('similarity','tactic_jaccard') else [lo, hi])
    # paired contrasts that carry the copying story
    contrasts = {}
    for a, b2, label in [('F_C','F','exposure: own proof vs failure notice only'),
                         ('F_CE','F_C','adding the correct diagnostic'),
                         ('F_C_other','F_C','replacing the predecessor'),
                         ('F_CE_shuf','F_CE','wrong vs correct diagnostic')]:
        if a not in per_problem or b2 not in per_problem: continue
        contrasts[f'{a}-{b2}'] = {'reading': label}
        for key in ['exact','near','sim','tac']:
            d = np.array(per_problem[a][key]) - np.array(per_problem[b2][key])
            m, lo, hi = boot(d)
            sc = 100 if key in ('exact','near') else 1
            contrasts[f'{a}-{b2}'][key] = dict(difference=sc*m, ci95=[sc*lo, sc*hi])
    out[model] = dict(n_problems=len(ids), arms=block, contrasts=contrasts)

    print(f'===== {model}  (n={len(ids)} problems, all retained samples, not conditioned on success) =====')
    print(f"  {'arm':11s} {'exact%':>8s} {'near.9%':>8s} {'sim':>7s} {'tactic':>7s}")
    for arm, v in block.items():
        print(f"  {arm:11s} {v['exact']['mean']:8.1f} {v['near_0.9']['mean']:8.1f} "
              f"{v['similarity']['mean']:7.3f} {v['tactic_jaccard']['mean']:7.3f}")
    print('  paired contrasts:')
    for name, v in contrasts.items():
        print(f"    {name:20s} exact {v['exact']['difference']:+6.1f} [{v['exact']['ci95'][0]:+6.1f},{v['exact']['ci95'][1]:+6.1f}]  "
              f"near {v['near']['difference']:+6.1f} [{v['near']['ci95'][0]:+6.1f},{v['near']['ci95'][1]:+6.1f}]  "
              f"sim {v['sim']['difference']:+.3f} [{v['sim']['ci95'][0]:+.3f},{v['sim']['ci95'][1]:+.3f}]  "
              f"tac {v['tac']['difference']:+.3f}")
    print()
json.dump(out, open(Path(__file__).parent/'out/r16_nearcopy.json','w'), indent=1)
