"""R4: characterize the replacement predecessor used by F_C_other.

owTS #2 and the meta-review: the unrelated-predecessor control is not described
as matched for length or error class, so the improvement under replacement could
partly reflect those properties rather than predecessor correspondence.

This reports, per model, on the paper's analysed bank:
  - how the donor is chosen and how often it is the same coarse error class
  - own vs replacement predecessor length distributions
  - the replacement effect restricted to same-class and length-matched donors
Bootstrap matches the released one-step procedure (10,000 resamples, seed 20260906).
"""
import collections, json, sys
from pathlib import Path
import numpy as np

from common import DATA
ROOT = DATA
sys.path.insert(0, str(ROOT))
import arms
from build_match import classify

SEED, B = 20260906, 10000
QDIRS = ['out','out_ext','out_ext2','out_ext3','out_noi']
DDIRS = ['out_m2','out_m2_ext','out_m2_ext2','out_m2_ext3','out_m2_noi']

def read(p): return [json.loads(l) for l in open(p) if l.strip()]

def boot(d):
    d = np.asarray(d, dtype=float)
    ix = np.random.default_rng(SEED).integers(0, len(d), (B, len(d)))
    q = np.quantile(d[ix].mean(axis=1), [.025, .975])
    return 100*d.mean(), 100*q[0], 100*q[1]

out = {}
for model, bankfile, dirs in [('qwen','bank.jsonl',QDIRS), ('deepseek','out_m2/bank2.jsonl',DDIRS)]:
    full = {r['id']: r for r in read(ROOT/bankfile)}                       # donor pool (incl. holdouts)
    bank = {k: v for k, v in full.items() if not v.get('is_framing_holdout')}  # analysed bank
    ids = sorted(bank)
    bp = None if model == 'qwen' else str(ROOT/'out_m2/bank2.jsonl')

    rows = collections.defaultdict(list)
    for d in dirs:
        for r in read(ROOT/d/'verified.jsonl'):
            if r['id'] in bank:
                rows[(r['arm'], r['id'])].append(r)
    def p4(arm, pid):
        rs = rows[(arm, pid)]
        assert len(rs) == 4, (arm, pid, len(rs))
        return int(any(r['status'] == 'PASS' for r in rs))

    same_class, ratios, donor_holdout, donor_in_bank = [], [], 0, 0
    recs = []
    for pid in ids:
        src = arms.shuffled_partner(pid, bp)
        own, rep = bank[pid]['predecessor_code'], full[src]['predecessor_code']
        oc, rc = classify(bank[pid]['predecessor_error']), classify(full[src]['predecessor_error'])
        sc = (oc == rc)
        ratio = len(rep)/max(len(own), 1)
        same_class.append(sc); ratios.append(ratio)
        if full[src].get('is_framing_holdout'): donor_holdout += 1
        if src in bank: donor_in_bank += 1
        recs.append(dict(pid=pid, src=src, same_class=sc, own_chars=len(own), rep_chars=len(rep),
                         ratio=ratio, own_class=oc, rep_class=rc,
                         d=p4('F_C_other', pid)-p4('F_C', pid)))

    r = np.array(ratios)
    ownl = np.array([x['own_chars'] for x in recs]); repl = np.array([x['rep_chars'] for x in recs])
    block = dict(
        n=len(ids),
        donor_rule='seeded cross-problem draw, random.Random(22072605); the same partner F_E_shuf/F_CE_shuf use',
        donors_from_analysed_bank=donor_in_bank, donors_that_are_framing_holdouts=donor_holdout,
        same_error_class=int(sum(same_class)),
        replacement_over_own_char_ratio=dict(
            min=float(r.min()), p25=float(np.percentile(r,25)), median=float(np.median(r)),
            p75=float(np.percentile(r,75)), max=float(r.max())),
        own_predecessor_chars=dict(median=float(np.median(ownl)), p25=float(np.percentile(ownl,25)), p75=float(np.percentile(ownl,75))),
        replacement_predecessor_chars=dict(median=float(np.median(repl)), p25=float(np.percentile(repl,25)), p75=float(np.percentile(repl,75))),
    )
    subsets = {
        'all': [x for x in recs],
        'same_error_class': [x for x in recs if x['same_class']],
        'length_matched_0.8_1.25': [x for x in recs if 0.8 <= x['ratio'] <= 1.25],
        'same_class_and_length_matched': [x for x in recs if x['same_class'] and 0.8 <= x['ratio'] <= 1.25],
    }
    block['F_C_other_minus_F_C'] = {}
    for name, s in subsets.items():
        if not s: continue
        m, lo, hi = boot([x['d'] for x in s])
        block['F_C_other_minus_F_C'][name] = dict(n=len(s), difference_pp=m, ci95_pp=[lo, hi])
    out[model] = block

    print(f"===== {model}  (analysed bank n={len(ids)}) =====")
    print(f"  donors drawn from the {len(full)}-row bank; {donor_holdout} donors are framing holdouts")
    print(f"  same coarse error class as own predecessor: {sum(same_class)}/{len(ids)}")
    print(f"  replacement/own char-length ratio: min {r.min():.2f}, quartiles "
          f"{np.percentile(r,25):.2f}/{np.median(r):.2f}/{np.percentile(r,75):.2f}, max {r.max():.2f}")
    print(f"  own predecessor chars median {np.median(ownl):.0f}; replacement median {np.median(repl):.0f}")
    for name, v in block['F_C_other_minus_F_C'].items():
        print(f"  F_C_other-F_C [{name:30s} n={v['n']:3d}] {v['difference_pp']:+7.2f} pp  "
              f"[{v['ci95_pp'][0]:+7.2f},{v['ci95_pp'][1]:+7.2f}]")
    print()
json.dump(out, open(Path(__file__).parent/'out/r4_replacement.json','w'), indent=1)
