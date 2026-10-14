"""R2: one-step matched-diagnostic quality, and the contrast on well-matched pairs.

PC L4 ("Wrong-diagnostic controls are approximate rather than semantic
controls; matching by coarse class and length does not remove other message
differences") and owTS #5, which lists exactly what class+length matching
leaves uncontrolled: lexical overlap, referenced identifiers, source
locations, number of sub-errors.

This quantifies the match quality per problem (token-length ratio, lexical
Jaccard, identifier overlap, byte-identity) and then re-estimates the
correct-versus-matched contrasts restricted to the best-matched pairs, so the
reader can see whether the result depends on the loose matches.
"""
import collections, json, re
from pathlib import Path
import numpy as np
from common import DATA

ROOT = DATA
SEED, B = 20260906, 10000
QDIRS = ['out','out_ext','out_ext2','out_ext3','out_noi']
DDIRS = ['out_m2','out_m2_ext','out_m2_ext2','out_m2_ext3','out_m2_noi']
TOK = re.compile(r"[A-Za-z_][A-Za-z0-9_']*")

def read(p): return [json.loads(l) for l in open(p) if l.strip()]
def toks(s): return collections.Counter(TOK.findall(s or ''))
def jac(a, b):
    if not a and not b: return 1.0
    u = sum((a | b).values())
    return sum((a & b).values())/u if u else 0.0

def boot(d):
    d = np.asarray(d, dtype=float)
    ix = np.random.default_rng(SEED).integers(0, len(d), (B, len(d)))
    q = np.quantile(d[ix].mean(axis=1), [.025, .975])
    return 100*float(d.mean()), 100*float(q[0]), 100*float(q[1])

out = {}
for model, bankfile, mapfile, dirs in [
        ('qwen','bank.jsonl','bank_match.json',QDIRS),
        ('deepseek','out_m2/bank2.jsonl','out_m2/bank2_match.json',DDIRS)]:
    full = {r['id']: r for r in read(ROOT/bankfile)}
    bank = {k:v for k,v in full.items() if not v.get('is_framing_holdout')}
    ids = sorted(bank)
    mp = json.load(open(ROOT/mapfile))['map']
    rows = collections.defaultdict(list)
    for d in dirs:
        for r in read(ROOT/d/'verified.jsonl'):
            if r['id'] in bank: rows[(r['arm'], r['id'])].append(r)
    def p4(arm, pid):
        rs = rows[(arm, pid)]
        return int(any(r['status']=='PASS' for r in rs)) if len(rs)==4 else None

    recs = []
    for pid in ids:
        m = mp.get(pid)
        if not m: continue
        own, src = bank[pid]['predecessor_error'], full[m['src']]['predecessor_error']
        ot, st = m.get('own_tok') or 1, m.get('src_tok') or 1
        ratio = max(ot, st)/max(1, min(ot, st))
        recs.append(dict(pid=pid, identical=(own == src), ratio=ratio,
                         lex=jac(toks(own), toks(src)),
                         same_class=(m.get('own_class') == m.get('src_class'))))
    r = np.array([x['ratio'] for x in recs]); lex = np.array([x['lex'] for x in recs])
    block = dict(n=len(recs), byte_identical=int(sum(x['identical'] for x in recs)),
                 same_class=int(sum(x['same_class'] for x in recs)),
                 token_ratio=dict(median=float(np.median(r)), p90=float(np.percentile(r,90)), max=float(r.max())),
                 lexical_jaccard=dict(median=float(np.median(lex)), p25=float(np.percentile(lex,25)),
                                      p75=float(np.percentile(lex,75))))
    subsets = {
        'all': recs,
        'excl_identical': [x for x in recs if not x['identical']],
        'well_matched (ratio<=1.25, not identical)': [x for x in recs if not x['identical'] and x['ratio'] <= 1.25],
        'low_lexical_overlap (<median, not identical)': [x for x in recs if not x['identical'] and x['lex'] < np.median(lex)],
    }
    block['contrasts'] = {}
    print(f'===== {model} (n={len(recs)}) =====')
    print(f"  byte-identical donors {block['byte_identical']}; same class {block['same_class']}/{len(recs)}")
    print(f"  token-length ratio median {np.median(r):.2f}, p90 {np.percentile(r,90):.2f}, max {r.max():.2f}")
    print(f"  lexical Jaccard of own vs donor message: median {np.median(lex):.3f} "
          f"(quartiles {np.percentile(lex,25):.3f}/{np.percentile(lex,75):.3f})")
    for name, sub in subsets.items():
        pids = [x['pid'] for x in sub]
        if len(pids) < 5: continue
        block['contrasts'][name] = {'n': len(pids)}
        line = f"  [{name:44s} n={len(pids):3d}]"
        for a, b2 in [('F_CE','F_CE_match'), ('F_E','F_E_match')]:
            d = [p4(a,p) - p4(b2,p) for p in pids if p4(a,p) is not None and p4(b2,p) is not None]
            if not d: continue
            m_, lo, hi = boot(d)
            key = f'{a}-{b2}'
            block['contrasts'][name][key] = dict(difference_pp=m_, ci95_pp=[lo, hi])
            line += f"  {key.replace('_match','_m'):16s} {m_:+6.1f} [{lo:+6.1f},{hi:+6.1f}]"
        print(line)
    out[model] = block
    print()
json.dump(out, open(Path(__file__).parent/'out/r2_match_quality.json','w'), indent=1)
