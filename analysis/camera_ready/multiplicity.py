"""Multiplicity sensitivity and the claims table (Appendix: statistical details).

Reproduces the "Multiplicity sensitivity" paragraph and Table tab:claims.

One-step: the 15 exploratory/interpreted comparisons per model of the earlier
R12 analysis, each with a two-sided paired problem-bootstrap p-value (fraction
of resamples on the far side of zero, doubled; 20,000 resamples,
random.Random(20260913)), then Holm and Benjamini-Hochberg within model.

Sequential: every contrast in tab:qwen-seq (21) and tab:seq_endpoints_deepseek
(20) at k=64, with exact paired randomization p-values over all 2^11 sign
assignments (as r10.py), then Holm and BH within model.

Claims table: deltas and intervals are looked up in the released outputs
(artifact generated/ files) and in the one-step bootstrap above, and the
adjustment column is recomputed from the two families.

Reads the frozen records under paper/artifact/ and the artifact's generated/
outputs (pass GENERATED=<dir> to point at a reproduce.py output directory);
writes out/multiplicity.json only.
"""
import itertools, json, os, random, sys
from pathlib import Path
import numpy as np

sys.dont_write_bytecode = True          # never leave __pycache__ inside paper/artifact
HERE = Path(__file__).resolve().parent
from artifact_path import ARTIFACT, GENERATED
sys.path.insert(0, str(ARTIFACT/'analysis'))
from analyze_sequential import load_model, ARMS   # noqa: E402

DATA = ARTIFACT/'experiments/decomp_rerun'

# ------------------------------------------------------------------ helpers
def read(path):
    with open(path) as f: return [json.loads(l) for l in f if l.strip()]

def holm(ps):
    m = len(ps); order = sorted(range(m), key=lambda i: ps[i]); adj = [0.]*m; cur = 0.
    for rank, i in enumerate(order):
        cur = max(cur, (m-rank)*ps[i]); adj[i] = min(1., cur)
    return adj

def bh(ps):
    m = len(ps); order = sorted(range(m), key=lambda i: ps[i]); adj = [0.]*m; cur = 1.
    for rank in range(m-1, -1, -1):
        i = order[rank]; cur = min(cur, m/(rank+1)*ps[i]); adj[i] = min(1., cur)
    return adj

def boot(vals, B=20000, seed=20260913):
    """Paired problem bootstrap of a per-problem difference vector (same stream as R12)."""
    rng = random.Random(seed); n = len(vals); ds = []
    for _ in range(B):
        ds.append(sum(vals[rng.randrange(n)] for _ in range(n))/n)
    obs = sum(vals)/n
    q = sum(1 for x in ds if (x <= 0 if obs >= 0 else x >= 0))/B
    ds.sort()
    return obs, min(1., 2*q), ds[int(.025*B)], ds[int(.975*B)-1]

SIGNS = np.array(list(itertools.product([1, -1], repeat=11)))
def exact_p(d):
    obs = abs(d.mean()); null = np.abs((SIGNS*d).mean(axis=1))
    return float((null >= obs-1e-12).mean())

# ------------------------------------------------------------------ one-step family (15 per model)
ONESTEP = {
    'qwen': dict(bank='bank.jsonl', dirs=['out', 'out_ext', 'out_ext2', 'out_ext3', 'out_noi']),
    'deepseek': dict(bank='out_m2/bank2.jsonl', dirs=['out_m2', 'out_m2_ext', 'out_m2_ext2', 'out_m2_ext3', 'out_m2_noi']),
}
A = [  # name, manuscript label, indicator function
    ('(FCE-FCEm)-(FE-FEm)', 'matched correspondence x proof exposure', lambda i, p: (i['F_CE'][p]-i['F_CE_match'][p])-(i['F_E'][p]-i['F_E_match'][p])),
    ('(FCE-FCEs)-(FE-FEs)', 'shuffled correspondence x proof exposure', lambda i, p: (i['F_CE'][p]-i['F_CE_shuf'][p])-(i['F_E'][p]-i['F_E_shuf'][p])),
    ('(FCE-FC)-(FCE_noI-FC_noI)', 'instruction x recovery', lambda i, p: (i['F_CE'][p]-i['F_C'][p])-(i['F_CE_noI'][p]-i['F_C_noI'][p])),
    ('FCE-FC', 'F_CE - F_C', lambda i, p: i['F_CE'][p]-i['F_C'][p]),
    ('FCE-FCEm', 'F_CE - F_CE_m', lambda i, p: i['F_CE'][p]-i['F_CE_match'][p]),
    ('FCE-FCEs', 'F_CE - F_CE_s', lambda i, p: i['F_CE'][p]-i['F_CE_shuf'][p]),
    ('FCEm-FCEs', 'F_CE_m - F_CE_s', lambda i, p: i['F_CE_match'][p]-i['F_CE_shuf'][p]),
    ('FE-FEm', 'F_E - F_E_m', lambda i, p: i['F_E'][p]-i['F_E_match'][p]),
    ('FE-FEs', 'F_E - F_E_s', lambda i, p: i['F_E'][p]-i['F_E_shuf'][p]),
    ('FCo-FC', 'F_C_o - F_C', lambda i, p: i['F_C_other'][p]-i['F_C'][p]),
    ('FCo-F', 'F_C_o - F', lambda i, p: i['F_C_other'][p]-i['F'][p]),
    ('FC-F', 'F_C - F (exposure penalty)', lambda i, p: i['F_C'][p]-i['F'][p]),
    ('FC_noI-F_noI', 'exposure penalty without instruction', lambda i, p: i['F_C_noI'][p]-i['F_noI'][p]),
    ('FCE_noI-FC_noI', 'recovery without instruction', lambda i, p: i['F_CE_noI'][p]-i['F_C_noI'][p]),
    ('FCE-I', 'F_CE - I', lambda i, p: i['F_CE'][p]-i['I'][p]),
]

def onestep(model):
    cfg = ONESTEP[model]
    pids = sorted(r['id'] for r in read(DATA/cfg['bank']) if not r.get('is_framing_holdout'))
    keep = set(pids); ind = {}
    for d in cfg['dirs']:
        for r in read(DATA/d/'verified.jsonl'):
            if r['id'] in keep:
                ind.setdefault(r['arm'], {}).setdefault(r['id'], 0)
                ind[r['arm']][r['id']] |= int(r['status'] == 'PASS')
    rows = []
    for name, label, fn in A:
        obs, p, lo, hi = boot([fn(ind, q) for q in pids])
        rows.append(dict(contrast=name, label=label, delta_pp=100*obs, ci95_pp=[100*lo, 100*hi], p_boot=p))
    for r, h, b in zip(rows, holm([r['p_boot'] for r in rows]), bh([r['p_boot'] for r in rows])):
        r['holm'] = h; r['bh'] = b
    return dict(problems=len(pids), resamples=20000, seed='random.Random(20260913)', comparisons=rows)

# ------------------------------------------------------------------ sequential family (all table contrasts)
SEQ = {  # manuscript label -> function of per-arm k=64 vectors (artifact labels: f=E, E_s=E_m)
    'CE-I': lambda v: v['CE']-v['I'], '(CE-CE_s)-(E-E_m)': lambda v: v['CE']-v['CE_s']-v['f']+v['E_s'],
    'C-I': lambda v: v['C']-v['I'], 'CE-C': lambda v: v['CE']-v['C'], 'CE-CE_s': lambda v: v['CE']-v['CE_s'],
    'CE-E': lambda v: v['CE']-v['f'], 'C-E': lambda v: v['C']-v['f'], 'CE_s-I': lambda v: v['CE_s']-v['I'],
    'E-I': lambda v: v['f']-v['I'], 'E-E_m': lambda v: v['f']-v['E_s'], 'E_m-I': lambda v: v['E_s']-v['I'],
    'E_m-noise': lambda v: v['E_s']-v['noise'], 'CE_s-E_m': lambda v: v['CE_s']-v['E_s'],
    'CE-CE_m': lambda v: v['CE']-v['CE_m'], '(CE-CE_m)-(E-E_m)': lambda v: v['CE']-v['CE_m']-v['f']+v['E_s'],
    'CE_m-I': lambda v: v['CE_m']-v['I'], 'CE_m-C': lambda v: v['CE_m']-v['C'], 'CE_m-CE_s': lambda v: v['CE_m']-v['CE_s'],
    'C-CE_s': lambda v: v['C']-v['CE_s'], 'CE_s-E': lambda v: v['CE_s']-v['f'], 'CE_m-E_m': lambda v: v['CE_m']-v['E_s'],
}
TABLE = {  # rows of tab:qwen-seq and tab:seq_endpoints_deepseek, in table order
    'qwen': ['CE-I', '(CE-CE_s)-(E-E_m)', 'C-I', 'CE-C', 'CE-CE_s', 'CE-E', 'C-E', 'CE_s-I', 'E-E_m', 'E_m-I', 'E_m-noise',
             'CE_s-E_m', 'CE-CE_m', '(CE-CE_m)-(E-E_m)', 'CE_m-I', 'CE_m-C', 'CE_m-CE_s', 'E-I', 'C-CE_s', 'CE_s-E', 'CE_m-E_m'],
    'deepseek': ['CE-I', 'C-I', 'CE-C', 'CE-CE_s', 'CE-E', 'C-E', 'CE_s-I', 'E-I', 'E-E_m', 'E_m-I', 'CE_s-E_m',
                 '(CE-CE_s)-(E-E_m)', 'CE-CE_m', '(CE-CE_m)-(E-E_m)', 'CE_m-I', 'CE_m-C', 'CE_m-CE_s', 'C-CE_s', 'CE_s-E', 'CE_m-E_m'],
}

def sequential(model):
    sub, arms = ('out', ARMS+['noise']) if model == 'qwen' else ('out_dsp', ARMS)
    ids, v, _, _ = load_model(sub, arms)
    assert len(ids) == 11
    rows = []
    for name in TABLE[model]:
        d = SEQ[name](v)[:, -1]
        rows.append(dict(contrast=name, delta_pp=float(100*d.mean()), exact_p=exact_p(d),
                         nonzero_problems=int((np.abs(d) > 1e-12).sum())))
    for r, h, b in zip(rows, holm([r['exact_p'] for r in rows]), bh([r['exact_p'] for r in rows])):
        r['holm'] = h; r['bh'] = b
    return dict(problems=11, sign_assignments=2048, comparisons=rows)

# ------------------------------------------------------------------ claims table
# (section, model, manuscript contrast, family, key in that family, status as printed)
CLAIMS = [
    ('Repair does not beat retrying', 'qwen', 'F_CE-I', 'onestep', 'FCE-I', 'Expl.'),
    ('Repair does not beat retrying', 'deepseek', 'F_CE-I', 'onestep', 'FCE-I', 'Expl.'),
    ('Repair does not beat retrying', 'qwen', 'CE-I', 'sequential', 'CE-I', 'Prim.'),
    ('Repair does not beat retrying', 'deepseek', 'CE-I', 'sequential', 'CE-I', 'Prim.'),
    ('Own failed proof harms', 'qwen', 'F_C-F', 'onestep', 'FC-F', 'Reg.'),
    ('Own failed proof harms', 'deepseek', 'F_C-F', 'onestep', 'FC-F', 'Expl.'),
    ('Own failed proof harms', 'qwen', 'F_C_o-F_C', 'onestep', 'FCo-FC', 'Expl.'),
    ('Own failed proof harms', 'deepseek', 'F_C_o-F_C', 'onestep', 'FCo-FC', 'Expl.'),
    ('Own failed proof harms', 'qwen', 'C-I', 'sequential', 'C-I', 'Reg.'),
    ('Own failed proof harms', 'deepseek', 'C-I', 'sequential', 'C-I', 'Reg.'),
    ('Correct diagnostic helps within repair (Qwen)', 'qwen', 'F_CE-F_C', 'onestep', 'FCE-FC', 'Expl.'),
    ('Correct diagnostic helps within repair (Qwen)', 'qwen', 'F_CE-F_CE_m', 'onestep', 'FCE-FCEm', 'Expl.'),
    ('Correct diagnostic helps within repair (Qwen)', 'qwen', 'CE-CE_m', 'sequential', 'CE-CE_m', 'Prim.'),
    ('Correct diagnostic helps within repair (Qwen)', 'qwen', 'CE-C', 'sequential', 'CE-C', 'Reg.'),
    ('Correct diagnostic helps within repair (Qwen)', 'qwen', 'CE-CE_s', 'sequential', 'CE-CE_s', 'Reg.'),
    ('Correct diagnostic helps within repair (Qwen)', 'deepseek', 'CE-CE_m', 'sequential', 'CE-CE_m', 'Prim.'),
    ('No correctness advantage without the failed proof', 'qwen', 'F_E-F_E_s pooled', None, None, 'Reg.'),
    ('No correctness advantage without the failed proof', 'qwen', 'E-E_m', 'sequential', 'E-E_m', 'Reg.'),
    ('No correctness advantage without the failed proof', 'deepseek', 'E-E_m', 'sequential', 'E-E_m', 'Reg.'),
    ('Interactions unresolved', 'qwen', '(CE-CE_m)-(E-E_m)', 'sequential', '(CE-CE_m)-(E-E_m)', 'Reg.'),
    ('Interactions unresolved', 'deepseek', '(CE-CE_m)-(E-E_m)', 'sequential', '(CE-CE_m)-(E-E_m)', 'Reg.'),
    ('Interactions unresolved', 'qwen', 'one-step, matched', 'onestep', '(FCE-FCEm)-(FE-FEm)', 'Expl.'),
    ('Interactions unresolved', 'qwen', 'one-step, instruction', 'onestep', '(FCE-FC)-(FCE_noI-FC_noI)', 'Expl.'),
]
# released-output interval for each sequential claim (artifact label of the contrast)
SEQ_KEY = {'CE-I': 'CE-I', 'C-I': 'C-I', 'CE-CE_m': 'CE-CE_m', 'CE-C': 'CE-C', 'CE-CE_s': 'CE-CE_s',
           'E-E_m': 'f-E_s', '(CE-CE_m)-(E-E_m)': 'matched_interaction'}

def survives(h, b):
    return 'Holm, BH' if h < .05 else 'BH only' if b < .05 else 'none'

def claims(fam):
    seqjson = json.loads((GENERATED/'sequential.json').read_text())
    pooled = json.loads((GENERATED/'pooled_diagnostics.json').read_text())
    inter = json.loads((GENERATED/'exploratory_onestep_interactions.json').read_text())
    out = []
    for section, model, label, family, key, status in CLAIMS:
        row = dict(section=section, model=model, contrast=label, status=status)
        if family:
            r = next(x for x in fam[family][model]['comparisons'] if x['contrast'] == key)
            row.update(delta_pp=r['delta_pp'], p=r.get('p_boot', r.get('exact_p')),
                       holm=r['holm'], bh=r['bh'], adjustment=survives(r['holm'], r['bh']))
            if family == 'onestep':
                row['ci95_pp_bootstrap_20260913'] = r['ci95_pp']
            else:
                row['ci95_pp_released'] = seqjson[model]['endpoint_contrasts'][SEQ_KEY[key]]['ci95_pp']
        if label == 'F_E-F_E_s pooled':   # registered precision estimate; no difference claimed
            row.update(delta_pp=pooled['pooled_difference_pp'], ci95_pp_released=pooled['problem_bootstrap_95_pp'])
        if label.startswith('one-step, '):  # the paper prints the released interaction intervals
            k = 'matched_diagnostic_by_proof_exposure' if label.endswith('matched') else 'instruction_by_diagnostic_recovery'
            row['ci95_pp_released'] = inter['results'][k]['ci95_pp']
        out.append(row)
    return out, dict(pooled_F_E_minus_F_E_s=dict(delta_pp=pooled['pooled_difference_pp'], ci95_pp=pooled['problem_bootstrap_95_pp']))

def main():
    fam = {'onestep': {m: onestep(m) for m in ONESTEP}, 'sequential': {m: sequential(m) for m in TABLE}}
    rows, extra = claims(fam)
    for name, block in fam.items():
        for model, b in block.items():
            print(f'===== {name} / {model} ({len(b["comparisons"])} comparisons)')
            for r in b['comparisons']:
                p = r.get('p_boot', r.get('exact_p'))
                print(f"  {r['contrast']:28s} {r['delta_pp']:+7.2f}  p={p:.4f}  Holm={r['holm']:.4f}  BH={r['bh']:.4f}")
    print('===== claims')
    for r in rows:
        ci = r.get('ci95_pp_released') or r.get('ci95_pp_bootstrap_20260913')
        ci = f'[{ci[0]:+.1f},{ci[1]:+.1f}]' if ci else ''
        print(f"  {r['model']:8s} {r['contrast']:22s} {r.get('delta_pp', float('nan')):+7.2f} {ci:16s} {r['status']:6s} {r.get('adjustment', '')}")
    (HERE/'out').mkdir(exist_ok=True)
    (HERE/'out/multiplicity.json').write_text(json.dumps(dict(families=fam, claims=rows, released_lookups=extra), indent=1)+'\n')

if __name__ == '__main__':
    main()
