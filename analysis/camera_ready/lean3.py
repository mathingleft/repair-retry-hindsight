"""Lean 3 predecessors: Table tab:lean3 and the Lean 3 numbers in the text.

Reproduces, from the frozen artifact only, the second paragraph of Appendix
app:drift ("Qwen's frozen bank contains 50 predecessors written in Lean 3
syntax ...") with Table tab:lean3, and the Lean 3 subgroup figures quoted in
Results (sections/results.tex), the Introduction and the Discussion.

Lean 3 detection rule. A frozen Qwen predecessor is "written in Lean 3 syntax"
when its proof body contains the Lean 3 tactic-block keyword `begin` as a whole
word. The body is the text after the first `:=` with `--` line comments and
`/- -/` block comments removed (the normalization of copy_rate.body(), before
whitespace collapsing). Lean 4 has no `begin ... end` blocks, so the verifier
rejects every such proof; on 48 of the 50 its first diagnostic is
"Unknown identifier `begin`". The same rule is applied to generated samples
("contains `begin`") and, for the sequential runs, to attempt-1 code.
A sequential "Lean 3 error" is a diagnostic that names `begin` as an unknown
identifier.

Uncertainty: paired problem bootstrap within each subset, 10,000 resamples
drawn with a fresh Python random.Random(0) per contrast and subset
(randrange, problem by problem), endpoints at sorted positions int(.025B) and
int(.975B). This single rule reproduces all ten tab:lean3 intervals. The numpy
default_rng(20260906) convention of r18.py differs by one grid step (1/117 pp)
on two "other problems" endpoints, which sit on a quantile boundary.
Exploratory and unadjusted.

Sequential conditioning: "trajectories whose bare first attempt was not
Lean 3" are trajectories whose attempt 1 failed and whose attempt-1 code is not
Lean 3 (repair trajectories); attempt-1 passes are excluded.
"""
import collections, json, random, re, sys
from pathlib import Path
import numpy as np

from common import ARTIFACT
ART = ARTIFACT
ROOT = ART/'experiments/decomp_rerun'
sys.path.insert(0, str(ROOT))
from copy_rate import body                      # released copy definition
from error_use import unknown_ident             # released unknown-identifier rule

SEED, B = 0, 10000
QDIRS = ['out','out_ext','out_ext2','out_ext3','out_noi']
CONTRASTS = [('F_C','F'), ('F_CE','F_C'), ('F_CE','F_CE_match'), ('F_C_other','F_C'), ('F_CE','I')]
SEQ_CODE_ARMS = ['C','CE','CE_s','CE_m']

def read(p): return [json.loads(l) for l in open(p) if l.strip()]

def raw_body(code):
    if code is None: return ''
    i = code.find(':='); b = code[i+2:] if i >= 0 else code
    b = re.sub(r'--[^\n]*', '', b)
    return re.sub(r'/-.*?-/', '', b, flags=re.S)

LEAN3 = re.compile(r'\bbegin\b')
def is_lean3(code): return bool(LEAN3.search(raw_body(code)))
BEGIN_ERR = 'Unknown identifier `begin`'

def boot(d):
    n = len(d); r = random.Random(SEED)
    m = sorted(sum(d[r.randrange(n)] for _ in range(n))/n for _ in range(B))
    return round(100*sum(d)/n, 1), [round(100*m[int(.025*B)], 1), round(100*m[int(.975*B)], 1)]

# ---------------------------------------------------------------- one-step
bank = {r['id']: r for r in read(ROOT/'bank.jsonl') if r.get('is_framing_holdout') not in (True, 'True')}
rows = collections.defaultdict(list)
for d in QDIRS:
    for r in read(ROOT/d/'verified.jsonl'):
        if r['id'] in bank: rows[(r['arm'], r['id'])].append(r)
pids = sorted(bank)
assert len(pids) == 167

lean3 = [p for p in pids if is_lean3(bank[p]['predecessor_code'])]
other = [p for p in pids if p not in lean3]
begin_err = [p for p in lean3 if BEGIN_ERR in bank[p]['predecessor_error']]
ident = [p for p in pids if (u := unknown_ident(bank[p]['predecessor_error'])) and u not in bank[p]['statement']]
pred = {p: body(bank[p]['predecessor_code']) for p in pids}

def p4(arm, p):
    rs = rows[(arm, p)]; assert len(rs) == 4, (arm, p, len(rs))
    return int(any(r['status'] == 'PASS' for r in rs))

def solved(arm, ps): return sum(p4(arm, p) for p in ps)

table = {}
for a, b in CONTRASTS:
    cell = {}
    for name, ps in [('lean3', lean3), ('other', other)]:
        m, ci = boot([p4(a, p) - p4(b, p) for p in ps])
        cell[name] = dict(difference_pp=m, ci95_pp=ci, solved_subtracted=solved(b, ps), solved_first=solved(a, ps))
    table[f'{a}-{b}'] = cell

def share(arm, ps, pred_fn):
    rs = [r for p in ps for r in rows[(arm, p)]]
    k = sum(pred_fn(r) for r in rs)
    return dict(count=k, samples=len(rs), percent=round(100*k/len(rs), 1))

writes_lean3 = {a: share(a, lean3, lambda r: is_lean3(r.get('lean_code'))) for a in ['F', 'F_C', 'F_CE', 'I']}
copy = {name: share('F_C', ps, lambda r: (b := body(r.get('lean_code'))) is not None and b == pred[r['id']])
        for name, ps in [('lean3', lean3), ('other', other)]}

onestep = dict(
    n_problems=len(pids), lean3_predecessors=len(lean3), lean3_problem_ids=lean3,
    lean3_with_begin_unknown_identifier=len(begin_err),
    lean3_without_begin_diagnostic={p: bank[p]['predecessor_error'][:160] for p in lean3 if p not in begin_err},
    identifier_subset=len(ident), begin_in_identifier_subset=len(set(begin_err) & set(ident)),
    I_solves=dict(lean3=solved('I', lean3), other=solved('I', other)),
    tab_lean3=table, samples_containing_begin_on_lean3=writes_lean3, F_C_exact_copy=copy)

# ---------------------------------------------------------------- sequential (Qwen)
def norm(r):
    return dict(id=r.get('id', r.get('problem_id')), traj=r.get('traj', r.get('trajectory_id')),
                attempt=r.get('attempt', r.get('attempt_id')), status=r.get('status', r.get('verification_result')),
                code=r.get('code', r.get('extracted_proof')), error=r.get('error', r.get('compiler_output')) or '')

seq = {'failed_attempts_with_lean3_error': {}, 'solved_given_attempt1_not_lean3': {}}
tot = [0, 0]
for arm in SEQ_CODE_ARMS:
    rs = [norm(r) for r in read(ROOT/'out'/f'depth_{arm}.jsonl')]
    fails = [r for r in rs if r['status'] != 'PASS']
    k = sum(BEGIN_ERR in r['error'] for r in fails); tot[0] += k; tot[1] += len(fails)
    seq['failed_attempts_with_lean3_error'][arm] = dict(count=k, failed=len(fails), percent=round(100*k/len(fails), 2))
    by = collections.defaultdict(list)
    for r in rs: by[(r['id'], r['traj'])].append(r)
    a1 = {t: next(r for r in g if r['attempt'] == 1) for t, g in by.items()}
    keep = [t for t in by if a1[t]['status'] != 'PASS' and not is_lean3(a1[t]['code'])]
    win = sum(any(r['status'] == 'PASS' for r in by[t]) for t in keep)
    seq['solved_given_attempt1_not_lean3'][arm] = dict(solved=win, trajectories=len(keep), of_total=len(by),
                                                       percent=round(100*win/len(keep), 1))
seq['failed_attempts_with_lean3_error']['pooled'] = dict(count=tot[0], failed=tot[1], percent=round(100*tot[0]/tot[1], 2))

out = dict(rule='predecessor/sample proof body (after first `:=`, comments removed) contains whole-word `begin`; '
                'sequential Lean 3 error = diagnostic containing "Unknown identifier `begin`"; '
                'sequential subset = attempt 1 failed and attempt-1 code not Lean 3',
           bootstrap=f'paired problem bootstrap within subset, B={B}, fresh random.Random({SEED}) per contrast/subset, sorted endpoints',
           onestep=onestep, sequential=seq)
Path(__file__).with_name('out').mkdir(exist_ok=True)
json.dump(out, open(Path(__file__).with_name('out')/'lean3.json', 'w'), indent=1)

print(f"Lean 3 predecessors {len(lean3)} ({len(begin_err)} with the `begin` diagnostic); "
      f"identifier subset {len(ident)}, of which `begin` {onestep['begin_in_identifier_subset']}")
print(f"I solves {onestep['I_solves']['lean3']}/{len(lean3)} vs {onestep['I_solves']['other']}/{len(other)}")
for k, c in table.items():
    print(f"  {k:14s}" + ''.join(f"  {c[s]['difference_pp']:+5.1f} [{c[s]['ci95_pp'][0]:+5.1f},{c[s]['ci95_pp'][1]:+5.1f}] "
                                 f"({c[s]['solved_subtracted']}->{c[s]['solved_first']})" for s in ['lean3', 'other']))
print('samples containing begin (Lean 3 bank):', {a: v['percent'] for a, v in writes_lean3.items()})
print('F_C exact copy:', {k: v['percent'] for k, v in copy.items()})
print('sequential Lean 3 errors:', {a: v['percent'] for a, v in seq['failed_attempts_with_lean3_error'].items()})
print('solved | attempt 1 not Lean 3:', {a: (v['solved'], v['trajectories'], v['percent']) for a, v in seq['solved_given_attempt1_not_lean3'].items()})
