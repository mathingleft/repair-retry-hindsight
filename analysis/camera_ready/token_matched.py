"""Sequential success at equal per-trajectory token budgets (Table tab:token_matched and its text).

Reads data/sequential_attempt_tokens.csv.xz (built by build_attempt_tokens.py; see its docstring for how each
count was obtained) and checks it against the released compact records before using it. Standard library only.

Method (ported from the reviewer-response analysis r05_tokens.py, 2026-10-12):
  * Each adaptive arm stops at its first verified proof and is capped at 64 attempts; a trajectory succeeds
    within budget T if its cumulative tokens at the first PASS are <= T.
  * Memoryless I draws from the 640 pooled draws per problem in 1,000 seeded random orders (random.Random(20261010),
    one stream per model, consumed problem by problem for total, input, then output accounting), stopping at the
    first PASS.
  * Censoring: a cell is reportable only below the smallest final spend of any failed trajectory in the arm and
    below the smallest 640-draw I spend over problems.
  * Contrasts: per-problem arm-minus-I differences; 95% paired problem bootstrap (10,000 resamples,
    random.Random(0), sorted endpoints) and the exact two-sided sign-flip test over all 2^11 assignments.
  * Output tokens of the rebuilt Qwen E/noise/assert (and DeepSeek E) arms are predicted by OLS of response tokens
    on extracted-proof tokens, fitted on the exact E_m rows, whose scaffold is identical to E's.
Raw labels: f = manuscript E, E_s = E_m, memI = I.
"""
import collections
import csv
import io
import json
import lzma
import random
import statistics
from pathlib import Path

from artifact_path import ARTIFACT

HERE = Path(__file__).resolve().parent
DATA = ARTIFACT/'experiments/decomp_rerun'
TABLE = HERE/'data/sequential_attempt_tokens.csv.xz'
OUT = HERE/'out'
B, SEED, NPERM, ISEED = 10000, 0, 1000, 20261010
BUDGETS = [2_000, 5_000, 10_000, 20_000, 50_000, 100_000, 200_000, 300_000, 500_000, 1_000_000]
MODELS = {'qwen': dict(depth='out', bank='depth/depth_bank.jsonl'),
          'deepseek': dict(depth='out_dsp', bank='depth/depth_bank_dsp.jsonl')}
NAME = {'f': 'E', 'E_s': 'E_m', 'memI': 'I'}
# Headline cells: the largest budget at which every listed arm (CE, E, and C in the text) is uncensored.
HEADLINE = {('qwen', 'total'): 200_000, ('qwen', 'input'): 20_000, ('qwen', 'output'): 20_000,
            ('deepseek', 'total'): 200_000, ('deepseek', 'input'): 50_000, ('deepseek', 'output'): 5_000}


def mean(xs):
    xs = list(xs); return sum(xs)/len(xs) if xs else float('nan')


def read_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(l) for l in f if l.strip()]


def boot_ci(pids, f, B=B, seed=SEED, alpha=0.05):
    rng = random.Random(seed); n = len(pids); ds = []
    for _ in range(B):
        S = [pids[rng.randrange(n)] for _ in range(n)]
        ds.append(sum(f(p) for p in S)/n)
    ds.sort()
    return ds[int(alpha/2*B)], ds[int((1-alpha/2)*B)-1]


def exact_flip_test(diffs):
    d = [x for x in diffs if x != 0]; n = len(d)
    if n == 0:
        return 1.0
    obs = abs(sum(d)); cnt = 0
    for mask in range(1 << n):
        s = sum(d[i] if (mask >> i) & 1 else -d[i] for i in range(n))
        cnt += abs(s) >= obs - 1e-12
    return cnt/(1 << n)


def load_table():
    rows = list(csv.DictReader(io.StringIO(lzma.decompress(TABLE.read_bytes()).decode())))
    for r in rows:
        for k in ('trajectory', 'attempt', 'passed', 'prompt_tokens', 'code_tokens'):
            r[k] = int(r[k])
        r['response_tokens'] = int(r['response_tokens']) if r['response_tokens'] != '' else None
    return rows


def check_against_release(model, rows):
    """Every table row must match a released compact record's outcome, and cover it exactly."""
    d = MODELS[model]['depth']
    by = collections.defaultdict(dict)
    for r in rows:
        by[r['arm']][(r['problem_id'], r['trajectory'], r['attempt'])] = r['passed']
    for arm, recs in by.items():
        if arm == 'memI':
            rel = {(x['problem_id'], x['trajectory_id'], x['attempt_id']): int(x['verification_result'] == 'PASS')
                   for x in read_jsonl(DATA/d/'depth_memI.jsonl')}
        else:
            rel = {(x['id'], x['traj'], x['attempt']): int(x['status'] == 'PASS') for x in read_jsonl(DATA/d/f'depth_{arm}.jsonl')}
        assert rel == recs, (model, arm, 'table does not match released records')


def ols(pairs):
    xs = [x for x, y in pairs]; ys = [y for x, y in pairs]; mx, my = mean(xs), mean(ys)
    b = sum((x-mx)*(y-my) for x, y in pairs)/sum((x-mx)**2 for x in xs); a = my-b*mx
    r2 = 1-sum((y-a-b*x)**2 for x, y in pairs)/sum((y-my)**2 for y in ys)
    return a, b, r2


def analyze(model, rows):
    pids = [r['id'] for r in read_jsonl(DATA/MODELS[model]['bank'])]
    res = dict(model=model)
    draws = collections.defaultdict(list)
    for r in sorted((r for r in rows if r['arm'] == 'memI'), key=lambda r: (r['problem_id'], int(r['row']))):
        draws[r['problem_id']].append((bool(r['passed']), r['prompt_tokens'], r['response_tokens']))
    exact = [a for a in ('C', 'CE', 'CE_s', 'CE_m', 'E_s') if any(r['arm'] == a for r in rows)]
    rebuilt = [a for a in ('f', 'noise', 'assert') if any(r['arm'] == a for r in rows)]
    # output calibration on exact E_m rows
    pairs = [(r['code_tokens'], r['response_tokens']) for r in rows if r['arm'] == 'E_s']
    a, b, r2 = ols(pairs)
    esk = collections.defaultdict(list)
    for r in rows:
        if r['arm'] == 'E_s':
            esk[r['attempt']].append(r['response_tokens'])
    res['output_calibration'] = dict(intercept=a, slope=b, r2=r2, n=len(pairs), E_m_mean_response=mean(y for x, y in pairs),
                                     validation={nm: dict(mean_error=mean((a+b*r['code_tokens'])-r['response_tokens'] for r in rows if r['arm'] == arm),
                                                          mean_response=mean(r['response_tokens'] for r in rows if r['arm'] == arm))
                                                 for nm, arm in (('I', 'memI'), ('CE', 'CE'), ('C', 'C'))})
    traj = {}
    for arm in exact+rebuilt:
        T = collections.defaultdict(list)
        for r in rows:
            if r['arm'] == arm:
                rt = r['response_tokens'] if arm in exact else a+b*r['code_tokens']
                T[(r['problem_id'], r['trajectory'])].append((r['attempt'], bool(r['passed']), r['prompt_tokens'], rt))
        traj[arm] = T
    res['rebuilt_mean_response'] = {NAME.get(arm, arm): dict(
        predicted=mean(x[3] for v in traj[arm].values() for x in v),
        E_m_per_attempt_mean_instead=mean(mean(esk[x[0]]) if esk.get(x[0]) else a for v in traj[arm].values() for x in v))
        for arm in rebuilt}
    arms = ['CE', 'C', 'CE_s', 'CE_m', 'f', 'E_s'] + (['noise', 'assert'] if model == 'qwen' else [])
    tokf = {'total': lambda pt, rt: pt+rt, 'input': lambda pt, rt: pt, 'output': lambda pt, rt: rt}
    rng = random.Random(ISEED)

    def cum(T, tok):
        out = {}
        for key, att in T.items():
            c = 0.0; first = None
            for k, ok, pt, rt in sorted(att):
                c += tok(pt, rt)
                if ok:
                    first = c; break
            out[key] = (first, c)
        return out

    def memI_first(pid, tok):
        ds = draws[pid]; res_ = []
        for _ in range(NPERM):
            c = 0.0; first = None
            for i in rng.sample(range(len(ds)), len(ds)):
                ok, pt, rt = ds[i]; c += tok(pt, rt)
                if ok:
                    first = c; break
            res_.append(first)
        return res_

    res['accounting'] = {}
    for which, tok in tokf.items():
        CU = {arm: cum(traj[arm], tok) for arm in arms}
        IF = {pid: memI_first(pid, tok) for pid in pids}
        lim_arm = {arm: min((c for f, c in CU[arm].values() if f is None), default=float('inf')) for arm in arms}
        spend640 = {pid: sum(tok(pt, rt) for ok, pt, rt in draws[pid]) for pid in pids}
        lim_I = min(spend640.values())

        def succ(arm, pid, T):
            v = [CU[arm][(pid, t)] for t in range(10) if (pid, t) in CU[arm]]
            return sum(1 for f, c in v if f is not None and f <= T)/len(v)

        def succI(pid, T):
            return sum(1 for f in IF[pid] if f is not None and f <= T)/len(IF[pid])

        cells = []
        for T in BUDGETS:
            for arm in arms:
                dv = {p: succ(arm, p, T)-succI(p, T) for p in pids}
                lo, hi = boot_ci(pids, dv.get)
                cells.append(dict(budget=T, arm=NAME.get(arm, arm), censored=not (T < lim_I and T < lim_arm[arm]),
                                  I_pct=100*mean(succI(p, T) for p in pids), arm_pct=100*mean(succ(arm, p, T) for p in pids),
                                  delta_pp=100*mean(dv.values()), ci95_pp=[100*lo, 100*hi], p_exact=exact_flip_test([dv[p] for p in pids])))
        res['accounting'][which] = dict(
            mean_per_attempt={**{NAME.get(arm, arm): mean(tok(pt, rt) for v in traj[arm].values() for k, ok, pt, rt in v) for arm in arms},
                              'I': mean(tok(pt, rt) for p in pids for ok, pt, rt in draws[p])},
            mean_cumulative_per_trajectory={**{NAME.get(arm, arm): mean(c for f, c in CU[arm].values()) for arm in arms},
                                            'I_64_draws_no_stop': mean(sum(tok(pt, rt) for ok, pt, rt in draws[p][:64]) for p in pids)},
            censoring_limits={**{NAME.get(arm, arm): lim_arm[arm] for arm in arms}, 'I_640_draws': lim_I},
            cells=cells)
        print(f'{model} {which}: done', flush=True)
    return res


def headline(results):
    """The table rows and the numbers the text quotes."""
    rows = []
    for (model, which), T in HEADLINE.items():
        acc = results[model]['accounting'][which]
        cell = {c['arm']: c for c in acc['cells'] if c['budget'] == T}
        assert not any(cell[a]['censored'] for a in ('CE', 'E', 'C')), (model, which, T)
        rows.append(dict(model=model, accounting=which, budget=T, I_pct=round(cell['CE']['I_pct'], 1),
                         **{f'{a}-I': dict(delta_pp=round(cell[a]['delta_pp'], 1), ci95_pp=[round(x, 1) for x in cell[a]['ci95_pp']],
                                            p_exact=round(cell[a]['p_exact'], 3)) for a in ('CE', 'E', 'C')}))
    q = results['qwen']['accounting']
    qtot = {c['budget']: c for c in q['total']['cells'] if c['arm'] == 'E'}
    qce = {c['budget']: c for c in q['total']['cells'] if c['arm'] == 'CE'}
    text = dict(
        per_attempt={m: dict(CE_input=results[m]['accounting']['input']['mean_per_attempt']['CE'],
                             CE_output=results[m]['accounting']['output']['mean_per_attempt']['CE'],
                             I_input=results[m]['accounting']['input']['mean_per_attempt']['I'],
                             I_output=results[m]['accounting']['output']['mean_per_attempt']['I']) for m in results},
        qwen_E_input_ratio_vs_64_I=q['input']['mean_cumulative_per_trajectory']['E']/q['input']['mean_cumulative_per_trajectory']['I_64_draws_no_stop'],
        qwen_E_total_ratio_vs_64_I=q['total']['mean_cumulative_per_trajectory']['E']/q['total']['mean_cumulative_per_trajectory']['I_64_draws_no_stop'],
        qwen_E_minus_I_50k_to_200k={T: dict(delta=qtot[T]['delta_pp'], ci=qtot[T]['ci95_pp']) for T in (50_000, 100_000, 200_000)},
        qwen_CE_minus_I_50k_to_200k={T: qce[T]['delta_pp'] for T in (50_000, 100_000, 200_000)},
        qwen_I_draws_at_200k_total=200_000/q['total']['mean_per_attempt']['I'])
    return rows, text


def main():
    table = load_table()
    results = {}
    for model in MODELS:
        rows = [r for r in table if r['model'] == model]
        check_against_release(model, rows)
        results[model] = analyze(model, rows)
    rows, text = headline(results)
    OUT.mkdir(exist_ok=True)
    (OUT/'token_matched.json').write_text(json.dumps(dict(table=rows, text=text, full=results), indent=1) + '\n')
    for r in rows:
        print(r['model'], r['accounting'], r['budget'], r['I_pct'], *(f"{a} {r[a]['delta_pp']:+.1f} {r[a]['ci95_pp']} p={r[a]['p_exact']}" for a in ('CE-I', 'E-I', 'C-I')))
    print(json.dumps(text, indent=1))


if __name__ == '__main__':
    main()
