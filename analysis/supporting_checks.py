"""Restore secondary checks from frozen rows; no generation or verification.

Run with sdvenv/bin/python analysis/supporting_checks.py.
Sequential similarity is descriptive: available first-eight proofs, up to 20
pairs per trajectory, 1500 normalized characters, problem-cluster bootstrap.
"""
import collections
import difflib
import itertools
import json
import random
import re
import sys

import numpy as np
import matplotlib.pyplot as plt
from results_figures import ROOT, PAPER, read, body, sign, save

sys.path.insert(0, str(ROOT))
from arms import shuffled_error


def main():
    bank = {r['id']: r for r in read(ROOT/'bank.jsonl') if not r.get('is_framing_holdout')}
    rows = collections.defaultdict(lambda: collections.defaultdict(list))
    for d in ['out', 'out_ext', 'out_ext2', 'out_noi']:
        for r in read(ROOT/d/'verified.jsonl'):
            if r['id'] in bank:
                rows[r['arm']][r['id']].append(r)
    assert all(set(groups) == set(bank) and all(len(rs) == 4 for rs in groups.values())
               for groups in rows.values())
    patterns = {}
    for p, b in bank.items():
        m = re.search(r"[Uu]nknown (?:identifier|constant)[ :]*[`'‘]?([^`'’\s]+)", b['predecessor_error'])
        if m and m[1] not in b['statement']:
            patterns[p] = re.compile(r"(?<![A-Za-z0-9_.'])"+re.escape(m[1])+r"(?![A-Za-z0-9_'])")
    strict = [p for p, pat in patterns.items() if not pat.search(shuffled_error(p))]
    assert len(patterns) == 68 and len(strict) == 62
    usage = {}
    for a in ['F_E_noI', 'F_E_shuf_noI']:
        usage[a] = np.array([sum(bool(patterns[p].search(body(r.get('lean_code')) or ''))
                                for r in rows[a][p]) for p in strict])
    subset = {a: sum(any(r['status']=='PASS' for r in rows[a][p]) for p in patterns)
              for a in ['I', 'F_C', 'F_CE', 'F_CE_shuf', 'F_C_noI', 'F_CE_noI']}
    c0, c1 = [np.array([sum(r['status']=='PASS' for r in rows[a][p]) for p in patterns])
              for a in ['F_C_noI', 'F_CE_noI']]
    coverage = {a: {p for p, rs in rows[a].items() if any(r['status']=='PASS' for r in rs)}
                for a in ['I', 'F_noI']}
    copies = {a: sum(body(r.get('lean_code')) == body(bank[p]['predecessor_code'])
                    for p, rs in rows[a].items() for r in rs) for a in ['F_C', 'F_CE']}
    passes = {a: sum(r['status']=='PASS' for rs in rows[a].values() for r in rs) for a in copies}
    report = {'strict_n': len(strict), 'strict_identifier_counts': {a: int(v.sum()) for a,v in usage.items()},
              'strict_sign_p': sign(*usage.values()), 'subgroup_successes': subset,
              'subgroup_no_instruction_sign_p': sign(c1, c0),
              'retry_overlap': len(coverage['I'] & coverage['F_noI']),
              'retry_union': len(coverage['I'] | coverage['F_noI']), 'copies': copies, 'passes': passes}
    plt.rcParams.update({'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
    fig, axs = plt.subplots(1, 2, figsize=(7.1, 2.7), layout='constrained')
    axs[0].bar(['Correct error', 'Shuffled error'], [v.sum()/248*100 for v in usage.values()], color=['#199779','#80b8a9'])
    axs[0].set_ylabel('Flagged identifier retained (%)')
    axs[0].set_title('Replacement error omits identifier (62 problems)')
    axs[0].set_ylim(0, 40)
    axs[1].bar(range(len(subset)), list(subset.values()), color=['#2776c8']+['#e47a37']*3+['#eebc99']*2)
    axs[1].set_xticks(range(len(subset)), ['I','C + I','CE + I','CEs + I','C, no I','CE, no I'], rotation=30, ha='right')
    axs[1].set_ylabel('Problems with a pass among four')
    axs[1].set_title('Unknown-identifier subgroup (68 problems)')
    save(fig, 'results_subgroup_controls')
    rng = random.Random(0)
    diversity = {}
    for a in ['blind','blind_F','blind_I','assert','noise','f']:
        bytraj = collections.defaultdict(list)
        for r in read(ROOT/'out'/f'depth_{a}.jsonl'):
            if r['attempt'] <= 8:
                bytraj[r['id'],r['traj']].append(body(r.get('code')))
        groups = collections.defaultdict(list)
        for (p,t), codes in bytraj.items():
            codes = [c for c in codes if c]
            if len(codes)<2:
                continue
            pairs = list(itertools.combinations(codes,2))
            pairs = rng.sample(pairs,min(20,len(pairs)))
            groups[p].append(np.mean([difflib.SequenceMatcher(None,x[:1500],y[:1500]).ratio() for x,y in pairs]))
        vals = np.array([[sum(v),len(v)] for p,v in sorted(groups.items())])
        ix = np.random.default_rng(20260906).integers(0,len(vals),(10000,len(vals)))
        draws = vals[ix].sum(axis=1)
        diversity[a] = {'mean':float(vals[:,0].sum()/vals[:,1].sum()),
                        'ci95':np.quantile(draws[:,0]/draws[:,1],[.025,.975]).tolist(),
                        'trajectories':int(vals[:,1].sum())}
    report['first_eight_similarity'] = diversity
    fig, ax = plt.subplots(figsize=(6.2,2.6),layout='constrained')
    means = np.array([v['mean'] for v in diversity.values()])
    ci = np.array([v['ci95'] for v in diversity.values()]).T
    ax.bar(range(6), means, color=['#999999','#86b6ef','#2776c8','#a88bbd','#ccb776','#199779'])
    ax.errorbar(range(6), means, yerr=np.maximum(0,[means-ci[0],ci[1]-means]), fmt='none',ecolor='#555555',capsize=3)
    ax.set_xticks(range(6),['Bare','Failure notice','Instruction','Markers','Noise','Diagnostics'])
    ax.set_ylabel('Within-trajectory proof similarity')
    save(fig,'results_sequential_diversity')
    (PAPER/'analysis'/'supporting_checks.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
