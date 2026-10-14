"""Adapt the original plotting functions, preserving their visual language.

Frozen source: baseline_snapshot, figures.py at baseline_snapshot.
Original image assets are never overwritten. No model or verifier calls.
"""
import collections
import importlib.util
import inspect
import json
import re
import sys
from math import comb
from pathlib import Path

import numpy as np
from results_figures import ROOT, DEPTH, PAPER, read, body, sequential, bootstrap

sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('original_figures', ROOT/'figures.py')
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
from arms import shuffled_error


def save(fig, out, name):
    audit = json.loads((PAPER/'analysis/results_audit.json').read_text())
    if name == 'fig4_discovery':
        # Every original-style discovery curve must match the frozen recount.
        lines = [line for line in fig.axes[0].lines if len(line.get_xdata()) == 64]
        arms = ['assert','noise','f','blind','blind_I','blind_F']
        assert len(lines) == len(arms)
        for line,arm in zip(lines,arms):
            assert np.isclose(line.get_ydata()[-1],audit['sequential'][arm]['pass64_percent'])
    if name == 'fig6_breadth_vs_depth':
        labels = {'compiler-error, L = 64':'f, L=64',
                  'compiler-error, L = 32':'f, L=32',
                  '“failed.”, L = 64':'assert, L=64'}
        for line in fig.axes[1].lines:
            key = labels.get(line.get_label())
            if key:
                expected = audit['allocation'][key]
                assert list(line.get_xdata()) == expected['budgets']
                assert np.allclose(np.asarray(line.get_ydata())*100, expected['difference_pp'])
    names = {'fig4_discovery': 'fig4_discovery_updated',
             'fig6_breadth_vs_depth': 'fig6_breadth_vs_depth_updated'}
    stem = names.get(name, name)
    for ext in ['pdf', 'png']:
        fig.savefig(PAPER/'figures'/f'{stem}.{ext}', dpi=220, bbox_inches='tight')
    old.plt.close(fig)
    print('wrote', stem, flush=True)


def subset(n, c, k):
    assert 0 <= k <= n
    return 1 - (comb(n-c, k)/comb(n, k) if n-c >= k else 0.)


def replace_once(text, before, after):
    assert text.count(before) == 1, before
    return text.replace(before, after)


def discovery():
    # Keep fig4's layout, palette, curve markers and endpoint whiskers.
    src = inspect.getsource(old.fig4)
    src = replace_once(src,
        'y = [sum(1 for v in first.values() if v <= k) for k in ks]; at8[arm] = y[7]; firsts[arm] = (first, keys)',
        'values = sequential(os.path.join(outdir, "out", f"depth_{arm}.jsonl"), arm.startswith("blind"))\n'
        '        y = values.mean(axis=0)*100; at8[arm] = sum(v <= 8 for v in first.values()); firsts[arm] = (first, keys)')
    src = replace_once(src,
        'ci = cluster_ci(list(byp.values()), lambda v: sum(1 for a in v if a <= 64), B=2000)',
        'bounds = bootstrap(values[:, -1])*100\n        ci = (y[-1], bounds[0], bounds[1])')
    src = src.replace('"   by k = 8: "', '"   observed counts by k = 8: "')
    src = src.replace('for i, ln in enumerate(lines): ax.text', 'for i, ln in enumerate([]): ax.text')
    src = src.replace('trajectories with a first pass by k (of 110)', 'pass@k (%)')
    src = src.replace('f"{name} {v}"', 'f"{name} {v:.1f}%"')
    ns = dict(vars(old), sequential=lambda path,iid: sequential(Path(path),iid), bootstrap=bootstrap)
    exec(compile(src, 'adapted_original_fig4', 'exec'), ns)
    ns['fig4'](str(ROOT), str(PAPER/'figures'))


def allocation():
    # Keep all three original panels. Only change estimators, their bootstrap,
    # admissible budgets and labels describing extrapolation.
    src = inspect.getsource(old.fig6)
    src = replace_once(src,
        'bre = lambda a, p, m: 1 - (1 - rate[a][p]) ** m',
        'bre = lambda a, p, m: subset(640, round(rate[a][p]*640), m)')
    src = replace_once(src,
        'dep = lambda a, p, m, L: 1 - (1 - pL(a, p, L)) ** (m / L)',
        'dep = lambda a, p, m, L: subset(10, round(pL(a,p,L)*10), m//L)')
    src = src.replace('xs = [m for m in ms if m >= L]', 'xs = [m for m in ms if m >= L and m//L <= 10]')
    src = src.replace('xx = [m for m in xs if m >= L]', 'xx = [m for m in xs if m >= L and m//L <= 10]')
    src = replace_once(src,
        'bs = sorted(statistics.mean(d[p] for p in rng.choices(pids, k=len(pids))) for _ in range(3000))\n'
        '            mid.append(statistics.mean(d.values())); lo.append(bs[75]); hi.append(bs[2924])',
        'ci = bootstrap(np.array([d[p] for p in pids]))\n'
        '            mid.append(statistics.mean(d.values())); lo.append(ci[0]); hi.append(ci[1])')
    src = src.replace('a  breadth vs depth at a fixed budget (hollow: extrapolated, T > 10)',
                      'a  breadth vs depth (subset estimates; T ≤ 10)')
    src = src.replace('expected fraction of the 11 problems solved', 'estimated success probability')
    ns = dict(vars(old), subset=subset, bootstrap=bootstrap)
    exec(compile(src, 'adapted_original_fig6', 'exec'), ns)
    ns['fig6'](str(ROOT), str(PAPER/'figures'))


def behavior():
    q, p = old.qwen(str(ROOT)), old.prover(str(ROOT))
    q.rows += read(ROOT/'out_ext2/verified.jsonl')
    p.rows += read(ROOT/'out_m2_ext2/verified.jsonl')
    old.FAMILY['F_CE_shuf'] = old.ORANGE
    old.TICK.update({'F_E_shuf':'shuf†', 'F_CE_shuf':'shuf†'})
    qa = ['bare','I','N','F','F_E','F_E_shuf','F_R','F_C','F_CE_shuf','F_CE']
    pa = [a for a in qa if a != 'F_R']
    fig, ax = old.plt.subplots(2,2,figsize=(10,6.6),
                              gridspec_kw={'hspace':.65,'wspace':.36})
    told = {'F_E':'own','F_E_shuf':"other's",'F_CE':'own','F_CE_shuf':"other's"}
    for a, model, bank, arms, title in [
        (ax[0,0],q,ROOT/'bank.jsonl',qa,'a  Qwen, instruction regime'),
        (ax[0,1],p,ROOT/'out_m2/bank2.jsonl',pa,'b  DeepSeek-Prover')]:
        stats,n = old.still_uses(model,str(bank),arms)
        old.panel3(a,arms,stats,n,title,arms.index('F_C')-1,told)
        for bar,arm in zip(a.patches,arms):
            if 'shuf' in arm:
                bar.set_hatch('///'); bar.set_edgecolor(old.SURF)
        a.set_ylabel('Flagged identifier retained (%)')
    bank = {r['id']:r for r in read(ROOT/'bank.jsonl')}
    patterns = {}
    for pid in q.pids:
        u = old.unknown_ident(bank[pid]['predecessor_error'])
        if u and u not in bank[pid]['statement']:
            patterns[pid] = re.compile(r"(?<![A-Za-z0-9_.'])"+re.escape(u)+r"(?![A-Za-z0-9_'])")
    nonmention = [pid for pid,pat in patterns.items() if not pat.search(shuffled_error(pid))]
    mention = [pid for pid in patterns if pid not in nonmention]
    assert (len(nonmention),len(mention)) == (62,6)
    groups = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in q.rows: groups[r['arm']][r['id']].append(r)
    cells = [('F_noI',nonmention),('F_E_shuf_noI',nonmention),
             ('F_E_noI',nonmention),('F_E_shuf_noI',mention)]
    cis = [old.cluster_ci([[100*bool(patterns[pid].search(body(r.get('lean_code')) or ''))
                            for r in groups[arm][pid]] for pid in ids],np.mean)
           for arm,ids in cells]
    bars = ax[1,0].bar(range(4),[c[0] for c in cis],
                      color=[old.BLUE,old.AQUA,old.AQUA,old.SURF],
                      edgecolor=[old.BLUE,old.SURF,old.AQUA,old.AQUA])
    bars[1].set_hatch('///')
    old.whiskers(ax[1,0],range(4),cis)
    for x,ci in enumerate(cis):
        ax[1,0].text(x,ci[2]+1.2,f'{ci[0]:.0f}%',ha='center',fontsize=9)
    ax[1,0].set_xticks(range(4),['Failure\nnotice','Shuffled error\n(no identifier)','Correct\nerror','Shuffled error\n(with identifier)'])
    ax[1,0].set_ylim(0,108)
    ax[1,0].set_ylabel('Flagged identifier retained (%)')
    ax[1,0].set_title('c  No instruction: identifier absent/present in replacement\nFirst three bars: 62 problems; last bar: 6',loc='left')
    ax[1,1].axis('off')
    ax[1,1].legend(handles=old.LEGEND,loc='center',fontsize=9)
    save(fig,None,'fig3_error_use_updated')


def code_depth():
    fig,ax = old.plt.subplots(figsize=(6.4,4.1))
    ref = sequential(ROOT/'out/depth_blind_I.jsonl',True)
    ax.plot(range(1,65),ref.mean(axis=0)*100,color=old.INK2,ls='--',
            marker='D',markevery=[0,1,3,7,15,31,63],ms=4,label='Instruction retry (other protocol)')
    ax.fill_between(range(1,65),*bootstrap(ref)*100,color=old.INK2,alpha=.07)
    for arm,ls,marker,label in [('C',':','s','Failed proofs only'),
        ('CE','-','o','Failed proofs + correct diagnostics'),
        ('CE_shuf','--','^','Failed proofs + shuffled diagnostics')]:
        v = sequential(DEPTH/f'depth_{arm}.jsonl')
        ax.plot(range(1,65),v.mean(axis=0)*100,color=old.ORANGE,ls=ls,
                marker=marker,ms=4,markevery=[0,1,3,7,15,31,63],label=label)
        ax.fill_between(range(1,65),*bootstrap(v)*100,color=old.ORANGE,alpha=.06)
    ax.set_xscale('log',base=2);ax.set_xticks([1,2,4,8,16,32,64],['1','2','4','8','16','32','64'])
    ax.set_ylim(0,100);ax.set_xlabel('Proof attempts k');ax.set_ylabel('pass@k (%)')
    ax.set_title('Code-bearing histories (exact 7k history budget)',loc='left')
    ax.legend(loc='upper left',fontsize=8)
    fig.text(.5,-.04,'Dashed gray retry curve: cross-protocol reference, not a matched control.',
             ha='center',fontsize=8,color=old.INK2)
    save(fig,None,'fig_code_depth')


if __name__ == '__main__':
    old.save = save
    behavior()
    discovery()
    allocation()
    code_depth()
