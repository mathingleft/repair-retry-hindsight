"""Recount frozen artifacts and build compact paper figures; no experimental writes.

Run with sdvenv/bin/python analysis/results_figures.py.
Intervals in figures: percentile problem bootstrap, 10000 resamples, seed 20260906.
Sequential bands are pointwise, problem-cluster intervals (not hierarchical).
Memoryless estimates pool all 640 iid draws within each problem. Bands resample
the resulting problem estimates; they are not the old ten-trajectory intervals.
The incomplete C trajectory is a lower-bound outcome, with completion sensitivity.
"""
import collections
import difflib
import json
import re
from math import comb
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1] / 'experiments/decomp_rerun'
DEPTH = ROOT / 'out'
PAPER = Path(__file__).resolve().parents[1] / 'generated'
SEED, B = 20260906, 10000

def read(path):
    return [json.loads(l) for l in path.open() if l.strip()]

def body(code):
    if code is None:
        return None
    b = code.split(':=', 1)[-1]
    b = re.sub(r'--[^\n]*', '', b)
    b = re.sub(r'/-.*?-/', '', b, flags=re.S)
    return re.sub(r'\s+', ' ', b).strip()

def bootstrap(values):
    values = np.asarray(values)
    ix = np.random.default_rng(SEED).integers(0, len(values), (B, len(values)))
    draws = values[ix].mean(axis=1)
    return np.quantile(draws, [.025, .975], axis=0)

def sign(x, y):
    b, c = int(sum(x > y)), int(sum(x < y))
    n = b+c
    return min(1., 2*sum(comb(n, i) for i in range(min(b,c)+1))/2**n)

def model(bankfile, dirs):
    bank = {r['id']: r for r in read(ROOT/bankfile) if not r.get('is_framing_holdout')}
    ids = sorted(bank)
    rows = collections.defaultdict(lambda: collections.defaultdict(list))
    seen = set()
    for d in dirs:
        for r in read(ROOT/d/'verified.jsonl'):
            if r['id'] not in bank:
                continue
            key = r['arm'], r['id'], r['sample_idx']
            assert key not in seen, key
            seen.add(key)
            rows[r['arm']][r['id']].append(r)
    counts, copies, summary = {}, {}, {}
    for a, groups in rows.items():
        assert set(groups) == set(ids) and all(len(v)==4 for v in groups.values()), a
        counts[a] = np.array([sum(r['status']=='PASS' for r in groups[p]) for p in ids])
        copies[a] = np.array([sum(body(r.get('lean_code'))==body(bank[p]['predecessor_code'])
                                 for r in groups[p])/4 for p in ids])
        # Bare predecessor-selection dependence makes DeepSeek bare copying
        # unsuitable for the central copying comparison; it is not plotted.
        summary[a] = {'problems': int(sum(counts[a]>0)), 'passes': int(sum(counts[a])),
                      'copy_percent': float(100*copies[a].mean()),
                      'distinct_mean': float(np.mean([len({body(r.get('lean_code')) for r in groups[p]
                                                        if body(r.get('lean_code')) is not None}) for p in ids]))}
    contrasts = {}
    for a, b in [('I','bare'),('F_C','F'),('F_C','I'),('F_CE','F_C'),
                 ('F_CE','F_CE_shuf'),('F_E','F_E_shuf'),('F','F_noI')]:
        diff = ((counts[a]>0).astype(int)-(counts[b]>0))*100
        contrasts[a+' - '+b] = {'difference_pp': float(diff.mean()),
                                'ci95_pp': bootstrap(diff).tolist(),
                                'pass_count_sign_p': sign(counts[a], counts[b])}
    unknown = {}
    for p in ids:
        m = re.search(r"[Uu]nknown (?:identifier|constant)[ :]*[`'‘]?([^`'’\s]+)", bank[p]['predecessor_error'])
        if m and m[1] not in bank[p]['statement']:
            unknown[p] = m[1]
    subset = {}
    for a, groups in rows.items():
        uses = []
        for p, u in unknown.items():
            pat = re.compile(r"(?<![A-Za-z0-9_.'])"+re.escape(u)+r"(?![A-Za-z0-9_'])")
            uses.extend(bool(pat.search(body(r.get('lean_code')) or '')) for r in groups[p])
        subset[a] = {'problems': sum(any(r['status']=='PASS' for r in groups[p]) for p in unknown),
                     'identifier_use_percent': 100*sum(uses)/len(uses)}
    similarity = {}
    for a in ['I','F_CE']:
        vals = [difflib.SequenceMatcher(None, (body(r.get('lean_code')) or '')[:3000],
                    body(bank[p]['predecessor_code'])[:3000]).ratio()
                for p in unknown for r in rows[a][p] if r['status']=='PASS']
        similarity[a] = {'n_passing_samples':len(vals), 'median':float(np.median(vals)),
                         'fraction_above_point8':float(np.mean(np.array(vals)>.8))}
    interaction = ((counts['F_CE']>0).astype(int)-(counts['F_CE_shuf']>0)
                   -(counts['F_E']>0)+(counts['F_E_shuf']>0))*100
    return ids, counts, copies, {'arms': summary, 'contrasts': contrasts,
        'union': int(sum(np.any(np.stack([c>0 for c in counts.values()]),axis=0))),
        'unknown_n': len(unknown), 'unknown_subset': subset, 'passing_similarity':similarity,
        'interaction_pp': float(interaction.mean()), 'interaction_ci95_pp': bootstrap(interaction).tolist()}

def sequential(path, memoryless=False):
    rows = read(path)
    groups = collections.defaultdict(list)
    seen = set()
    for r in rows:
        key = r['id'],r['traj'],r['attempt']
        assert key not in seen, key
        seen.add(key)
        groups[r['id']].append(r)
    assert len(groups)==11
    out = []
    for p in sorted(groups):
        rs = groups[p]
        assert len({r['traj'] for r in rs})==10
        if memoryless:
            assert len(rs)==640
            c = sum(r['status']=='PASS' for r in rs)
            out.append([1-(comb(640-c,k)/comb(640,k) if 640-c>=k else 0.) for k in range(1,65)])
        else:
            out.append([len({r['traj'] for r in rs if r['status']=='PASS' and r['attempt']<=k})/10
                        for k in range(1,65)])
    return np.array(out)

def save(fig, name):
    for ext in ['pdf','png']:
        fig.savefig(PAPER/'figures'/f'{name}.{ext}', dpi=220, bbox_inches='tight')
    plt.close(fig)

def main():
    plt.rcParams.update({'font.size':8, 'axes.spines.top':False, 'axes.spines.right':False,
                         'pdf.fonttype':42, 'axes.titlesize':9})
    models = [model('bank.jsonl',['out','out_ext','out_ext2','out_noi']),
              model('out_m2/bank2.jsonl',['out_m2','out_m2_ext','out_m2_ext2','out_m2_noi'])]
    titles = ['Qwen3.6-27B','DeepSeek-Prover-V2-7B']
    arms = ['bare','I','F','F_E','F_E_shuf','F_C','F_CE','F_CE_shuf']
    labels = ['bare','I','F',r'$F_E$',r'$F_{E_s}$',r'$F_C$',r'$F_{CE}$',r'$F_{CE_s}$']
    colors = ['#969696','#2776c8','#2776c8','#199779','#199779','#e47a37','#e47a37','#e47a37']
    fig, axs = plt.subplots(2,2,figsize=(7.1,3.65), layout='constrained')
    for j,(ids,counts,copies,summary) in enumerate(models):
        for i in [0,1]:
            ix = list(range(8)) if i==0 else [1,2,5,6,7]
            vals = np.stack([(counts[arms[x]]>0)*100 if i==0 else copies[arms[x]]*100 for x in ix],axis=1)
            means,ci = vals.mean(axis=0),bootstrap(vals)
            bars = axs[i,j].bar(range(len(ix)),means,color=[colors[x] for x in ix],width=.65)
            for bar,x in zip(bars,ix):
                if 'shuf' in arms[x]: bar.set_hatch('///');bar.set_edgecolor('white')
            axs[i,j].errorbar(range(len(ix)),means,yerr=np.maximum(0,np.array([means-ci[0],ci[1]-means])),fmt='none',ecolor='#555555',capsize=2,lw=.7)
            axs[i,j].set_xticks(range(len(ix)),[labels[x] for x in ix])
            axs[i,j].set_ylim(0,42 if i==0 else 75)
            axs[i,j].set_ylabel('pass@4 (%)' if i==0 else 'Copied samples (%)')
            axs[i,j].grid(axis='y',alpha=.18);axs[i,j].set_axisbelow(True)
        axs[0,j].set_title(f'{titles[j]} (n={len(ids)})')
        axs[0,j].axhline(100*np.mean(counts['I']>0),ls='--',c='#555555',lw=.8)
    save(fig,'results_onestep')
    fig,axs=plt.subplots(2,2,figsize=(7.1,4.2),layout='constrained')
    for j,m in enumerate(models):
        s=m[3]
        aa=['I','F','F_E','F_E_shuf','F_C','F_CE','F_CE_shuf']
        if j==0:aa.insert(4,'F_R')
        for i,key in enumerate(['identifier_use_percent','distinct_mean']):
            vals=[s['unknown_subset'][a][key] if i==0 else s['arms'][a][key] for a in aa]
            axs[i,j].bar(range(len(aa)),vals,color=['#e47a37' if 'C' in a else '#199779' for a in aa])
            axs[i,j].set_xticks(range(len(aa)),[a.replace('_shuf','s') for a in aa],rotation=45,ha='right')
            axs[i,j].set_ylabel('Flagged identifier retained (%)' if i==0 else 'Distinct proofs among four')
        axs[0,j].set_title(f'{titles[j]}: {s["unknown_n"]} identifier problems')
    save(fig,'results_behavior')
    fig,axs=plt.subplots(1,2,figsize=(7.1,2.7),layout='constrained')
    for j,m in enumerate(models):
        aa=['F','F_E','F_E_shuf','F_C','F_CE']
        x=np.arange(len(aa));n=len(m[0])
        for shift,suffix,label in [(-.18,'_noI','Without instruction'),(.18,'','With instruction')]:
            vals=[100*m[3]['arms'][a+suffix]['problems']/n for a in aa]
            axs[j].bar(x+shift,vals,.35,label=label,color='#86b6ef' if suffix else '#2776c8')
        axs[j].set_xticks(x,[a.replace('_shuf','s') for a in aa]);axs[j].set_ylim(0,35)
        axs[j].set_ylabel('pass@4 (%)');axs[j].set_title(titles[j]);axs[j].legend(fontsize=7)
    save(fig,'results_framing')
    fig,axs = plt.subplots(1,2,figsize=(7.1,2.6),layout='constrained')
    protocols = [[('Instruction retry','blind_I','#333333',True),('Bare retry','blind','#aaaaaa',True),
                  ('Failure-notice retry','blind_F','#8866aa',True),('Diagnostic history','f','#199779',False),
                  ('Marker history','assert','#2776c8',False),('Noise history','noise','#86b6ef',False)],
                 [('Proofs only','C','#e47a37',False),('Proofs + correct error','CE','#199779',False),
                  ('Proofs + shuffled error','CE_shuf','#2776c8',False)]]
    depth_summary = {}
    for j,protocol in enumerate(protocols):
        for label,arm,color,iid in protocol:
            path = (ROOT/'out' if j==0 else DEPTH)/f'depth_{arm}.jsonl'
            values = sequential(path,iid);mean=values.mean(axis=0)*100;ci=bootstrap(values)*100
            axs[j].plot(range(1,65),mean,label=label,color=color,lw=1.3,ls='--' if iid else '-')
            # Shade all right-panel curves; on the crowded left show endpoint
            # intervals separately in a supplementary figure below.
            if j==1: axs[j].fill_between(range(1,65),*ci,color=color,alpha=.10)
            depth_summary[arm]={'pass64_percent':float(mean[-1]),'problem_ci95_percent':ci[:,-1].tolist()}
        axs[j].set_xscale('log',base=2);axs[j].set_xticks([1,4,16,64],['1','4','16','64'])
        axs[j].set_ylim(0,100);axs[j].set_ylabel('pass@k (%)');axs[j].set_xlabel('Proof attempts k')
        axs[j].legend(fontsize=6.3,loc='upper left');axs[j].grid(alpha=.15)
    axs[0].set_title('Diagnostic-only protocol (~7k history)')
    axs[1].set_title('Code-bearing protocol (exact 7k history)')
    save(fig,'results_sequential')
    fig,ax=plt.subplots(figsize=(5.3,2.5),layout='constrained')
    keys=list(depth_summary)[:6]
    for i,a in enumerate(keys):
        s=depth_summary[a];m=s['pass64_percent'];lo,hi=s['problem_ci95_percent']
        ax.errorbar(m,i,xerr=[[m-lo],[hi-m]],fmt='o',capsize=3,color=protocols[0][i][2])
    ax.set_yticks(range(6),[x[0] for x in protocols[0]]);ax.invert_yaxis()
    ax.set_xlabel('pass@64 (%) with 95% problem-bootstrap intervals');ax.set_xlim(0,100)
    save(fig,'results_sequential_intervals')
    # Budget allocation: unbiased subset estimator for both independent draws
    # and independent whole trajectories. Do not extrapolate beyond 10 trajectories.
    iid=read(ROOT/'out/depth_blind_I.jsonl');pids=sorted({r['id'] for r in iid})
    def u(n,c,k): return 1-(comb(n-c,k)/comb(n,k) if n-c>=k else 0.)
    counts={p:sum(r['status']=='PASS' for r in iid if r['id']==p) for p in pids}
    fig,axs=plt.subplots(1,2,figsize=(7.1,2.6),layout='constrained');allocation={}
    for arm,length,color in [('f',32,'#199779'),('f',64,'#14664f'),('assert',64,'#2776c8'),('noise',64,'#86b6ef')]:
        v=sequential(ROOT/'out'/f'depth_{arm}.jsonl')[:,length-1]
        ms=[m for m in [32,64,128,256,640] if m>=length and m//length<=10]
        diffs=[];means=[];cis=[]
        for budget in ms:
            depth=np.array([u(10,int(round(c*10)),budget//length) for c in v])
            breadth=np.array([u(640,counts[p],budget) for p in pids])
            d=(depth-breadth)*100;diffs.append(float(d.mean()));means.append(float(depth.mean()*100));cis.append(bootstrap(d))
        label=f'{arm}, L={length}';allocation[label]={'budgets':ms,'difference_pp':diffs,'ci95_pp':[c.tolist() for c in cis]}
        axs[0].plot(ms,means,'o-',color=color,label=label);axs[1].plot(ms,diffs,'o-',color=color)
        cc=np.array(cis);axs[1].fill_between(ms,cc[:,0],cc[:,1],color=color,alpha=.10)
    ms=[32,64,128,256,640]
    axs[0].plot(ms,[100*np.mean([u(640,counts[p],m) for p in pids]) for m in ms],'--',color='#333333',label='Instruction retry')
    for ax in axs:ax.set_xscale('log',base=2);ax.set_xlabel('Total proof-attempt budget');ax.grid(alpha=.15)
    axs[0].set_ylabel('Estimated success (%)');axs[0].legend(fontsize=6.5)
    axs[1].set_ylabel('Depth minus instruction retry (pp)');axs[1].axhline(0,color='#555555',lw=.7)
    save(fig,'results_allocation')
    summary={'seed':SEED,'resamples':B,'models':{t:m[3] for t,m in zip(titles,models)},'sequential':depth_summary,'allocation':allocation}
    (PAPER/'analysis/results_audit.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({'models':{t:{'union':m[3]['union'],'unknown_n':m[3]['unknown_n']} for t,m in zip(titles,models)},'sequential':depth_summary},indent=2))

if __name__=='__main__':
    main()
