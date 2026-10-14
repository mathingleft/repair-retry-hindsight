"""Rebuild seven included one-step figures from frozen proof records.

Recovered original plotting functions retain their published estimators. The main
EXT3 plot and shallow subgroup panel are rebuilt from their stated definitions.
"""
import collections
import functools
import json
import random
import statistics
import sys
import numpy as np
from common import DATA,OUTPUT,read,csvout
sys.path.insert(0,str(DATA))
import figures as old
import original_style_figures as updated
from matplotlib.patches import Patch

OUT=OUTPUT/'figures'
PLOT_ROWS=[]


def save(fig,out,name):
    OUT.mkdir(parents=True,exist_ok=True)
    for ext in ['png','pdf']:fig.savefig(OUT/f'{name}.{ext}',dpi=220,bbox_inches='tight')
    old.plt.close(fig);print('Figure:',name,flush=True)


def cluster_ci(groups,stat,B=2000,seed=0):
    """Vectorized equivalent of the original flatten-after-problem-resampling CI."""
    groups=[g for g in groups if g];rng=random.Random(seed)
    ix=np.array([rng.choices(range(len(groups)),k=len(groups)) for _ in range(B)])
    observed=float(stat([v for g in groups for v in g]))
    if stat([1,1,7])==1: # original output-length median
        assert len({len(g) for g in groups})==1
        draws=np.median(np.asarray(groups)[ix].reshape(B,-1),axis=1)
    else:
        scale=float(stat([1,1,7]))/3
        sums=np.array([sum(g) for g in groups]);sizes=np.array([len(g) for g in groups])
        draws=scale*sums[ix].sum(axis=1)/sizes[ix].sum(axis=1)
    draws.sort();return observed,float(draws[int(.025*B)]),float(draws[int(.975*B)])


def models():
    q=old.qwen(str(DATA));p=old.prover(str(DATA))
    for m,ext in [(q,'out_ext2'),(p,'out_m2_ext2')]:
        m.rows+=read(DATA/ext/'verified.jsonl')
        for r in m.rows:
            if r['status']=='PASS' and r['id'] in m.pids:m.solved[r['arm']].add(r['id'])
    return q,p


def main_ext3(q,p):
    aa=['bare','I','F','F_E','F_E_shuf','F_E_match','F_C','F_C_other','F_CE','F_CE_shuf','F_CE_match']
    labels=['bare','I','F',r'$F_E$',r'$F_{E_s}$',r'$F_{E_m}$',r'$F_C$',r'$F_{C_o}$',r'$F_{CE}$',r'$F_{CE_s}$',r'$F_{CE_m}$']
    fig,axs=old.plt.subplots(1,2,figsize=(9,3.6),sharey=True)
    for ax,m,ext in zip(axs,[q,p],['out_ext3','out_m2_ext3']):
        allrows=m.rows+read(DATA/ext/'verified.jsonl');solved=collections.defaultdict(set)
        for r in allrows:
            if r['status']=='PASS' and r['id'] in m.pids:solved[r['arm']].add(r['id'])
        vals=np.array([[100*int(pid in solved[a]) for a in aa] for pid in m.pids])
        ix=np.random.default_rng(20260906).integers(0,len(m.pids),(10000,len(m.pids)))
        ci=np.quantile(vals[ix].mean(axis=1),[.025,.975],axis=0);means=vals.mean(axis=0)
        colors=[old.GRAY if a=='bare' else old.BLUE if a in ['I','F'] else old.ORANGE if 'C' in a else old.AQUA for a in aa]
        bars=ax.bar(range(len(aa)),means,color=colors,width=.72)
        for a,b in zip(aa,bars):
            hatch='///' if 'shuf' in a else '...' if 'match' in a else 'xx' if 'other' in a else ''
            if hatch:b.set_hatch(hatch);b.set_edgecolor('white')
        ax.errorbar(range(len(aa)),means,yerr=[means-ci[0],ci[1]-means],fmt='none',color='#666',capsize=2,lw=.8)
        ax.axhline(means[1],color='#666',ls='--',lw=1);ax.set_xticks(range(len(aa)),labels);ax.set_title(f'{m.name} (n={len(m.pids)})');ax.set_ylim(0,42)
        for i,a in enumerate(aa):PLOT_ROWS.append(dict(figure='results_onestep',model=m.name,arm=a,mean=float(means[i]),lo=float(ci[0,i]),hi=float(ci[1,i])))
    axs[0].set_ylabel('pass@4 (%)')
    fig.legend(handles=[Patch(color=old.GRAY,label='bare'),Patch(color=old.BLUE,label='retry / failure'),Patch(color=old.AQUA,label='diagnostic, no code'),Patch(color=old.ORANGE,label='code-bearing')],loc='lower center',ncol=4,bbox_to_anchor=(.5,-.08))
    fig.legend(handles=[Patch(facecolor='white',edgecolor='#666',hatch='///',label=r'$s$: shuffled wrong diagnostic'),Patch(facecolor='white',edgecolor='#666',hatch='...',label=r'$m$: matched wrong diagnostic'),Patch(facecolor='white',edgecolor='#666',hatch='xx',label=r'$o$: other failed proof')],loc='lower center',ncol=3,bbox_to_anchor=(.5,-.16))
    fig.tight_layout();save(fig,None,'results_onestep')


# Manuscript notation for the appendix one-step figures (artifact labels: Table tab:aliases).
MS_TICK={'bare':'bare','I':'$I$','N':'random\nwords','F':'$F$','F_E':'$F_E$','F_E_shuf':'$F_{E_s}$',
         'F_R':'$F_R$','F_C':'$F_C$','F_CE_shuf':'$F_{CE_s}$','F_CE':'$F_{CE}$'}
MS_ANATOMY={'failure sentence':'failure notice','compiler error':'own diagnostic',
            "another problem's error":'shuffled diagnostic','failed code':'failed proof',
            'instruction line':'retry instruction'}


def manuscript_labels():
    """Relabel the recovered figure helpers; estimators and bar values are unchanged."""
    old.tick=lambda a:MS_TICK.get(a,a)
    old.ANATOMY[:]=[(MS_ANATOMY.get(name,name),members) for name,members in old.ANATOMY]
    draw=old.bracket
    def neutral(ax,x0,x1,y,text,color=old.INK):  # report the contrast, not an interpretation of it
        draw(ax,x0,x1,y,'$F_{CE}-F_C$ = '+text.split(':')[0].replace('-','−'),color)
    old.bracket=neutral
    old.LEGEND[:]=[Patch(color=old.GRAY,label='bare'),Patch(color=old.BLUE,label='retry / failure notice'),
                   Patch(color=old.BLUE_LIGHT,label='random words'),Patch(color=old.AQUA,label='diagnostic or reflection, no code'),
                   Patch(color=old.ORANGE,label='code-bearing')]


def appendix(q,p):
    old.FAMILY['F_CE_shuf']=old.ORANGE
    for name,members in old.ANATOMY:
        if name in ['failure sentence',"another problem's error",'failed code','instruction line','task (statement)']:
            members.add('F_CE_shuf')
    manuscript_labels()
    shuffled=Patch(facecolor=old.GRAY_LIGHT,edgecolor=old.SURF,hatch='///',label='hatched: shuffled wrong diagnostic')
    qa=['bare','I','N','F','F_E','F_E_shuf','F_R','F_C','F_CE_shuf','F_CE'];pa=[a for a in qa if a!='F_R']
    fig=old.plt.figure(figsize=(10,5.1));gs=fig.add_gridspec(2,2,height_ratios=[1.25,3],hspace=.4,wspace=.25)
    ax=fig.add_subplot(gs[0,0]);old.anatomy(ax,qa)
    ax.set_title('prompt anatomy of each arm: [failure notice] [blocks] [retry instruction] [task]',loc='left',color=old.INK2,fontsize=7.5,pad=20)
    old.panel1(fig.add_subplot(gs[1,0]),q,qa,6,f'a  Qwen3.6-27B, {len(q.pids)} problems, k = 4',84)
    old.panel1(fig.add_subplot(gs[1,1]),p,pa,5,f'b  DeepSeek-Prover-V2-7B, {len(p.pids)} problems, k = 4',58)
    for ax,aa in zip(fig.axes[1:],[qa,pa]):
        for a,bar in zip(aa,ax.patches):
            if 'shuf' in a:bar.set_hatch('///');bar.set_edgecolor(old.SURF)
    fig.legend(handles=old.LEGEND+[shuffled],loc='upper left',ncol=1,handlelength=1.2,bbox_to_anchor=(.6,.97),frameon=False)
    save(fig,None,'fig1_decomposition')
    old.fig5(str(OUT),q,p)
    # Bar sets as published: no F_E_s on either model, no random words on DeepSeek-Prover.
    sets=[(q,[a for a in qa if a!='F_E_shuf']),(p,[a for a in pa if a not in ('N','F_E_shuf')])]
    fig,axs=old.plt.subplots(2,2,figsize=(10,6.2),gridspec_kw={'hspace':.6,'wspace':.2,'width_ratios':[9,7]})
    simq=None
    for j,(m,aa) in enumerate(sets):
        cp,di,sim=old.copy_stats(m,aa)
        if j==0:simq=old.copy_stats(m,qa)[2]
        for i,stat in enumerate([cp,di]):
            old.panel2(axs[i,j],m,aa,stat,'%' if i==0 else '',f"{'ab'[i]}  {m.name}" if j==0 else m.name,
                       80 if i==0 else 4.8,{'bare','I','F_C','F_CE_shuf','F_CE'},(lambda x:f'{x:.1f}') if i==0 else (lambda x:f'{x:.2f}'))
            for a,b in zip(aa,axs[i,j].patches):
                if 'shuf' in a:b.set_hatch('///');b.set_edgecolor(old.SURF)
            if j==0:axs[i,j].set_ylabel('Copied samples (%)' if i==0 else 'Distinct proofs among four')
            for a in aa:PLOT_ROWS.append(dict(figure='fig2_copy_diversity' if i==0 else 'fig2_diversity',model=m.name,arm=a,mean=stat[a][0],lo=stat[a][1],hi=stat[a][2]))
    fig.legend(handles=old.LEGEND+[shuffled,Patch(facecolor=old.GRAY,edgecolor=old.SURF,hatch='///',label='hatched bare: no predecessor in the prompt')],
               loc='upper center',ncol=3,handlelength=1.2,bbox_to_anchor=(.5,1.04),frameon=False)
    save(fig,None,'fig2_copy_diversity');old.figA3(str(DATA),str(OUT),q,simq)
    updated.save=save;updated.old.cluster_ci=cluster_ci;updated.behavior()
    # Rebuild the original 68-problem shallow-error subgroup figure.
    bank={r['id']:r for r in read(DATA/'bank.jsonl')}
    ids=[pid for pid in q.pids if (u:=old.unknown_ident(bank[pid]['predecessor_error'])) and u not in bank[pid]['statement']]
    assert len(ids)==68
    sub=old.Model(q.name,q.rows,set(ids),q.pred)
    aa=['I','F','F_E','F_E_shuf','F_C','F_CE_shuf','F_CE','F_C_noI','F_CE_noI']
    ci=old.solved_ci(sub,aa);fig,ax=old.plt.subplots(figsize=(8,3))
    for i,a in enumerate(aa):
        ax.bar(i,ci[a][0],color='white' if a.endswith('_noI') else old.FAMILY[a],edgecolor=old.FAMILY[a])
        PLOT_ROWS.append(dict(figure='fig7_shallow_bucket',model=q.name,arm=a,mean=ci[a][0],lo=ci[a][1],hi=ci[a][2]))
    labels=[r'$I$',r'$F$',r'$F_E$',r'$F_{E_s}$',r'$F_C$',r'$F_{CE_s}$',r'$F_{CE}$',r'$F_C$ (no $I$)',r'$F_{CE}$ (no $I$)']
    old.whiskers(ax,range(len(aa)),[ci[a] for a in aa]);ax.set_xticks(range(len(aa)),labels,rotation=25,ha='right');ax.set_ylabel('Problems solved among 68');ax.axhline(ci['I'][0],color=old.INK2,ls='--');save(fig,None,'fig7_shallow_bucket')


def main():
    old.save=save;old.cluster_ci=cluster_ci
    # Memoize normalized repeated predecessors; no estimator changes.
    old.body=functools.lru_cache(maxsize=50000)(old.body)
    q,p=models()
    with old.plt.rc_context({'font.size':11,'axes.titlesize':13,'axes.labelsize':11,'xtick.labelsize':11,'ytick.labelsize':11,'legend.fontsize':10}):
        main_ext3(q,p)
    appendix(q,p);csvout('onestep_figure_data.csv',PLOT_ROWS)


if __name__=='__main__':main()
