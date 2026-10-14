"""Plot saved problem-level curves; export every pointwise confidence bound."""
import csv
import json
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import OUTPUT, csvout
from analyze_sequential import ARMS,effects,python_indices

BLUE,ORANGE,AQUA='#2a78d6','#eb6834','#1baf7a'


def main():
    with (OUTPUT/'per_problem_curves.csv').open() as f:rows=list(csv.DictReader(f))
    summaries=json.loads((OUTPUT/'sequential.json').read_text());bounds=[]
    fig,axs=plt.subplots(2,2,figsize=(7.2,5.5),layout='constrained')
    for row,model in enumerate(['qwen','deepseek']):
        ids=sorted({r['problem_id'] for r in rows if r['model']==model});pi={p:i for i,p in enumerate(ids)}
        v={a:np.zeros((11,64)) for a in ARMS}
        for r in rows:
            if r['model']==model and r['arm'] in v:v[r['arm']][pi[r['problem_id']],int(r['k'])-1]=float(r['success_probability'])
        ix=python_indices(random.Random(0),2000,11,choices=True) if model=='qwen' else np.random.default_rng(20260913).integers(0,11,(200000,11))
        weights=np.stack([(ix==j).sum(axis=1) for j in range(11)],axis=1)/11
        def band(name,values):
            # Weighted problem counts avoid allocating a B x 11 x 64 tensor.
            draws=weights@values
            if model=='qwen':
                draws.sort(axis=0);ci=draws[[50,1950]]*100
            else:ci=np.quantile(draws,[.025,.975],axis=0)*100
            mean=values.mean(axis=0)*100
            for k in range(64):bounds.append(dict(model=model,curve=name,k=k+1,mean=float(mean[k]),lo=float(ci[0,k]),hi=float(ci[1,k])))
            return mean,ci
        display={'f':'E','E_s':'E_m'}
        # Match the manuscript palette: blue is retry/failure framing, orange is
        # code-bearing context, and aqua is diagnostic context without code.
        for arm,color,ls in [('I',BLUE,'--'),('C',ORANGE,':'),('CE',ORANGE,'-'),('CE_m',ORANGE,'-.'),('CE_s',ORANGE,(0,(3,1,1,1))),('f',AQUA,'-'),('E_s',AQUA,'--')]:
            mean,ci=band(arm,v[arm]);axs[row,0].plot(range(1,65),mean,color=color,ls=ls,label='$'+display.get(arm,arm)+'$')
            axs[row,0].fill_between(range(1,65),*ci,color=color,alpha=.12)
        for name,color in [('CE-CE_m',ORANGE),('f-E_s',AQUA)]:
            label={'CE-CE_m':r'$CE-CE_m$','f-E_s':r'$E-E_m$'}[name]
            mean,ci=band(name,effects(v)[name]);axs[row,1].plot(range(1,65),mean,color=color,label=label)
            axs[row,1].fill_between(range(1,65),*ci,color=color,alpha=.12)
        axs[row,1].axhline(0,c='#888',lw=.7)
        for col in range(2):
            ax=axs[row,col];ax.set_xscale('log',base=2);ax.set_xticks([1,2,4,8,16,32,64],[1,2,4,8,16,32,64]);ax.set_xlabel('Proof attempts k');ax.grid(alpha=.15);ax.legend(fontsize=9)
        axs[row,0].set_ylim(0,100);axs[row,0].set_ylabel('pass@k (%)');axs[row,1].set_ylabel('Paired difference (pp)')
        model_name='Qwen3.6-27B' if model=='qwen' else 'DeepSeek-Prover-V2-7B'
        e=summaries[model]['endpoint_contrasts']['CE-I']
        axs[row,0].set_title(model_name+f"\nCE−I (64): {e['difference_pp']:+.1f} pp [{e['ci95_pp'][0]:+.1f}, {e['ci95_pp'][1]:+.1f}]",fontsize=9.5)
    out=OUTPUT/'figures';out.mkdir(exist_ok=True)
    for ext in ['png','pdf']:fig.savefig(out/f'fig_sequential_final.{ext}',dpi=220,bbox_inches='tight')
    plt.close(fig);csvout('sequential_figure_bounds.csv',bounds)
    print('Regenerated final sequential figure and all plotted confidence bounds.')


if __name__=='__main__':main()
