"""Rebuild both seven-arm sequential analyses from frozen retained outcomes."""
from collections import defaultdict
from math import comb
import random
import numpy as np
from common import DATA, read, normalize, dump, csvout

ARMS=['I','C','CE','CE_m','CE_s','f','E_s']
KS=[1,2,4,8,16,32,64]


def pooled(n,c,k):return 1-(comb(n-c,k)/comb(n,k) if n-c>=k else 0.)


def load_model(sub,arms=ARMS):
    curves={}; details={}; conditional={}
    for arm in arms:
        rows=[normalize(r) for r in read(DATA/sub/f'depth_{"memI" if arm=="I" else arm}.jsonl')]
        groups=defaultdict(list)
        for r in rows:groups[r['id']].append(r)
        ids=sorted(groups);values=[];cond=[];passed=0
        for pid in ids:
            rs=groups[pid]
            if arm=='I':
                c=sum(r['status']=='PASS' for r in rs);passed+=c
                values.append([pooled(len(rs),c,k) for k in range(1,65)])
                cond.append([pooled(len(rs),c,k-1) for k in range(1,65)])
            else:
                first={r['traj']:r['attempt'] for r in rs if r['status']=='PASS'}
                eligible={r['traj'] for r in rs if r['attempt']==1 and r['status']!='PASS'}
                passed+=len(first)
                values.append([sum(k0<=k for k0 in first.values())/10 for k in range(1,65)])
                cond.append([sum(t in eligible and k0<=k for t,k0 in first.items())/len(eligible) if eligible else float('nan') for k in range(1,65)])
        curves[arm]=np.array(values);conditional[arm]=np.array(cond)
        details[arm]=dict(rows=len(rows),passes_or_solved_trajectories=passed,pass64_percent=float(100*curves[arm][:,-1].mean()))
    return ids,curves,conditional,details


def effects(v):
    return {'CE-I':v['CE']-v['I'],'C-I':v['C']-v['I'],'CE-C':v['CE']-v['C'],
            'CE-CE_m':v['CE']-v['CE_m'],'CE_m-C':v['CE_m']-v['C'],
            'CE_m-CE_s':v['CE_m']-v['CE_s'],'CE_m-E_s':v['CE_m']-v['E_s'],
            'CE-CE_s':v['CE']-v['CE_s'],'f-E_s':v['f']-v['E_s'],
            'interaction':v['CE']-v['CE_s']-v['f']+v['E_s'],'CE_s-E_s':v['CE_s']-v['E_s'],
            'matched_interaction':v['CE']-v['CE_m']-v['f']+v['E_s'],
            'CE_s-I':v['CE_s']-v['I'],'C-CE_s':v['C']-v['CE_s'],'CE-f':v['CE']-v['f'],
            'CE_s-f':v['CE_s']-v['f'],'E_s-I':v['E_s']-v['I'],'f-I':v['f']-v['I'],'CE_m-I':v['CE_m']-v['I'],'C-f':v['C']-v['f']}


def python_indices(rng,b,n,choices=False):
    return np.array([rng.choices(range(n),k=n) if choices else [rng.randrange(n) for _ in range(n)] for _ in range(b)])


def interval(d,ix,original=False):
    bs=np.sort(d[ix].mean(axis=1),axis=0)
    return (bs[[int(.025*len(ix)),int(.975*len(ix))]] if original else np.quantile(bs,[.025,.975],axis=0))*100


def main():
    report={};csv=[];horizons=[];contrastrows=[]
    for model,sub in [('qwen','out'),('deepseek','out_dsp')]:
        arms=ARMS+['noise'] if model=='qwen' else ARMS
        ids,v,cond,details=load_model(sub,arms);eff=effects(v)
        if model=='qwen':eff['E_s-noise']=v['E_s']-v['noise']
        if model=='deepseek':ix=np.random.default_rng(20260913).integers(0,11,(200000,11))
        else:ix=python_indices(random.Random(0),10000,11)
        ep={name:dict(difference_pp=float(d[:,-1].mean()*100),ci95_pp=interval(d[:,-1],ix,model=='qwen').tolist()) for name,d in eff.items()}
        # Registered CE_m analysis: one joint paired-problem index matrix in the
        # helper's stable problem order, Random(0), 10,000 resamples, and sorted
        # percentile endpoints. Preserve the established randrange draw stream.
        review_ix=python_indices(random.Random(0),10000,11)
        for name in ['CE-CE_m','f-E_s','matched_interaction','CE_m-C','CE_m-CE_s','CE_m-E_s','CE_m-I']:
            d=eff[name][:,-1]
            ep[name]=dict(difference_pp=float(d.mean()*100),ci95_pp=interval(d,review_ix,True).tolist())
        if model=='qwen':
            # Replay RNG consumption in the established analyze_memI.py:
            # 7 baseline horizons, then each arm's 7 contrasts and 6 sensitivities.
            rng=random.Random(22072606)
            for _ in KS:python_indices(rng,10000,11)
            for arm in ['C','CE','CE_s','blind_I','f']:
                for k in KS:
                    idx=python_indices(rng,10000,11)
                    if k==64 and arm+'-I' in ep:ep[arm+'-I']['ci95_pp']=interval(v[arm][:,63]-v['I'][:,63],idx,True).tolist()
                for k in KS[1:]:
                    idx=python_indices(rng,10000,11)
                    if k==64 and arm in cond:ep[arm+'-I_conditional']=dict(difference_pp=float(100*(cond[arm][:,63]-cond['I'][:,63]).mean()),ci95_pp=interval(cond[arm][:,63]-cond['I'][:,63],idx,True).tolist())
            # Original CODEARMS report included these five arms in sorted order.
            # Preserve its RNG stream for the additional appendix pairwise rows.
            original=['C','CE','CE_s','blind_I','f']
            for i,a in enumerate(original):
                for b in original[i+1:]:
                    idx=python_indices(rng,10000,11);name=a+'-'+b
                    if name in ep:ep[name]['ci95_pp']=interval(v[a][:,63]-v[b][:,63],idx,True).tolist()
            # E_s versus I was a separate one-arm invocation.
            rng=random.Random(22072606)
            for _ in KS:python_indices(rng,10000,11)
            for k in KS:
                idx=python_indices(rng,10000,11)
                if k==64:ep['E_s-I']['ci95_pp']=interval(v['E_s'][:,63]-v['I'][:,63],idx,True).tolist()
        for name,item in ep.items():contrastrows.append(dict(model=model,contrast=name,difference_pp=item['difference_pp'],lo=item['ci95_pp'][0],hi=item['ci95_pp'][1]))
        for arm in arms:
            for j,pid in enumerate(ids):
                for k in range(1,65):csv.append(dict(model=model,problem_id=pid,arm=arm,k=k,success_probability=float(v[arm][j,k-1])))
            for k in KS:horizons.append(dict(model=model,arm=arm,k=k,success_percent=float(v[arm][:,k-1].mean()*100),solved_trajectories='' if arm=='I' else round(float(v[arm][:,k-1].sum()*10))))
        report[model]=dict(problem_ids=ids,arms=details,endpoint_contrasts=ep)
    dump('sequential_ce.json',report);pass
    for model,r in report.items():print(model,{a:round(d['pass64_percent'],2) for a,d in r['arms'].items()})


if __name__=='__main__':main()
