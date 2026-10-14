"""R8: first-failure-conditional endpoints for BOTH models.

The released script computes conditional contrasts only for Qwen. This adds
DeepSeek using DeepSeek's own established index matrix (200,000 resamples,
seed 20260913, unsorted-quantile endpoints), matching analyze_sequential.py.
"""
import json, random
import numpy as np
from as_ce import load_model, python_indices, interval, ARMS
from pathlib import Path

out={}
for model,sub in [('qwen','out'),('deepseek','out_dsp')]:
    arms=ARMS+['noise'] if model=='qwen' else ARMS
    ids,v,cond,details=load_model(sub,arms)
    if model=='deepseek':
        ix=np.random.default_rng(20260913).integers(0,11,(200000,11)); orig=False
    else:
        ix=python_indices(random.Random(0),10000,11); orig=True
    res={}
    for name,a,b in [('CE-I','CE','I'),('C-I','C','I'),('CE_s-I','CE_s','I'),
                     ('E-I','f','I'),('E_m-I','E_s','I'),('CE_m-I','CE_m','I')]:
        d=cond[a][:,63]-cond[b][:,63]
        lo,hi=interval(d,ix,orig)
        res[name]=dict(difference_pp=float(d.mean()*100), ci95_pp=[float(lo),float(hi)])
    # eligible-trajectory denominators, for the caption
    res['_unconditional_pass64_percent']={a:round(float(v[a][:,63].mean()*100),2) for a in arms}
    out[model]=res
json.dump(out,open(Path(__file__).parent/'out/r8_conditional.json','w'),indent=1)
for m,r in out.items():
    print('=====',m,'(conditional on failure of attempt 1)')
    for k,x in r.items():
        if k.startswith('_'): continue
        lo,hi=x['ci95_pp']
        print(f'  {k:10s} {x["difference_pp"]:+7.2f} pp  [{lo:+7.2f},{hi:+7.2f}]')
