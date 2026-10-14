"""Recompute token summaries, correction accounting, repetitions and batch sensitivity."""
import collections
import csv
import json
from math import sqrt
import statistics
import sys
from common import DATA,OUTPUT,ARTIFACT,read,normalize,dump,csvout
sys.path.insert(0,str(DATA))
import analyze_confirm_error as confirm
from copy_rate import body


def main():
    countsfile=OUTPUT/'token_counts.csv'
    if not countsfile.exists():countsfile=ARTIFACT/'data/token_counts.csv'
    with countsfile.open() as f:rows=list(csv.DictReader(f))
    tokens={}
    for arm in sorted({r['arm'] for r in rows}):
        group=[r for r in rows if r['arm']==arm];tokens[arm]={}
        for field in ['input_tokens','output_tokens']:
            vals=[int(r[field]) for r in group]
            tokens[arm][field]=dict(n=len(vals),total=sum(vals),mean=statistics.mean(vals),median=statistics.median(vals))
    dump('tokens.json',tokens)
    blocks,integ=confirm.load_blocks(str(DATA/'out_confirm_error'))
    ids=confirm.pids_of(str(DATA/'bank.jsonl'));d,v,df,r,_,_=confirm.primary(blocks,ids)
    blockvals=[x[3] for x in confirm.block_level(blocks,ids)]
    bv=statistics.variance(blockvals)/len(blockvals)
    batch=dict(difference_pp=d*100,ci95_pp=[100*x for x in confirm.ci(d,v,df)[:2]],ci90_pp=[100*x for x in confirm.ci(d,v,df,.90)[:2]],block_sensitivity95_pp=[100*x for x in confirm.ci(d,bv,len(blockvals)-1)[:2]],infra_unresolved=sum(x['infra'] for x in integ.values()))
    dump('precision_batches.json',batch)
    correction=read(DATA/'out/sorry_reverify.jsonl')
    flipped=[r for r in correction if r['old']=='PASS' and r['new']!='PASS']
    dump('integrity.json',dict(reverified_rows=len(correction),flipped_passes=len(flipped),flips_by_file=dict(collections.Counter(r['file'].split('/')[-2]+'/'+r['file'].split('/')[-1] for r in flipped))))
    rep={}
    for arm in ['C','CE','CE_s','f','E_s','assert','noise']:
        groups=collections.defaultdict(list)
        for r in read(DATA/'out'/f'depth_{arm}.jsonl'):groups[r['id'],r['traj']].append(r)
        n=0;repeat=0
        for rs in groups.values():
            seen=set()
            for r in sorted(rs,key=lambda r:r['attempt']):
                # The paper measures byte-identical saved proof bodies, among
                # attempts after the first; one-step copy uses normalization.
                b=r.get('code') or ''
                if r['attempt']>1:n+=1;repeat+=bool(b) and b in seen
                if b:seen.add(b)
        rep[arm]=dict(repeats=repeat,eligible_attempts=n,percent=100*repeat/n)
    dump('sequential_repetition.json',rep)
    # Each row after attempt 1 adds one new wrong diagnostic. Counting every
    # retained history entry again would overcount donors carried across prompts.
    draws=[r['donor'] for r in read(DATA/'out_codearms/E_s_donors.jsonl')]
    ratios=[r['src_tok']/r['own_tok'] for r in draws]
    import numpy as np
    dump('diagnostic_matching.json',dict(draws=len(draws),same_class=sum(r['own_class']==r['src_class'] for r in draws),tactic_pool=sum(r['pool']=='tactic_pool' for r in draws),ratio_quartiles=np.quantile(ratios,[.25,.5,.75]).tolist()))
    import results_figures as historical
    old_i=historical.sequential(DATA/'out/depth_blind_I.jsonl',True)
    f=historical.sequential(DATA/'out/depth_f.jsonl',False)
    d=100*(f[:,-1]-old_i[:,-1])
    dump('historical_depth.json',dict(f_minus_old_instruction_pp=float(d.mean()),ci95_pp=historical.bootstrap(d).tolist(),note='Earlier memoryless instruction pool; not the final same-stack I baseline.'))
    # All GPU allocations in the inherited ledger; productive subtraction is
    # a historical label, not a claim all remaining jobs produced useful samples.
    jobs=[]
    for line in (DATA/'figs/sacct_dump_parsable.txt').read_text().splitlines():
        fields=line.split('|')
        if len(fields)<5 or 'gres/gpu=' not in fields[3]:continue
        import re
        m=re.search(r'gres/gpu=(\d+)',fields[3]);gpu=int(m[1]);elapsed=fields[2];days=0
        if '-' in elapsed:ds,elapsed=elapsed.split('-');days=int(ds)
        h,mi,s=map(int,elapsed.split(':'));hours=(days*86400+h*3600+mi*60+s)/3600
        jobs.append(dict(job=fields[0],name=fields[1],gpus=gpu,hours=hours,gpu_hours=gpu*hours,state=fields[4]))
    csvout('gpu_allocations.csv',jobs)
    categories=collections.defaultdict(float)
    for row in jobs:
        name=row['name']
        if row['state']=='TIMEOUT' and row['gpu_hours']==48:continue
        if 'depth' in name:category='Depth experiments'
        elif name in ['decomp-noi','decomp-noi-r2-g2','decomp-m2noi']:category='No-instruction ladder'
        elif 'ext2' in name:category='Code plus shuffled diagnostic'
        elif 'm2' in name:category='DeepSeek-Prover one-step'
        elif name.startswith('probe') or name=='nvidia-smi':category='Probes'
        else:category='Qwen one-step'
        categories[category]+=row['gpu_hours']
    total=sum(r['gpu_hours'] for r in jobs)
    assert abs(sum(categories.values())-(total-48))<1e-8
    dump('compute.json',dict(ledger_gpu_hours=total,ledger_minus_known_48h_timeout=total-48,categories_gpu_hours=dict(categories),deepseek_overlapping_subtotal=sum(r['gpu_hours'] for r in jobs if 'm2' in r['name']),coverage='Earlier ledger only; excludes later sequential and precision campaigns; retained failed/cancelled allocations are included.'))
    print('Recomputed batch sensitivity, token summaries, correction ledger and repetition.')


if __name__=='__main__':main()
