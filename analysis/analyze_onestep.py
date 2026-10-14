"""Recount all one-step conditions, behavioral checks, and paired contrasts."""
from collections import defaultdict
import json
import re
import sys
import numpy as np
from common import DATA,OUTPUT,QDIRS,DDIRS,read,dump,csvout
import results_figures as source


def main():
    result={};table=[];features=[];tests=[]
    for name,bank,dirs in [('qwen','bank.jsonl',QDIRS),('deepseek','out_m2/bank2.jsonl',DDIRS)]:
        ids,counts,copies,summary=source.model(bank,dirs)
        result[name]=summary
        pairs=[(a,'I') for a in counts if a!='I']
        pairs += [(a,a+'_noI') for a in counts if a+'_noI' in counts]
        pairs += [(a,'bare') for a in counts if a.endswith('_noI')]
        for a,b in pairs:
            x,y=counts[a],counts[b]
            tests.append(dict(model=name,arm_a=a,arm_b=b,wins=int((x>y).sum()),losses=int((x<y).sum()),sign_p=source.sign(x,y),solved_wins=int(((x>0)&(y==0)).sum()),solved_losses=int(((y>0)&(x==0)).sum()),mcnemar_p=source.sign(x>0,y>0)))
        for arm,c in counts.items():
            table.append(dict(model=name,arm=arm,problems=len(ids),solved=int((c>0).sum()),passes=int(c.sum()),samples=len(ids)*4))
            for j,pid in enumerate(ids):features.append(dict(model=name,arm=arm,problem_id=pid,passes=int(c[j]),solved=int(c[j]>0),copy_fraction=float(copies[arm][j])))
        if name=='qwen':
            result[name]['seven_retry_union']=int(np.any(np.stack([counts[a]>0 for a in ['I','N','N2','N3','N4','N5','F_noI']]),axis=0).sum())
            result[name]['instruction_failure_notice_overlap']=int(((counts['I']>0)&(counts['F_noI']>0)).sum())
            result[name]['instruction_failure_notice_union']=int(((counts['I']>0)|(counts['F_noI']>0)).sum())
            original=[a for a in counts if a not in ['F_C_other','F_E_match','F_CE_match']]
            result[name]['original_union']=int(np.any(np.stack([counts[a]>0 for a in original]),axis=0).sum())
            fresh=read(DATA/'out_noi_r2/verified.jsonl')
            result[name]['fresh_F_noI']={'solved':len({r['id'] for r in fresh if r['status']=='PASS'}),'passes':sum(r['status']=='PASS' for r in fresh)}
            fc=np.array([sum(r['id']==pid and r['status']=='PASS' for r in fresh) for pid in ids])
            result[name]['fresh_F_noI']['sign_p_vs_F']=source.sign(fc,counts['F'])
    # Strict identifier subgroup used by the updated figure.
    sys.path.insert(0,str(DATA));import arms
    bank={r['id']:r for r in read(DATA/'bank.jsonl') if not r['is_framing_holdout']}
    patterns={}
    for pid,b in bank.items():
        u=re.search(r"[Uu]nknown (?:identifier|constant)[ :]*[`'‘]?([^`'’\s]+)",b['predecessor_error'])
        if u and u[1] not in b['statement']:patterns[pid]=re.compile(r"(?<![A-Za-z0-9_.'])"+re.escape(u[1])+r"(?![A-Za-z0-9_'])")
    strict=[pid for pid,pat in patterns.items() if not pat.search(arms.shuffled_error(pid))]
    groups=defaultdict(lambda:defaultdict(list))
    for d in QDIRS:
        for r in read(DATA/d/'verified.jsonl'):
            if r['id'] in bank:groups[r['arm']][r['id']].append(r)
    vals={a:np.array([sum(bool(patterns[p].search(source.body(r.get('lean_code')) or '')) for r in groups[a][p]) for p in strict]) for a in ['F_E_noI','F_E_shuf_noI']}
    result['strict_identifier']={'n':len(strict),'counts':{a:int(v.sum()) for a,v in vals.items()},'sign_p':source.sign(*vals.values())}
    subgroup={a:np.array([sum(r['status']=='PASS' for r in groups[a][p]) for p in patterns]) for a in ['F_C_noI','F_CE_noI']}
    result['strict_identifier']['no_instruction_subgroup_sign_p']=source.sign(subgroup['F_CE_noI'],subgroup['F_C_noI'])
    dump('onestep.json',result);csvout('onestep_counts.csv',table);csvout('onestep_problem_features.csv',features);csvout('onestep_paired_tests.csv',tests)
    print('Recounted all one-step arms, pooled coverage, copying, and identifier summaries.')


if __name__=='__main__':main()
