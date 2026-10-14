"""Audit the matched wrong-diagnostic sequential arm from compact donor ledgers."""
from collections import defaultdict
import math
from common import DATA, read, dump


def quantile(values, q):
    values=sorted(values); x=(len(values)-1)*q; lo=math.floor(x); hi=math.ceil(x)
    return values[lo] if lo==hi else values[lo]*(hi-x)+values[hi]*(x-lo)


def main():
    report={}
    for model,sub,rich in [('qwen','out','out_codearms'),('deepseek','out_dsp','out_dsp_codearms')]:
        rows=read(DATA/sub/'depth_CE_m.jsonl')
        donors=read(DATA/rich/'CE_m_donors.jsonl')
        expected={(r['id'],r['traj'],r['attempt']) for r in rows if r['attempt']>1}
        keys=[(r['id'],r['traj'],r['attempt']) for r in donors]
        assert len(keys)==len(set(keys)) and set(keys)==expected
        ratios=[max(r['donor']['own_tok'],r['donor']['src_tok'])/min(r['donor']['own_tok'],r['donor']['src_tok']) for r in donors]
        histories=defaultdict(list); recurrent=0
        for r in sorted(rows,key=lambda x:(x['id'],x['traj'],x['attempt'])):
            key=r['id'],r['traj']
            if r['attempt']>1 and r.get('code') and r['code'] in histories[key]:recurrent+=1
            if r.get('code'):histories[key].append(r['code'])
        report[model]=dict(
            rows=len(rows), donor_rows=len(donors),
            same_class=sum(r['donor']['own_class']==r['donor']['src_class'] for r in donors),
            pooled_family_fallbacks=sum(r['donor']['pool']!=r['donor']['own_class'] for r in donors),
            identical_messages=sum(r['source_error_identical_to_own'] for r in donors),
            cross_problem_sources=sum(r['source_problem_differs'] for r in donors),
            length_ratio=dict(median=quantile(ratios,.5),p90=quantile(ratios,.9),maximum=max(ratios)),
            exact_code_recurrence=dict(count=recurrent,denominator=len(donors),percent=100*recurrent/len(donors)))
    dump('ce_m_matching.json',report)
    for model,r in report.items():print(model,r)


if __name__=='__main__':main()
