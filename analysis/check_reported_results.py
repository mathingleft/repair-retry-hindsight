"""Compare sample-level recounts against frozen reported result tables."""
import csv
import json
import re
from common import ARTIFACT, OUTPUT, dump


def load(name):
    return json.loads((OUTPUT/name).read_text())


def number(cell):
    return float(re.search(r'[+-]?\d+(?:\.\d+)?', cell)[0])


def main():
    reference=json.loads((ARTIFACT/'expected/reported_results.json').read_text())
    checks=[]

    def table(label):
        return reference['tables'][label]

    def agree(actual, reported, label, tolerance=.051):
        assert abs(actual-reported)<=tolerance,(label,actual,reported)
        checks.append(label)

    one=load('onestep.json')
    for row in table('tab:qwen'):
        arm=row[0]
        if arm not in one['qwen']['arms'] and arm!='N2--N5':continue
        arms=['N2','N3','N4','N5'] if arm=='N2--N5' else [arm]
        for field,column in [('problems',2),('passes',3)]:
            reported=[int(v) for v in re.findall(r'\d+',row[column])]
            assert reported==[one['qwen']['arms'][a][field] for a in arms],row
            checks.append(f'Qwen {arm} {field}')

    ext={'F_{C_o}':'F_C_other','F_{E_m}':'F_E_match','F_{CE_m}':'F_CE_match'}
    for row in table('tab:ext3'):
        arm=ext.get(row[0].strip('$'))
        if not arm:continue
        for model,offset in [('qwen',1),('deepseek',3)]:
            agree(one[model]['arms'][arm]['problems'],number(row[offset]),f'{model} {arm} solved',0)
            agree(one[model]['arms'][arm]['passes'],number(row[offset+1]),f'{model} {arm} passes',0)

    noi=['bare','N_noI','F_noI','F_E_noI','F_E_shuf_noI','F_R_noI','F_C_noI','F_CE_noI']
    rows=[r for r in table('tab:noi') if r[0] not in ['scaffold']]
    assert len(rows)==len(noi)
    for row,a in zip(rows,noi):
        b='I' if a=='bare' else a.removesuffix('_noI')
        for arm,col in [(a,1),(b,2)]:agree(one['qwen']['arms'][arm]['problems'],number(row[col]),f'no instruction table {arm}',0)

    prover=[('bare',None),('N',None),('F','F_noI'),('F_E','F_E_noI'),('F_E_shuf','F_E_shuf_noI'),('F_C','F_C_noI'),('F_CE','F_CE_noI'),('F_CE_shuf',None)]
    rows=[r for r in table('tab:prover') if r[0]!='scaffold']
    assert len(rows)==len(prover)
    for row,(a,b) in zip(rows,prover):
        agree(one['deepseek']['arms'][a]['problems'],number(row[1]),f'prover table {a}',0)
        if b:agree(one['deepseek']['arms'][b]['problems'],number(row[2]),f'prover table {b}',0)

    seq=load('sequential.json')
    for label,model in [('tab:seq_endpoints','qwen'),('tab:seq_endpoints_deepseek','deepseek')]:
        for row in table(label):
            arm=row[0].strip('$')
            if arm.startswith('('):arm='interaction'
            if arm not in seq[model]['endpoint_contrasts']:continue
            result=seq[model]['endpoint_contrasts'][arm]
            agree(result['difference_pp'],number(row[1]),f'{model} {arm} effect')
            ci=[float(v) for v in re.findall(r'[+-]?\d+\.\d+',row[2])]
            assert len(ci)==2,row
            for i in range(2):agree(result['ci95_pp'][i],ci[i],f'{model} {arm} CI {i}')

    with (OUTPUT/'sequential_horizons.csv').open() as f:
        horizons={(r['model'],r['arm'],int(r['k'])):r for r in csv.DictReader(f)}
    for label,model in [('tab:seq_depth_final','qwen'),('tab:seq_depth_deepseek','deepseek')]:
        for row in table(label):
            arm=row[0].replace('$','').replace(' (pooled)','')
            if (model,arm,1) not in horizons:continue
            for k,cell in zip([1,2,4,8,16,32,64],row[1:]):
                r=horizons[model,arm,k]
                value=float(r['success_percent'] if arm=='I' else r['solved_trajectories'])
                agree(value,number(cell),f'{model} {arm} at {k}',.051 if arm=='I' else 0)

    # Main and appendix behavioral statements: use separate definitions for
    # normalized one-step copying, success-conditioned similarity, and repetition.
    for model,a,b in [('qwen',16.6,.6),('deepseek',60.8,1.6)]:
        for arm,pct in [('F_C',a),('F_C_other',b)]:
            agree(one[model]['arms'][arm]['copy_percent'],pct,f'{model} {arm} copying')
    for arm,count,pct in [('F_C',11,70.6),('F_CE',29,33.1),('F_CE_shuf',20,59.6),('I',35,17.3)]:
        rec=one['qwen']['unknown_subset'][arm]
        agree(rec['problems'],count,f'unknown identifier {arm} solved',0)
        agree(rec['identifier_use_percent'],pct,f'unknown identifier {arm} retention')
    for k,n in [('seven_retry_union',69),('instruction_failure_notice_overlap',38),('instruction_failure_notice_union',53),('original_union',80)]:
        agree(one['qwen'][k],n,k,0)
    compute=load('compute.json')
    for category,reported in reference['compute_gpu_hours'].items():
        agree(compute['categories_gpu_hours'][category],reported,'compute '+category,.0051)
    dump('reported_result_checks.json',dict(status='PASS',numerical_checks=len(checks),checks=checks))
    print(f'PASS: {len(checks)} numerical comparisons against frozen reported results.')


if __name__=='__main__':main()
