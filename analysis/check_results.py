"""Regression checks against manuscript counts and recovered historical analyses."""
import json
import math
from common import ARTIFACT, OUTPUT, dump


def close(a,b,path='root'):
    if isinstance(b,dict):
        for k,v in b.items():close(a[k],v,path+'.'+k)
    elif isinstance(b,list):
        assert len(a)==len(b),path
        for i,(x,y) in enumerate(zip(a,b)):close(x,y,path+f'[{i}]')
    elif isinstance(b,(float,int)):
        assert math.isclose(a,b,abs_tol=1e-9,rel_tol=1e-9),(path,a,b)
    else:assert a==b,(path,a,b)


def main():
    load=lambda name:json.loads((OUTPUT/name).read_text())
    expected=lambda name:json.loads((ARTIFACT/'expected'/name).read_text())
    one=load('onestep.json')
    counts={
      'qwen':dict(bare=15,I=45,F=33,F_C=18,F_E=41,F_CE=40,F_R=42,N=49,N2=48,N3=43,N4=42,N5=44,S=30,P=30,D=35,F_E_shuf=39,F_CE_shuf=26,F_E_match=47,F_CE_match=27,F_C_other=42,F_noI=46,F_E_noI=32,F_E_shuf_noI=29,F_C_noI=23,F_CE_noI=28,F_R_noI=41,N_noI=30),
      'deepseek':dict(bare=29,I=34,F=35,F_C=14,F_E=30,F_CE=12,F_E_shuf=32,N=32,F_CE_shuf=16,F_E_match=32,F_CE_match=12,F_C_other=33,F_noI=32,F_E_noI=32,F_E_shuf_noI=32,F_C_noI=8,F_CE_noI=11)}
    for model,arms in counts.items():
        for arm,n in arms.items():assert one[model]['arms'][arm]['problems']==n,(model,arm)
    close(one['strict_identifier'],dict(n=62,counts={'F_E_noI':66,'F_E_shuf_noI':64}))
    assert one['qwen']['fresh_F_noI']['solved']==45
    assert load('integrity.json')['flipped_passes']==224
    for name in ['pooled_diagnostics.json','feedback_bounds.json']:close(load(name),expected(name))
    tokens=load('tokens.json')
    for arm,record in expected('token_cost_audit.json')['one_step'].items():
        for old,new in [('prompt_tokens','input_tokens'),('visible_output_tokens','output_tokens')]:
            close(tokens[arm][new],{k:record[old][k] for k in ['n','total','mean','median']})
    seq=load('sequential.json')
    for model,vals in [('qwen',[62.8,22.7,44.5,20.9,26.4,63.6,57.3]),('deepseek',[75.3,24.5,24.5,22.7,22.7,67.3,81.8])]:
        for arm,v in zip(['I','C','CE','CE_m','CE_s','f','E_s'],vals):assert round(seq[model]['arms'][arm]['pass64_percent'],1)==v,(model,arm)
    intervals={
        'qwen':{'CE-I':[-41.8,5.3],'C-I':[-62.2,-17.0],'CE_s-I':[-58.4,-12.7],'CE-C':[10.9,32.7],'CE-CE_s':[8.2,28.2],'C-CE_s':[-14.5,7.3],'CE-f':[-31.8,-5.5],'CE_s-f':[-50.,-25.5],'CE_s-E_s':[-49.1,-13.6],'f-E_s':[-6.4,19.1],'interaction':[-8.2,29.1],'E_s-noise':[-16.4,11.8],'E_s-I':[-26.2,13.7]},
        'deepseek':{'CE-I':[-66.4,-34.5],'C-I':[-67.6,-33.6],'CE_s-I':[-68.9,-35.6],'CE-C':[-3.6,3.6],'CE-CE_s':[-10.,13.6],'f-E_s':[-30.,-2.7],'CE_s-E_s':[-76.4,-40.],'CE-f':[-61.8,-23.6],'interaction':[-1.8,34.5]}}
    for model,contrasts in intervals.items():
        for name,want in contrasts.items():assert [round(x,1) for x in seq[model]['endpoint_contrasts'][name]['ci95_pp']]==want,(model,name)
    new_intervals={
      'qwen':{'CE-CE_m':[12.7,35.5],'f-E_s':[-6.4,19.1],'matched_interaction':[-1.8,34.5],'CE_m-C':[-10.9,9.1],'CE_m-CE_s':[-17.3,7.3],'CE_m-E_s':[-50.,-21.8]},
      'deepseek':{'CE-CE_m':[-5.5,9.1],'f-E_s':[-30.,-2.7],'matched_interaction':[.9,35.5],'CE_m-C':[-10.,6.4],'CE_m-CE_s':[-12.7,11.8],'CE_m-E_s':[-71.8,-44.5]}}
    for model,contrasts in new_intervals.items():
        for name,want in contrasts.items():assert [round(x,1) for x in seq[model]['endpoint_contrasts'][name]['ci95_pp']]==want,(model,name)
    audit=load('ce_m_matching.json')
    for model,want in [('qwen',(5755,5662,93,4343)),('deepseek',(5793,5002,791,4792))]:
        got=audit[model]
        assert (got['donor_rows'],got['same_class'],got['pooled_family_fallbacks'],got['exact_code_recurrence']['count'])==want,model
        assert got['identical_messages']==0 and got['cross_problem_sources']==got['donor_rows']
    for arm,v in dict(C=73.9,CE_s=78.3,CE=65.7,f=0.,E_s=.4,assert_=.2).items():
        arm='assert' if arm=='assert_' else arm
        assert round(load('sequential_repetition.json')[arm]['percent'],1)==v,arm
    assert round(load('compute.json')['ledger_gpu_hours'],1)==219.9
    close(load('diagnostic_matching.json'),dict(draws=4055,same_class=4004,tactic_pool=51))
    assert [round(v,2) for v in load('diagnostic_matching.json')['ratio_quartiles']]==[1.,1.,1.]
    interactions=load('exploratory_onestep_interactions.json')['results']
    expected_interactions={
      'matched_diagnostic_by_proof_exposure':(11.38,[2.40,19.76]),
      'instruction_by_diagnostic_recovery':(10.18,[2.40,17.96])}
    for name,(estimate,interval) in expected_interactions.items():
        assert round(interactions[name]['estimate_pp'],2)==estimate,name
        assert [round(v,2) for v in interactions[name]['ci95_pp']]==interval,name
    dump('checks.json',dict(status='PASS',checks=['all 44 one-step condition counts','strict identifier control','fresh seed replicate','224 corrected verdicts','historical pooled and simultaneous-bound analyses','two exploratory one-step interactions','all historical one-step token summaries','both seven-arm sequential endpoints','CE_m donor matching and recurrence audit','byte-identical sequential repetition','partial GPU ledger']))
    print('PASS: frozen results agree with manuscript counts and recovered reference analyses.')


if __name__=='__main__':main()
