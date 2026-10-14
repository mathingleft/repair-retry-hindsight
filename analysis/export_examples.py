"""Export exact sample provenance for the appendix's qualitative examples."""
import re
import sys
from common import DATA, read, dump
sys.path.insert(0,str(DATA))
import arms


def main():
    q={r['id']:r for r in read(DATA/'bank.jsonl')}
    d={r['id']:r for r in read(DATA/'out_m2/bank2.jsonl')}
    records=[]
    specs=[('out','mathd_numbertheory_293','F_C',0),
           ('out','mathd_algebra_329','F_CE',0),
           ('out_noi','aime_1997_p9','F_E_shuf_noI',0),
           ('out_m2','mathd_algebra_171','F_CE',1)]
    for directory,pid,arm,k in specs:
        matches=[r for r in read(DATA/directory/'verified.jsonl') if (r['id'],r['arm'],r['sample_idx'])==(pid,arm,k)]
        assert len(matches)==1
        r=matches[0];bank=d if directory=='out_m2' else q
        records.append(dict(source=f'experiments/decomp_rerun/{directory}/verified.jsonl',
                            row=r,predecessor=bank[pid],proof_body=r['lean_code'].split(':=',1)[1]))
    assert records[0]['row']['lean_code']==records[0]['predecessor']['predecessor_code']
    assert records[0]['row']['status']=='FAIL'
    assert records[1]['row']['status']=='PASS'
    assert 'begin' in records[1]['predecessor']['predecessor_code']
    assert re.search(r'\bbegin\b',records[2]['proof_body'])
    wrong=arms.shuffled_error('aime_1997_p9')
    assert not re.search(r'\bbegin\b',wrong)
    records[2]['displayed_wrong_diagnostic']=wrong
    assert records[3]['row']['status']=='FAIL' and 'add_right_neg' in records[3]['proof_body']
    pid='algebra_2varlineareq_fp3zeq11_3tfm1m5zeqn68_feqn10_zeq7'
    rs=[r for r in read(DATA/'out/verified.jsonl') if r['id']==pid and r['arm'] in ['F_R','F_E']]
    reflection=next(r for r in read(DATA/'out/reflections.jsonl') if r['id']==pid)
    for arm in ['F_R','F_E']:
        group=[r for r in rs if r['arm']==arm];assert len(group)==4
        if arm=='F_R':
            assert all(r['status']=='FAIL' for r in group)
            assert all(not re.search(r'\bbegin\b',r['lean_code'].split(':=',1)[1]) for r in group)
        else:
            assert [r['sample_idx'] for r in group if re.search(r'\bbegin\b',r['lean_code'].split(':=',1)[1])]==[0]
            assert sorted(r['sample_idx'] for r in group if r['status']=='PASS')==[1,3]
    dump('behavioral_examples.json',dict(individual_examples=records,reflection_example=dict(problem_id=pid,rows=rs,reflection=reflection)))
    print('PASS: appendix examples located and checked against exact saved samples.')


if __name__=='__main__':main()
