"""Reconstruct the frozen analysis banks using the retained selection evidence."""
from collections import Counter, defaultdict
import random
from common import DATA, ARTIFACT, DDIRS, read, dump, csvout


def main():
    q=read(DATA/'bank.jsonl'); qids={b['id'] for b in q if not b['is_framing_holdout']}
    assert len(q)==171 and len(qids)==167
    holdouts=sorted(b['id'] for b in q if b['is_framing_holdout'])
    assert holdouts==sorted(['algebra_amgm_sumasqdivbgeqsuma','amc12a_2013_p4','imo_1963_p5','mathd_algebra_276'])
    assert all(0 <= b['base_pass_rate_k256'] < .25 for b in q)
    historical=read(ARTIFACT/'selection/reconstructed_source_manifest.jsonl')
    hm={r['id']:r for r in historical}
    matched=0
    for b in q:
        if b['id'] in hm:
            h=hm[b['id']]
            assert abs(h['base_pass_rate']-b['base_pass_rate_k256'])<1e-8
            assert h['predecessor_rollout_idx']==b['predecessor_rollout_idx']
            matched+=1
    # DeepSeek: use all 167 original bare problems, before restricting to its bank.
    bare=defaultdict(list)
    for r in read(DATA/'out_m2/verified.jsonl'):
        if r['arm']=='bare':bare[r['id']].append(r)
    assert set(bare)==qids and all(len(v)==4 for v in bare.values())
    selected={}; allpass=[]; degenerate=[]
    for pid,rs in bare.items():
        fails=sorted([r for r in rs if r['status']!='PASS'],key=lambda r:r['sample_idx'])
        if not fails:allpass.append(pid);continue
        real=[r for r in fails if not (r.get('error') or '').startswith(('no code extracted','no proof body parsed'))]
        if not real:degenerate.append(pid);continue
        selected[pid]=real[0]
    dbank=read(DATA/'out_m2/bank2.jsonl')
    assert set(selected)=={b['id'] for b in dbank} and len(allpass)==30 and not degenerate
    for b in dbank:
        r=selected[b['id']]
        assert (r['lean_code'],r.get('error') or '',r['sample_idx'])==(b['predecessor_code'],b['predecessor_error'],b['predecessor_rollout_idx'])
    # The amended depth selection uses all available 17 one-step arms (68 draws/problem).
    one=defaultdict(lambda:[0,0]); keys=set()
    for directory in DDIRS:
        for r in read(DATA/directory/'verified.jsonl'):
            if r['id'] not in selected:continue
            key=(r['arm'],r['id'],r['sample_idx']);assert key not in keys;keys.add(key)
            one[r['id']][1]+=1;one[r['id']][0]+=r['status']=='PASS'
    recorded_one=__import__('json').loads((DATA/'out_dsp/prover_onestep_passes.json').read_text())
    assert dict(one)=={p:recorded_one[p] for p in selected}
    screen=read(DATA/'out_dsp/cold_screen.jsonl')
    n=Counter(r['id'] for r in screen); c=Counter(r['id'] for r in screen if r['status']=='PASS')
    assert set(n)==set(selected) and set(n.values())=={32} and len(screen)==4384
    qdepth=read(DATA/'depth/depth_bank.jsonl');targets=sorted([r['base_pass']/r['base_n'] for r in qdepth],reverse=True)
    order={p:i for i,p in enumerate(random.Random(22072606).sample(sorted(n),len(n)))}
    pool=[p for p in n if c[p]>0 or one[p][0]>0];picked=[];used=set()
    for target in targets:
        pid=min((p for p in pool if p not in used),key=lambda p:(abs(c[p]/n[p]-target),0 if one[p][0]>0 else 1,order[p]))
        used.add(pid);picked.append(dict(id=pid,base_pass=c[pid],base_n=n[pid],onestep_pass=one[pid][0],onestep_n=one[pid][1],qwen_target=round(target,4)))
    frozen=read(DATA/'depth/depth_bank_dsp.jsonl')
    assert len(picked)==11 and [r['id'] for r in picked]==[r['id'] for r in frozen]
    for a,b in zip(picked,frozen):assert all(a[k]==b[k] for k in a)
    csvout('deepseek_depth_selection.csv',picked)
    csvout('qwen_onestep_selection.csv',[dict(id=b['id'],included=b['id'] in qids,recorded_screening_rate=b['base_pass_rate_k256'],predecessor_index=b['predecessor_rollout_idx']) for b in q])
    from common import OUTPUT
    import json
    for name,rows in [('qwen_bank.jsonl',[b for b in q if b['id'] in qids]),('deepseek_bank.jsonl',dbank),('qwen_depth_bank.jsonl',qdepth),('deepseek_depth_bank.jsonl',frozen)]:
        (OUTPUT/name).write_text(''.join(json.dumps(r)+'\n' for r in rows))
    dump('selection.json',dict(qwen_frozen_source=171,qwen_onestep=167,holdouts=holdouts,
        historical_manifest_overlap=matched,recorded_rate_mean=sum(b['base_pass_rate_k256'] for b in q if b['id'] in qids)/167,
        recorded_zero_count=sum(b['base_pass_rate_k256']==0 for b in q if b['id'] in qids),
        deepseek_onestep=137,deepseek_all_bare_pass_exclusions=sorted(allpass),deepseek_degenerate_exclusions=degenerate,
        deepseek_depth=11,screen_rows=len(screen),screen_non_ok=sum(r['gen_status']!='ok' or r['infrastructure_status']!='ok' for r in screen),
        depth_bank_overlap=sorted({r['id'] for r in qdepth}&used),
        qwen_selection_limit='The upstream adaptive screening pool is not bundled; the frozen predecessor manifest and exact exclusions reproduce the submitted bank. The historical field name k256 does not certify a uniform 256-draw screening procedure.',
        qwen_depth_selection='Inherited fixed head-to-head list with amc12b_2021_p1 excluded; selected IDs and recorded 256-draw counts are frozen. No randomized selection rule is asserted.'))
    print('Selection reproduced: Qwen 167; DeepSeek 137; amended DeepSeek depth 11.')


if __name__=='__main__':main()
