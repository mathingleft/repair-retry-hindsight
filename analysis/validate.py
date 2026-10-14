"""Validate frozen file hashes, problem/sample grids, and full/slim verdict identity."""
from collections import defaultdict
import json
from common import ARTIFACT, DATA, QDIRS, DDIRS, read, normalize, digest, dump
import selection


def main():
    package=ARTIFACT/'manifests/package.json'
    omitted=set(json.loads(package.read_text()).get('omitted_optional_full_traces',[])) if package.exists() else set()
    release=ARTIFACT/'manifests/release.json'
    if release.exists():
        for r in json.loads(release.read_text()):assert digest(ARTIFACT/r['path'])==r['sha256'],r['path']
    manifest=json.loads((ARTIFACT/'manifests/inputs.json').read_text())
    for r in manifest:assert digest(ARTIFACT/r['path'])==r['sha256'],r['path']
    report={'hashed_files':len(manifest),'onestep':{},'sequential':{}}
    for model,bank,dirs in [('qwen','bank.jsonl',QDIRS+['out_noi_r2']),('deepseek','out_m2/bank2.jsonl',DDIRS)]:
        pids={r['id'] for r in read(DATA/bank) if not r.get('is_framing_holdout')}
        for directory in dirs:
            groups=defaultdict(list)
            for r in read(DATA/directory/'verified.jsonl'):
                if r['id'] in pids:groups[r['arm'],r['id']].append(r['sample_idx'])
            arms={a for a,p in groups}
            assert all((a,p) in groups for a in arms for p in pids),directory
            assert all(sorted(g)==list(range(4)) for g in groups.values()),directory
            report['onestep'][directory]={'problems':len(pids),'arms':sorted(arms),'rows':sum(map(len,groups.values()))}
    for model,sub,bank,rich in [('qwen','out','depth/depth_bank.jsonl','out_codearms'),('deepseek','out_dsp','depth/depth_bank_dsp.jsonl','out_dsp_codearms')]:
        pids={r['id'] for r in read(DATA/bank)}
        for arm in ['C','CE','CE_m','CE_s','f','E_s','memI']:
            rows=[normalize(r) for r in read(DATA/sub/f'depth_{arm}.jsonl')];groups=defaultdict(list);keys={}
            for r in rows:
                key=r['id'],r['traj'],r['attempt'];assert key not in keys;keys[key]=r
                groups[r['id'],r['traj']].append(r)
            assert {p for p,t in groups}==pids and all(sum(p==pid for p,t in groups)==10 for pid in pids)
            for rs in groups.values():
                rs.sort(key=lambda r:r['attempt']);assert [r['attempt'] for r in rs]==list(range(1,len(rs)+1))
                passes=[r['attempt'] for r in rs if r['status']=='PASS']
                if arm=='memI':assert len(rs)==64
                elif passes:assert passes==[len(rs)]
                else:assert len(rs)==64
            fullpath=DATA/rich/f'{arm}_rows.jsonl.gz'
            if arm in ['C','CE','CE_s','E_s'] and str(fullpath.relative_to(ARTIFACT)) not in omitted:
                full=[normalize(r) for r in read(fullpath)]
                assert len(full)==len(keys)
                assert len({(r['id'],r['traj'],r['attempt']) for r in full})==len(full)
                for r in full:
                    key=r['id'],r['traj'],r['attempt'];assert r==keys[key],(model,arm,key)
            report['sequential'][model+'/'+arm]={'rows':len(rows),'problems':len(pids),'trajectories':len(groups)}
    donors=read(DATA/'out_codearms/E_s_donors.jsonl')
    donor_keys=[(r['id'],r['traj'],r['attempt']) for r in donors]
    expected_keys={(r['id'],r['traj'],r['attempt']) for r in read(DATA/'out/depth_E_s.jsonl') if r['attempt']>1}
    assert len(donor_keys)==len(set(donor_keys)) and set(donor_keys)==expected_keys
    source_errors={f"{r['id']}|{r['traj']}|{r['attempt']}":r for r in read(DATA/'out/depth_f.jsonl')}
    for r in donors:
        source=source_errors[r['donor']['src']]
        assert source['status']=='FAIL' and source['id']!=r['id']
    report['diagnostic_donor_rows']=len(donors)
    for model,sub,rich in [('qwen','out','out_codearms'),('deepseek','out_dsp','out_dsp_codearms')]:
        donors=read(DATA/rich/'CE_m_donors.jsonl')
        donor_keys=[(r['id'],r['traj'],r['attempt']) for r in donors]
        expected_keys={(r['id'],r['traj'],r['attempt']) for r in read(DATA/sub/'depth_CE_m.jsonl') if r['attempt']>1}
        assert len(donor_keys)==len(set(donor_keys)) and set(donor_keys)==expected_keys
        assert all(r['source_problem_differs'] for r in donors)
        assert not any(r['source_error_identical_to_own'] for r in donors)
        report['sequential'][model+'/CE_m_donors']={'rows':len(donors),'cross_problem':len(donors),'identical_messages':0}
    precision_ids={r['id'] for r in read(DATA/'bank.jsonl') if not r.get('is_framing_holdout')}
    for batch in range(1,5):
        path=DATA/'out_confirm_error'/f'b{batch}'/'verified.jsonl'
        rows=read(path)
        expected={(arm,pid,k) for arm in ['F_E','F_E_shuf'] for pid in precision_ids for k in range(4)}
        keys=[(r['arm'],r['id'],r['sample_idx']) for r in rows]
        assert len(keys)==len(expected) and set(keys)==expected,path
    report['precision_rows']=4*len(expected)
    report['optional_full_trace_files_omitted_by_package']=sorted(omitted)
    selection.main();dump('validation.json',report)
    print(f'Validated {len(manifest)} frozen files, both one-step banks, and both complete seven-arm sequential grids.')


if __name__=='__main__':main()
