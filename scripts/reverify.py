"""Reverify frozen proofs into a new file; no source verdict is overwritten."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'analysis'),str(ROOT/'experiments/decomp_rerun')]
from common import DATA,QDIRS,DDIRS,read,normalize

SEQUENTIAL_ARMS = ['C', 'CE', 'CE_s', 'CE_m', 'f', 'E_s', 'memI']


def collect_jobs(model, family):
    """Enumerate frozen records without contacting a verifier or changing data."""
    jobs = []
    if family == 'sequential':
        bank = {r['id']: r for r in read(DATA/'depth'/('depth_bank.jsonl' if model == 'qwen' else 'depth_bank_dsp.jsonl'))}
        package = json.loads((ROOT/'manifests/package.json').read_text())
        if set(SEQUENTIAL_ARMS) != set(package['reported_sequential_arms']['raw_records']):
            raise ValueError('Reverification arm inventory differs from reported package arms')
        for arm in SEQUENTIAL_ARMS:
            for raw in read(DATA/('out' if model == 'qwen' else 'out_dsp')/f'depth_{arm}.jsonl'):
                r = normalize(raw)
                jobs.append((f"{arm}|{r['id']}|{r['traj']}|{r['attempt']}", r['id'], r['code'], r['status']))
    else:
        if family in ('precision', 'fresh-noi') and model != 'qwen':
            raise ValueError(f'{family} is Qwen only')
        bank = {r['id']: r for r in read(DATA/('bank.jsonl' if model == 'qwen' else 'out_m2/bank2.jsonl')) if not r.get('is_framing_holdout')}
        if family == 'precision':
            dirs = [f'out_confirm_error/b{i}' for i in range(1, 5)]
        elif family == 'fresh-noi':
            dirs = ['out_noi_r2']
        elif family == 'onestep':
            dirs = QDIRS + ['out_noi_r2'] if model == 'qwen' else DDIRS
        else:
            raise ValueError(f'Unknown family: {family}')
        for directory in dirs:
            for r in read(DATA/directory/'verified.jsonl'):
                if r['id'] in bank:
                    jobs.append((f"{directory}|{r['arm']}|{r['id']}|{r['sample_idx']}", r['id'], r.get('lean_code'), r['status']))
    if len({j[0] for j in jobs}) != len(jobs):
        raise ValueError('Duplicate reverification keys')
    if any(j[1] not in bank for j in jobs):
        raise ValueError('Reverification record outside selected bank')
    return bank, jobs


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['qwen','deepseek'],required=True)
    p.add_argument('--family',choices=['onestep','sequential','precision','fresh-noi'],required=True)
    p.add_argument('--kimina-url');p.add_argument('--output',type=Path)
    p.add_argument('--dry-run',action='store_true',help='Report full coverage without a verifier, output file, or runner dependencies')
    p.add_argument('--timeout',type=int,default=120);p.add_argument('--workers',type=int,default=8);p.add_argument('--limit',type=int)
    a=p.parse_args()
    if a.limit is not None and a.limit < 1:p.error('--limit must be positive')
    bank,jobs=collect_jobs(a.model,a.family)
    if a.dry_run:
        from collections import Counter
        print(json.dumps(dict(model=a.model,family=a.family,records=len(jobs),
            recorded_successes=sum(j[3]=='PASS' for j in jobs),
            missing_saved_proofs=sum(not j[2] for j in jobs),
            by_source=dict(Counter(j[0].split('|')[0] for j in jobs)),
            selected_records=min(a.limit,len(jobs)) if a.limit else len(jobs)),indent=2))
        return
    if not a.kimina_url or not a.output:p.error('--kimina-url and --output are required unless --dry-run')
    if a.output.exists():raise SystemExit('Output already exists; choose a new path.')
    from verify_run import verify_codes,verdict
    from lean_extract import extract_lean_code
    a.output.parent.mkdir(parents=True,exist_ok=True)
    if a.limit:jobs=jobs[:a.limit]
    def work(job):
        key,pid,code,old=job
        if not code:return dict(key=key,old=old,new='FAIL',error='no saved proof')
        proof=extract_lean_code(code,bank[pid]['statement'])
        result=verify_codes(a,[{'custom_id':key,'proof':proof}]);new,error=verdict(result.get(key,{'error':'missing verifier result'}))
        return dict(key=key,old=old,new=new,error=error,changed=old!=new)
    with a.output.open('x') as f,ThreadPoolExecutor(max_workers=a.workers) as ex:
        for r in ex.map(work,jobs):f.write(json.dumps(r)+'\n');f.flush()
    print('Reverified',len(jobs),'proofs into',a.output)


if __name__=='__main__':main()
