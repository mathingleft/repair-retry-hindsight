"""Portable orchestration of the original generation/verification runners.

Uses a caller-provided model service and verifier. Never launches either service
implicitly. Outputs must be outside the frozen experiment directory.
"""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import shlex

ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'experiments/decomp_rerun'


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['qwen','deepseek'],required=True)
    p.add_argument('--family',choices=['onestep','precision','sequential','fresh-noi'],required=True)
    p.add_argument('--base-url',required=True);p.add_argument('--kimina-url',required=True)
    p.add_argument('--served-model',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--concurrency',type=int,default=8)
    p.add_argument('--dry-run',action='store_true',help='Print commands without launching or creating output files.')
    a=p.parse_args();a.output=a.output.resolve()
    if a.output==DATA or DATA in a.output.parents:raise SystemExit('Choose an output outside the frozen data tree.')
    if not a.dry_run:a.output.mkdir(parents=True,exist_ok=True)
    os.environ['DECOMP_MODEL_DIR']=str(ROOT/'tokenizers'/a.model)
    os.environ['DECOMP_TASK_HEADER']='1' if a.model=='deepseek' else '0'
    os.environ['DECOMP_DEPTH_BANK']=str(DATA/'depth'/('depth_bank.jsonl' if a.model=='qwen' else 'depth_bank_dsp.jsonl'))
    os.environ['DECOMP_SERVED_MODEL']=a.served_model
    bank=DATA/('bank.jsonl' if a.model=='qwen' else 'out_m2/bank2.jsonl')
    def run(script,*args):
        cmd=[sys.executable,str(DATA/script),*map(str,args)]
        print(shlex.join(cmd),flush=True)
        if not a.dry_run:subprocess.run(cmd,check=True)
    if a.family in ['onestep','precision','fresh-noi']:
        if a.family in ['precision','fresh-noi'] and a.model!='qwen':raise SystemExit('This campaign is Qwen only.')
        arms='bare,I,F,F_C,F_E,F_CE,F_E_shuf,F_CE_shuf,F_C_other,F_E_match,F_CE_match,N,F_noI,F_C_noI,F_E_noI,F_CE_noI,F_E_shuf_noI'
        if a.model=='qwen':arms+=',F_R,F_R_noI,N2,N3,N4,N5,S,P,D,N_noI'
        configs=[(a.output,22072601,arms)] if a.family=='onestep' else [(a.output/f'b{i}',seed,'F_E,F_E_shuf') for i,seed in enumerate([22073001,22073101,22073201,22073301],1)]
        if a.family=='fresh-noi':configs=[(a.output,22072701,'F_noI')]
        for out,seed,aa in configs:
            if not a.dry_run:out.mkdir(exist_ok=True)
            base=['--base-url',a.base_url,'--model',a.served_model,'--bank',bank,'--outdir',out,'--concurrency',a.concurrency]
            if 'F_R' in aa:run('run_gen.py','--phase','reflections',*base)
            run('run_gen.py','--phase','main','--arms',aa,'--seed-base',seed,*base)
            run('verify_run.py','--kimina-url',a.kimina_url,'--outdir',out,'--bank',bank)
    else:
        mem=a.output/'memoryless'
        if not a.dry_run:mem.mkdir(exist_ok=True)
        run('depth/memoryless_I.py','--phase','gen','--outdir',mem,'--base-url',a.base_url,'--model',a.served_model,'--concurrency',a.concurrency)
        run('depth/memoryless_I.py','--phase','verify','--outdir',mem,'--kimina-url',a.kimina_url)
        run('depth/depth_run.py','--arm','f','--outdir',a.output,'--base-url',a.base_url,'--kimina-url',a.kimina_url,'--concurrency',a.concurrency)
        # CE_s uses fixed partners; CE_m consumes the completed CE error pool;
        # E_s consumes f. Keep CE_s as the ordinary cross-problem shuffle.
        for arm in ['C','CE','CE_s','CE_m','E_s']:
            run('depth/code_arms_run.py','--arm',arm,'--outdir',a.output/'full','--depth-outdir',a.output,'--base-url',a.base_url,'--model',a.served_model,'--kimina-url',a.kimina_url,'--concurrency',a.concurrency,'--ce-log',a.output/'full/CE_rows.jsonl','--f-log',a.output/'depth_f.jsonl')
    print('Fresh experiment outputs:',a.output)


if __name__=='__main__':main()
