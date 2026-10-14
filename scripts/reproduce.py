"""Single entry point for CPU-only reproduction; never writes frozen inputs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
ANALYSIS=ROOT/'analysis'
OUT=ROOT/'generated'


def run(name,*args,cwd=None,stdout=None):
    print('Running',name,flush=True)
    target=ANALYSIS/name if (ANALYSIS/name).exists() else ROOT/'experiments/decomp_rerun'/name
    if stdout:
        with (OUT/stdout).open('w') as stream:subprocess.run([sys.executable,str(target),*args],cwd=cwd or ROOT,stdout=stream,check=True)
    else:subprocess.run([sys.executable,str(target),*args],cwd=cwd or ROOT,check=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--no-figures',action='store_true');a=p.parse_args()
    from unpack_data import main as unpack_data
    unpack_data()
    OUT.mkdir(exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR',str(OUT/'mpl-cache'))
    os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MPLBACKEND']='Agg'
    run('validate.py');run('check_reverification_coverage.py');run('analyze_onestep.py');run('analyze_exploratory_interactions.py');run('analyze_sequential.py');run('analyze_ce_m_matching.py')
    run('pooled_diagnostics.py',stdout='pooled_diagnostics.json');run('feedback_bounds.py',stdout='feedback_bounds.json');run('supporting.py')
    data=ROOT/'experiments/decomp_rerun'
    run('analyze.py','out','--exclude','algebra_amgm_sumasqdivbgeqsuma,amc12a_2013_p4,imo_1963_p5,mathd_algebra_276',cwd=data,stdout='original_family.txt')
    for model,args in [('qwen',['out','out_ext','out_ext2','out_ext3']),('deepseek',['out_m2','out_m2_ext','out_m2_ext2','out_m2_ext3','out_m2/bank2.jsonl'])]:
        run('analyze_ext3.py',*args,cwd=data,stdout=model+'_ext3.txt')
        ext2=args[:3]+(args[4:] if model=='deepseek' else [])
        run('analyze_ext2.py',*ext2,cwd=data,stdout=model+'_ext2.txt')
    if not a.no_figures:run('plot_onestep.py');run('plot_sequential.py')
    run('export_examples.py');run('check_results.py');run('check_reported_results.py')
    print('Reproduction complete. Outputs:',OUT)


if __name__=='__main__':main()
