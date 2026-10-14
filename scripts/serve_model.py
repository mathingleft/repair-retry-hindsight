"""Explicit, pinned vLLM launch recipe; run inside the chosen GPU environment."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['qwen','deepseek'],required=True)
    p.add_argument('--tp',type=int,required=True);p.add_argument('--kv-cache',choices=['auto','fp8'],default='auto')
    p.add_argument('--port',type=int,default=8017);p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    pins=json.loads((ROOT/'environments/revisions.json').read_text());q=a.model=='qwen'
    repo='Qwen/Qwen3.6-27B' if q else 'deepseek-ai/DeepSeek-Prover-V2-7B'
    cmd=['vllm','serve',repo,'--revision',pins[a.model],'--tokenizer',str(ROOT/'tokenizers'/a.model),
         '--served-model-name','qwen3.6-27b' if q else 'dsp-v2-7b','--dtype','bfloat16','--tensor-parallel-size',str(a.tp),
         '--kv-cache-dtype',a.kv_cache,'--max-model-len','131072' if q else '32768','--max-num-seqs','64',
         '--disable-custom-all-reduce','--port',str(a.port),'--generation-config','vllm',
         '--override-generation-config',json.dumps(dict(temperature=.7,top_k=20 if q else -1,top_p=.95 if q else 1.0))]
    print(shlex.join(cmd),flush=True)
    if not a.dry_run:subprocess.run(cmd,check=True)


if __name__=='__main__':main()
