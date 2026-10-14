"""CPU-only tokenizer recount from original visible responses and rebuilt prompts."""
import collections
import os
import sys
from common import DATA,ARTIFACT,read,csvout


def main():
    os.environ.pop('DECOMP_TASK_HEADER',None)
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(ARTIFACT/'tokenizers/qwen',local_files_only=True)
    sys.path.insert(0,str(DATA));import arms
    bank={r['id']:r for r in read(DATA/'bank.jsonl') if not r['is_framing_holdout']}
    refl={r['id']:r['reflection'] for r in read(DATA/'out/reflections.jsonl')}
    def ntok(t):return len(tokenizer.encode(t or '',add_special_tokens=False))
    def prompt_tokens(t):return ntok(tokenizer.apply_chat_template([{'role':'user','content':t}],tokenize=False,add_generation_prompt=True,enable_thinking=False))
    inputs={};rows=[]
    for directory in ['out','out_ext','out_ext2','out_noi']:
        for r in read(DATA/directory/'rollouts.jsonl'):
            if r['id'] not in bank:continue
            key=r['arm'],r['id']
            if key not in inputs:inputs[key]=prompt_tokens(arms.build_prompt(r['arm'],bank[r['id']],refl.get(r['id'])))
            rows.append(dict(kind='proof',arm=r['arm'],problem_id=r['id'],sample_idx=r['sample_idx'],input_tokens=inputs[key],output_tokens=ntok(r.get('text'))))
        print('Retokenized',directory,flush=True)
    for pid,b in bank.items():rows.append(dict(kind='reflection',arm='reflection',problem_id=pid,sample_idx=0,input_tokens=prompt_tokens(arms.build_reflector_prompt(b)),output_tokens=ntok(refl[pid])))
    csvout('token_counts.csv',rows)


if __name__=='__main__':main()
