"""Build the per-attempt token table behind the token-matched sequential comparison.

Writes data/sequential_attempt_tokens.csv.xz: one row per sequential attempt (both models, every reported
arm plus Qwen's noise and assert controls) and one row per memoryless I draw, with prompt, response, and
extracted-proof token counts. token_matched.py reads only this table and the compact released records.

Sources, per arm (raw labels; manuscript E = f, E_m = E_s, I = memI):
  memI                    exact runner-logged counts from the released depth_memI.jsonl records
  C, CE, CE_s, CE_m, E_s  exact runner-logged counts from the full sequential traces
                          (out_codearms/<arm>_rows.jsonl.gz, out_dsp_codearms/<arm>_rows.jsonl.gz). These ten
                          files are listed as omitted in manifests/package.json; point FULL_TRACES at a directory
                          holding them (default: the artifact's experiments/decomp_rerun).
  f, noise, assert        only compact records were retained: prompts are re-rendered with the runner's own
                          depth_run.render() history and FIFO-eviction rule, prefixed by the model's chat template,
                          and counted with the bundled tokenizer; response tokens are left blank (token_matched.py
                          predicts them from code_tokens).
code_tokens is the extracted proof counted with the bundled tokenizer (no special tokens).

Counting conventions follow the runners: prompt = prompt_token_count, else prompt_token_count_tokenizer;
response = response_token_count if logged, else response_token_count_tokenizer. As a check of the re-rendering,
the first 1,500 compact CE attempts of each model are re-rendered with code_arms_run.render() and compared with
the logged tokenizer prompt counts (written to out/attempt_tokens_build.json).

Needs `transformers` (tokenizer and chat template only; no model weights) plus the runner modules' imports
(`aiohttp`, `requests`). The table itself is committed, so token_matched.py does not need any of these.
"""
import collections
import csv
import gzip
import io
import json
import lzma
import os
import sys
from pathlib import Path

from artifact_path import ARTIFACT

HERE = Path(__file__).resolve().parent
DATA = ARTIFACT/'experiments/decomp_rerun'
TRACES = Path(os.environ.get('FULL_TRACES', DATA))
TABLE = HERE/'data/sequential_attempt_tokens.csv.xz'
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1'); sys.dont_write_bytecode = True
sys.path.insert(0, str(DATA/'depth')); sys.path.insert(0, str(DATA))
import depth_run, code_arms_run  # noqa: E402  (runner prompt builders)

MODELS = {
    'qwen': dict(depth='out', rows='out_codearms', bank='depth/depth_bank.jsonl', tok='tokenizers/qwen', header=False,
                 rebuilt=['f', 'noise', 'assert']),
    'deepseek': dict(depth='out_dsp', rows='out_dsp_codearms', bank='depth/depth_bank_dsp.jsonl', tok='tokenizers/deepseek',
                     header=True, rebuilt=['f']),
}
RUNNER_ARMS = ['C', 'CE', 'CE_s', 'CE_m', 'E_s']
COLUMNS = ['model', 'arm', 'source', 'problem_id', 'trajectory', 'attempt', 'row', 'passed',
           'prompt_tokens', 'response_tokens', 'code_tokens']


def read(path):
    op = gzip.open if str(path).endswith('.gz') else open
    with op(path, 'rt', encoding='utf-8') as f:
        return [json.loads(l) for l in f if l.strip()]


class Counter:
    def __init__(self, model):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(str(ARTIFACT/MODELS[model]['tok']), local_files_only=True)

    def text(self, s):
        return len(self.tok(s, add_special_tokens=False)['input_ids']) if s else 0

    def chat(self, user_msg):
        # both runners passed enable_thinking=False; the prover template ignores it
        s = self.tok.apply_chat_template([{'role': 'user', 'content': user_msg}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)
        return len(self.tok(s, add_special_tokens=False)['input_ids'])


def set_header(model):
    if MODELS[model]['header']:
        os.environ['DECOMP_TASK_HEADER'] = '1'
    else:
        os.environ.pop('DECOMP_TASK_HEADER', None)


def main():
    rows, checks = [], {}
    for model, cfg in MODELS.items():
        tc = Counter(model); set_header(model)
        bank = {r['id']: r for r in read(DATA/cfg['bank'])}
        seen = collections.Counter()
        for d in read(DATA/cfg['depth']/'depth_memI.jsonl'):
            pid = d['problem_id']
            rows.append(dict(model=model, arm='memI', source='exact', problem_id=pid, trajectory=d['trajectory_id'],
                             attempt=d['attempt_id'], row=seen[pid], passed=int(d['verification_result'] == 'PASS'),
                             prompt_tokens=d['prompt_token_count'] or 0, response_tokens=d['response_token_count'] or 0,
                             code_tokens=tc.text(d['extracted_proof'])))
            seen[pid] += 1
        for arm in RUNNER_ARMS:
            full = read(TRACES/cfg['rows']/f'{arm}_rows.jsonl.gz')
            full.sort(key=lambda d: (d['problem_id'], d['trajectory_id'], d['attempt_id']))
            compact = {(d['id'], d['traj'], d['attempt']): d['status'] for d in read(DATA/cfg['depth']/f'depth_{arm}.jsonl')}
            assert len(compact) == len(full), (model, arm, len(compact), len(full))
            for d in full:
                key = (d['problem_id'], d['trajectory_id'], d['attempt_id'])
                ok = d['verification_result'] == 'PASS'
                assert (compact[key] == 'PASS') == ok, (model, arm, key)   # full and released compact records agree
                pt = d.get('prompt_token_count') or d.get('prompt_token_count_tokenizer') or 0
                rt = d.get('response_token_count') if d.get('response_token_count') is not None else d.get('response_token_count_tokenizer') or 0
                rows.append(dict(model=model, arm=arm, source='exact', problem_id=key[0], trajectory=key[1], attempt=key[2],
                                 row='', passed=int(ok), prompt_tokens=pt, response_tokens=rt,
                                 code_tokens=tc.text(d.get('extracted_proof'))))
        for arm in cfg['rebuilt']:
            comp = read(DATA/cfg['depth']/f'depth_{arm}.jsonl')
            comp.sort(key=lambda d: (d['id'], d['traj'], d['attempt']))
            hist = collections.defaultdict(list)
            for d in comp:
                key = (d['id'], d['traj'])
                pt = tc.chat(depth_run.render(arm, bank[d['id']], hist[key]))
                rows.append(dict(model=model, arm=arm, source='rebuilt_prompt', problem_id=d['id'], trajectory=d['traj'],
                                 attempt=d['attempt'], row='', passed=int(d['status'] == 'PASS'), prompt_tokens=pt,
                                 response_tokens='', code_tokens=tc.text(d.get('code'))))
                hist[key].append((d.get('error') or '')[:1200])          # the runner stored capped diagnostics
        # re-rendering check on logged CE prompts
        logged = {(d['problem_id'], d['trajectory_id'], d['attempt_id']): d['prompt_token_count_tokenizer']
                  for d in read(TRACES/cfg['rows']/'CE_rows.jsonl.gz')}
        comp = read(DATA/cfg['depth']/'depth_CE.jsonl'); comp.sort(key=lambda d: (d['id'], d['traj'], d['attempt']))
        hist = collections.defaultdict(list); diffs = []
        for d in comp[:1500]:
            key = (d['id'], d['traj'])
            up, _ = code_arms_run.render('CE', bank[d['id']], hist[key])
            diffs.append(tc.chat(up) - logged[(d['id'], d['traj'], d['attempt'])])
            hist[key].append((d.get('code'), (d.get('error') or '')[:1200]))
        checks[model] = dict(ce_attempts_checked=len(diffs), max_abs_diff=max(abs(x) for x in diffs),
                             mean_diff=sum(diffs)/len(diffs))
        print(model, 'CE prompt re-render check:', checks[model], flush=True)
    TABLE.parent.mkdir(exist_ok=True)
    buf = io.StringIO(); w = csv.DictWriter(buf, fieldnames=COLUMNS, lineterminator='\n'); w.writeheader(); w.writerows(rows)
    TABLE.write_bytes(lzma.compress(buf.getvalue().encode(), preset=9))
    (HERE/'out').mkdir(exist_ok=True)
    (HERE/'out/attempt_tokens_build.json').write_text(json.dumps(dict(
        rows=len(rows), by_arm={f'{m}/{a}': n for (m, a), n in sorted(collections.Counter((r['model'], r['arm']) for r in rows).items())},
        prompt_rerender_check=checks), indent=2) + '\n')
    print('wrote', TABLE, len(rows), 'rows')


if __name__ == '__main__':
    main()
