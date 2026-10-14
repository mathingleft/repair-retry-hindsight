"""Audit token-cost coverage without modifying experimental artifacts.

Run with a Python environment containing transformers, e.g. vllm028/bin/python.
Counts are tokenizer reconstructions, NOT API usage, wall time, or FLOPs.
Sequential extracted-code counts omit prose and cannot stand in for output cost.
"""
import argparse
from collections import defaultdict
import importlib.util
import json
import os
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1] / 'experiments/decomp_rerun'
DEPTH = ROOT / 'out'
# The audit used a local snapshot of Qwen/Qwen3.6-27B at revision 6a9e13bd; the bundled
# tokenizer files are that revision's (restored by scripts/unpack_data.py).
TOKENIZER = os.environ.get('QWEN_TOKENIZER', str(Path(__file__).resolve().parents[1]/'tokenizers/qwen'))


def rows(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def summary(values):
    return dict(n=len(values), total=sum(values), mean=statistics.mean(values),
                median=statistics.median(values), maximum=max(values)) if values else dict(n=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tokenizer', default=TOKENIZER)
    args = parser.parse_args()
    # Qwen's original builder used no DeepSeek task header.
    os.environ.pop('DECOMP_TASK_HEADER', None)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    def ntok(text):
        return len(tokenizer.encode(text or '', add_special_tokens=False))
    def prompt_tokens(text):
        return ntok(tokenizer.apply_chat_template([dict(role='user', content=text)],
                    tokenize=False, add_generation_prompt=True, enable_thinking=False))

    spec = importlib.util.spec_from_file_location('audit_arms', ROOT / 'arms.py')
    arms = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(arms)
    bank = {r['id']: r for r in rows(ROOT / 'bank.jsonl') if not r['is_framing_holdout']}
    reflections = {r['id']: r for r in rows(ROOT / 'out/reflections.jsonl') if r['id'] in bank}
    report = dict(tokenizer=args.tokenizer, caveats=[
        'API usage was not retained; one-step counts reconstruct visible text and rendered prompts.',
        'Output retokenization can differ from generated token IDs; invisible stop tokens are omitted.',
        'Each one-step condition uses all four sampled outputs, without early stopping.',
        'Reflection is generated once per problem and shared across its four proof samples.',
        'Sequential output logs retain extracted code only, not complete generated responses.',
        'Prompt tokens measure input volume, not equivalent compute to output tokens.',
        'Failure-bank creation, verifier cost, infrastructure retries, and unresolved unlogged requests are excluded.',
        'No DeepSeek tokenizer count is substituted with Qwen tokenization.'
    ], one_step={}, sequential={}, deepseek_coverage={})
    for directory in ['out', 'out_ext', 'out_ext2', 'out_noi']:
        groups = defaultdict(list)
        for row in rows(ROOT / directory / 'rollouts.jsonl'):
            if row['id'] in bank:
                groups[row['arm']].append(row)
        for arm, group in groups.items():
            keys = [(r['id'], r['sample_idx']) for r in group]
            assert len(set(keys)) == len(keys), (directory, arm, 'duplicate keys')
            input_counts, output_counts = [], []
            for row in group:
                prompt = arms.build_prompt(arm, bank[row['id']],
                          reflection=reflections.get(row['id'], {}).get('reflection'))
                input_counts.append(prompt_tokens(prompt))
                if row.get('text') is not None:
                    output_counts.append(ntok(row['text']))
            item = dict(source=str(ROOT / directory / 'rollouts.jsonl'), rows=len(group),
                        problems=len({r['id'] for r in group}),
                        prompt_tokens=summary(input_counts),
                        visible_output_tokens=summary(output_counts),
                        missing_outputs=len(group)-len(output_counts))
            if arm.startswith('F_R'):
                ids = {r['id'] for r in group}
                item['reflection_prompt_tokens_once_per_problem'] = summary([
                    prompt_tokens(arms.build_reflector_prompt(bank[pid])) for pid in ids])
                item['reflection_output_tokens_once_per_problem'] = summary([
                    ntok(reflections[pid]['reflection']) for pid in ids])
            report['one_step'][arm] = item
        print('Processed', directory, flush=True)
    for directory in ['out_m2', 'out_m2_ext', 'out_m2_ext2', 'out_m2_noi']:
        groups = defaultdict(list)
        for row in rows(ROOT / directory / 'verified.jsonl'):
            groups[row['arm']].append(row)
        for arm, group in groups.items():
            report['deepseek_coverage'][arm] = dict(rows=len(group),
                full_text_available=sum(r.get('text') is not None for r in group),
                api_usage_available=sum('usage' in r for r in group), tokenizer_not_audited=True)
    paths = [ROOT / 'out' / f'depth_{arm}.jsonl' for arm in ['assert','noise','f','blind','blind_I','blind_F']]
    paths += [DEPTH / f'depth_{arm}.jsonl' for arm in ['C','CE','CE_shuf']]
    for path in paths:
        data = rows(path)
        keys = [(r['id'], r['traj'], r['attempt']) for r in data]
        assert len(set(keys)) == len(keys), (path, 'duplicate keys')
        item = dict(source=str(path), rows=len(data),
                    full_output_available=sum('text' in r for r in data),
                    extracted_code_tokens_NOT_output_cost=summary([ntok(r['code']) for r in data if r.get('code')]),
                    recorded_prompt_tokens=summary([r['prompt_tokens'] for r in data if 'prompt_tokens' in r]),
                    recorded_history_tokens=summary([r['history_tokens'] for r in data if 'history_tokens' in r]))
        trajectories = defaultdict(list)
        for row in data:
            trajectories[(row['id'], row['traj'])].append(row)
        costs = []
        for group in trajectories.values():
            total = 0
            for row in sorted(group, key=lambda r: r['attempt']):
                total += row.get('prompt_tokens', 0)
                if row['status'] == 'PASS':
                    break
            if 'prompt_tokens' in group[0]:
                costs.append(total)
        item['recorded_input_tokens_through_success_or_observed_end'] = summary(costs)
        report['sequential'][path.stem] = item
        print('Processed', path.name, flush=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print('Wrote', args.output)


if __name__ == '__main__':
    main()
