"""CPU-only regression check of records offered to the optional Lean workflow."""
import sys
from collections import Counter
from pathlib import Path

from common import DATA, dump, read

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from reverify import collect_jobs


def main():
    report = {}
    for model in ('qwen', 'deepseek'):
        for family in ('onestep', 'sequential', 'precision', 'fresh-noi'):
            if model == 'deepseek' and family in ('precision', 'fresh-noi'):
                continue
            bank, jobs = collect_jobs(model, family)
            counts = Counter(j[0].split('|')[0] for j in jobs)
            if family == 'sequential':
                sub = 'out' if model == 'qwen' else 'out_dsp'
                # Independent inventory: includes every reported raw arm.
                expected = {arm: len(read(DATA/sub/f'depth_{arm}.jsonl'))
                            for arm in ('memI', 'C', 'CE', 'CE_m', 'CE_s', 'f', 'E_s')}
                assert dict(counts) == expected, (model, counts, expected)
                cem = [j for j in jobs if j[0].startswith('CE_m|')]
                assert (len(cem), sum(j[3] == 'PASS' for j in cem)) == (
                    (5865, 23) if model == 'qwen' else (5903, 25))
            if model == 'qwen' and family in ('onestep', 'fresh-noi'):
                expected = sum(r['id'] in bank for r in read(DATA/'out_noi_r2/verified.jsonl'))
                assert counts['out_noi_r2'] == expected == 668
            report[f'{model}/{family}'] = dict(records=len(jobs), by_source=dict(counts))
    dump('reverification_coverage.json', dict(status='PASS', families=report,
         scope='Record enumeration only; no Lean execution or new verdicts.'))
    print('PASS: reverification covers all seven sequential arms and fresh failure-notice replicate.')


if __name__ == '__main__':
    main()
