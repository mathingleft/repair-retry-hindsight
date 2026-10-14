"""Per-sample success versus instruction-only retrying (Appendix, "Per-sample success").

Pass@4 counts a problem once however many of its four samples pass. Here the
outcome is each problem's per-sample pass rate (passing samples / 4), and every
one-step condition is compared with I, paired by problem. Exploratory and
unadjusted. Percentile problem bootstrap, numpy default_rng(20260906), 10,000
resamples (one fresh index matrix per contrast, the camera-ready one-step
convention). Reads only frozen records under paper/artifact/; writes
out/per_sample.json.
"""
import collections, json
from pathlib import Path
import numpy as np

from artifact_path import ARTIFACT
ROOT = ARTIFACT/'experiments/decomp_rerun'
OUT = Path(__file__).resolve().parent / 'out'
SEED, B = 20260906, 10000
MODELS = {'qwen': ('bank.jsonl', ['out', 'out_ext', 'out_ext2', 'out_ext3', 'out_noi']),
          'deepseek': ('out_m2/bank2.jsonl', ['out_m2', 'out_m2_ext', 'out_m2_ext2', 'out_m2_ext3', 'out_m2_noi'])}
DIAGNOSTIC = {'F_E', 'F_E_shuf', 'F_E_match', 'F_R', 'F_E_noI', 'F_E_shuf_noI', 'F_R_noI'}
CODE = {'F_C', 'F_CE', 'F_CE_shuf', 'F_CE_match', 'F_C_other', 'F_C_noI', 'F_CE_noI'}
CONTENT_FREE = {'N', 'N2', 'N3', 'N4', 'N5', 'N_noI', 'F', 'F_noI', 'S', 'P', 'D', 'bare'}


def read(p): return [json.loads(l) for l in open(p) if l.strip()]


def main():
    result = {}
    for model, (bankfile, dirs) in MODELS.items():
        ids = [r['id'] for r in read(ROOT / bankfile) if not r.get('is_framing_holdout')]
        keep = set(ids); passes = collections.defaultdict(collections.Counter); n = collections.Counter()
        for d in dirs:
            for r in read(ROOT / d / 'verified.jsonl'):
                if r['id'] in keep:
                    n[r['arm'], r['id']] += 1; passes[r['arm']][r['id']] += r['status'] == 'PASS'
        arms = sorted({a for a, _ in n})
        rate = {a: np.array([passes[a][p] / 4 for p in ids]) for a in arms
                if all(n[a, p] == 4 for p in ids)}
        block = {}
        for a in sorted(rate):
            if a == 'I': continue
            d = rate[a] - rate['I']
            ix = np.random.default_rng(SEED).integers(0, len(ids), (B, len(ids)))
            q = np.quantile(d[ix].mean(axis=1), [.025, .975])
            group = 'diagnostic' if a in DIAGNOSTIC else 'code' if a in CODE else 'content_free' if a in CONTENT_FREE else 'other'
            block[a] = dict(group=group, diff_pp=100 * float(d.mean()), ci95_pp=[100 * float(q[0]), 100 * float(q[1])])
        result[model] = dict(problems=len(ids), I_rate_pct=100 * float(rate['I'].mean()), contrasts=block)
        print(f"== {model} ({len(ids)} problems), I per-sample rate {result[model]['I_rate_pct']:.1f}%")
        for a, v in sorted(block.items(), key=lambda kv: (kv[1]['group'], -kv[1]['diff_pp'])):
            lo, hi = v['ci95_pp']
            print(f"  {v['group']:12s} {a:14s} {v['diff_pp']:+6.1f} [{lo:+5.1f},{hi:+5.1f}]{'  *' if lo > 0 or hi < 0 else ''}")
    OUT.mkdir(exist_ok=True)
    (OUT / 'per_sample.json').write_text(json.dumps(result, indent=1) + '\n')


if __name__ == '__main__':
    main()
