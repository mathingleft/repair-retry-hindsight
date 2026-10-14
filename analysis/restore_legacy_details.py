"""Recount secondary details and regenerate the original similarity/length panels.

Reads frozen artifacts only. Uses the source plotting definitions for Fig. A3
(all parseable samples, not the success-conditioned subgroup statistic).
"""
import collections
import importlib.util
import json
import sys

from results_figures import ROOT, PAPER, read, save

sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('source_figures', ROOT/'figures.py')
source = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source)


def main():
    q = source.qwen(str(ROOT))
    arms = ['bare', 'I', 'N', 'F', 'F_E', 'F_R', 'F_C', 'F_CE']
    _, _, sim = source.copy_stats(q, arms)
    source.save = lambda fig, out, name: save(fig, 'results_similarity_length')
    source.figA3(str(ROOT), str(PAPER/'figures'), q, sim)
    log = read(ROOT/'out'/'sorry_reverify.jsonl')
    flips = collections.Counter(r['file'] for r in log if r['old']=='PASS' and r['new']!='PASS')
    report = {'similarity': sim, 'reverified_rows': len(log), 'flips_by_file': dict(flips)}
    depth = {}
    for a in ['assert','noise','f','blind','blind_I','blind_F']:
        rs = read(ROOT/'out'/f'depth_{a}.jsonl')
        success = {(r['id'],r['traj']) for r in rs if r['status']=='PASS' and r['attempt']<=8}
        depth[a] = success
    report['early_depth'] = {}
    for a,b in [('assert','f'),('f','blind')]:
        wins,losses = len(depth[a]-depth[b]),len(depth[b]-depth[a])
        report['early_depth'][a+' - '+b] = [len(depth[a]),len(depth[b]),wins,losses]
    report['depth_problem_counts'] = {}
    for a in ['blind','blind_I','blind_F']:
        c = collections.Counter(r['id'] for r in read(ROOT/'out'/f'depth_{a}.jsonl') if r['status']=='PASS')
        report['depth_problem_counts'][a] = {p:c[p] for p in ['algebra_amgm_sumasqdivbgeqsuma','amc12a_2013_p4']}
    (PAPER/'analysis'/'restored_details_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
