"""Interaction of diagnostic content x own-code presence in the sequential (depth) arms.

    python depth/analyze_interaction.py [outdir=out] [--nocode E_s|noise] [--code CE_s|CE_m] [--boot 10000] [--seed 0]

Question: does the semantic content of the verifier error help more when the model is conditioned on its own failed
proof than when it only receives error feedback without the proof?

    Delta_interaction(k) = (CE - CE_s)(k) - (f - X)(k)      X = the code-free wrong-diagnostic contrast (--nocode)

  CE    : own failed-code history + own (correct) diagnostic          (out/depth_CE.jsonl)
  CE_s  : own failed-code history + another problem's diagnostic       (out/depth_CE_s.jsonl)  [default --code]
  CE_m  : own failed-code history + class+length-matched wrong error  (out/depth_CE_m.jsonl)  [PREREG_CE_m_2026-09-12]
  f     : diagnostic-only history, own (correct) diagnostic            (out/depth_f.jsonl)
  E_s   : diagnostic-only history, class+length-matched wrong error    (out/depth_E_s.jsonl)   [default; PREREG_E_s_seq]
  noise : diagnostic-only scaffold, content-free random-word sentence  (out/depth_noise.jsonl) [post-hoc 2026-09-08 version]

Estimator per arm and problem i: p_arm,i(k) = #{trajectories with a PASS at attempt <= k} / #trajectories, the
depth-arm estimator of analyze_memI.py. Point estimate = mean over the 11 problems of the per-problem interaction.
Uncertainty: paired problem bootstrap — resample the 11 problems with replacement, carry every trajectory of each
sampled problem in all four arms, compute (CE - CE_s) and (f - noise) within the replicate, subtract; 10,000
resamples; percentile 95% CI. k in {1, 2, 4, 8, 16, 32, 64}; k = 64 is the headline. The two component effects
are reported with the same bootstrap, plus the paired sign test (problem as unit) on the per-problem interaction.
"""
import argparse, collections, json, os, random, sys
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
KS = [1, 2, 4, 8, 16, 32, 64]
ARMS = {"CE": "depth_CE.jsonl", "f": "depth_f.jsonl"}


def load(path):
    """-> {pid: {traj: first_pass_attempt or None}}"""
    out = collections.defaultdict(dict)
    for l in open(path):
        try: d = json.loads(l)
        except Exception: continue
        pid, t, k, st = d["id"], d["traj"], d["attempt"], d["status"]
        e = out[pid].setdefault(t, None)
        if st == "PASS" and (e is None or k < e): out[pid][t] = k
    return out


def p(trajs, k):
    return sum(1 for e in trajs.values() if e is not None and e <= k) / len(trajs)


def sign_test(pos, neg):
    n = pos + neg
    if n == 0: return 1.0
    m = min(pos, neg)
    return min(1.0, 2 * sum(comb(n, i) for i in range(m + 1)) / 2 ** n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir", nargs="?", default=os.path.join(os.path.dirname(HERE), "out"))
    ap.add_argument("--nocode", default="E_s", help="code-free wrong-diagnostic arm: E_s (preregistered) or noise")
    ap.add_argument("--code", default="CE_s", help="code-bearing wrong-diagnostic arm: CE_s (partner, position-matched) or CE_m (class+length matched)")
    ap.add_argument("--boot", type=int, default=10000); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    X = a.nocode; Y = a.code
    bank = [json.loads(l)["id"] for l in open((os.environ.get("DECOMP_DEPTH_BANK") or os.path.join(HERE, "depth_bank.jsonl"))) if l.strip()]
    arms = {n: load(os.path.join(a.outdir, f)) for n, f in ARMS.items()}
    arms[X] = load(os.path.join(a.outdir, f"depth_{X}.jsonl")); arms[Y] = load(os.path.join(a.outdir, f"depth_{Y}.jsonl"))
    for n, A in arms.items():
        miss = [pid for pid in bank if pid not in A]
        assert not miss, f"{n}: missing problems {miss}"
        print(f"{n:6s} {sum(len(A[pid]) for pid in bank):4d} trajectories, {sum(1 for pid in bank for e in A[pid].values() if e is not None):3d} solved by 64")
    print(f"\nDelta_interaction(k) = (CE - {Y}) - (f - {X}); paired problem bootstrap, n={len(bank)} problems, "
          f"{a.boot} resamples, seed {a.seed}\n")
    print(f"   k   CE     {Y:6s} f      {X:6s} | CE-{Y:5s} 95% CI            | f-{X:5s}  95% CI            | INTERACTION  95% CI            sign(+/-) p")
    rng = random.Random(a.seed)
    samples = [[rng.choice(bank) for _ in bank] for _ in range(a.boot)]
    for k in KS:
        P = {n: {pid: p(arms[n][pid], k) for pid in bank} for n in arms}
        def stat(pids, f):
            return sum(f(pid) for pid in pids) / len(pids)
        eff = {"code": lambda pid: P["CE"][pid] - P[Y][pid],
               "nocode": lambda pid: P["f"][pid] - P[X][pid]}
        eff["inter"] = lambda pid: eff["code"](pid) - eff["nocode"](pid)
        pt = {n: stat(bank, f) for n, f in eff.items()}
        boots = {n: sorted(stat(s, f) for s in samples) for n, f in eff.items()}
        ci = {n: (b[int(0.025 * a.boot)], b[int(0.975 * a.boot) - 1]) for n, b in boots.items()}
        per = [eff["inter"](pid) for pid in bank]
        pos, neg = sum(1 for x in per if x > 0), sum(1 for x in per if x < 0)
        tag = "  <- HEADLINE" if k == 64 else ""
        print(f"  {k:3d}  {stat(bank, lambda q: P['CE'][q]):.3f}  {stat(bank, lambda q: P[Y][q]):.3f}  "
              f"{stat(bank, lambda q: P['f'][q]):.3f}  {stat(bank, lambda q: P[X][q]):.3f}  | "
              f"{100*pt['code']:+6.1f}pp [{100*ci['code'][0]:+6.1f},{100*ci['code'][1]:+6.1f}] | "
              f"{100*pt['nocode']:+6.1f}pp [{100*ci['nocode'][0]:+6.1f},{100*ci['nocode'][1]:+6.1f}] | "
              f"{100*pt['inter']:+7.1f}pp [{100*ci['inter'][0]:+6.1f},{100*ci['inter'][1]:+6.1f}]  {pos}/{neg} p={sign_test(pos, neg):.3f}{tag}")
    print(f"\nper-problem at k=64 (CE, {Y}, f, {X} | CE-{Y} | f-{X} | interaction):")
    for pid in bank:
        v = {n: p(arms[n][pid], 64) for n in arms}
        c, nc = v["CE"] - v[Y], v["f"] - v[X]
        print(f"  {pid:32s} {v['CE']:.2f} {v[Y]:.2f} {v['f']:.2f} {v[X]:.2f} | {c:+.2f} | {nc:+.2f} | {c-nc:+.2f}")


if __name__ == "__main__":
    main()
