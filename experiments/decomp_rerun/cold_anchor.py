"""Cold-pool consistency check: does the recovered ~256-draw cold pool predict the rerun's `bare` arm?

Expected solved-at-k=4 under iid draws from the pool: sum_i 1-(1-p_i)^4 (Poisson-binomial sd alongside);
expected passes: sum_i 4 p_i. Also reports the two transcriptions of the pool present in the repo
(bank.jsonl `base_pass_rate_k256` vs depth/depth_bank.jsonl `base_pass/base_n`) and the new-stack cold draws
that the depth arms already contain (attempt 1 of every trajectory is the bare task prompt).

    python cold_anchor.py [outdir=out]
"""
import collections, json, math, os, sys

def main(outdir="out"):
    here = os.path.dirname(os.path.abspath(__file__))
    bank = {r["id"]: r for r in map(json.loads, open(os.path.join(here, "bank.jsonl")))}
    bare = collections.defaultdict(list)
    for l in open(os.path.join(outdir, "verified.jsonl")):
        x = json.loads(l)
        if x["arm"] == "bare": bare[x["id"]].append(x["status"] == "PASS")
    pids = sorted(bare); p = {q: bank[q]["base_pass_rate_k256"] for q in pids}
    es = sum(1 - (1 - p[q]) ** 4 for q in pids); sd = math.sqrt(sum((1 - (1 - p[q]) ** 4) * (1 - p[q]) ** 4 for q in pids))
    os_ = sum(any(bare[q]) for q in pids)
    print(f"{len(pids)} problems; pool zero-rate problems {sum(1 for q in pids if p[q] == 0)}, max pool rate {max(p.values()):.3f}")
    print(f"bare solved: expected from pool sum(1-(1-p)^4) = {es:.1f} (sd {sd:.2f}), observed {os_}  (z = {(os_ - es) / sd:+.1f})")
    print(f"bare passes: expected sum(4p) = {sum(4 * p[q] for q in pids):.1f}, observed {sum(sum(bare[q]) for q in pids)} / {4 * len(pids)}")
    z = [q for q in pids if p[q] == 0]
    print(f"pool-zero problems solved by bare: {sum(1 for q in z if any(bare[q]))} / {len(z)}: {[q for q in z if any(bare[q])]}")
    # two transcriptions of the pool
    dbank = [json.loads(l) for l in open(os.path.join(here, "depth", "depth_bank.jsonl"))]
    print("\npool transcriptions for the 11 depth problems: depth_bank base_pass/base_n | bank.jsonl base_pass_rate_k256")
    for b in dbank:
        r2 = bank.get(b["id"], {}).get("base_pass_rate_k256")
        flag = "" if r2 is None or abs(b["base_pass"] / b["base_n"] - r2) < 0.005 else "   <-- differ"
        print(f"  {b['id']:34} {b['base_pass']:3}/{b['base_n']} = {b['base_pass'] / b['base_n']:.4f} | {r2}{flag}")
    # new-stack cold draws inside the depth arms
    print("\nnew-stack cold draws (depth attempt 1 = bare task prompt), vs pool expectation per draw:")
    for a in ("assert", "f", "noise"):
        path = os.path.join(outdir, f"depth_{a}.jsonl")
        if not os.path.exists(path): continue
        n = c = 0; exp = 0.0
        for l in open(path):
            x = json.loads(l)
            if x["attempt"] == 1:
                n += 1; c += x["status"] == "PASS"
                b = next(b for b in dbank if b["id"] == x["id"]); exp += b["base_pass"] / b["base_n"]
        print(f"  {a:7} {c}/{n} passes; pool (depth_bank rates) expects {exp:.1f}")

if __name__ == "__main__":
    main(*sys.argv[1:2])
