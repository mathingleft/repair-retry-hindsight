"""Fresh-seed replicate of F_noI (out_noi_r2/, seeds 22072701+k) against the original F_noI (out_noi/), I and F (out/).

    python analyze_noi_r2.py
"""
import collections, json, os, random
from analyze_ext import load, contrast
HERE = os.path.dirname(os.path.abspath(__file__))
def main():
    npass = collections.defaultdict(collections.Counter); solved = collections.defaultdict(set); seen = collections.Counter()
    load("out", npass, solved, seen); load("out_ext", npass, solved, seen); load("out_noi", npass, solved, seen)
    # the replicate is stored under the same arm name; rename on load
    for l in open(os.path.join(HERE, "out_noi_r2", "verified.jsonl")):
        d = json.loads(l); seen["F_noI_r2"] += 1
        if d["status"] == "PASS": npass["F_noI_r2"][d["id"]] += 1; solved["F_noI_r2"].add(d["id"])
    pids = sorted(json.loads(l)["id"] for l in open(os.path.join(HERE, "bank.jsonl")) if not json.loads(l).get("is_framing_holdout"))
    print("arm        rows  solved  passes")
    for a in ("bare", "I", "F", "F_noI", "F_noI_r2"): print(f"{a:9s} {seen[a]:5d} {len(solved[a]):6d} {sum(npass[a].values()):7d}")
    rng = random.Random(1)
    print("\nreplicate (fresh seeds) vs original and vs the reference arms:")
    for x, y in (("F_noI_r2", "F_noI"), ("F_noI_r2", "I"), ("F_noI_r2", "F"), ("F_noI_r2", "bare"), ("F_noI", "I")):
        print(contrast(x, y, npass, solved, pids, rng))
    both = solved["F_noI"] & solved["F_noI_r2"]; print(f"\nproblems solved by both F_noI runs: {len(both)}; by either: {len(solved['F_noI'] | solved['F_noI_r2'])}")
if __name__ == "__main__":
    main()
