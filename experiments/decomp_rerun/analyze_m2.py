"""Second-model analysis: bare / I / F / F_E / F_C / F_CE on the model's own bank (bank2.jsonl).

    python analyze_m2.py [outdir]        # default out_m2/

Contrasts (per-problem pass-count sign test = primary, McNemar on solved, paired bootstrap CI), plus
copy rate of the frozen predecessor per arm. Problems = bank2 (those with a failing cold sample).
"""
import collections, difflib, json, os, random, re, statistics, sys
from math import comb
HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = ("bare", "I", "F", "F_E", "F_C", "F_CE")
def sign(b, c):
    n = b + c
    if n == 0: return 1.0
    k = min(b, c); return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)
def body(code):
    if code is None: return None
    i = code.find(":="); b = code[i + 2:] if i >= 0 else code
    b = re.sub(r"--[^\n]*", "", b); b = re.sub(r"/-.*?-/", "", b, flags=re.S)
    return re.sub(r"\s+", " ", b).strip()
def main(outdir):
    bank2 = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(outdir, "bank2.jsonl"))}
    pids = sorted(bank2); pred = {p: body(bank2[p]["predecessor_code"]) for p in pids}
    npass = collections.defaultdict(collections.Counter); solved = collections.defaultdict(set)
    rows = collections.defaultdict(lambda: collections.defaultdict(list)); seen = collections.Counter()
    for l in open(os.path.join(outdir, "verified.jsonl")):
        try: d = json.loads(l)
        except Exception: continue
        if d["id"] not in bank2: continue
        seen[d["arm"]] += 1; rows[d["arm"]][d["id"]].append(d)
        if d["status"] == "PASS": npass[d["arm"]][d["id"]] += 1; solved[d["arm"]].add(d["id"])
    print(f"model: {bank2[pids[0]].get('source_model')}   problems: {len(pids)}   base-zero (0/4 cold): {sum(bank2[p]['base_pass_rate_k4']==0 for p in pids)}")
    print("arm    rows  solved  passes  exact-copy-of-predecessor  mean-sim")
    for a in ARMS:
        if not seen[a]: continue
        # bare: the predecessor IS one of its own 4 samples; exclude that sample from the copy statistics
        bs = [(body(d.get("lean_code")), pred[p]) for p in pids for d in rows[a][p]
              if not (a == "bare" and d["sample_idx"] == bank2[p]["predecessor_rollout_idx"])]
        cp = sum(b is not None and b == q for b, q in bs)
        sim = statistics.mean(difflib.SequenceMatcher(None, (b or "")[:3000], q[:3000]).ratio() for b, q in bs)
        print(f"  {a:5s} {seen[a]:4d}  {len(solved[a]):4d}   {sum(npass[a].values()):4d}     {cp:3d} ({100*cp/max(len(bs),1):4.1f}%)             {sim:.3f}")
    rng = random.Random(1); B = 20000
    def have(*a):
        for x in a:
            if seen[x] != 4 * len(pids): print(f"  [skip: {x} has {seen[x]} rows, need {4*len(pids)}]"); return False
        return True
    print("\ncontrasts (pass-count sign test primary)")
    for x, y in (("I", "bare"), ("F", "bare"), ("F", "I"), ("F_E", "I"), ("F_E", "F"), ("F_C", "F"), ("F_C", "I"), ("F_CE", "F_E"), ("F_CE", "I")):
        if not have(x, y): continue
        b = sum(npass[x][p] > npass[y][p] for p in pids); c = sum(npass[x][p] < npass[y][p] for p in pids)
        bx = len(solved[x] - solved[y]); cy = len(solved[y] - solved[x])
        obs = (len(solved[x]) - len(solved[y])) / len(pids); ds = []
        for _ in range(B):
            S = [pids[rng.randrange(len(pids))] for _ in pids]
            ds.append(sum((p in solved[x]) - (p in solved[y]) for p in S) / len(pids))
        ds.sort()
        print(f"  {x:5s} vs {y:5s}: solved {len(solved[x]):3d} vs {len(solved[y]):3d} ({100*obs:+5.1f}pp [{100*ds[int(.025*B)]:+5.1f}, {100*ds[int(.975*B)]:+5.1f}])  "
              f"pass-count {b}/{c} p={sign(b,c):.3g}   McNemar {bx}/{cy} p={sign(bx,cy):.3g}")
    if have("F_C", "F"):
        cpr = {a: {p: sum(body(d.get("lean_code")) == pred[p] for d in rows[a][p]) for p in pids} for a in ("F", "F_C", "F_E", "F_CE")}
        for x, y in (("F_C", "F"), ("F_CE", "F_E")):
            b = sum(cpr[x][p] > cpr[y][p] for p in pids); c = sum(cpr[x][p] < cpr[y][p] for p in pids)
            print(f"  copy-rate {x} > {y} on {b} problems, < on {c}, p={sign(b,c):.3g}")
    print("\nnote: bare is selected against by construction (a problem enters bank2 only if >=1 of its 4 cold samples failed); bare copy stats exclude the predecessor sample.")
if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "out_m2"))
