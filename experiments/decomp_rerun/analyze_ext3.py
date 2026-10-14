"""EXT3 analysis (post-hoc, 2026-09-06): F_C_other, F_E_match, F_CE_match.

    python analyze_ext3.py [main] [ext] [ext2] [ext3] [bank]
    # Qwen  : python analyze_ext3.py out out_ext out_ext2 out_ext3
    # prover: python analyze_ext3.py out_m2 out_m2_ext out_m2_ext2 out_m2_ext3 out_m2/bank2.jsonl

Same machinery as analyze_ext2.py (exact paired sign test on per-problem pass counts, McNemar on solved,
paired bootstrap 95% CI on the solved-rate difference).
  F_C_other  vs F_C       : wrong problem's failed code vs own failed code (code CONTENT, code present)
  F_C_other  vs F         : does an unrelated failed proof hurt/help relative to no code at all?
  F_E_match  vs F_E       : class+length-matched wrong error vs own error (code absent)
  F_E_match  vs F_E_shuf  : matched vs random wrong error — is the error's value its class/shape?
  F_CE_match vs F_CE      : same, code present
  F_CE_match vs F_CE_shuf : same, code present
  F_CE_match vs F_C       : does a matched wrong error add/hurt on top of the code?
Copy channel: exact body copies of the OWN failed proof (F_C, F_CE, F_CE_match) and of the PARTNER's failed
proof (F_C_other): does the model transplant the unrelated proof?
"""
import collections, json, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_ext2 import body, sign, load, contrast
import arms

MATCH_OK = ("F_E_match", "F_CE_match")

def main(main_dir, ext_dir, ext2_dir, ext3_dir, bank=None):
    npass = collections.defaultdict(collections.Counter); solved = collections.defaultdict(set)
    seen = collections.Counter()
    for d in (main_dir, ext_dir, ext2_dir, ext3_dir): load(d, npass, solved, seen)
    bank = bank or os.path.join(os.path.dirname(os.path.abspath(__file__)), "bank.jsonl")
    rows = {json.loads(l)["id"]: json.loads(l) for l in open(bank)}
    pids = sorted(p for p, r in rows.items() if not r.get("is_framing_holdout"))
    rng = random.Random(1)
    ms = json.load(open(arms.match_path(bank)))["summary"]
    print(f"bank {bank}: {len(pids)} problems")
    print(f"match map {os.path.basename(arms.match_path(bank))}: classes {ms['classes']}, reuse max {ms['reuse']['max']}, "
          f"same-class {100*ms['same_class_frac']:.0f}%, tok-ratio median {ms['tok_ratio_median']:.2f} p90 {ms['tok_ratio_p90']:.2f}, "
          f"identical-text pairs {ms.get('identical_text_pairs')}")
    print("arm          rows  solved  passes")
    order = ("F", "F_C", "F_C_other", "F_E", "F_E_shuf", "F_E_match", "F_CE", "F_CE_shuf", "F_CE_match")
    for a in order:
        if seen[a]: print(f"  {a:10s} {seen[a]:4d}  {len(solved[a]):4d}   {sum(npass[a].values()):4d}")
    def have(*a):
        for x in a:
            if seen[x] != 4 * len(pids):
                print(f"  [skip: {x} has {seen[x]} rows, need {4*len(pids)}]"); return False
        return True
    print("\ncode content: wrong problem's failed code vs own")
    for x, y in (("F_C_other", "F_C"), ("F_C_other", "F"), ("F_C", "F")):
        if have(x, y): print(contrast(x, y, npass, solved, pids, rng))
    print("\nerror content, code ABSENT: matched wrong error vs own vs random wrong")
    for x, y in (("F_E_match", "F_E"), ("F_E_match", "F_E_shuf"), ("F_E_match", "F"), ("F_E", "F_E_shuf")):
        if have(x, y): print(contrast(x, y, npass, solved, pids, rng))
    print("\nerror content, code PRESENT: matched wrong error vs own vs random wrong")
    for x, y in (("F_CE_match", "F_CE"), ("F_CE_match", "F_CE_shuf"), ("F_CE_match", "F_C"), ("F_CE", "F_CE_shuf")):
        if have(x, y): print(contrast(x, y, npass, solved, pids, rng))
    print("  reading: match ~= own but shuf < own -> the error's worth is its class/length, not problem-specific detail;")
    print("           match ~= shuf < own          -> the specific content matters.")
    # copy channel
    own = {p: body(rows[p]["predecessor_code"]) for p in rows}
    partner = {p: body(rows[arms.shuffled_partner(p, bank)]["predecessor_code"]) for p in rows}
    ncopy = collections.defaultdict(collections.Counter); ncopy_partner = collections.defaultdict(collections.Counter)
    ntot = collections.Counter()
    for d in (main_dir, ext_dir, ext2_dir, ext3_dir):
        p = os.path.join(d, "verified.jsonl")
        if not os.path.exists(p): continue
        for l in open(p):
            r = json.loads(l)
            if r["arm"] not in ("F_C", "F_CE", "F_CE_shuf", "F_CE_match", "F_C_other") or r["id"] not in rows: continue
            ntot[r["arm"]] += 1; b = body(r.get("lean_code"))
            if b == own[r["id"]]: ncopy[r["arm"]][r["id"]] += 1
            if b == partner[r["id"]]: ncopy_partner[r["arm"]][r["id"]] += 1
    print("\nexact copies of a failed proof body (body-normalised): own predecessor / partner's (F_C_other's shown code)")
    for a in ("F_C", "F_CE", "F_CE_shuf", "F_CE_match", "F_C_other"):
        if ntot[a]:
            c = sum(ncopy[a].values()); cp = sum(ncopy_partner[a].values())
            print(f"  {a:10s} own {c:3d}/{ntot[a]} = {100*c/ntot[a]:4.1f}%   partner {cp:3d}/{ntot[a]} = {100*cp/ntot[a]:4.1f}%")
    for x, y in (("F_CE_match", "F_CE"), ("F_CE_match", "F_CE_shuf")):
        if ntot[x] and ntot[y]:
            b_ = sum(ncopy[x][p] > ncopy[y][p] for p in pids); c_ = sum(ncopy[x][p] < ncopy[y][p] for p in pids)
            print(f"  {x} vs {y} own-copies per problem: {b_}/{c_} p={sign(b_, c_):.3g}")

if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0] if len(a) > 0 else "out", a[1] if len(a) > 1 else "out_ext", a[2] if len(a) > 2 else "out_ext2",
         a[3] if len(a) > 3 else "out_ext3", a[4] if len(a) > 4 else None)
