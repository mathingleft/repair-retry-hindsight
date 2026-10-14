"""Does the model use the compiler error's content? (free analysis on existing rows, 2026-08-28)

    python error_use.py [outdir]

1. Error-content tokens: identifiers named in the predecessor error (backticked/quoted names and
   identifier-like tokens, minus Lean/English boilerplate and anything already in the statement).
   For each rollout, the fraction of those tokens that appear in the proof body. Paired per-problem
   comparison F_E vs F (and N): if F_E does not mention the error's content more than F, the error is
   not being used in any way visible in the output.
2. Unknown-identifier bucket: the error names an identifier that does not exist. Fraction of proof
   bodies that STILL use that identifier, per arm. If F_E uses it as often as F, the model did not read
   the one line that would have told it not to.
3. F_E vs F by predecessor failure bucket (solved-only counts and the per-problem pass-count sign test).
"""
import collections, json, os, re, statistics, sys
from math import comb
HERE = os.path.dirname(os.path.abspath(__file__))
STOP = set("""error unknown identifier expected token unexpected goals unsolved goal type mismatch failed the
this that with for and not has have but was were are its into from term application function argument
does did could cannot unable find contradiction case cases linarith nlinarith simp omega norm_num decide
ring field_simp positivity aesop tauto exact apply intro intros constructor rfl refl use by at rw rewrite
motive is depends metavariables metavariable sorry declaration uses proof theorem lemma example prop
nat int real rat complex set finset instance synthesize failed to synthesize invalid tactic
begin end obtain rcases rintro rename_i show calc have let fun match then else if omega push_cast
lean mathlib line column bool true false some none option list sum prod
""".split())
def body(code):
    if code is None: return ""
    i = code.find(":="); b = code[i + 2:] if i >= 0 else code
    b = re.sub(r"--[^\n]*", "", b); b = re.sub(r"/-.*?-/", "", b, flags=re.S); return b
def sign(b, c):
    n = b + c
    if n == 0: return 1.0
    k = min(b, c); return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)
def content_tokens(err, statement):
    toks = set(re.findall(r"[A-Za-z_][A-Za-z0-9_.']{2,}", err))
    named = set(re.findall(r"[`'‘]([^`'’\n]{2,60})[`'’]", err))
    toks = {t for t in toks | named if t.lower() not in STOP and not re.fullmatch(r"h\d*|x|y|z|n|m|k", t)}
    return {t for t in toks if t not in statement}
def unknown_ident(err):
    m = re.search(r"[Uu]nknown (?:identifier|constant)[ :]*[`'‘]?([^`'’\s]+)", err)
    return m.group(1) if m else None
def main(outdir):
    bank = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(HERE, "bank.jsonl"))}
    rows = collections.defaultdict(lambda: collections.defaultdict(list))
    for sub in ("", "../out_ext", "../out_noi"):        # main arms + extension arms + the no-instruction ladder (2026-08-29)
        path = os.path.join(outdir, sub, "verified.jsonl") if sub else os.path.join(outdir, "verified.jsonl")
        if not os.path.exists(path): continue
        for l in open(path):
            d = json.loads(l); rows[d["arm"]][d["id"]].append(d)
    pids = sorted(rows["F"])
    ARMS = [a for a in ("bare", "I", "F", "F_E", "F_E_shuf", "F_R", "F_C", "F_CE", "N", "N2", "S", "D",
                        "F_noI", "F_E_noI", "F_E_shuf_noI", "F_R_noI", "F_C_noI", "F_CE_noI", "N_noI") if rows[a]]
    print("1. fraction of error-content tokens that appear in the proof body (mean over rollouts; problems with >=1 token)")
    frac = {a: {} for a in ARMS}; ntok = {}
    for p in pids:
        ct = content_tokens(bank[p]["predecessor_error"], bank[p]["statement"]); ntok[p] = len(ct)
        if not ct: continue
        for a in ARMS:
            vals = [sum(t in body(d.get("lean_code")) for t in ct) / len(ct) for d in rows[a][p]]
            if vals: frac[a][p] = statistics.mean(vals)
    P = [p for p in pids if ntok[p]]
    print(f"   problems with content tokens: {len(P)} / {len(pids)}; median tokens {statistics.median(ntok[p] for p in P)}")
    for a in ARMS: print(f"   {a:5s} mean {statistics.mean(frac[a][p] for p in P):.3f}")
    for x, y in (("F_E", "F"), ("F_E", "N"), ("F_CE", "F_C"), ("F_R", "F")):
        if x in frac and y in frac:
            b = sum(frac[x][p] > frac[y][p] for p in P); c = sum(frac[x][p] < frac[y][p] for p in P)
            print(f"   {x} > {y} on {b} problems, < on {c}, sign p={sign(b,c):.3g}")
    print("\n2. unknown-identifier problems: fraction of proof bodies that still use the identifier the error said was unknown")
    U = {p: unknown_ident(bank[p]["predecessor_error"]) for p in pids}
    U = {p: u for p, u in U.items() if u and u not in bank[p]["statement"]}
    print(f"   problems: {len(U)}   examples: {list(U.values())[:6]}")
    still = {a: {} for a in ARMS}
    for a in ARMS:
        for p, u in U.items():
            pat = re.compile(r"(?<![A-Za-z0-9_.'])" + re.escape(u) + r"(?![A-Za-z0-9_'])")
            vals = [bool(pat.search(body(d.get("lean_code")))) for d in rows[a][p]]
            if vals: still[a][p] = statistics.mean(vals)
    for a in ARMS:
        v = list(still[a].values()); print(f"   {a:5s} still uses it: {100*statistics.mean(v):5.1f}% of rollouts; on {sum(x>0 for x in v)}/{len(v)} problems at least once")
    for x, y in (("F_E", "F"), ("F_E", "I"), ("F", "I"), ("N", "I"), ("F_E", "N"), ("F_CE", "F_C"),
                 # priming contrasts (2026-08-29): does an error block / the instruction line RAISE use of the flagged identifier?
                 ("F_E_noI", "F_noI"), ("F_E_shuf_noI", "F_noI"), ("F", "F_noI"), ("F_E", "F_noI"), ("F_E_noI", "F_E")):
        if x in still and y in still:
            b = sum(still[x][p] < still[y][p] for p in U); c = sum(still[x][p] > still[y][p] for p in U)
            print(f"   {x} uses it LESS than {y} on {b} problems, more on {c}, sign p={sign(b,c):.3g}")
    print("\n3. F_E vs F by predecessor failure bucket (solved counts; per-problem pass-count sign test)")
    npass = {a: {p: sum(d["status"] == "PASS" for d in rows[a][p]) for p in pids} for a in ARMS}
    byb = collections.defaultdict(list)
    for p in pids: byb[bank[p]["bucket"]].append(p)
    for bk, ps in sorted(byb.items(), key=lambda kv: -len(kv[1])):
        sF = sum(npass["F"][p] > 0 for p in ps); sE = sum(npass["F_E"][p] > 0 for p in ps)
        b = sum(npass["F_E"][p] > npass["F"][p] for p in ps); c = sum(npass["F_E"][p] < npass["F"][p] for p in ps)
        print(f"   {bk:15s} n={len(ps):3d}  solved F={sF:2d} F_E={sE:2d}  passes F={sum(npass['F'][p] for p in ps):3d} F_E={sum(npass['F_E'][p] for p in ps):3d}  sign {b}/{c} p={sign(b,c):.2g}")
if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "out"))
