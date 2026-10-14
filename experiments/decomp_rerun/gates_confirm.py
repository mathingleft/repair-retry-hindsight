"""Launch gates for the confirmatory error-content precision run.

  PREREG: PREREG_CONFIRM_ERROR_CONTENT_2026-08-30.md  (sections 5, 7, 14)

GATE 2 — prompt fidelity : F_E and F_E_shuf prompts are byte-identical once the diagnostic text is
                           removed, for all 167 problems; bank.jsonl md5 matches the committed bank.
GATE 3 — shuffle fidelity: the random.Random(22072605) map is reproduced byte-for-byte and checksummed.

Exits non-zero on any failure. Run BEFORE generation; the sbatch aborts if it fails.
"""
import hashlib, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import arms

BANK_MD5_EXPECTED = os.environ.get("CONFIRM_BANK_MD5")      # set once measured, then pinned
SHUF_MD5_EXPECTED = os.environ.get("CONFIRM_SHUF_MD5")

def md5(b):
    return hashlib.md5(b if isinstance(b, bytes) else b.encode()).hexdigest()

fail = []
bank_path = os.path.join(HERE, "bank.jsonl")
bank_md5 = md5(open(bank_path, "rb").read())
bank = {}
for l in open(bank_path):
    d = json.loads(l); bank[d["id"]] = d
pids = sorted(i for i in bank if not bank[i].get("is_framing_holdout"))

print(f"GATE: bank.jsonl md5 = {bank_md5}   ({len(bank)} rows, {len(pids)} analysis problems)")
if len(pids) != 167: fail.append(f"expected 167 analysis problems, got {len(pids)}")
if BANK_MD5_EXPECTED and bank_md5 != BANK_MD5_EXPECTED:
    fail.append(f"bank md5 {bank_md5} != pinned {BANK_MD5_EXPECTED}")

# ---- GATE 3: shuffle map ----
shuf = {i: arms.shuffled_error(i) for i in sorted(bank)}
shuf_md5 = md5(json.dumps({k: md5(v) for k, v in sorted(shuf.items())}, sort_keys=True))
print(f"GATE 3: shuffle map md5 (over all {len(shuf)} bank ids) = {shuf_md5}")
if SHUF_MD5_EXPECTED and shuf_md5 != SHUF_MD5_EXPECTED:
    fail.append(f"shuffle map md5 {shuf_md5} != pinned {SHUF_MD5_EXPECTED}")
selfkeep = [i for i in pids if shuf[i] == bank[i]["predecessor_error"]]
print(f"GATE 3: problems assigned their OWN error text: {len(selfkeep)} (must be 0)")
if selfkeep: fail.append(f"{len(selfkeep)} problems kept their own error: {selfkeep[:5]}")
# document (do NOT fail on) the known non-derangement — see PREREG section 0
import collections
mult = collections.Counter(shuf[i] for i in pids)
print(f"GATE 3: NOTE (known, documented in PREREG section 0): assignment is a draw WITH REPLACEMENT, "
      f"not a derangement — {len(mult)} distinct texts used, max reuse {max(mult.values())}")

# ---- GATE 2: prompt fidelity ----
bad = []
for i in pids:
    p = bank[i]
    a = arms.build_prompt("F_E", p)
    b = arms.build_prompt("F_E_shuf", p)
    # strip the diagnostic text from each; the remainder must be byte-identical
    ea, eb = p["predecessor_error"], shuf[i]
    ra, rb = a.replace(ea, "<<ERR>>", 1), b.replace(eb, "<<ERR>>", 1)
    if ra != rb:
        bad.append(i)
    for nm, s in (("F_E", a), ("F_E_shuf", b)):
        if "You previously attempted to prove this theorem and it did not verify." not in s: fail.append(f"{nm}/{i}: missing failure sentence")
        if "Lean compiler feedback from the failed attempt:" not in s: fail.append(f"{nm}/{i}: missing error header")
        if "Produce a correct, complete proof." not in s: fail.append(f"{nm}/{i}: missing instruction line")
        if "This was the failed proof:" in s: fail.append(f"{nm}/{i}: code block leaked into an error-only arm")
print(f"GATE 2: prompts identical modulo the diagnostic: {len(pids)-len(bad)}/{len(pids)}")
if bad: fail.append(f"{len(bad)} problems whose F_E/F_E_shuf prompts differ beyond the diagnostic: {bad[:5]}")
if os.environ.get("DECOMP_TASK_HEADER") == "1":
    fail.append("DECOMP_TASK_HEADER=1 — must be unset for Qwen (PREREG section 5)")

print()
if fail:
    print("GATES FAILED:")
    for f in fail: print("  -", f)
    sys.exit(1)
print("ALL GATES PASS")
print(json.dumps({"bank_md5": bank_md5, "shuffle_map_md5": shuf_md5, "n_problems": len(pids)}))
