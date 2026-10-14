"""Build <bank>_match.json for the F_E_match / F_CE_match arms (EXT3, 2026-09-06).

For every bank problem pick ANOTHER problem's compiler error such that
  1. same coarse Lean error class (classify(): the first error message of the text, see CLASSES),
  2. token count as close as possible (cost = |log(tok_src / tok_own)|, tokens from the model's own tokenizer),
  3. not the same theorem (different id; identical error TEXT from another theorem is penalised, used only when
     unavoidable — count in summary.identical_text_pairs),
  4. balanced reuse: min-cost PERFECT matching within each class (Hungarian), so every error is used as a source
     exactly once. Classes with a single member fall back to the "tactic" family (all non-syntax classes pooled).

    python build_match.py bank.jsonl $VERICODE_SCRATCH/models/Qwen3.6-27B
    python build_match.py out_m2/bank2.jsonl $VERICODE_SCRATCH/models/DeepSeek-Prover-V2-7B
    python build_match.py --selftest      # shuffled_partner reproduces the old shuffled_error byte for byte
"""
import collections, json, math, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import arms

# ordered: first pattern that matches the FIRST message wins
CLASSES = [
    ("syntax",        r"Unknown identifier `begin`|unexpected token|expected command|unexpected end of input|Invalid `end`|expected ',|expected term|expected '"),
    ("unsolved",      r"^unsolved goals"),
    ("linarith",      r"^n?linarith failed"),
    ("omega",         r"^omega could not"),
    ("no_goals",      r"^No goals"),
    ("rewrite",       r"^Tactic `rewrite` failed|^Tactic `rw`|motive is not type correct|did not find instance of the pattern"),
    ("simp",          r"^Tactic `simp` failed|simp made no progress|`field_simp` made no progress|^norm_num|`norm_num` failed|`ring` failed|ring failed|`nlinarith`|`positivity` failed|failed to prove positivity"),
    ("type",          r"[Tt]ype mismatch|Application type mismatch|has type"),
    ("unknown",       r"^Unknown (identifier|constant)|unknown (identifier|constant)|unknown free variable"),
    ("resource",      r"maximum recursion depth|deterministic\) timeout|heartbeats|timed out|timeout"),
    ("crash",         r"verifier crash"),
]
def first_msg(err):
    t = err.strip()
    # messages are joined with "; " on the first line; the goal state (if any) follows on later lines
    return re.split(r";\s|\n", t, 1)[0]
def classify(err):
    m = first_msg(err)
    for name, pat in CLASSES:
        if re.search(pat, m): return name
    return "other"

def hungarian(cost):
    """min-cost perfect matching on a square cost matrix (e-maxx O(n^3)); returns row->col."""
    n = len(cost); INF = float("inf")
    u = [0.0] * (n + 1); v = [0.0] * (n + 1); p = [0] * (n + 1); way = [0] * (n + 1)
    for i in range(1, n + 1):
        p[0] = i; j0 = 0; minv = [INF] * (n + 1); used = [False] * (n + 1)
        while True:
            used[j0] = True; i0 = p[j0]; delta = INF; j1 = 0
            for j in range(1, n + 1):
                if not used[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]: minv[j] = cur; way[j] = j0
                    if minv[j] < delta: delta = minv[j]; j1 = j
            for j in range(n + 1):
                if used[j]: u[p[j]] += delta; v[j] -= delta
                else: minv[j] -= delta
            j0 = j1
            if p[j0] == 0: break
        while True:
            j1 = way[j0]; p[j0] = p[j1]; j0 = j1
            if j0 == 0: break
    ans = [0] * n
    for j in range(1, n + 1):
        if p[j]: ans[p[j] - 1] = j - 1
    return ans

def build(bank_path, model_dir):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    bank = arms._load_bank(bank_path)
    ids = sorted(bank)
    ntok = {i: len(tok(bank[i]["predecessor_error"], add_special_tokens=False)["input_ids"]) for i in ids}
    cls = {i: classify(bank[i]["predecessor_error"]) for i in ids}
    groups = collections.defaultdict(list)
    for i in ids: groups[cls[i]].append(i)
    # singletons: pool into the tactic family so a same-class partner exists
    pooled = [c for c, g in groups.items() if len(g) < 2]
    for c in pooled:
        for i in groups.pop(c): groups["tactic_pool"].append(i)
    if len(groups.get("tactic_pool", [])) == 1:      # still alone: pool with 'other'-like tactic classes
        for c in ("other", "unsolved", "linarith"):
            if c in groups: groups["tactic_pool"] += groups.pop(c); break
    BIG = 1e6; SAME_TXT = 50.0
    m = {}
    for c, g in sorted(groups.items()):
        n = len(g); assert n >= 2, (c, g)
        # self is forbidden; an identical error TEXT from another theorem is legal under the criteria but makes the
        # match prompt byte-identical to F_E, so it carries a large finite penalty (SAME_TXT) and is used only when
        # no derangement avoids it (e.g. a 2-member class with equal texts). Count reported in the summary.
        cost = [[BIG if a == b else SAME_TXT if bank[a]["predecessor_error"] == bank[b]["predecessor_error"]
                 else abs(math.log(max(ntok[b], 1) / max(ntok[a], 1))) for b in g] for a in g]
        asg = hungarian(cost)
        for r, cidx in enumerate(asg):
            a, b = g[r], g[cidx]
            assert a != b, (a, b)
            m[a] = {"src": b, "class": c, "own_class": cls[a], "src_class": cls[b],
                    "own_tok": ntok[a], "src_tok": ntok[b], "cost": round(cost[r][cidx], 4)}
    out = arms.match_path(bank_path)
    use = collections.Counter(v["src"] for v in m.values())
    ratios = sorted(max(v["own_tok"], v["src_tok"]) / max(1, min(v["own_tok"], v["src_tok"])) for v in m.values())
    summary = {
        "bank": os.path.basename(bank_path), "model_tokenizer": model_dir, "n": len(m),
        "classes": {c: len(g) for c, g in sorted(groups.items())},
        "pooled_singleton_classes": pooled,
        "reuse": {"max": max(use.values()), "min": min(use.values()), "distinct_sources": len(use)},
        "tok_ratio_median": ratios[len(ratios) // 2], "tok_ratio_p90": ratios[int(.9 * len(ratios))],
        "tok_ratio_max": ratios[-1],
        "same_class_frac": sum(v["own_class"] == v["src_class"] for v in m.values()) / len(m),
        "self_pairs": sum(k == v["src"] for k, v in m.items()),
        "identical_text_pairs": sum(bank[k]["predecessor_error"] == bank[v["src"]]["predecessor_error"] for k, v in m.items()),
        "seed_note": "deterministic: Hungarian min-cost perfect matching within class, cost=|log tok ratio|",
    }
    json.dump({"summary": summary, "map": m}, open(out, "w"), indent=1)
    print(json.dumps(summary, indent=1)); print("->", out)
    print("\nclass counts:", dict(collections.Counter(cls.values())))
    for i in ids[:5]:
        print(f"  {i:28s} {cls[i]:9s} tok {ntok[i]:6d} <- {m[i]['src']:28s} tok {m[i]['src_tok']:6d}")

def selftest():
    """shuffled_partner must reproduce the pre-refactor shuffled_error draw byte for byte."""
    import random
    for path in ("bank.jsonl", "out_m2/bank2.jsonl"):
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
        if not os.path.exists(p): print("skip", path); continue
        bank = {}
        for l in open(p):
            d = json.loads(l); bank[d["id"]] = d["predecessor_error"]
        rng = random.Random(22072605); old = {}
        for pid in sorted(bank):
            others = [q for q in sorted(bank) if bank[q] != bank[pid]]
            old[pid] = bank[rng.choice(others)]
        assert all(arms.shuffled_error(pid, p) == old[pid] for pid in bank), path
        assert all(arms.shuffled_partner(pid, p) != pid for pid in bank)
        print(f"selftest OK {path}: {len(bank)} ids, shuffled_error unchanged, partner != self")

if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]: selftest()
    else: build(sys.argv[1], sys.argv[2])
