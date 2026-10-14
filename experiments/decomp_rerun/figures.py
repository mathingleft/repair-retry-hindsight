"""Paper figures for the decomposition rerun (outline §11). Reads only verified/raw files in this directory.

    python figures.py [outdir=figs]

ONE COLOR LANGUAGE, fixed across every figure:
    gray   = nothing / a null (bare, blind, blind_I)
    blue   = trigger only — no verifier feedback, no code (I, F, assert); light blue = random words (N, noise)
             (F is the MEANINGFUL failure sentence; "trigger only" describes what the prompt lacks, not its meaning)
    green  = verifier feedback carried (F_E, wrong error, F_R, f)
    orange = the failed code in context (F_C, F_CE, gf)
Provenance (preregistered vs post-hoc) is a dagger in the tick label, never a color.

Main text:  fig1 decomposition (both models, I reference line, rescue brackets)
            fig2 copy rate + distinct-of-4 (both models; bare hatched = re-derivation, not transcription)
            fig3 does it read the error: proof still uses the identifier the compiler called unknown
            fig4 depth discovery, new run, both measured nulls, k <= 8 region marked
Appendix:   figA1 repetition compounding (old-run curve muted; within-old-run contrast only)
            figA2 old-run f vs gf crossing (transcribed; no synthetic null)
            figA3 similarity to the failed proof + output length
"""
import collections, difflib, json, os, random, re, statistics, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from copy_rate import body
from error_use import unknown_ident, body as ebody

# ---- palette (dataviz reference instance, light mode) ----
BLUE, ORANGE, AQUA, BLUE_LIGHT = "#2a78d6", "#eb6834", "#1baf7a", "#86b6ef"
GRAY, GRID, AXIS, INK, INK2, SURF = "#898781", "#e1e0d9", "#c3c2b7", "#0b0b0b", "#52514e", "#fcfcfb"
GRAY_LIGHT = "#b3b1a9"
FAMILY = {"bare": GRAY, "blind": GRAY, "blind_I": INK2, "blind_F": GRAY_LIGHT,
          "I": BLUE, "F": BLUE, "assert": BLUE, "N": BLUE_LIGHT, "noise": BLUE_LIGHT,
          "F_E": AQUA, "F_E_shuf": AQUA, "F_R": AQUA, "f": AQUA,
          "F_C": ORANGE, "F_CE": ORANGE, "gf": ORANGE,
          "N_noI": BLUE_LIGHT, "F_noI": BLUE, "F_E_noI": AQUA, "F_E_shuf_noI": AQUA, "F_R_noI": AQUA, "F_C_noI": ORANGE, "F_CE_noI": ORANGE}
LEGEND = [Patch(color=GRAY, label="nothing (null)"), Patch(color=BLUE, label="trigger only (no feedback, no code)"),
          Patch(color=BLUE_LIGHT, label="random words"), Patch(color=AQUA, label="verifier feedback (error / reflection)"),
          Patch(color=ORANGE, label="failed code in context")]
POSTHOC = {"I", "N", "F_E_shuf"}
TICK = {"F_E_shuf": "wrong\nerror"}
def tick(a): return TICK.get(a, a) + ("†" if a in POSTHOC else "")

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
    "axes.axisbelow": True, "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "legend.frameon": False, "legend.fontsize": 8, "text.color": INK, "axes.labelcolor": INK2,
    "hatch.linewidth": 1.0, "pdf.fonttype": 42, "savefig.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.04,
})

def load(path): return [json.loads(l) for l in open(path) if l.strip()]
def save(fig, out, name):
    for ext in ("pdf", "png"): fig.savefig(os.path.join(out, f"{name}.{ext}"))
    plt.close(fig); print("wrote", name)

# ---------------------------------------------------------------- data
class Model:
    def __init__(self, name, rows, pids, pred, pred_idx=None):
        self.name, self.rows, self.pids, self.pred, self.pred_idx = name, rows, sorted(pids), pred, pred_idx or {}
        self.solved = collections.defaultdict(set)
        for r in rows:
            if r["status"] == "PASS" and r["id"] in pids: self.solved[r["arm"]].add(r["id"])
    def arm_rows(self, a, exclude_pred=False):
        R = [r for r in self.rows if r["arm"] == a and r["id"] in set(self.pids)]
        if exclude_pred: R = [r for r in R if self.pred_idx.get(r["id"]) != r["sample_idx"]]
        return R

def qwen(outdir):
    rows = load(os.path.join(outdir, "out", "verified.jsonl")) + load(os.path.join(outdir, "out_ext", "verified.jsonl"))
    noi = os.path.join(outdir, "out_noi", "verified.jsonl")
    if os.path.exists(noi): rows += load(noi)          # the six no-instruction arms (2026-08-29)
    bank = {r["id"]: r for r in load(os.path.join(HERE, "bank.jsonl"))}
    pids = {r["id"] for r in rows if r["arm"] == "bare"}
    return Model("Qwen3.6-27B", rows, pids, {p: body(bank[p]["predecessor_code"]) for p in pids})

def prover(outdir):
    rows = load(os.path.join(outdir, "out_m2", "verified.jsonl"))
    for sub in ("out_m2_ext", "out_m2_noi"):           # prover F_E_shuf + N; prover no-instruction ladder (2026-08-29)
        ext = os.path.join(outdir, sub, "verified.jsonl")
        if os.path.exists(ext): rows += load(ext)
    bank2 = {r["id"]: r for r in load(os.path.join(outdir, "out_m2", "bank2.jsonl"))}
    return Model("DeepSeek-Prover-V2-7B", rows, set(bank2), {p: body(bank2[p]["predecessor_code"]) for p in bank2},
                 {p: bank2[p].get("predecessor_rollout_idx") for p in bank2})

def cluster_ci(groups, stat, B=2000, seed=0):
    """95% problem-bootstrap interval: resample problems (each carrying all its samples/trajectories), recompute stat."""
    rng = random.Random(seed); groups = [g for g in groups if g]; obs = stat([v for g in groups for v in g])
    bs = sorted(stat([v for g in rng.choices(groups, k=len(groups)) for v in g]) for _ in range(B))
    return obs, bs[int(.025 * B)], bs[int(.975 * B)]

def whiskers(ax, x, cis, color=INK2):
    lo = [c[0] - c[1] for c in cis]; hi = [c[2] - c[0] for c in cis]
    ax.errorbar(x, [c[0] for c in cis], yerr=[lo, hi], fmt="none", ecolor=color, elinewidth=0.9, capsize=2.5, capthick=0.9, zorder=3)

def solved_ci(m, arms, B=4000, seed=0):
    rng = random.Random(seed); out = {}
    for a in arms:
        obs = len(m.solved[a]); bs = sorted(sum(p in m.solved[a] for p in rng.choices(m.pids, k=len(m.pids))) for _ in range(B))
        out[a] = (obs, bs[int(.025 * B)], bs[int(.975 * B)])
    return out

# ---------------------------------------------------------------- fig1
def bracket(ax, x0, x1, y, text, color=INK):
    ax.plot([x0, x0, x1, x1], [y - 1.2, y, y, y - 1.2], color=INK2, lw=0.8, solid_capstyle="round")
    ax.text((x0 + x1) / 2, y + 0.8, text, ha="center", va="bottom", fontsize=8, color=color)

def panel1(ax, m, arms, sep_after, title, ylim):
    st = solved_ci(m, arms); x = np.arange(len(arms)); vals = [st[a][0] for a in arms]
    ax.bar(x, vals, width=0.58, color=[FAMILY[a] for a in arms], linewidth=0)
    ax.errorbar(x, vals, yerr=[[st[a][0] - st[a][1] for a in arms], [st[a][2] - st[a][0] for a in arms]],
                fmt="none", ecolor=INK2, elinewidth=0.9, capsize=2.5, capthick=0.9)
    for xi, a in zip(x, arms):
        ax.text(xi, st[a][2] + ylim * 0.012, str(st[a][0]), ha="center", va="bottom", fontsize=8, color=INK)
    # reference: the instruction line alone
    yI = st["I"][0]; ax.axhline(yI, color=INK2, lw=0.9, ls=(0, (4, 3)), zorder=0.5)
    ax.text(-0.42, yI + ylim * 0.012, f"I = {yI}", ha="left", va="bottom", fontsize=7.5, color=INK2)
    # separator: code enters the context
    ax.axvline(sep_after + 0.5, color=AXIS, lw=0.8, zorder=0.5)
    ax.text(sep_after + 0.55, ylim * 0.985, "failed code\nin context →", ha="left", va="top", fontsize=7, color=INK2)
    # the decisive contrast: does the error rescue the code?
    i, j = arms.index("F_C"), arms.index("F_CE"); d = st["F_CE"][0] - st["F_C"][0]
    top = max(st["F_C"][2], st["F_CE"][2]) + ylim * 0.10
    bracket(ax, i, j, top, f"{d:+d}: the error\nrescues the code" if d > 5 else f"{d:+d}: it does not")
    ax.set_xticks(x); ax.set_xticklabels([tick(a) for a in arms], fontsize=8); ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, ylim); ax.set_title(title, loc="left", color=INK); ax.set_ylabel(f"problems solved (of {len(m.pids)})")

ANATOMY = [("failure sentence", {"F", "F_E", "F_E_shuf", "F_R", "F_C", "F_CE"}), ("random words", {"N"}),
           ("compiler error", {"F_E", "F_CE"}), ("another problem's error", {"F_E_shuf"}), ("failed code", {"F_C", "F_CE"}),
           ("reflection", {"F_R"}), ("instruction line", {"I", "N", "F", "F_E", "F_E_shuf", "F_R", "F_C", "F_CE"}),
           ("task (statement)", {"bare", "I", "N", "F", "F_E", "F_E_shuf", "F_R", "F_C", "F_CE"})]

def anatomy(ax, arms):
    """Prompt-anatomy strip: one row per prompt component, one column per arm (aligned with the bars below)."""
    ax.set_xlim(-0.5, len(arms) - 0.5); ax.set_ylim(-0.5, len(ANATOMY) - 0.5); ax.invert_yaxis()
    for r, (name, members) in enumerate(ANATOMY):
        ax.text(-0.62, r, name, ha="right", va="center", fontsize=6.6, color=INK2)
        for c, a in enumerate(arms):
            ax.text(c, r, "✓" if a in members else "·", ha="center", va="center", fontsize=7.5 if a in members else 6, color=INK if a in members else AXIS)
    for c, a in enumerate(arms):          # column headers = the arm names, same order as the bars below
        ax.text(c, -0.95, tick(a), ha="center", va="bottom", fontsize=6.6, color=INK, clip_on=False, linespacing=0.95)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    ax.set_title("prompt anatomy of each arm: [sentence] [blocks] [instruction line] [task]", loc="left", color=INK2, fontsize=7.5, pad=20)

def fig1(out, Q, P):
    fig = plt.figure(figsize=(8.6, 4.3))
    gs = fig.add_gridspec(2, 2, width_ratios=[9, 8], height_ratios=[1.25, 3], wspace=0.24, hspace=0.35)
    a0 = fig.add_subplot(gs[0, 0]); a1 = fig.add_subplot(gs[1, 0]); a2 = fig.add_subplot(gs[1, 1])
    qarms = ["bare", "I", "N", "F", "F_E", "F_E_shuf", "F_R", "F_C", "F_CE"]
    anatomy(a0, qarms)
    panel1(a1, Q, qarms, 6, "a  Qwen3.6-27B, 167 problems, k = 4", 84)
    parms = [a for a in ["bare", "I", "N", "F", "F_E", "F_E_shuf", "F_C", "F_CE"] if P.solved.get(a) is not None or any(r["arm"] == a for r in P.rows[:4000])]
    parms = [a for a in ["bare", "I", "N", "F", "F_E", "F_E_shuf", "F_C", "F_CE"] if any(r["arm"] == a for r in P.rows)]
    panel1(a2, P, parms, parms.index("F_C") - 1, "b  DeepSeek-Prover-V2-7B, 137 problems, k = 4", 58)
    fig.legend(handles=LEGEND, loc="upper center", ncol=5, handlelength=1.2, bbox_to_anchor=(0.5, 1.03), frameon=False)
    save(fig, out, "fig1_decomposition")

# ---------------------------------------------------------------- fig2
def copy_stats(m, arms):
    """Per arm: (obs, lo, hi) for copy rate (%), distinct-of-4 (mean), similarity (mean) — 95% problem-bootstrap."""
    copy, dist, sim = {}, {}, {}
    for a in arms:
        R = m.arm_rows(a, exclude_pred=(a == "bare"))
        if not R: continue
        cp = collections.defaultdict(list); sm = collections.defaultdict(list); byp = collections.defaultdict(list)
        for r in R:
            b = body(r.get("lean_code")); cp[r["id"]].append(int(b is not None and b == m.pred[r["id"]]))
            if b is not None: sm[r["id"]].append(difflib.SequenceMatcher(None, b[:3000], m.pred[r["id"]][:3000]).ratio())
        for r in m.arm_rows(a): byp[r["id"]].append(body(r.get("lean_code")))
        copy[a] = cluster_ci([cp[p] for p in m.pids], lambda v: 100 * statistics.mean(v))
        sim[a] = cluster_ci([sm[p] for p in m.pids], statistics.mean)
        dist[a] = cluster_ci([[len({b for b in byp[p] if b is not None})] for p in m.pids], statistics.mean)
    return copy, dist, sim

def panel2(ax, m, arms, d, unit, title, ylim, labelled, fmt):
    x = np.arange(len(arms)); v = [d[a][0] for a in arms]
    bars = ax.bar(x, v, width=0.58, color=[FAMILY[a] for a in arms], linewidth=0)
    bars[arms.index("bare")].set_hatch("///"); bars[arms.index("bare")].set_edgecolor(SURF)
    whiskers(ax, x, [d[a] for a in arms])
    for xi, a, vi in zip(x, arms, v):
        if a in labelled: ax.text(xi, d[a][2] + ylim * 0.015, fmt(vi) + unit, ha="center", va="bottom", fontsize=7.5, color=INK)
    ax.set_xticks(x); ax.set_xticklabels([tick(a) for a in arms], fontsize=8); ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, ylim); ax.set_title(title, loc="left", color=INK, fontsize=9)

def fig2(out, Q, P):
    qa = ["bare", "I", "N", "F", "F_E", "F_R", "F_C", "F_CE"]; pa = ["bare", "I", "F", "F_E", "F_C", "F_CE"]
    cq, dq, sq = copy_stats(Q, qa); cp, dp, sp = copy_stats(P, pa)
    fig, ax = plt.subplots(2, 2, figsize=(7.4, 4.6), gridspec_kw=dict(width_ratios=[8, 6], wspace=0.22, hspace=0.55))
    top = max(max(c[2] for c in cq.values()), max(c[2] for c in cp.values())) * 1.15
    panel2(ax[0, 0], Q, qa, cq, "%", "a  Qwen3.6-27B", top, {"bare", "I", "F_C", "F_CE"}, lambda v: f"{v:.1f}")
    panel2(ax[0, 1], P, pa, cp, "%", "DeepSeek-Prover-V2-7B", top, {"bare", "I", "F_C", "F_CE"}, lambda v: f"{v:.1f}")
    panel2(ax[1, 0], Q, qa, dq, "", "b  Qwen3.6-27B", 4.8, {"bare", "I", "F_C", "F_CE"}, lambda v: f"{v:.2f}")
    panel2(ax[1, 1], P, pa, dp, "", "DeepSeek-Prover-V2-7B", 4.8, {"bare", "I", "F_C", "F_CE"}, lambda v: f"{v:.2f}")
    ax[0, 0].set_ylabel("% of samples reproducing\nthe failed proof verbatim"); ax[1, 0].set_ylabel("distinct proofs among\nthe 4 samples (mean)")
    for a in (ax[0, 1], ax[1, 1]): a.set_yticklabels([])
    fig.legend(handles=LEGEND + [Patch(facecolor=GRAY, edgecolor=SURF, hatch="///", label="bare: re-derives its default proof (nothing in context)")],
               loc="upper center", ncol=3, handlelength=1.2, bbox_to_anchor=(0.5, 1.05), frameon=False)
    save(fig, out, "fig2_copy_diversity")
    return sq

# ---------------------------------------------------------------- fig3
def still_uses(m, bankfile, arms, key="predecessor_error"):
    """Per arm: (obs, lo, hi) % of samples whose proof body still uses the identifier the error called unknown."""
    bank = {r["id"]: r for r in load(bankfile)}
    U = {p: unknown_ident(bank[p][key]) for p in m.pids if p in bank}
    U = {p: u for p, u in U.items() if u and u not in bank[p]["statement"]}
    byarm = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in m.rows: byarm[r["arm"]][r["id"]].append(r)
    out = {}
    for a in arms:
        groups = []
        for p, u in U.items():
            pat = re.compile(r"(?<![A-Za-z0-9_.'])" + re.escape(u) + r"(?![A-Za-z0-9_'])")
            rs = byarm[a][p]
            if rs: groups.append([statistics.mean(bool(pat.search(ebody(d.get("lean_code")))) for d in rs)])
        out[a] = cluster_ci(groups, lambda v: 100 * statistics.mean(v))
    return out, len(U)

def panel3(ax, arms, still, n, title, sep_after, told, labels=None):
    x = np.arange(len(arms)); v = [still[a][0] for a in arms]
    ax.bar(x, v, width=0.58, color=[FAMILY[a] for a in arms], linewidth=0)
    whiskers(ax, x, [still[a] for a in arms])
    for xi, a, vi in zip(x, arms, v): ax.text(xi, still[a][2] + 1.2, f"{vi:.0f}%", ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(x); ax.set_xticklabels(labels or [tick(a) for a in arms], fontsize=8); ax.tick_params(axis="x", length=0)
    ax.text(-0.55, -0.2, "error shown:", ha="right", va="top", fontsize=7, color=INK2, transform=ax.get_xaxis_transform(), clip_on=False)
    for xi, a in zip(x, arms):
        ax.text(xi, -0.2, told.get(a, "–"), ha="center", va="top", fontsize=7, color=INK2, transform=ax.get_xaxis_transform(), clip_on=False)
    ax.axvline(sep_after + 0.5, color=AXIS, lw=0.8, zorder=0.5)
    ax.set_ylim(0, 108); ax.set_title(f"{title}, {n} problems", loc="left", color=INK, fontsize=9)

def fig3(out, Q, P):
    told = {"F_E": "own", "F_CE": "own", "F_E_shuf": "another\nproblem's"}
    sq, nq = still_uses(Q, os.path.join(HERE, "bank.jsonl"), ["bare", "I", "N", "F", "F_E", "F_E_shuf", "F_R", "F_C", "F_CE"])
    parms3 = [a for a in ["bare", "I", "N", "F", "F_E", "F_E_shuf", "F_C", "F_CE"] if any(r["arm"] == a for r in P.rows)]
    sp, npr = still_uses(P, os.path.join(HERE, "out_m2", "bank2.jsonl"), parms3)
    print(f"fig3: {nq} Qwen / {npr} prover unknown-identifier problems")
    narms = [a for a in ["F_noI", "F_E_noI", "F_E_shuf_noI", "F_R_noI", "F_C_noI", "F_CE_noI"] if any(r["arm"] == a for r in Q.rows)]
    sn, _ = still_uses(Q, os.path.join(HERE, "bank.jsonl"), narms)
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(14.0, 3.3), gridspec_kw=dict(width_ratios=[9, 8, 6], wspace=0.22))
    panel3(a1, ["bare", "I", "N", "F", "F_E", "F_E_shuf", "F_R", "F_C", "F_CE"], sq, nq, "a  Qwen, instruction regime (`begin`)", 6, told)
    panel3(a2, parms3, sp, npr, "b  DeepSeek-Prover (nonexistent lemmas)", parms3.index("F_C") - 1, told)
    told_n = {"F_E_noI": "own", "F_CE_noI": "own", "F_E_shuf_noI": "another\nproblem's"}
    panel3(a3, narms, sn, nq, "c  Qwen, no instruction line", narms.index("F_C_noI") - 1, told_n,
           labels=[a[:-4].replace("F_E_shuf", "wrong\nerror") + "†" for a in narms])
    a1.set_ylabel("% of samples still using the identifier\nthe compiler called unknown")
    fig.legend(handles=LEGEND, loc="upper center", ncol=5, handlelength=1.2, bbox_to_anchor=(0.5, 1.06), frameon=False)
    save(fig, out, "fig3_error_use")

# ---------------------------------------------------------------- fig4
def depth_first(path):
    first = {}; keys = set()
    if not os.path.exists(path): return first, keys
    for l in open(path):
        try: x = json.loads(l)
        except Exception: continue
        k = (x["id"], x["traj"]); keys.add(k)
        if x["status"] == "PASS": first[k] = min(first.get(k, 10**9), x["attempt"])
    return first, keys

def fig4(outdir, out):
    ks = list(range(1, 65))
    spec = [("assert", "assert: “attempt n: failed.”", BLUE, "o"), ("noise", "noise: random-word sentence", BLUE_LIGHT, "^"),
            ("f", "f: the compiler error", AQUA, "o"), ("blind", "blind: 64 i.i.d. bare-prompt draws", GRAY, "s"),
            ("blind_I", "blind_I: 64 i.i.d. instruction-prompt draws", INK2, "D"),
            ("blind_F", "blind_F: 64 i.i.d. sentence-only draws", GRAY_LIGHT, "v")]
    fig, ax = plt.subplots(figsize=(5.8, 4.2)); series = []; ends = []; at8 = {}; firsts = {}
    offs = {"assert": 1.12, "noise": 1.24, "f": 1.37, "blind": 1.52, "blind_I": 1.68, "blind_F": 1.86}     # endpoint whisker x, right of the curves
    for arm, lab, c, m in spec:
        first, keys = depth_first(os.path.join(outdir, "out", f"depth_{arm}.jsonl"))
        if len(keys) < 110: continue
        y = [sum(1 for v in first.values() if v <= k) for k in ks]; at8[arm] = y[7]; firsts[arm] = (first, keys)
        ax.plot(ks, y, color=c, lw=2, marker=m, ms=5.5, mfc=c, mec=SURF, mew=1.2, markevery=[0, 1, 3, 7, 15, 31, 63], solid_joinstyle="round")
        byp = collections.defaultdict(list)
        for (pid, t) in keys: byp[pid].append(first.get((pid, t), 10**9))
        ci = cluster_ci(list(byp.values()), lambda v: sum(1 for a in v if a <= 64), B=2000)
        ax.errorbar([64 * offs[arm]], [ci[0]], yerr=[[ci[0] - ci[1]], [ci[2] - ci[0]]], fmt=m, ms=4, mfc=c, mec=SURF, mew=0.8,
                    ecolor=c, elinewidth=1.0, capsize=2, capthick=0.9, zorder=2.5, clip_on=False)
        ends.append((y[-1], arm, c)); series.append((lab, c, m))
    ax.axvspan(0.9, 8, color=INK, alpha=0.045, lw=0, zorder=0)
    ax.text(0.98, 113, "first 8 attempts", fontsize=7.5, color=INK2, ha="left", va="top")
    ax.text(0.98, 106, "   by k = 8: " + " · ".join(f"{a} {at8[a]}" for a in ("assert", "noise", "f", "blind", "blind_I", "blind_F") if a in at8), fontsize=7.5, color=INK2, ha="left", va="top")
    def paired8(x, y):   # trajectory-paired sign test at k = 8, x vs y
        fx, kx = firsts[x]; fy, ky = firsts[y]; keys = kx & ky
        a = sum(1 for t in keys if fx.get(t, 10**9) <= 8 and not fy.get(t, 10**9) <= 8)
        b = sum(1 for t in keys if fy.get(t, 10**9) <= 8 and not fx.get(t, 10**9) <= 8)
        n = a + b; kk = min(a, b); pv = min(1.0, 2 * sum(__import__("math").comb(n, i) for i in range(kk + 1)) / 2 ** n) if n else 1.0
        return f"{x} {'>' if a > b and pv < 0.05 else '=' if pv >= 0.05 else '<'} {y} ({a}/{b}, p = {pv:.2g})"
    lines = []
    for null in ("blind", "blind_I", "blind_F"):
        if null in firsts: lines.append("; ".join(paired8(x, null) for x in ("f", "assert") if x in firsts))
    for i, ln in enumerate(lines): ax.text(0.98, 99 - 7 * i, "   " + ln, fontsize=7.5, color=INK2, ha="left", va="top")
    ax.set_xscale("log", base=2); ax.set_xticks([1, 2, 4, 8, 16, 32, 64]); ax.set_xticklabels(["1", "2", "4", "8", "16", "32", "64"])
    ax.set_xlim(0.9, 260); ax.set_ylim(0, 116); ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.text(64 * 1.10, 115, "95% intervals at k = 64\n(problem bootstrap, n = 11)", fontsize=6.5, color=INK2, ha="left", va="top")
    ax.set_xlabel("attempt k"); ax.set_ylabel("trajectories with a first pass by k (of 110)")
    ends.sort(); ylast = -10
    for v, name, c in ends:
        yy = max(v, ylast + 6); ylast = yy
        ax.annotate(f"{name} {v}", (64 * 1.86, v), xytext=(64 * 2.1, yy), textcoords="data", fontsize=8, color=INK, va="center",
                    arrowprops=dict(arrowstyle="-", color=AXIS, lw=0.7, shrinkA=0, shrinkB=2) if abs(yy - v) > 0.5 else None)
    ax.legend(handles=[Line2D([], [], color=c, lw=2, marker=m, mfc=c, mec=SURF, ms=5.5, label=lab) for lab, c, m in series],
              loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, handlelength=2.2, frameon=False)
    save(fig, out, "fig4_discovery")

# ---------------------------------------------------------------- fig5
def panel5(ax, M, title, ylim):
    scaff = [("nothing", "bare", "I"), ("random\nwords", "N_noI", "N"), ("failure\nsentence", "F_noI", "F"), ("+ compiler\nerror", "F_E_noI", "F_E"),
             ("+ wrong\nerror", "F_E_shuf_noI", "F_E_shuf"), ("+ reflection", "F_R_noI", "F_R"), ("+ failed\ncode", "F_C_noI", "F_C"), ("+ code\n+ error", "F_CE_noI", "F_CE")]
    present = {r["arm"] for r in M.rows}
    scaff = [t for t in scaff if t[1] in present and t[2] in present]
    arms = [a for t in scaff for a in t[1:]]; st = solved_ci(M, arms)
    x = np.arange(len(scaff)); w = 0.36
    for i, (lab, a0, a1) in enumerate(scaff):
        col = FAMILY[a1] if a1 != "I" else BLUE; col0 = GRAY if a0 == "bare" else col
        ax.bar(i - w / 2 - 0.02, st[a0][0], width=w, facecolor=SURF, edgecolor=col0, linewidth=1.6)
        ax.bar(i + w / 2 + 0.02, st[a1][0], width=w, color=col, linewidth=0)
        for xx, a in ((i - w / 2 - 0.02, a0), (i + w / 2 + 0.02, a1)):
            ax.errorbar([xx], [st[a][0]], yerr=[[st[a][0] - st[a][1]], [st[a][2] - st[a][0]]], fmt="none", ecolor=INK2, elinewidth=0.9, capsize=2.2, capthick=0.9, zorder=3)
            ax.text(xx, st[a][2] + ylim * 0.015, str(st[a][0]), ha="center", va="bottom", fontsize=7.5, color=INK)
    ax.axhline(st["I"][0], color=INK2, lw=0.9, ls=(0, (4, 3)), zorder=0.5); ax.text(len(scaff) - 0.55, st["I"][0] + ylim * 0.015, f"I = {st['I'][0]}", ha="right", va="bottom", fontsize=7.5, color=INK2)
    k = next(i for i, t in enumerate(scaff) if t[2] == "F_C")
    ax.axvline(k - 0.5, color=AXIS, lw=0.8, zorder=0.5); ax.text(k - 0.45, ylim * 0.96, "failed code\nin context →", ha="left", va="top", fontsize=7, color=INK2)
    ax.set_xticks(x); ax.set_xticklabels([t[0] for t in scaff], fontsize=7.5); ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, ylim); ax.set_ylabel(f"problems solved (of {len(M.pids)})"); ax.set_title(title, loc="left", color=INK, fontsize=9)

def fig5(out, Q, P):
    """Instruction line x prompt content: each scaffold with and without "Produce a correct, complete proof."."""
    has_p = any(r["arm"].endswith("_noI") for r in P.rows)
    if has_p:
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.0, 3.5), gridspec_kw=dict(width_ratios=[8, 6], wspace=0.18))
        panel5(a1, Q, "a  Qwen3.6-27B, 167 problems, k = 4", 66); panel5(a2, P, "b  DeepSeek-Prover-V2-7B, 137 problems, k = 4", 48)
    else:
        fig, a1 = plt.subplots(figsize=(8.4, 3.3)); panel5(a1, Q, "Qwen3.6-27B, 167 problems, k = 4 — each scaffold without and with the instruction line", 66)
    fig.legend(handles=LEGEND, loc="upper center", ncol=5, handlelength=1.2, bbox_to_anchor=(0.5, 1.12), frameon=False)
    fig.legend(handles=[Patch(facecolor=SURF, edgecolor=INK2, linewidth=1.4, label="outlined: without \"Produce a correct, complete proof.\""),
                        Patch(facecolor=INK2, label="filled: with it")], loc="upper center", ncol=2, handlelength=1.2, bbox_to_anchor=(0.5, 1.05), frameon=False)
    save(fig, out, "fig5_instruction_x_content")

# ---------------------------------------------------------------- fig6
def fig6(outdir, out):
    """Breadth vs depth at a fixed attempt budget m per problem (the original construction, on this run's data).
    breadth(m) = mean_i 1-(1-p̂_i)^m, p̂_i the i.i.d. per-draw rate from 640 draws; depth(m, L) = mean_i 1-(1-p_L,i)^(m/L),
    p_L,i the fraction of the 10 observed trajectories solved within L attempts; T = m/L > 10 is extrapolated (hollow).
    Marginal bands are deliberately omitted (n = 11 all-or-nothing problems make them +-0.3 and meaningless for the
    comparison); panel b carries the PAIRED difference depth - breadth with its 95% problem-bootstrap band."""
    import itertools, difflib
    D = {a: [json.loads(l) for l in open(os.path.join(outdir, "out", f"depth_{a}.jsonl"))] for a in ("assert", "noise", "f", "blind", "blind_I", "blind_F")
         if os.path.exists(os.path.join(outdir, "out", f"depth_{a}.jsonl"))}
    pids = sorted({r["id"] for r in D["f"]})
    rate = {a: {p: sum(1 for r in D[a] if r["id"] == p and r["status"] == "PASS") / 640 for p in pids} for a in D if a.startswith("blind")}
    first = {a: {} for a in ("assert", "noise", "f")}
    for a in first:
        for r in D[a]:
            if r["status"] == "PASS": first[a][(r["id"], r["traj"])] = min(first[a].get((r["id"], r["traj"]), 99), r["attempt"])
    pL = lambda a, p, L: sum(1 for t in range(10) if first[a].get((p, t), 99) <= L) / 10
    bre = lambda a, p, m: 1 - (1 - rate[a][p]) ** m
    dep = lambda a, p, m, L: 1 - (1 - pL(a, p, L)) ** (m / L)
    ms = [8, 16, 32, 64, 128, 256, 640]
    fig, (ax, axd, ax2) = plt.subplots(1, 3, figsize=(12.0, 3.6), gridspec_kw=dict(width_ratios=[6, 4, 3], wspace=0.32))
    for a, lab, c, m_ in (("blind_I", "breadth: i.i.d. instruction-prompt draws", INK2, "D"), ("blind_F", "breadth: i.i.d. sentence-only draws", GRAY_LIGHT, "v"), ("blind", "breadth: i.i.d. bare-prompt draws", GRAY, "s")):
        if a in rate: ax.plot(ms, [statistics.mean(bre(a, p, m) for p in pids) for m in ms], color=c, lw=2, marker=m_, ms=5.5, mfc=c, mec=SURF, mew=1.2, label=lab)
    for a, L, c, m_, lab in (("f", 32, AQUA, "o", "depth: compiler-error trajectories, L = 32"), ("f", 64, AQUA, "o", "depth: compiler-error trajectories, L = 64"),
                           ("assert", 64, BLUE, "o", "depth: “failed.” trajectories, L = 64"), ("noise", 64, BLUE_LIGHT, "^", "depth: random-word trajectories, L = 64")):
        xs = [m for m in ms if m >= L]; ys = [statistics.mean(dep(a, p, m, L) for p in pids) for m in xs]; obs = [m / L <= 10 for m in xs]
        ax.plot(xs, ys, color=c, lw=2, ls="-" if L == 64 else (0, (4, 2)), label=lab, zorder=2)
        ax.plot([x for x, o in zip(xs, obs) if o], [y for y, o in zip(ys, obs) if o], ls="none", marker=m_, ms=5.5, mfc=c, mec=SURF, mew=1.2, zorder=3)
        ax.plot([x for x, o in zip(xs, obs) if not o], [y for y, o in zip(ys, obs) if not o], ls="none", marker=m_, ms=5.5, mfc=SURF, mec=c, mew=1.4, zorder=3)
    ax.set_xscale("log", base=2); ax.set_xticks(ms); ax.set_xticklabels([str(m) for m in ms]); ax.set_xlim(7, 730); ax.set_ylim(0, 1.0)
    ax.set_xlabel("attempt budget per problem, m"); ax.set_ylabel("expected fraction of the 11 problems solved")
    ax.set_title("a  breadth vs depth at a fixed budget (hollow: extrapolated, T > 10)", loc="left", color=INK, fontsize=9)
    ax.legend(loc="lower right", handlelength=2.4, fontsize=7, borderaxespad=0.3)
    # panel b: paired difference depth(m, L) - breadth_I(m), 95% problem-bootstrap band
    rng = random.Random(0); xs = [m for m in ms if m >= 32]
    axd.axhline(0, color=AXIS, lw=0.9, zorder=0.5)
    for a, L, c, m_, lab in (("f", 64, AQUA, "o", "compiler-error, L = 64"), ("f", 32, AQUA, "o", "compiler-error, L = 32"), ("assert", 64, BLUE, "o", "“failed.”, L = 64")):
        xx = [m for m in xs if m >= L]; mid, lo, hi = [], [], []
        for m in xx:
            d = {p: dep(a, p, m, L) - bre("blind_I", p, m) for p in pids}
            bs = sorted(statistics.mean(d[p] for p in rng.choices(pids, k=len(pids))) for _ in range(3000))
            mid.append(statistics.mean(d.values())); lo.append(bs[75]); hi.append(bs[2924])
        axd.fill_between(xx, lo, hi, color=c, alpha=0.12 if L == 64 else 0.07, lw=0, zorder=1)
        axd.plot(xx, mid, color=c, lw=2, ls="-" if L == 64 else (0, (4, 2)), marker=m_, ms=5, mfc=c, mec=SURF, mew=1.2, label=lab, zorder=3)
    axd.set_xscale("log", base=2); axd.set_xticks(xs); axd.set_xticklabels([str(m) for m in xs]); axd.set_xlim(28, 730); axd.set_ylim(-0.35, 0.4)
    axd.set_xlabel("attempt budget per problem, m"); axd.set_ylabel("depth − breadth (instruction draws)")
    axd.set_title("b  paired difference, 95% problem bootstrap", loc="left", color=INK, fontsize=9)
    axd.legend(loc="upper left", handlelength=2.2, fontsize=7, borderaxespad=0.3)
    # panel c: diversity within the first 8 attempts
    rng = random.Random(0)
    def simset(codes):
        cs = [c for c in codes if c]
        if len(cs) < 2: return None
        pairs = list(itertools.combinations(cs, 2)); pairs = rng.sample(pairs, min(20, len(pairs)))
        return sum(difflib.SequenceMatcher(None, x[:1500], y[:1500]).ratio() for x, y in pairs) / len(pairs)
    arms = [a for a in ("blind", "blind_F", "blind_I", "assert", "noise", "f") if a in D]; vals = {}
    for a in arms:
        byt = collections.defaultdict(list)
        for r in D[a]:
            if r["attempt"] <= 8: byt[(r["id"], r["traj"])].append(body(r.get("code")))
        groups = collections.defaultdict(list)
        for (pid, t), v in byt.items():
            sv = simset(v)
            if sv is not None: groups[pid].append(sv)
        vals[a] = cluster_ci(list(groups.values()), statistics.mean)
    x = np.arange(len(arms)); ax2.bar(x, [vals[a][0] for a in arms], width=0.58, color=[FAMILY[a] for a in arms], linewidth=0)
    whiskers(ax2, x, [vals[a] for a in arms])
    ax2.set_xticks(x); ax2.set_xticklabels(arms, fontsize=7.5, rotation=30, ha="right"); ax2.tick_params(axis="x", length=0)
    ax2.set_ylim(0, 0.5); ax2.set_ylabel("mean pairwise similarity\nof the first 8 attempts"); ax2.set_title("c  diversity", loc="left", color=INK, fontsize=9)
    save(fig, out, "fig6_breadth_vs_depth")

# ---------------------------------------------------------------- appendix
def depth_repeat(path):
    seen = collections.defaultdict(set); rep = tot = 0
    for l in open(path):
        x = json.loads(l); c = body(x.get("code"))
        if c is None: continue
        k = (x["id"], x["traj"]); tot += 1
        if c in seen[k]: rep += 1
        seen[k].add(c)
    return 100 * rep / max(tot, 1)

def figA1(outdir, out, Q):
    cqci, _, _ = copy_stats(Q, ["F", "F_E", "F_R", "F_C", "F_CE"]); cq = {a: c[0] for a, c in cqci.items()}
    old_x = [5, 10, 20, 64]; old_gf = [26.9, 39.4, 52.9, 73.7]; old_f = [1.2, 0.8, 0.5, 0.7]      # CLAIMS_SPINE §3 Leg B, transcribed
    new = {a: depth_repeat(os.path.join(outdir, "out", f"depth_{a}.jsonl")) for a in ("assert", "noise", "f")}
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    ax.plot(old_x, old_gf, color=ORANGE, lw=1.6, ls=(0, (4, 3)), alpha=0.7, marker="s", ms=5, mfc=SURF, mec=ORANGE, mew=1.4)
    ax.plot(old_x, old_f, color=AQUA, lw=1.6, ls=(0, (4, 3)), alpha=0.7, marker="s", ms=5, mfc=SURF, mec=AQUA, mew=1.4)
    ax.errorbar([1, 1], [cq["F_C"], cq["F_CE"]], yerr=[[cqci[a][0] - cqci[a][1] for a in ("F_C", "F_CE")], [cqci[a][2] - cqci[a][0] for a in ("F_C", "F_CE")]],
                fmt="none", ecolor=ORANGE, elinewidth=0.9, capsize=2, zorder=2.5)
    ax.plot([1, 1], [cq["F_C"], cq["F_CE"]], ls="none", marker="o", ms=7, mfc=ORANGE, mec=SURF, mew=1.5)
    nocode = statistics.mean(cq[a] for a in ("F", "F_E", "F_R"))
    ax.plot([1], [nocode], ls="none", marker="o", ms=7, mfc=AQUA, mec=SURF, mew=1.5)
    ax.plot([64] * 3, [new["assert"], new["noise"], new["f"]], ls="none", marker="o", ms=7, mfc=AQUA, mec=SURF, mew=1.5)
    ax.annotate("F_C", (1, cq["F_C"]), xytext=(6, 2), textcoords="offset points", fontsize=8, color=INK)
    ax.annotate("F_CE", (1, cq["F_CE"]), xytext=(6, -3), textcoords="offset points", fontsize=8, color=INK)
    ax.annotate("F, F_E, F_R", (1, nocode), xytext=(6, 4), textcoords="offset points", fontsize=8, color=INK)
    ax.annotate("gf, old run", (64, 73.7), xytext=(-4, 6), textcoords="offset points", fontsize=8, color=INK, ha="right")
    ax.annotate("f, old run", (7, 1.0), xytext=(0, 9), textcoords="offset points", fontsize=8, color=INK, ha="center")
    ax.annotate("assert, noise, f — this run", (64, max(new.values())), xytext=(-8, 22), textcoords="offset points", fontsize=8, color=INK, ha="right",
                arrowprops=dict(arrowstyle="-", color=AXIS, lw=0.7, shrinkA=0, shrinkB=3))
    ax.set_xscale("log", base=2); ax.set_xticks([1, 2, 4, 8, 16, 32, 64]); ax.set_xticklabels(["1", "2", "4", "8", "16", "32", "64"])
    ax.set_xlabel("failed attempts in context"); ax.set_ylabel("% of attempts repeating an earlier proof"); ax.set_ylim(0, 82); ax.set_xlim(0.8, 90)
    ax.legend(handles=[Line2D([], [], color=ORANGE, lw=2, label="failed code in the window"),
                       Line2D([], [], color=AQUA, lw=2, label="errors / markers / words only"),
                       Line2D([], [], ls="none", marker="o", mfc=GRAY, mec=SURF, ms=7, label="this run, measured"),
                       Line2D([], [], ls=(0, (4, 3)), color=GRAY, marker="s", mfc=SURF, mec=GRAY, ms=5, label="old run, transcribed (own contrast only)")],
              loc="upper left", handlelength=2.0, borderaxespad=0.3)
    save(fig, out, "figA1_compounding")

def figA2(out):
    k = [2, 4, 8, 16, 32, 64]; gf = [.073, .245, .364, .427, .500, .545]; f = [.091, .164, .245, .455, .627, .736]     # CLAIMS_SPINE §2, transcribed
    fig, ax = plt.subplots(figsize=(4.2, 2.9))
    for y, c, lab in ((gf, ORANGE, "gf: error + failed code"), (f, AQUA, "f: error only")):
        ax.plot(k, y, color=c, lw=2, marker="s", ms=5.5, mfc=c, mec=SURF, mew=1.2, label=lab)
        ax.annotate(f"{y[-1]:.2f}", (64, y[-1]), xytext=(5, 0), textcoords="offset points", fontsize=8, color=INK, va="center")
    ax.set_xscale("log", base=2); ax.set_xticks(k); ax.set_xticklabels([str(v) for v in k]); ax.set_xlim(1.8, 90); ax.set_ylim(0, 0.8)
    ax.set_xlabel("attempt k"); ax.set_ylabel("fraction of trajectories with a first pass by k")
    ax.set_title("old run, transcribed — within-run contrast only", loc="left", color=INK2, fontsize=8.5)
    ax.legend(loc="upper left", handlelength=2.2, borderaxespad=0.3)
    save(fig, out, "figA2_old_run_crossing")

def figA3(outdir, out, Q, sim):
    arms = ["bare", "I", "N", "F", "F_E", "F_R", "F_C", "F_CE"]
    L = collections.defaultdict(list)
    for l in open(os.path.join(outdir, "out", "rollouts.jsonl")):
        try: d = json.loads(l)
        except Exception: continue
        if d["arm"] in arms: L[d["arm"]].append(len(d.get("text") or ""))
    Lp = collections.defaultdict(lambda: collections.defaultdict(list))
    for l in open(os.path.join(outdir, "out", "rollouts.jsonl")):
        try: d = json.loads(l)
        except Exception: continue
        if d["arm"] in arms: Lp[d["arm"]][d["id"]].append(len(d.get("text") or ""))
    length = {a: cluster_ci([Lp[a][p] for p in Q.pids], statistics.median) for a in arms}
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.6), gridspec_kw=dict(wspace=0.3))
    for a_, d, title, fmt, top in ((ax[0], sim, "a  similarity to the failed proof (mean)", lambda v: f"{v:.2f}", 0.66),
                                   (ax[1], length, "b  output length (median chars)", lambda v: f"{v:,.0f}", 4300)):
        x = np.arange(len(arms)); v = [d[a][0] for a in arms]
        a_.bar(x, v, width=0.58, color=[FAMILY[a] for a in arms], linewidth=0)
        whiskers(a_, x, [d[a] for a in arms])
        for xi, a, vi in zip(x, arms, v):
            if a in {"bare", "I", "F_C", "F_CE"}: a_.text(xi, d[a][2] + top * 0.015, fmt(vi), ha="center", va="bottom", fontsize=7.5, color=INK)
        a_.set_xticks(x); a_.set_xticklabels([tick(a) for a in arms], fontsize=8); a_.tick_params(axis="x", length=0)
        a_.set_ylim(0, top); a_.set_title(title, loc="left", color=INK, fontsize=9)
    fig.legend(handles=LEGEND, loc="upper center", ncol=5, handlelength=1.2, bbox_to_anchor=(0.5, 1.06), frameon=False)
    save(fig, out, "figA3_similarity_length")

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "figs"); os.makedirs(out, exist_ok=True)
    Q, P = qwen(HERE), prover(HERE)
    fig1(out, Q, P)
    sim = fig2(out, Q, P)
    fig3(out, Q, P)
    fig4(HERE, out)
    fig5(out, Q, P)
    fig6(HERE, out)
    figA1(HERE, out, Q)
    figA2(out)
    figA3(HERE, out, Q, sim)
