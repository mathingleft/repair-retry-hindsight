"""Select problems into a difficulty band from a pass-rate source.

Productizes the previously-manual step of hand-building a band file (e.g.
`hard_band_zero.jsonl`): point this at something that measured per-problem
pass rate, give it a band `[lo, hi]`, and it writes a `{id, statement}` JSONL
of the problems whose pass rate falls in that band — ready to feed straight
into `build_replay_prompts.py --problems` or any eval `--input`.

Why a band (not just "hardest"): to *detect* a prompt effect you need problems
with measurable headroom. Saturated (≈100%) and floor (≈0%) problems can't move,
so an A/B there is uninformative. The mid-band (default 5–50%) is where a lift
is visible.

Sources (auto-detected):
  * a directory            — looks for verified.jsonl, then attempts.jsonl
  * an eval verified.jsonl — rows {id, statement, status, ...}
  * a TTT attempts.jsonl   — rows {problem_id, arm, status, ...}; pass `--arm
                             parallel` to score off the i.i.d. baseline (the
                             clean difficulty measure — the sequential arm
                             accumulates context and is NOT a difficulty proxy)
  * an eval summary.json   — uses problems_by_difficulty[].pass_rate

Statements come from the source when present; otherwise pass `--problems` (a
pool JSONL with {id, statement}) to join them in (TTT attempts carry no
statement).

Examples
--------
Mid-band off a TTT run's parallel arm::

    python3 experiments/data-local/select_problems.py \\
      --from     $SCRATCH/runs/ttt-hard26-v49-20260524/goedel-32b__v49__B8_curate/ttt_logs/attempts.jsonl \\
      --arm      parallel \\
      --problems $SCRATCH/data/hard_problems_v49.jsonl \\
      --band     0.05 0.5 \\
      --out      $SCRATCH/data/midband_v49.jsonl

Mid-band off a base eval::

    python3 experiments/data-local/select_problems.py \\
      --from $SCRATCH/eval/.../verified.jsonl --band 0.05 0.5 \\
      --out  $SCRATCH/data/midband.jsonl
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "experiments"))

from globals.utils import read_jsonl, write_jsonl, log    # noqa: E402


def _resolve_source(src: str) -> Path:
    """A dir resolves to its verified.jsonl (preferred) or attempts.jsonl."""
    p = Path(src)
    if p.is_dir():
        for cand in ("verified.jsonl", "ttt_logs/attempts.jsonl", "attempts.jsonl"):
            if (p / cand).exists():
                return p / cand
        raise FileNotFoundError(f"{p}: no verified.jsonl or attempts.jsonl found")
    if not p.exists():
        raise FileNotFoundError(p)
    return p


def pass_rates_from_summary(path: Path) -> dict[str, dict]:
    """eval/ttt summary.json -> {id: {pass_rate, n_pass, n_attempts, statement}}."""
    import json
    s = json.loads(path.read_text())
    probs = s.get("problems_by_difficulty") or s.get("problems") or []
    out: dict[str, dict] = {}
    for pr in probs:
        pid = pr.get("id")
        if pid is None:
            continue
        out[pid] = {
            "pass_rate":   pr.get("pass_rate", pr.get("P_par", pr.get("p_par"))),
            "n_pass":      pr.get("n_pass"),
            "n_attempts":  pr.get("n_attempts"),
            "statement":   pr.get("statement"),
        }
    return out


def pass_rates_from_attempts(path: Path, arm: str | None) -> dict[str, dict]:
    """per-attempt rows (verified.jsonl or attempts.jsonl) -> per-problem pass rate.

    `id` is read from `id` or `problem_id`. If `arm` is given, only rows with
    that `arm` are counted (use `parallel` for the TTT i.i.d. baseline).
    """
    n_pass: dict[str, int]  = defaultdict(int)
    n_tot:  dict[str, int]  = defaultdict(int)
    stmt:   dict[str, str]  = {}
    for r in read_jsonl(str(path)):
        if arm is not None and r.get("arm") != arm:
            continue
        pid = r.get("id") or r.get("problem_id")
        if pid is None:
            continue
        n_tot[pid]  += 1
        n_pass[pid] += int(r.get("status") == "PASS")
        if r.get("statement"):
            stmt[pid] = r["statement"]
    return {
        pid: {"pass_rate": n_pass[pid] / n_tot[pid], "n_pass": n_pass[pid],
              "n_attempts": n_tot[pid], "statement": stmt.get(pid)}
        for pid in n_tot
    }


def load_pass_rates(src: str, arm: str | None) -> dict[str, dict]:
    path = _resolve_source(src)
    if path.suffix == ".json":
        if arm is not None:
            log(f"  NOTE --arm {arm!r} ignored for summary.json source")
        return pass_rates_from_summary(path)
    return pass_rates_from_attempts(path, arm)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="source", required=True,
                    help="dir / verified.jsonl / attempts.jsonl / summary.json")
    ap.add_argument("--arm", default=None,
                    help="filter per-attempt rows to this arm (e.g. 'parallel' for TTT)")
    ap.add_argument("--band", nargs=2, type=float, default=[0.05, 0.5],
                    metavar=("LO", "HI"), help="keep pass_rate in [LO, HI] inclusive")
    ap.add_argument("--problems", default=None,
                    help="pool JSONL {id, statement} to join statements from "
                         "(required when the source carries no statements)")
    ap.add_argument("--out", required=True, help="output problem JSONL {id, statement}")
    args = ap.parse_args()

    lo, hi = args.band
    rates = load_pass_rates(args.source, args.arm)
    log(f"  source: {args.source}  ({'arm='+args.arm+', ' if args.arm else ''}{len(rates)} problems scored)")

    # Join statements from the pool when missing on the source rows.
    pool: dict[str, str] = {}
    if args.problems:
        pool = {r["id"]: r["statement"] for r in read_jsonl(args.problems) if "statement" in r}

    rows, missing_stmt, in_band = [], [], []
    for pid, d in rates.items():
        pr = d["pass_rate"]
        if pr is None or not (lo <= pr <= hi):
            continue
        in_band.append((pid, pr, d.get("n_pass"), d.get("n_attempts")))
        stmt = d.get("statement") or pool.get(pid)
        if not stmt:
            missing_stmt.append(pid)
            continue
        rows.append({"id": pid, "statement": stmt, "pass_rate": round(pr, 4)})

    rows.sort(key=lambda r: r["pass_rate"])
    write_jsonl(args.out, rows)

    log(f"  band [{lo}, {hi}]: {len(in_band)} problems in band, {len(rows)} written → {args.out}")
    for pid, pr, npass, ntot in sorted(in_band, key=lambda x: x[1]):
        frac = f"{npass}/{ntot}" if npass is not None and ntot else ""
        log(f"    {pr:6.1%}  {frac:>8}  {pid}")
    if missing_stmt:
        log(f"  WARNING {len(missing_stmt)} in-band ids had no statement (source lacks it; "
            f"pass --problems with a pool containing them): {missing_stmt[:8]}")


if __name__ == "__main__":
    main()
