"""Reproduce the two exploratory one-step interactions adopted in the revision.

The unit is a Qwen problem and each arm outcome is pass@4: whether at least one
of the four retained continuations verifies. A bootstrap draw resamples problems
and carries every arm in an estimand together, preserving all pairing.
"""
from collections import defaultdict
import random

from common import DATA, QDIRS, read, dump


SEED = 20260913
RESAMPLES = 20_000
DEFINITIONS = {
    "matched_diagnostic_by_proof_exposure": {
        "formula": "(F_CE - F_CE_match) - (F_E - F_E_match)",
        "manuscript_formula": "(F_CE - F_CE_m) - (F_E - F_E_m)",
        "arms": ["F_CE", "F_CE_match", "F_E", "F_E_match"],
        "signs": [1, -1, -1, 1],
    },
    "instruction_by_diagnostic_recovery": {
        "formula": "(F_CE - F_C) - (F_CE_noI - F_C_noI)",
        "manuscript_formula": "(F_CE - F_C) - (F_CE_noI - F_C_noI)",
        "arms": ["F_CE", "F_C", "F_CE_noI", "F_C_noI"],
        "signs": [1, -1, -1, 1],
    },
}


def load_pass4():
    rows = defaultdict(list)
    for directory in QDIRS:
        for row in read(DATA / directory / "verified.jsonl"):
            if row.get("arm") in {a for d in DEFINITIONS.values() for a in d["arms"]}:
                rows[row["arm"], row["id"]].append(row)

    # Preserve the frozen bank order used by the review calculation. Bootstrap
    # draws are exchangeable in principle, but a finite seeded draw stream can
    # shift percentile endpoints if the problem order is changed.
    problem_ids = [
        row["id"] for row in read(DATA / "bank.jsonl")
        if not row.get("is_framing_holdout")
    ]
    assert len(problem_ids) == 167, len(problem_ids)
    values = {}
    for definition in DEFINITIONS.values():
        for arm in definition["arms"]:
            assert all((arm, pid) in rows for pid in problem_ids), arm
            for pid in problem_ids:
                samples = rows[arm, pid]
                keys = [r["sample_idx"] for r in samples]
                assert len(keys) == 4 and sorted(keys) == [0, 1, 2, 3], (arm, pid, keys)
                assert len(set(keys)) == 4, (arm, pid)
            values[arm] = [int(any(r["status"] == "PASS" for r in rows[arm, pid])) for pid in problem_ids]
    return problem_ids, values


def effect(values, definition, indices):
    return 100 * sum(
        sign * values[arm][i]
        for i in indices
        for arm, sign in zip(definition["arms"], definition["signs"])
    ) / len(indices)


def main():
    problem_ids, values = load_pass4()
    rng = random.Random(SEED)
    # Use Random.choices exactly as in the review calculation. It is
    # distributionally equivalent to repeated randrange calls but consumes a
    # different seeded RNG stream, which matters for exact finite-run endpoints.
    draws = [rng.choices(range(len(problem_ids)), k=len(problem_ids)) for _ in range(RESAMPLES)]
    results = {}
    for name, definition in DEFINITIONS.items():
        observed = effect(values, definition, range(len(problem_ids)))
        distribution = sorted(effect(values, definition, draw) for draw in draws)
        lo_index = int(0.025 * RESAMPLES)
        hi_index = int(0.975 * RESAMPLES) - 1
        results[name] = {
            **definition,
            "arm_solved_counts": {arm: sum(values[arm]) for arm in definition["arms"]},
            "estimate_pp": observed,
            "ci95_pp": [distribution[lo_index], distribution[hi_index]],
        }
    output = {
        "status": "exploratory_unadjusted",
        "unit": "problem",
        "outcome": "per-problem pass@4 indicator",
        "model": "Qwen3.6-27B",
        "problem_count": len(problem_ids),
        "problem_ids": problem_ids,
        "seed": SEED,
        "resamples": RESAMPLES,
        "interval": {
            "method": "paired problem bootstrap with sorted percentile endpoints",
            "lower_index_zero_based": lo_index,
            "upper_index_zero_based": hi_index,
            "all_estimand_arms_carried_in_each_draw": True,
        },
        "results": results,
    }
    dump("exploratory_onestep_interactions.json", output)
    for name, result in results.items():
        print(name, round(result["estimate_pp"], 2), [round(v, 2) for v in result["ci95_pp"]])


if __name__ == "__main__":
    main()
