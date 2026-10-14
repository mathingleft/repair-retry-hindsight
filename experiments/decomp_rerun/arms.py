"""Prompt builders for the 167-problem content-decomposition rerun.

Six arms. Every non-bare arm shares the IDENTICAL failure sentence; arms differ
only in which blocks are appended after it. This is a reconstruction of the original
purged 2873630 design (bare/failure_only/compiler/reflection) extended with the
{code}x{error} cells so code-presence and error-presence are separated.

Phrasing is modeled on the surviving probe-v2 baked prompts (v1/v4).
"""
import os

FAIL_SENT = "You previously attempted to prove this theorem and it did not verify."
TASK = "Complete the following Lean 4 code:\n\n```lean4\n{statement}\n```"

# Second-model runs (2026-08-28, DeepSeek-Prover-V2-7B): that prover's native prompt puts the Mathlib
# header inside the code block. DECOMP_TASK_HEADER=1 prepends it; unset (default) leaves every existing
# prompt byte-identical. Same string lean_extract.py splices in front of the statement.
_HEADER = ("import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 400000\n\n"
           "open BigOperators Real Nat Topology Rat\n\n")

def _task(problem):
    # statements in bank.jsonl end with `:= sorry`; strip the sorry so the
    # model writes the proof (house style, matches null16k/probe prompts).
    import os
    s = problem["statement"]
    if s.rstrip().endswith("sorry"):
        s = s.rstrip()[: -len("sorry")].rstrip()
    if os.environ.get("DECOMP_TASK_HEADER") == "1":
        s = _HEADER + s
    return TASK.format(statement=s)

def _code_block(problem):
    return f"This was the failed proof:\n\n```lean4\n{problem['predecessor_code']}\n```"

def _error_block(problem):
    return f"Lean compiler feedback from the failed attempt:\n{problem['predecessor_error']}"

def _reflection_block(reflection):
    return f"Reflection on what went wrong:\n{reflection}"

def build_prompt(arm, problem, reflection=None):
    # `_noI` suffix (2026-08-28 evening, NOI_ARMS): the same arm with the instruction line omitted. N_noI predates
    # the general rule and is unchanged by it (base N, no instruction).
    noI = arm.endswith("_noI"); base = arm[:-4] if noI else arm
    parts = []
    if base == "N":
        parts.append(NOISE_SENT)
    elif base in NOISE_SENTS:
        parts.append(NOISE_SENTS[base])
    elif base in POSTHOC_SENT:
        parts.append(POSTHOC_SENT[base])
    elif base == "I":
        pass                                   # instruction line only
    elif base != "bare":
        parts.append(FAIL_SENT)
    if base in ("F_C", "F_CE", "F_CE_shuf", "F_CE_match"):
        parts.append(_code_block(problem))
    if base == "F_C_other":
        parts.append(_code_block({"predecessor_code": shuffled_code(problem["id"])}))
    if base in ("F_E", "F_CE"):
        parts.append(_error_block(problem))
    if base in ("F_E_shuf", "F_CE_shuf"):
        parts.append(_error_block({"predecessor_error": shuffled_error(problem["id"])}))
    if base in ("F_E_match", "F_CE_match"):
        parts.append(_error_block({"predecessor_error": matched_error(problem["id"])}))
    if base == "F_R":
        assert reflection, f"F_R needs a reflection for {problem['id']}"
        parts.append(_reflection_block(reflection))
    parts.append("Produce a correct, complete proof." if (base != "bare" and not noI) else None)
    parts.append(_task(problem))
    return "\n\n".join(p for p in parts if p)

# NO-INSTRUCTION ARMS (post-hoc, 2026-08-28 ~23:50 PT, user request): the feedback ladder with the instruction
# line removed. Every existing arm except bare carries "Produce a correct, complete proof."; these are the same
# scaffolds without it, so X vs X_noI is the instruction's worth on top of X, and X_noI vs bare is the feedback
# content's worth with no instruction to saturate it (F_E_noI vs F_noI: does the error help when nothing else
# lifts the model? F_E_shuf_noI: the wrong-error control in the same regime). Generated into out_noi/ by
# cluster/gpu_run_noi.sbatch; analysed by analyze_noi.py. Not in ARMS (row-count guards).
NOI_ARMS = ["F_noI", "F_E_noI", "F_R_noI", "F_C_noI", "F_CE_noI", "F_E_shuf_noI"]

ARMS = ["bare", "F", "F_E", "F_C", "F_CE", "F_R", "N", "I", "S", "P", "D", "N2", "N3", "N4", "N5"]

# EXTENSION ARMS (post-hoc, 2026-08-28 ~02:45 PT, added while the I/S/P/D + N2-N5 chain was still running).
# Deliberately NOT in ARMS: the running gen job's retry loop and the verify guards derive their row counts from
# len(ARMS). These are generated into out_ext/ by cluster/gpu_run_ext.sbatch and analysed by analyze_ext.py.
#   F_E_shuf : the F_E scaffold, but the error block is ANOTHER problem's compiler error (fixed derangement of
#              the 171 bank ids under random.Random(22072605), no problem keeps its own error). If
#              F_E_shuf ~= F_E, the error's content is interchangeable with a wrong error: the F_E twin of N.
#   N_noI    : NOISE_SENT + task with NO instruction line. Completes the 2x2 {sentence: none/N} x
#              {instruction line: no/yes} with bare (none/no), I (none/yes), N (N/yes).
EXT_ARMS = ["F_E_shuf", "N_noI"]
# EXT2 (post-hoc, 2026-08-31, user request): F_CE_shuf = the F_CE scaffold verbatim (failure sentence +
# the problem's OWN failed code + error block + instruction line) but the error text is ANOTHER problem's —
# the SAME seeded derangement as F_E_shuf (random.Random(22072605) over the bank in use, so per problem the
# drawn error is byte-identical to the F_E_shuf arm's). Tests whether error content matters once the failed
# code is present. Run on BOTH models: Qwen -> out_ext2/ (bank.jsonl), prover -> out_m2_ext2/ (bank2 via
# SHUF_BANK). Not in ARMS (row-count guards).
EXT2_ARMS = ["F_CE_shuf"]
# EXT3 (post-hoc, 2026-09-06, user request). Three arms, both models (Qwen -> out_ext3/, prover -> out_m2_ext3/):
#   F_C_other  : the F_C scaffold verbatim (failure sentence + code block + instruction + task) but the code block
#                holds the failed proof of ANOTHER problem — the SAME partner the seeded derangement gives
#                F_E_shuf / F_CE_shuf (random.Random(22072605); see shuffled_partner), so per problem the wrong
#                code comes from the very problem whose wrong error the shuf arms showed. Wrong code vs own code.
#   F_E_match  : the F_E scaffold, error block = another problem's error MATCHED to the problem's own error on
#                (1) coarse Lean error class, (2) token count, (3) not the same theorem, (4) each error reused
#                as evenly as possible (min-cost perfect matching within class; build_match.py, bank_match.json).
#   F_CE_match : the F_CE scaffold (own failed code) with that same matched error.
#   match vs shuf isolates the error's *specific content* from its class + length: if F_E_match ~= F_E but
#   F_E_shuf < F_E, the error's value is its class/shape, not its problem-specific detail.
EXT3_ARMS = ["F_C_other", "F_E_match", "F_CE_match"]
_SHUF = {}
_MATCH = {}
SHUF_BANK = None      # set by run_gen.py to its --bank so a second model's F_E_shuf draws from ITS OWN bank's errors (2026-08-29)
def _bank_path(bank_path=None):
    import os
    return bank_path or SHUF_BANK or os.path.join(os.path.dirname(os.path.abspath(__file__)), "bank.jsonl")
def _load_bank(path):
    import json
    return {json.loads(l)["id"]: json.loads(l) for l in open(path) if l.strip()}
def shuffled_partner(problem_id, bank_path=None):
    """Id of the ANOTHER-problem partner used by F_E_shuf / F_CE_shuf / F_C_other: seeded draw under
    random.Random(22072605) over the bank's sorted ids, candidates = problems whose error text differs from the
    problem's own (so never itself). Byte-identical rng sequence to the pre-2026-09-06 shuffled_error(), which
    only kept the drawn error text (checked in build_match.py --selftest)."""
    import random
    path = _bank_path(bank_path)
    if path not in _SHUF:
        bank = {k: v["predecessor_error"] for k, v in _load_bank(path).items()}
        rng = random.Random(22072605); m = {}
        for pid in sorted(bank):
            others = [q for q in sorted(bank) if bank[q] != bank[pid]]
            m[pid] = rng.choice(others)
        _SHUF[path] = m
    return _SHUF[path][problem_id]
def shuffled_error(problem_id, bank_path=None):
    """Error text of ANOTHER problem in the same bank (see shuffled_partner). Default bank = bank.jsonl (unchanged
    since 2026-08-28); SHUF_BANK / bank_path selects e.g. out_m2/bank2.jsonl for the prover."""
    path = _bank_path(bank_path)
    return _load_bank(path)[shuffled_partner(problem_id, path)]["predecessor_error"]
def shuffled_code(problem_id, bank_path=None):
    """Failed proof code of the SAME partner problem F_E_shuf drew (F_C_other)."""
    path = _bank_path(bank_path)
    return _load_bank(path)[shuffled_partner(problem_id, path)]["predecessor_code"]
def match_path(bank_path=None):
    p = _bank_path(bank_path)
    return (p[:-len(".jsonl")] if p.endswith(".jsonl") else p) + "_match.json"
def matched_partner(problem_id, bank_path=None):
    """Id of the class/length-matched error source (build_match.py -> <bank>_match.json)."""
    import json
    mp = match_path(bank_path)
    if mp not in _MATCH:
        _MATCH[mp] = {k: v["src"] for k, v in json.load(open(mp))["map"].items()}
    return _MATCH[mp][problem_id]
def matched_error(problem_id, bank_path=None):
    path = _bank_path(bank_path)
    return _load_bank(path)[matched_partner(problem_id, path)]["predecessor_error"]

# POST-HOC ablations of the failure sentence (2026-08-27 ~23:30 PT, user request; not prereg). All use the
# F scaffold (sentence slot + "Produce a correct, complete proof." + task); sentences are 13 Qwen3.6 tokens
# like FAIL_SENT:
#   I : no sentence at all (instruction line only)      -> isolates the repair-instruction line
#   S : same sentence but the attempt DID verify         -> isolates "you failed" vs "a prior attempt exists"
#   P : "Someone" instead of "You" (one-word edit)       -> isolates second-person framing
#   D : difficulty/caution cue, no failure claim         -> isolates "be careful / try harder"
POSTHOC_SENT = {
    "S": "You previously attempted to prove this theorem and it did fully verify.",
    "P": "Someone previously attempted to prove this theorem and it did not verify.",
    "D": "This theorem is quite difficult and requires a careful, complete proof.",
}
# POST-HOC (2026-08-27 23:30 PT, added after the six prereg arms were generated and analysed; not in the
# preregistration): N = the F scaffold with the failure sentence replaced by random words of the SAME
# Qwen3.6 token count (13). Isolates "the model is told it failed" from "13 extra tokens + the repair
# instruction". Words sampled from /usr/share/dict/words with random.Random(22072600); fixed for all problems.
NOISE_SENT = "Ennui inward evacuates catchier sequel tubing lubbers."
# N2-N5 (post-hoc, 2026-08-28 ~00:20 PT): four more independent random-word sentences, each word drawn
# uniformly from /usr/share/dict/words (lowercase alpha, 3-9 letters) under random.Random(seed), resampled
# until exactly 13 Qwen3.6 tokens. Seeds 22072601..22072604. Replicates of N to show the N result is not
# specific to one string.
NOISE_SENTS = {
    "N2": "Rosaries fanfares wanton rice dale dirtballs.",
    "N3": "Agreeable halting metaphor stunned obsidian minuend chose.",
    "N4": "Freelance fossil plenary vocatives erode cinched.",
    "N5": "Blue retriever workloads abortions panties mavens whittled.",
}
# bare  : problem only
# F     : + failure sentence            (content-free)
# F_E   : + compiler error              (original 'compiler' arm, code ABSENT)
# F_C   : + failed proof code           (new cell: code without error)
# F_CE  : + code + error                (new cell: the standard practice)
# F_R   : + model-written reflection    (original 'reflection' arm)

REFLECTOR_PROMPT = """You are an expert Lean 4 proof coach. A proof attempt failed to verify.

Theorem:
```lean4
{statement}
```

Failed proof:
```lean4
{predecessor_code}
```

Lean compiler feedback:
{predecessor_error}

Write a 3-5 sentence coaching critique of what went wrong and what to do differently on the next attempt. Do not write any Lean code."""

def build_reflector_prompt(problem):
    return REFLECTOR_PROMPT.format(**problem)
