"""Verbatim copy of the pipeline's proof-file assembly (globals/utils.py: get_lean_preamble,
_extract_proof_body, extract_lean_code) so the rerun verifies the SAME file shape as the
original run: standard preamble + dataset statement up to `:=` + model proof body. The frozen
predecessors in bank.jsonl have exactly this shape. Copied (not imported) because globals/utils
pulls in the GPU stack, which the kimina-env verify job does not have."""
import re

PREAMBLE = ("import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 400000\n\n"
            "open BigOperators Real Nat Topology Rat\n\n")

def get_lean_preamble() -> str:
    return PREAMBLE

_LEAN_BLOCK_RE = re.compile(r"```lean4?\n(.*?)```", re.DOTALL)
_TACTIC_LEADING_TOKENS = ("by ", "by\n", "(", "fun ", "λ ", "⟨")

def _extract_proof_body(candidate: str, name: str) -> str:
    decl_re = re.compile(
        rf'(?:^|\n)\s*(?:theorem|lemma|def|abbrev)\s+{re.escape(name)}\b.*?:=(.*)',
        re.DOTALL,
    )
    m = decl_re.search(candidate)
    return m.group(1).strip() if m else ""

def extract_lean_code(proof_text: str, statement: str | None) -> str:
    if statement is None:
        return proof_text if proof_text.endswith("\n") else proof_text + "\n"
    stmt_decls = list(re.finditer(
        r'(?:^|\n)\s*(?:theorem|lemma|def|abbrev)\s+(\w+).*?:=',
        statement,
        re.DOTALL,
    ))
    if not stmt_decls:
        return statement.strip() + "\n"
    last_decl = stmt_decls[-1]
    name = last_decl.group(1)
    stmt_head = statement[:last_decl.end()]
    stmt_head = re.sub(r'(?m)^[ \t]*import[ \t]+\S+[ \t]*\n?', '', stmt_head).lstrip("\n")
    blocks = _LEAN_BLOCK_RE.findall(proof_text or "")
    candidate = (blocks[-1] if blocks else (proof_text or "")).strip()
    body = _extract_proof_body(candidate, name)
    if not body and candidate:
        if any(candidate.startswith(tok) for tok in _TACTIC_LEADING_TOKENS):
            body = candidate
    if not body:
        body = "extract_lean_code_no_body_parsed"
    return get_lean_preamble() + stmt_head + " " + body + "\n"
