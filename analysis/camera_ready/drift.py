"""Toolchain drift: classify the unknown identifiers named by Lean diagnostics (app:drift, tab:drift).

    bash fetch_mathlib.sh && sbatch drift_names.sbatch     # libraries + constant-name dumps
    python drift.py [CACHE]                                # -> out/drift.json

Each diagnostic contributes its first "Unknown identifier/constant" name, extracted with the
artifact's own rule (decomp_rerun/error_use.py: unknown_ident). Names are checked against the
constant sets of two environments (dump_names.lean):
  new  the verifier's library, Mathlib 8a178386 / Lean v4.29.0
  old  DeepSeek-Prover-V2's evaluation-era library, DeepSeek's pinned mathlib4 fork
       (xinhjBrant/mathlib4 2f65ba7f, Lean v4.9.0-rc1; Mathlib content of upstream d1d1e4b7)
Names resolve as Lean would under the proof's `open` lines (the preamble opens BigOperators
Real Nat Topology Rat). Categories, first match wins:
  lean3      the Lean 3 keyword `begin`
  other      resolves in the new library, i.e. the name exists but was rejected for another reason
  renamed    resolves in the old library but not the new (rename target from the deprecation
             records of the intermediate releases, or of the old library itself, where one exists)
  namespace  resolves in neither, but NS.name is a constant of the new library for a namespace NS
             the theorem or proof mentions (ℂ counts as Complex, ℝ as Real, and so on)
  local      a bare name bound in the statement or the proof (hypothesis or variable)
  never      in neither library
A trailing projection (`.mp`, `.1`, ...) is stripped once before lookup when the full name is absent.
Reads only the frozen records under paper/artifact/.
"""
import collections
import json
import re
import os
import sys
from pathlib import Path

ARTIFACT = Path(__file__).resolve().parents[2] / 'artifact'
DATA = ARTIFACT / 'experiments/decomp_rerun'
OUT = Path(__file__).resolve().parent / 'out'
CACHE = Path(sys.argv[1] if len(sys.argv) > 1 else os.environ.get('DRIFT_CACHE', 'mathlib-drift-cache'))
MARKER = 'extract_lean_code_no_body_parsed'   # runner's no-proof-body placeholder
STATEMENT_PARSE = "unexpected token 'in'; expected ','"  # dataset statement the verifier cannot parse
QDIRS = ['out', 'out_ext', 'out_ext2', 'out_ext3', 'out_noi']
DDIRS = ['out_m2', 'out_m2_ext', 'out_m2_ext2', 'out_m2_ext3', 'out_m2_noi']
CATS = ['lean3', 'renamed', 'namespace', 'never', 'local', 'other']


def read(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(l) for l in f if l.strip()]


def unknown_ident(err):  # verbatim from decomp_rerun/error_use.py
    m = re.search(r"[Uu]nknown (?:identifier|constant)[ :]*[`'‘]?([^`'’\s]+)", err)
    return m.group(1) if m else None


def names(env):
    have, dep = {}, {}
    for line in open(CACHE / f'{env}.names.tsv', encoding='utf-8'):
        n, new, prot = line.rstrip('\n').split('\t')
        have[n] = prot == '1'
        if new: dep[n] = new
    return have, dep


def release_renames():
    """old name -> new name from the deprecation records of releases v4.10-v4.28: both
    `@[deprecated ...] alias old := new` and `@[deprecated new ...] theorem old`. Cached."""
    cache = CACHE / 'renames.json'
    if cache.exists(): return json.loads(cache.read_text())
    ren = {}
    decl = re.compile(r'(?:(?:protected|private|noncomputable|nonrec)\s+)*(?:theorem|lemma|def|abbrev|instance)\s+(\S+)')
    for path in sorted((CACHE / 'releases').glob('v4.*/Mathlib/**/*.lean')):
        text = path.read_text(encoding='utf-8', errors='replace')
        if 'deprecated' not in text: continue
        ns, target = [], None   # target: pending deprecation (new name or '' if none given)
        for line in text.splitlines():
            s = line.strip()
            if m := re.match(r'namespace\s+(\S+)', s): ns.extend(m.group(1).split('.')); continue
            if m := re.match(r'end\s+(\S+)', s):
                k = len(m.group(1).split('.'))
                if ns[-k:] == m.group(1).split('.'): del ns[-k:]
                continue
            if m := re.search(r'@\[[^\]]*deprecated\s*([^\s(\]]*)', s):
                target = m.group(1)
                s = s[s.index(']', m.end() - 1) + 1:].strip() if ']' in s[m.end() - 1:] else ''
            if target is None or not s: continue
            p = '.'.join(ns)
            def full(n): return n.removeprefix('_root_.') if n.startswith('_root_.') else (p + '.' + n if p else n)
            if m := re.match(r'(?:protected\s+)?alias\s+(\S+)\s*:=\s*(\S+)', s):
                ren.setdefault(full(m.group(1)), m.group(2) if '.' in m.group(2) else full(m.group(2))); target = None
            elif m := decl.match(s):
                t = '' if not target or target.startswith('"') else (target if '.' in target else full(target))
                ren.setdefault(full(m.group(1)), t); target = None
            elif not s.startswith(('@[', '/-', '--')): target = None
    cache.write_text(json.dumps(ren, ensure_ascii=False))
    return ren


# A name is local when a binder or tactic introduces it: (x : T), fun x, intro x, have x, ⟨x, y⟩, ...
BINDER_RES = [r"[({⦃]\s*([^:(){}⦃⦄\n]*?)\s*:", r"\b(?:fun|λ|∀|∃!?)\s+([^,:=→↦\n]+)",
              r"\b(?:intro|intros|rintro|introv|by_contra!?|generalize|set|let|have|haveI|obtain|next)\b([^\n:;]*)",
              r"\bwith\b([^\n:;]*)", r"\bcase\s+\S+([^\n=]*)=>", r"⟨([^⟩\n]*)⟩"]
TOKEN = re.compile(r"[^\s,|⟨⟩()\[\]{}@:=]+")


def bound_names(text):
    return {t for pat in BINDER_RES for m in re.finditer(pat, text) for t in TOKEN.findall(m.group(1))}


PROJ = {'mp', 'mpr', '1', '2', 'symm', 'trans', 'le', 'ge', 'lt', 'ne', "ne'", 'elim', 'left', 'right',
        'resolve_left', 'resolve_right', 'out', 'some', 'mono', 'not', 'antisymm', 'trans_le', 'trans_lt'}
TYPE_NS = {'ℂ': 'Complex', 'ℝ': 'Real', 'ℕ': 'Nat', 'ℤ': 'Int', 'ℚ': 'Rat', 'ℝ≥0': 'NNReal', 'ZMod': 'ZMod', 'Finset': 'Finset'}


def is_local(x, statement, code):
    return '.' not in x and x in bound_names(statement + '\n' + (code or ''))


def resolve(x, opens, env):
    """The constant `x` names in env under the opened namespaces, else None (Lean's lookup order
    aside, any match counts). A trailing projection is stripped once if the full name is absent."""
    stems = [x] + ([x.rsplit('.', 1)[0]] if '.' in x and x.rsplit('.', 1)[1] in PROJ else [])
    for st in stems:
        if st in env: return st
        for ns in opens:  # `open` does not expose protected declarations
            if env.get(f'{ns}.{st}') is False: return f'{ns}.{st}'
    return None


def classify(x, statement, code, OLD, NEW, OLDDEP, ren):
    if x == 'begin': return 'lean3', None
    opens = [o for line in re.findall(r'^open\s+(.+)$', code or '', flags=re.M) for o in line.split()]
    if n := resolve(x, opens, NEW): return 'other', n
    if o := resolve(x, opens, OLD): return 'renamed', ren.get(o) or OLDDEP.get(o)
    text = statement + '\n' + (code or '')
    spaces = set(re.findall(r'\b([A-Z]\w*)\.', text)) | {ns for sym, ns in TYPE_NS.items() if sym in text}
    for ns in sorted(spaces):
        if f'{ns}.{x}' in NEW: return 'namespace', f'{ns}.{x}'
    if is_local(x, statement, code): return 'local', None
    return 'never', None


def main():
    NEW, _ = names('new'); OLD, OLDDEP = names('old'); ren = release_renames()
    qbank = {r['id']: r for r in read(DATA / 'bank.jsonl')}
    dbank = {r['id']: r for r in read(DATA / 'out_m2/bank2.jsonl')}
    qpids = {r['id'] for r in read(DATA / 'out/verified.jsonl') if r['arm'] == 'bare'}
    res = {'libraries': {'new': 'mathlib4 8a178386ffc0f5fef0b77738bb5449d50efeea95, Lean v4.29.0',
                         'old': 'xinhjBrant/mathlib4 2f65ba7f1a9144b20c8e7358513548e317d26de1 '
                                '(upstream d1d1e4b72c6e009edd97cabe1d661b021fe8eeae), Lean v4.9.0-rc1',
                         'constants': {'new': len(NEW), 'old': len(OLD)}, 'deprecation_aliases': len(ren)}}
    for model, bank, pids, dirs, seq in [('qwen', qbank, qpids, QDIRS, 'out'), ('deepseek', dbank, set(dbank), DDIRS, 'out_dsp')]:
        pred, idents = collections.Counter(), collections.defaultdict(collections.Counter)
        for p in sorted(pids):
            u = unknown_ident(bank[p]['predecessor_error'])
            if u and u not in bank[p]['statement']:
                c, tgt = classify(u, bank[p]['statement'], bank[p]['predecessor_code'], OLD, NEW, OLDDEP, ren)
                pred[c] += 1; idents[('pred', c)][f'{u} -> {tgt}' if tgt else u] += 1
        samp, failed, parse = collections.Counter(), 0, 0
        for d in dirs:
            for r in read(DATA / d / 'verified.jsonl'):
                e = r.get('error') or ''
                if r['status'] == 'PASS' or r['id'] not in pids or MARKER in e: continue
                failed += 1; parse += STATEMENT_PARSE in e
                if not (u := unknown_ident(e)): continue
                c, tgt = classify(u, bank[r['id']]['statement'], r.get('lean_code'), OLD, NEW, OLDDEP, ren)
                samp[c] += 1; idents[('samples', c)][f'{u} -> {tgt}' if tgt else u] += 1
        seqc, seqf = collections.Counter(), 0
        for path in sorted((DATA / seq).glob('depth_*.jsonl')):
            if path.stem in ('depth_memI', 'depth_blind_I'): continue  # memoryless pools: no history
            for r in read(path):
                e = r.get('error') or ''
                if r.get('status') == 'PASS' or MARKER in e: continue
                seqf += 1
                if u := unknown_ident(e):
                    seqc[classify(u, bank[r['id']]['statement'] if r['id'] in bank else '', r.get('code'), OLD, NEW, OLDDEP, ren)[0]] += 1
        n = sum(samp.values()); sn = sum(seqc.values())
        res[model] = {
            'predecessors': {c: pred[c] for c in CATS} | {'total': sum(pred.values())},
            'failed_samples': {c: samp[c] for c in CATS} | {'total': n},
            'failed_one_step_samples': failed,
            'share_failures_with_unknown_identifier': round(n / failed, 4),
            'statement_parse_failures': parse,
            'share_failures_with_unknown_identifier_excl_statement_parse': round(n / (failed - parse), 4),
            'share_unknown_renamed': round(samp['renamed'] / n, 4) if n else None,
            'sequential_adaptive': {'failed_attempts': seqf, 'with_unknown_identifier': sn,
                                    'share_failures': round(sn / seqf, 4) if seqf else None,
                                    'share_renamed': round(seqc['renamed'] / sn, 4) if sn else None,
                                    'by_category': {c: seqc[c] for c in CATS}},
            'identifiers': {f'{k[0]}:{k[1]}': dict(v.most_common()) for k, v in sorted(idents.items())},
        }
    OUT.mkdir(exist_ok=True)
    (OUT / 'drift.json').write_text(json.dumps(res, indent=2, ensure_ascii=False) + '\n')
    for m in ('qwen', 'deepseek'):
        print(m, res[m]['predecessors'], res[m]['failed_samples'], res[m]['share_failures_with_unknown_identifier_excl_statement_parse'],
              res[m]['share_unknown_renamed'], res[m]['sequential_adaptive']['share_failures'], res[m]['sequential_adaptive']['share_renamed'])


if __name__ == '__main__':
    main()
