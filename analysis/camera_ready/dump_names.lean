-- Print every constant in the Mathlib environment, one per line, with the target of
-- its deprecation attribute if any and whether it is protected (not reachable through `open`):
--   name<TAB>new_name_or_empty<TAB>1_if_protected
-- Run from a Mathlib checkout:  lake env lean dump_names.lean  (writes $DRIFT_OUT)
import Mathlib
open Lean Elab Command in
#eval show CommandElabM Unit from do
  let env ← getEnv
  let out := (← IO.getEnv "DRIFT_OUT").getD "names.tsv"
  let h ← IO.FS.Handle.mk out .write
  for (n, _) in env.constants.toList do
    let dep := match Lean.Linter.deprecatedAttr.getParam? env n with
      | some e => (e.newName?.map toString).getD "?"
      | none => ""
    h.putStrLn s!"{n}\t{dep}\t{if isProtected env n then 1 else 0}"
