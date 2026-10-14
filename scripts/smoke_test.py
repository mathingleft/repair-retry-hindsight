"""Small meaningful checks; optional live verifier probe, no GPU required."""
import argparse
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'analysis'),str(ROOT/'experiments/decomp_rerun')]


def main():
    p=argparse.ArgumentParser();p.add_argument('--kimina-url');a=p.parse_args()
    from lean_extract import extract_lean_code
    from verify_run import verdict,verify_codes
    from analyze_sequential import pooled
    for warning in ["declaration uses `sorry`","declaration uses 'sorry'",'declaration uses sorry']:
        assert verdict({'response':{'messages':[{'severity':'warning','data':warning}]}})[0]=='FAIL'
    assert verdict({'error':'REPL crashed'})[0]=='FAIL'
    assert verdict({'response':{'messages':[{'severity':'error','data':'type mismatch'}]}})[0]=='FAIL'
    assert verdict({'response':{'messages':[]}})[0]=='PASS'
    statement='theorem fixed_goal : (2 : Nat) + 2 = 5 := sorry'
    assembled=extract_lean_code('theorem fixed_goal : (2 : Nat) + 2 = 4 := by norm_num',statement)
    assert '2 = 5 :=' in assembled and '2 = 4 :=' not in assembled
    assert pooled(640,0,64)==0 and pooled(640,640,64)==1
    assert abs(pooled(4,1,2)-.5)<1e-12
    if a.kimina_url:
        items=[]
        for key,stmt,proof in [('valid','theorem fixed_goal : (2 : Nat) + 2 = 4 := sorry','by norm_num'),('false',statement,'by norm_num'),('admitted',statement,'by admit')]:
            items.append(dict(custom_id=key,proof=extract_lean_code(proof,stmt)))
        results=verify_codes(SimpleNamespace(kimina_url=a.kimina_url,timeout=120),items)
        got={k:verdict(v)[0] for k,v in results.items()}
        assert got=={'valid':'PASS','false':'FAIL','admitted':'FAIL'},got
    print('PASS: sorry warning variants, errors, fixed theorem assembly, and pooled estimator boundaries.'+(' Live Lean probes passed.' if a.kimina_url else ' No live service requested.'))


if __name__=='__main__':main()
