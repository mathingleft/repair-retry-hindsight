"""Build pinned public Lean dependencies in a NEW directory (network required)."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);a=p.parse_args()
    dest=a.directory.resolve()
    if dest.exists():raise SystemExit('Choose a new directory; existing services are never overwritten.')
    if not shutil.which('elan'):raise SystemExit('Install elan and put its bin directory on PATH first.')
    pins=json.loads((ROOT/'environments/revisions.json').read_text())
    def run(*cmd,cwd=None):subprocess.run(list(cmd),cwd=cwd,check=True)
    run('git','clone','https://github.com/project-numina/kimina-lean-server.git',str(dest))
    run('git','checkout','--detach',pins['kimina'],cwd=dest)
    for name,url in [('mathlib4','https://github.com/leanprover-community/mathlib4.git'),('repl','https://github.com/leanprover-community/repl.git')]:
        run('git','clone',url,str(dest/name))
        run('git','checkout','--detach',pins['mathlib' if name=='mathlib4' else 'repl'],cwd=dest/name)
        for filename in ['lake-manifest.json','lean-toolchain']:
            shutil.copyfile(ROOT/'environments'/name/filename,dest/name/filename)
    run('elan','toolchain','install','leanprover/lean4:'+pins['lean'])
    run('lake','exe','cache','get',cwd=dest/'mathlib4')
    run('lake','build',cwd=dest/'mathlib4');run('lake','build',cwd=dest/'repl')
    run(sys.executable,'-m','venv',str(dest/'.venv'))
    run(str(dest/'.venv/bin/python'),'-m','pip','install','-e',str(dest)+'[server]')
    run(str(dest/'.venv/bin/prisma'),'generate',cwd=dest)
    print('Built service at',dest,'; see ENVIRONMENT.md for startup and probes.')


if __name__=='__main__':main()
