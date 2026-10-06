"""Record reproducible engineering evidence; never declares scientific success."""
from datetime import datetime, timezone
import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from recursive_ssd.io import atomic_json, digest, read_json, source_manifest


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",default="research/suite-v3/verification")
    args=parser.parse_args()
    out=ROOT/args.output
    out.mkdir(parents=True,exist_ok=True)
    commands=[[sys.executable,"-m","pytest","-q"],
        [sys.executable,"-m","pip","check"],
        [sys.executable,"-m","compileall","-q","recursive_ssd","scripts","tests"],
        [sys.executable,"-m","recursive_ssd.cli","--help"],
        [sys.executable,"scripts/research.py","--help"],
        [sys.executable,"-m","recursive_ssd.suite","--help"],
        ["bash","-n","scripts/setup.sh","scripts/run_8h.sh","scripts/python_env.sh"]]
    checks=[]
    for i,command in enumerate(commands):
        result=subprocess.run(command,cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        log=out/f"check-{i+1}.log"
        # Path aliases are disclosed; stdout text and exit code otherwise retained.
        content=result.stdout.replace(str(ROOT),"<PROJECT>").replace(str(Path(sys.prefix)),"<VERIFY_ENV>")
        log.write_text(content)
        checks.append({"command":["python" if x==sys.executable else x for x in command],
                       "exit_code":result.returncode,"log":str(log.relative_to(ROOT)),"sha256":digest(log)})
        if result.returncode:
            raise SystemExit(f"Check failed: {command}; see {log}")
    environment={"python":platform.python_version(),"platform":platform.platform(),
        "packages":{d.metadata["Name"]:d.version for d in importlib.metadata.distributions()},
        "device_scope":"CPU; no target CUDA GPU available; official native scorer qualification is a separate retained check"}
    atomic_json(out/"environment.json",environment)
    atomic_json(out/"engineering-checks.json",{"checked_at":datetime.now(timezone.utc).isoformat(),
        "source":source_manifest(ROOT),"checks":checks,"log_path_aliases":["<PROJECT>","<VERIFY_ENV>"],
        "scope":"engineering only; GPU and scientific method outcomes pending"})
    print(json.dumps({"checks":len(checks),"status":"passed","scope":"engineering only"}))


if __name__=="__main__":
    main()
