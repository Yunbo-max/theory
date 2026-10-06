import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import time
from .io import atomic_json, digest, event, object_hash, read_json, source_manifest


def launch(args):
    from .data import verify_data
    from .evaluation import evaluator_info
    from .runner import hardware, report, validate_config
    config=validate_config(read_json(args.config))
    run=Path(args.run_dir).resolve()
    artifacts=Path(args.artifacts).resolve()
    run.mkdir(parents=True,exist_ok=True)
    with (run/"run.lock").open("w") as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("another supervisor owns this run directory")
        hardware_info=hardware()
        verify_data(artifacts,config)
        preflight=read_json(artifacts/"preflight/preflight.json")
        if preflight["status"]!="passed" or preflight["config_sha256"]!=object_hash(config):
            raise RuntimeError("run preflight for this exact configuration first")
        if preflight["hardware"]["name"]!=hardware_info["name"]:
            raise RuntimeError("GPU differs from preflight")
        evaluator=evaluator_info()
        source=source_manifest(Path(__file__).resolve().parents[1])
        if preflight.get("evaluator")!=evaluator or preflight.get("source_sha256")!=source["sha256"]:
            raise RuntimeError("native environment/source differs from preflight; rerun preflight")
        if not 0<args.hours<=8:
            raise ValueError("this authorization caps each run window at eight hours")
        start=datetime.fromisoformat(args.start_at).timestamp() if args.start_at else time.time()
        if args.start_at and datetime.fromisoformat(args.start_at).tzinfo is None:
            raise ValueError("--start-at must include a timezone offset, e.g. +01:00")
        if (run/"run.json").exists():
            state=read_json(run/"run.json")
            if state["config"]!=config or state["artifacts_manifest_sha"]!=digest(artifacts/"manifest.json"):
                raise ValueError("resume config/data differs from frozen run")
            if state.get("evaluator")!=evaluator or state["source"]["sha256"]!=source["sha256"]:
                raise ValueError("resume source/native evaluator differs; preserve the original run and deadline")
            start=state["start_epoch"]
        else:
            state={"config":config,"start_epoch":start,"end_epoch":start+args.hours*3600,
                "hours":args.hours,"hardware":hardware_info,"artifacts_manifest_sha":digest(artifacts/"manifest.json"),
                "evaluator":evaluator,"source":source,"runtime_version":"native-v1",
                "scientific_gate_status":"PENDING native run; no PASS asserted"}
            atomic_json(run/"run.json",state)
        while time.time()<start:
            print(f"Waiting until {datetime.fromtimestamp(start).astimezone().isoformat()}",flush=True)
            time.sleep(min(30,start-time.time()))
        remaining=state["end_epoch"]-time.time()
        if remaining<=60:
            report(run)
            print("The persisted window has ended. No new GPU work started.")
            return
        command=[sys.executable,"-m","recursive_ssd.cli","worker","--run-dir",str(run),"--artifacts",str(artifacts)]
        child_env=os.environ.copy()
        run_id=object_hash({"run":str(run),"end":state["end_epoch"]})[:20]
        child_env["RECURSIVE_SSD_RUN_ID"]=run_id
        # Child gets a shorter soft deadline; process-group termination bounds a
        # stuck kernel/evaluator. The same run directory never resets the deadline.
        with (run/"worker.log").open("a") as log:
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=child_env)
            try:
                code=child.wait(timeout=max(1,remaining-30))
            except (subprocess.TimeoutExpired,KeyboardInterrupt):
                os.killpg(child.pid,signal.SIGTERM)
                try:
                    code=child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL)
                    code=child.wait(timeout=5)
                event(run,"supervisor_stop",returncode=code)
            else:
                if code:
                    event(run,"worker_failed",returncode=code,log="worker.log")
        report(run)
        print(f"Saved {run/'REPORT.md'}; return code {code}.")
        if code and code not in (-signal.SIGTERM,-signal.SIGKILL):
            raise SystemExit(code)


def collect(run, output):
    run=Path(run).resolve()
    output=Path(output).resolve()
    output.parent.mkdir(parents=True,exist_ok=True)
    # Results only; weights remain local and are never blindly committed to GitHub.
    selected=[p for p in run.rglob("*") if p.is_file() and p.suffix in {".json",".jsonl",".md",".log"}]
    with tarfile.open(output,"w:gz") as archive:
        for path in selected:
            archive.add(path,arcname=str(Path(run.name)/path.relative_to(run)),recursive=False)
    print(json.dumps({"bundle":str(output),"sha256":digest(output),"files":len(selected)}))


def main():
    parser=argparse.ArgumentParser(description="Recursive self-policy distillation pilot")
    sub=parser.add_subparsers(dest="command",required=True)
    for command in ("prepare","qualify","preflight","run","worker","report","collect"):
        p=sub.add_parser(command)
        if command in {"prepare","preflight","run"}:
            p.add_argument("--config",default="configs/2080ti_8h.json")
        if command in {"prepare","qualify","preflight","run","worker"}:
            p.add_argument("--artifacts",default="artifacts")
        if command in {"run","worker","report","collect"}:
            p.add_argument("--run-dir",default="runs/2080ti-native-8h")
        if command=="qualify":
            p.add_argument("--output",default="artifacts/native-qualification")
        if command=="prepare":
            p.add_argument("--skip-model",action="store_true",help="data-only preparation; insufficient for GPU run")
        if command=="run":
            p.add_argument("--hours",type=float,default=8)
            p.add_argument("--start-at",help="ISO timestamp with timezone; otherwise start now")
        if command=="collect":
            p.add_argument("--output",default="returns/2080ti-native-8h.tar.gz")
    args=parser.parse_args()
    parser.error('The archived pilot CLI cannot start workloads. Use scripts/research.py for the complete native suite and its research-autopilot execution owner.')
    if args.command=="prepare":
        from .data import prepare
        from .runner import validate_config
        prepare(args.artifacts,validate_config(read_json(args.config)),not args.skip_model)
        print("Pinned prompt-only train data and official benchmark prepared." if args.skip_model else
              "Pinned prompt-only train data, official benchmark and model prepared.")
    elif args.command=="qualify":
        from .evaluation import official_score
        from .io import Deadline, jsonl, read_jsonl
        output=Path(args.output)
        output.mkdir(parents=True,exist_ok=True)
        problems=read_jsonl(Path(args.artifacts)/"HumanEvalPlus-dev.jsonl")[:2]
        if len(problems)!=2:
            raise ValueError("qualification requires two released development tasks")
        path=output/"problems.jsonl"
        jsonl(path,problems)
        result=official_score(None,path,output,1,Deadline(time.time()+600),qualify=True)
        print(json.dumps({"status":"passed","backend":"native_python","task_ids":list(result["per_task"]),
                          "scope":"official reference-solution scorer check; not model accuracy or GPU readiness"}))
    elif args.command=="preflight":
        from .runner import preflight
        print(json.dumps(preflight(read_json(args.config),Path(args.artifacts),Path(args.artifacts)/"preflight"),indent=2))
    elif args.command=="run":
        launch(args)
    elif args.command=="worker":
        from .runner import worker, report
        from .io import DeadlineReached
        try:
            worker(Path(args.run_dir),Path(args.artifacts))
        except DeadlineReached as exc:
            event(args.run_dir,"budget_stop",reason=str(exc))
            report(Path(args.run_dir))
    elif args.command=="report":
        from .runner import report
        report(Path(args.run_dir))
    elif args.command=="collect":
        collect(args.run_dir,args.output)


if __name__=="__main__":
    main()
