"""Official native scoring in Docker. No generated code is executed by the host."""
import math
import os
from pathlib import Path
import shutil
import subprocess
import uuid
import numpy as np
from .io import atomic_json, jsonl, read_json, read_jsonl, stable_seed

IMAGE="recursive-ssd-eval:0.3.1"


def docker_info():
    result=subprocess.run(["docker","image","inspect",IMAGE,"--format","{{.Id}}"],
        text=True,capture_output=True,check=True,timeout=30)
    return result.stdout.strip()


def inventory(raw, problems, expected_samples):
    counts={p["task_id"]:0 for p in problems}
    identities=set()
    for row in raw:
        key=(row["task_id"],row["sample_id"])
        if row["task_id"] not in counts or key in identities:
            raise ValueError("unexpected task or duplicate sample in evaluation")
        identities.add(key)
        counts[row["task_id"]]+=1
    if any(n!=expected_samples for n in counts.values()):
        raise ValueError("incomplete native evaluation inventory; cannot report pass rate")
    return counts


def official_score(raw_path, problems_path, output, expected_samples, deadline, qualify=False):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    result_path=output/"samples_eval_results.json"
    if result_path.exists():
        return summarize(result_path,expected_samples)
    raw=read_jsonl(raw_path) if raw_path is not None else []
    problems=read_jsonl(problems_path)
    if not qualify:
        inventory(raw,problems,expected_samples)
    inputs=output/"inputs"
    outputs=output/"native"
    inputs.mkdir(exist_ok=True)
    outputs.mkdir(exist_ok=True)
    shutil.copyfile(problems_path,inputs/"problems.jsonl")
    if raw_path is not None:
        shutil.copyfile(raw_path,inputs/"raw.jsonl")
    container="recursive-ssd-"+uuid.uuid4().hex[:12]
    command=["docker","run","--rm","--name",container,
        "--label",f"recursive-ssd.run={os.environ.get('RECURSIVE_SSD_RUN_ID','manual-qualification')}",
        "--network","none","--read-only",
        "--cap-drop=ALL","--security-opt=no-new-privileges","--pids-limit","256",
        "--memory","3g","--cpus","2","--user",f"{os.getuid()}:{os.getgid()}",
        "--tmpfs","/tmp:rw,nosuid,nodev,size=1g",
        "--mount",f"type=bind,src={inputs.resolve()},dst=/input,readonly",
        "--mount",f"type=bind,src={outputs.resolve()},dst=/output",IMAGE]
    if qualify:
        command.append("--qualify")
    try:
        deadline.check(45)
        with (output/"evaluator.log").open("w") as log:
            completed=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,
                timeout=max(1,min(1800,deadline.remaining()-20)))
        if completed.returncode:
            raise RuntimeError(f"official evaluator failed; inspect {output/'evaluator.log'}")
    finally:
        # A timed-out/interrupted docker client need not stop its container.
        subprocess.run(["docker","rm","-f",container],capture_output=True,timeout=20)
    shutil.copyfile(outputs/"samples_eval_results.json",result_path)
    result=summarize(result_path,expected_samples)
    if set(result["per_task"])!={p["task_id"] for p in problems}:
        raise ValueError("official output task inventory mismatch")
    if qualify and any(v["plus_correct"]!=expected_samples for v in result["per_task"].values()):
        raise RuntimeError("official canonical qualification failed")
    atomic_json(output/"metrics.json",result)
    return result


def pass_at_k(n, correct, k):
    """Same unbiased estimator used by EvalPlus; never substitute task pooling."""
    if not 1<=k<=n or not 0<=correct<=n:
        raise ValueError("invalid pass@k counts")
    if n-correct<k:
        return 1.
    return 1.-math.prod(1.-k/i for i in range(n-correct+1,n+1))


def summarize(result_path, expected_samples):
    raw=read_json(result_path)
    per_task={}
    for task,rows in raw["eval"].items():
        if len(rows)!=expected_samples:
            raise ValueError("missing native test outcomes")
        base=sum(row["base_status"]=="pass" for row in rows)
        plus=sum(row["base_status"]=="pass" and row["plus_status"]=="pass" for row in rows)
        per_task[task]={"n":len(rows),"base_correct":base,"plus_correct":plus,
            "base_pass1":base/len(rows),"plus_pass1":plus/len(rows),
            f"plus_pass{expected_samples}":pass_at_k(len(rows),plus,expected_samples)}
    if not per_task:
        raise ValueError("empty native evaluation")
    return {"benchmark":"HumanEval+ v0.1.10 pilot subset", "task_count":len(per_task),
        "samples_per_task":expected_samples,"per_task":per_task,
        "base_pass1":sum(x["base_pass1"] for x in per_task.values())/len(per_task),
        "plus_pass1":sum(x["plus_pass1"] for x in per_task.values())/len(per_task),
        f"plus_pass{expected_samples}":sum(x[f"plus_pass{expected_samples}"] for x in per_task.values())/len(per_task),
        "scientific_verdict":"INCONCLUSIVE: exploratory subset, one seed, confirmation pending"}


def paired_interval(a,b,key="plus_pass1",repeats=4000):
    if set(a["per_task"])!=set(b["per_task"]):
        raise ValueError("paired comparison requires identical task inventories")
    tasks=sorted(a["per_task"])
    delta=np.array([a["per_task"][t][key]-b["per_task"][t][key] for t in tasks])
    rng=np.random.default_rng(42017)
    means=delta[rng.integers(0,len(delta),(repeats,len(delta)))].mean(1)
    return {"mean_delta":float(delta.mean()),"task_bootstrap_95":[float(x) for x in np.quantile(means,[.025,.975])],
            "scope":"task-cluster pilot interval; not seed uncertainty or multiple-testing correction"}
