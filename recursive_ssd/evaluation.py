"""Pinned official EvalPlus in a bounded native Python child process."""
import importlib.metadata
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import uuid
import numpy as np
import psutil
from .io import atomic_json, digest, object_hash, read_json, read_jsonl


def evaluator_info():
    """Bind actual interpreter, installed dependencies and official scorer bytes."""
    expected={"evalplus":"0.3.1","tree-sitter":"0.23.2","tree-sitter-python":"0.23.6"}
    for package,version in expected.items():
        try:
            actual=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError("Install this project's [evaluation] dependencies in the active Python environment") from exc
        if actual!=version:
            raise RuntimeError(f"native evaluator requires {package}=={version}, found {actual}")
    distribution=importlib.metadata.distribution("evalplus")
    code={str(p):digest(distribution.locate_file(p)) for p in distribution.files or []
          if str(p).startswith("evalplus/") and str(p).endswith(".py")}
    if not code:
        raise RuntimeError("cannot fingerprint official EvalPlus source")
    return {"backend":"evalplus-native","python":platform.python_version(),
            "interpreter":str(Path(sys.executable).resolve()),
            "packages":{d.metadata["Name"]:d.version for d in importlib.metadata.distributions()},
            "official_code_sha256":object_hash(code),"official_code_files":code,
            "wrapper_sha256":digest(Path(__file__).with_name("native_score.py")),
            "parallel":2,"min_time_limit":1,"gt_time_limit_factor":4,
            "max_memory_bytes":4*1024**3}


def run_native_process(command, work, timeout):
    """Resource cleanup, not an OS security sandbox. Inherit the worker group.

    The supervisor can terminate the entire worker process group on its hard
    deadline. On a local scorer timeout also stop its descendant processes.
    """
    work=Path(work).resolve()
    work.mkdir(parents=True,exist_ok=True)
    tmp=work/"tmp"
    tmp.mkdir(exist_ok=True)
    env={k:os.environ[k] for k in ("PATH","LANG","LC_ALL","LD_LIBRARY_PATH") if k in os.environ}
    env.update(CUDA_VISIBLE_DEVICES="",TOKENIZERS_PARALLELISM="false",OMP_NUM_THREADS="1",
               MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",TMPDIR=str(tmp),
               XDG_CACHE_HOME=str(work/"cache"),HF_HUB_OFFLINE="1",
               EVALPLUS_MAX_MEMORY_BYTES=str(4*1024**3))
    started=time.time()
    status="interrupted"
    child=None
    try:
        with (work/"evaluator.log").open("w") as log:
            child=subprocess.Popen(command,cwd=work,env=env,stdout=log,stderr=subprocess.STDOUT)
            code=child.wait(timeout=timeout)
        status="completed" if code==0 else "failed"
        if code:
            raise RuntimeError(f"official evaluator failed ({code}); inspect {work/'evaluator.log'}")
        return code
    except subprocess.TimeoutExpired:
        status="timeout"
        raise
    finally:
        if child is not None and child.poll() is None:
            try:
                parent=psutil.Process(child.pid)
                processes=parent.children(recursive=True)+[parent]
                for process in processes:
                    try:
                        process.terminate()
                    except psutil.NoSuchProcess:
                        pass
                _,alive=psutil.wait_procs(processes,timeout=2)
                for process in alive:
                    try:
                        process.kill()
                    except psutil.NoSuchProcess:
                        pass
                child.wait(timeout=3)
            except psutil.NoSuchProcess:
                pass
        atomic_json(work/"execution.json",{"argv":command,"cwd":str(work),
            "started_epoch":started,"ended_epoch":time.time(),"status":status,
            "returncode":child.returncode if child is not None else None,
            "timeout_seconds":timeout,"backend":"native_python",
            "environment_scope":"allowlisted process environment; no GPU; not a security sandbox"})


def inventory(raw, problems, expected_samples):
    counts={p["task_id"]:0 for p in problems}
    if not counts or len(counts)!=len(problems) or type(expected_samples) is not int or expected_samples<1:
        raise ValueError("invalid native problem/sample inventory")
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
    output=Path(output).resolve()
    output.mkdir(parents=True,exist_ok=True)
    result_path=output/"samples_eval_results.json"
    receipt_path=output/"native_receipt.json"
    if result_path.exists() and not receipt_path.exists():
        raise ValueError("unbound cached native result: provenance receipt missing; retain it and use a new attempt directory")
    raw=read_jsonl(raw_path) if raw_path is not None else []
    problems=read_jsonl(problems_path)
    if qualify and expected_samples!=1:
        raise ValueError("canonical qualification uses one native reference per task")
    inventory([{"task_id":p["task_id"],"sample_id":0} for p in problems] if qualify else raw,
              problems,expected_samples)
    info=evaluator_info()
    identity={"problems_sha256":digest(problems_path),"raw_sha256":digest(raw_path) if raw_path is not None else None,
              "expected_samples":expected_samples,"qualify":qualify,"evaluator":info}
    if result_path.exists():
        receipt=read_json(receipt_path)
        if receipt["identity"]!=identity or receipt["result_sha256"]!=digest(result_path):
            raise ValueError("cached native result provenance changed; create a new experiment identity")
    else:
        outputs=output/"native"/("attempt-"+uuid.uuid4().hex[:12])
        inputs=outputs/"inputs"
        inputs.mkdir(parents=True)
        shutil.copyfile(problems_path,inputs/"problems.jsonl")
        command=[sys.executable,"-I",str(Path(__file__).with_name("native_score.py")),
                 "--problems",str(inputs/"problems.jsonl"),"--output",str(outputs)]
        if qualify:
            command.append("--qualify")
        else:
            shutil.copyfile(raw_path,inputs/"raw.jsonl")
            command.extend(["--raw",str(inputs/"raw.jsonl")])
        deadline.check(45)
        run_native_process(command,outputs,max(1,min(1800,deadline.remaining()-20)))
        # Atomic cache publication; raw official output and every attempt remain.
        atomic_json(result_path,read_json(outputs/"samples_eval_results.json"))
        atomic_json(receipt_path,{"identity":identity,"result_sha256":digest(result_path),
                    "native_output":str((outputs/"samples_eval_results.json").relative_to(output)),
                    "execution":str((outputs/"execution.json").relative_to(output))})
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
