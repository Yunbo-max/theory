#!/usr/bin/env python3
"""Finite single-host DAG scheduler around the existing bounded native runner.

Linux/POSIX execution, Python 3.11+, standard library only. A report window is
not a job timeout. No scientific decision, LLM call, SSH login or cloud purchase
is made by this program. Validate by default; execution binds a reviewed digest.
"""
import argparse
import copy
import fcntl
import json
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
import _autoresearch as C
import run_experiments as R

TERMINAL = {"completed", "failed", "blocked", "budget_exhausted"}


def plan_digest(plan):
    return C.hashed({key:value for key,value in plan.items() if key != "plan_digest"})


def workload_digest(native):
    """Identity of the measured workload, without run/attempt directory identity."""
    return C.hashed({key:native[key] for key in ("purpose","evidence_mode","protocol_digest","jobs","provenance")})


def validate_plan(root, plan):
    root = C.root_path(root); C.validate(plan,"harness-plan")
    if plan["plan_digest"] != plan_digest(plan): C.fail("HARNESS_PLAN_DIGEST_MISMATCH")
    C.safe_path(root,plan["output_root"])
    if not Path(plan["pool_dir"]).is_absolute(): C.fail("HARNESS_ABSOLUTE_POOL_REQUIRED")
    C.root_path(plan["pool_dir"])
    tasks = {task["task_id"]:task for task in plan["tasks"]}
    if len(tasks) != len(plan["tasks"]): C.fail("HARNESS_DUPLICATE_TASK")
    if len({task["idea_id"] for task in tasks.values()}) > 20: C.fail("HARNESS_TOO_MANY_IDEAS")
    runs = set(); gpu_seconds = 0
    for task in tasks.values():
        if not set(task["depends_on"]).issubset(tasks): C.fail("HARNESS_DEPENDENCY_UNKNOWN")
        native = C.load_file(C.verify_ref(root,task["plan_ref"])); R.validate_plan(root,native)
        identity = (native["output_root"],native["run_id"])
        if identity in runs: C.fail("HARNESS_DUPLICATE_NATIVE_RUN")
        runs.add(identity)
        resource = task["resources"]; limits = plan["limits"]
        if resource["cpu_cores"] > limits["cpu_cores"] or resource["ram_mib"] > limits["ram_mib"]:
            C.fail("HARNESS_TASK_EXCEEDS_HOST_BUDGET")
        if resource["gpu_count"] > len(plan["gpus"]["uuids"]): C.fail("HARNESS_GPU_INVENTORY_INSUFFICIENT")
        gpu_seconds += resource["gpu_count"] * native["limits"]["wall_time_seconds"]
        if resource["allow_gpu_share"]:
            if resource["gpu_count"] != 1 or resource["gpu_peak_mib"] is None or not resource["memory_profile_ref"]:
                C.fail("HARNESS_SHARING_PROFILE_REQUIRED")
            profile = C.load_file(C.verify_ref(root,resource["memory_profile_ref"])); C.validate(profile,"gpu-memory-profile")
            if profile["workload_digest"] != workload_digest(native) or profile["reservation_mib"] != resource["gpu_peak_mib"]:
                C.fail("HARNESS_MEMORY_PROFILE_MISMATCH")
            if profile["workload_digest"] not in profile["co_location_workload_digests"]:
                C.fail("HARNESS_SHARING_COHORT_UNQUALIFIED")
            if profile["peak_mib"] > profile["reservation_mib"]: C.fail("HARNESS_MEMORY_PROFILE_MISMATCH")
            if plan["gpus"]["max_tasks_per_gpu"] > profile["qualified_max_concurrency"]:
                C.fail("HARNESS_SHARING_CONCURRENCY_UNQUALIFIED")
            if not set(plan["gpus"]["uuids"]).issubset(profile["device_uuids"]):
                C.fail("HARNESS_SHARING_DEVICE_UNQUALIFIED")
            for ref in profile["measurement_refs"] + profile["throughput_evidence_refs"]: C.verify_ref(root,ref)
        elif resource["memory_profile_ref"] is not None:
            C.verify_ref(root,resource["memory_profile_ref"])
        if not resource["gpu_count"] and (resource["gpu_peak_mib"] is not None or resource["allow_gpu_share"]):
            C.fail("HARNESS_CPU_TASK_GPU_CONFLICT")
    if gpu_seconds > plan["limits"]["max_gpu_task_seconds"]: C.fail("HARNESS_GPU_TIME_BUDGET_EXCEEDED")
    visiting = set(); visited = set()
    def visit(name):
        if name in visiting: C.fail("HARNESS_DEPENDENCY_CYCLE")
        if name in visited: return
        visiting.add(name)
        for dep in tasks[name]["depends_on"]: visit(dep)
        visiting.remove(name); visited.add(name)
    for name in tasks: visit(name)
    return plan


def make_plan(root, *, batch_id, tasks, pool_dir=None, limits, gpus=None, output_root="runs/harness"):
    limits = copy.deepcopy(limits); limits.setdefault("window_seconds",8*3600)
    plan = C.envelope("harness-plan",batch_id=batch_id,tasks=copy.deepcopy(tasks),limits=limits,
        pool_dir=str(pool_dir or (Path.home()/".local/state/research-autopilot")),output_root=output_root,
        gpus=copy.deepcopy(gpus or {"uuids":[],"safety_margin_mib":1024,"max_tasks_per_gpu":1}))
    plan["plan_digest"] = plan_digest(plan)
    return validate_plan(root,plan)


def _atomic(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temp.open("x",encoding="utf-8") as stream:
        stream.write(C.canonical(value)+"\n"); stream.flush(); os.fsync(stream.fileno())
    os.replace(temp,path)
    fd = os.open(str(path.parent),os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)


def _lock(path):
    path.parent.mkdir(parents=True,exist_ok=True)
    fd = os.open(str(path),os.O_CREAT|os.O_RDWR,0o600)
    try: fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd); return None
    except BaseException:
        os.close(fd); raise
    return fd


def _owned_lock(path, owner):
    """Join only this batch's surviving workers; changing owner needs exclusivity.

    The separate exclusive driver.lock serializes this handshake and all host
    admissions. Workers inherit shared ownership locks, not the driver lock.
    """
    path.parent.mkdir(parents=True,exist_ok=True)
    fd = os.open(str(path),os.O_CREAT|os.O_RDWR,0o600)
    owner_path = path.with_name(path.name+".owner.json")
    try:
        current = C.load_file(owner_path) if owner_path.is_file() else None
        if current == owner: fcntl.flock(fd,fcntl.LOCK_SH|fcntl.LOCK_NB)
        else:
            fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
            _atomic(owner_path,owner)
            fcntl.flock(fd,fcntl.LOCK_SH)
    except BlockingIOError:
        os.close(fd); return None
    except BaseException:
        os.close(fd); raise
    return fd


def _identity(pid):
    try:
        stat = Path("/proc/%s/stat" % pid).read_text().rsplit(")",1)[1].split()
        if stat[0] == "Z": return None
        return {"pid":int(pid),"start_ticks":stat[19],"boot_id":Path("/proc/sys/kernel/random/boot_id").read_text().strip()}
    except (OSError,IndexError): return None


def _self_identity():
    # Read the procfs identity, not a wrapper's virtual Popen PID. They are the
    # same on the target Linux host, but can differ in engineering sandboxes.
    return _identity(int(Path("/proc/self/stat").read_text().split()[0]))


def _alive(identity):
    return bool(identity and _identity(identity["pid"]) == identity)


def _descendants(identities):
    parents = {}
    for path in Path("/proc").glob("[0-9]*/stat"):
        try: parents[int(path.parent.name)] = int(path.read_text().rsplit(")",1)[1].split()[1])
        except (OSError,ValueError,IndexError): continue
    own = {identity["pid"] for identity in identities if _alive(identity)}
    while True:
        new = {pid for pid,parent in parents.items() if parent in own} - own
        if not new: return own
        own.update(new)


def _nvidia_query(fields, kind="gpu"):
    result = subprocess.run(["nvidia-smi","--query-"+kind+"="+fields,"--format=csv,noheader,nounits"],
                            capture_output=True,text=True,timeout=5,check=True)
    return [[part.strip() for part in line.split(",")] for line in result.stdout.splitlines() if line.strip()]


def gpu_snapshot(identities=()):
    """Live UUID-based NVIDIA admission telemetry; unavailable/ambiguous fails closed."""
    try:
        rows = _nvidia_query("uuid,memory.total,memory.used")
        apps = _nvidia_query("gpu_uuid,pid,used_gpu_memory","compute-apps")
        own = _descendants(identities); own_usage = {}; foreign = {}
        for device,pid,used in apps:
            if int(pid) in own: own_usage[device] = own_usage.get(device,0) + int(used)
            else: foreign.setdefault(device,[]).append(int(pid))
        return [{"uuid":device,"total_mib":int(total),"used_mib":int(used),
                 "foreign_mib":max(0,int(used)-own_usage.get(device,0)),"foreign_pids":foreign.get(device,[])}
                for device,total,used in rows]
    except (OSError,ValueError,subprocess.SubprocessError): return []


def select_devices(resource, active, snapshot, policy):
    if resource["gpu_count"] == 0: return []
    available = []
    for gpu in snapshot:
        device = gpu["uuid"]
        if device not in policy["uuids"] or gpu["foreign_pids"]: continue
        occupants = [item for item in active if device in item["devices"]]
        if occupants and (not resource["allow_gpu_share"] or resource["gpu_count"] != 1 or
                          any(not item["resources"]["allow_gpu_share"] for item in occupants)): continue
        if occupants and any(resource.get("_workload_digest") not in item["resources"].get("_co_location",[None]) or
                item["resources"].get("_workload_digest") not in resource.get("_co_location",[None]) for item in occupants): continue
        if len(occupants) >= policy["max_tasks_per_gpu"]: continue
        peak = resource["gpu_peak_mib"]
        if peak is None:
            if occupants or gpu["foreign_mib"] > policy["safety_margin_mib"]: continue
        else:
            reserved = sum(item["resources"]["gpu_peak_mib"] or gpu["total_mib"] for item in occupants)
            actual_own = max(0,gpu["used_mib"]-gpu["foreign_mib"])
            if occupants and actual_own > reserved: continue
            if max(reserved,actual_own) + peak + gpu["foreign_mib"] + policy["safety_margin_mib"] > gpu["total_mib"]: continue
        available.append(device)
    count = resource["gpu_count"]
    return available[:count] if len(available) >= count else None


def _available_ram():
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"): return int(line.split()[1]) // 1024
    except (OSError,ValueError): pass
    return 0


def _worker(root, batch_root, task_id, deadline, pool_fd, devices, expected_plan_digest, resource_fds):
    """The inherited execution lock survives driver death until this worker ends."""
    task_dir = C.safe_path(batch_root,"tasks/"+task_id)
    frozen = C.load_file(batch_root/"plan.json")
    if frozen["plan_digest"] != expected_plan_digest or plan_digest(frozen) != expected_plan_digest:
        C.fail("HARNESS_WORKER_PLAN_DIGEST_MISMATCH")
    task = next(item for item in frozen["tasks"] if item["task_id"] == task_id)
    if C.load_file(task_dir/"task.json") != task: C.fail("HARNESS_WORKER_TASK_DIGEST_MISMATCH")
    native = C.load_file(C.verify_ref(root,task["plan_ref"]))
    result_path = task_dir/"result.json"
    _atomic(task_dir/"process.json",_self_identity())
    context_path = task_dir/"execution-context.json"
    _atomic(context_path,{"batch_plan_digest":expected_plan_digest,"task_id":task_id,"devices":devices,
        "declared_resources":task["resources"],"process":_self_identity(),"observed_at":C.now(),
        "CUDA_VISIBLE_DEVICES":",".join(devices),"native_plan_digest":native["plan_digest"],"gate_advanced":False})
    def save_result(value):
        value.setdefault("completed_at",C.now()); value["completed_monotonic"] = time.monotonic()
        value["execution_context_ref"] = C.reference(root,context_path)
        _atomic(result_path,value)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(devices)
    def stop(signum,frame): raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop); signal.signal(signal.SIGALRM,stop)
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        save_result({"status":"budget_exhausted","reason_code":"HARNESS_HARD_DEADLINE","gate_advanced":False}); return
    signal.setitimer(signal.ITIMER_REAL,remaining)
    try:
        receipt = R.run_plan(root,native,authorizer=lambda scope:
            scope.get("plan_digest") == native["plan_digest"] and scope.get("run_id") == native["run_id"],
            process_fds=tuple(resource_fds))
        ref = C.reference(root,C.safe_path(root,native["output_root"])/native["run_id"]/"receipt.json")
        outputs = []; reason = None
        if receipt["status"] == "completed":
            for job in native["jobs"]:
                done = [item for item in receipt["attempts"] if item["trial_id"] == job["trial_id"] and item["status"] == "completed"]
                if len(done) != 1: C.fail("HARNESS_ATTEMPT_INVENTORY_MISMATCH")
                for path in job["output_paths"]:
                    output = C.safe_path(root,done[0]["attempt_path"]+"/workspace/"+path)
                    if not output.is_file(): reason = "HARNESS_DECLARED_OUTPUT_MISSING"
                    else:
                        relative = output.relative_to(root).as_posix()
                        output_ref = next((item for item in done[0]["output_refs"] if item["path"] == relative),None)
                        if output_ref is None: C.fail("HARNESS_DECLARED_OUTPUT_RECEIPT_REQUIRED")
                        outputs.append(output_ref)
        status = "completed" if receipt["status"] == "completed" and reason is None else "failed"
        if receipt["status"] in {"interrupted","budget_exhausted"}: status = "budget_exhausted" if time.monotonic() >= deadline else "failed"
        save_result({"status":status,"reason_code":reason or receipt["status"],"receipt_ref":ref,
            "output_refs":outputs,"completed_at":receipt["completed_at"],"gate_advanced":False})
    except KeyboardInterrupt:
        save_result({"status":"budget_exhausted","reason_code":"HARNESS_HARD_DEADLINE","gate_advanced":False})
    except (C.Failure,OSError) as error:
        save_result({"status":"failed","reason_code":getattr(error,"code","HARNESS_WORKER_IO_ERROR"),"gate_advanced":False})
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        os.close(pool_fd)


def _verify_result(root, task, result):
    verified = {}
    def verify(ref):
        if ref["path"] in verified:
            if verified[ref["path"]] != ref["sha256"]: C.fail("HARNESS_OUTPUT_DIGEST_CONFLICT")
            return C.safe_path(root,ref["path"])
        path = C.verify_ref(root,ref); verified[ref["path"]] = ref["sha256"]; return path
    if result.get("receipt_ref"):
        receipt = C.load_file(verify(result["receipt_ref"])); C.validate(receipt,"experiment-run-receipt")
        native = C.load_file(verify(task["plan_ref"]))
        run_root = C.safe_path(root,native["output_root"])/native["run_id"]
        if C.safe_path(root,result["receipt_ref"]["path"]) != run_root/"receipt.json":
            C.fail("HARNESS_CANONICAL_RECEIPT_REQUIRED")
        if receipt["run_id"] != native["run_id"] or receipt["plan_digest"] != native["plan_digest"]:
            C.fail("HARNESS_RECEIPT_IDENTITY_MISMATCH")
        if any(receipt[key] != native[key] for key in ("purpose","evidence_mode","provenance")):
            C.fail("HARNESS_RECEIPT_IDENTITY_MISMATCH")
        for attempt in receipt["attempts"]:
            if C.safe_path(root,attempt["attempt_path"]) != run_root/attempt["attempt_id"]:
                C.fail("HARNESS_ATTEMPT_PATH_MISMATCH")
            if C.load_file(run_root/attempt["attempt_id"]/"attempt.json") != attempt:
                C.fail("HARNESS_ATTEMPT_RECEIPT_MISMATCH")
            if attempt["trial_id"] not in {job["trial_id"] for job in native["jobs"]}:
                C.fail("HARNESS_ATTEMPT_INVENTORY_MISMATCH")
            if attempt["status"] == "completed" and attempt["exit_code"] != 0:
                C.fail("HARNESS_ATTEMPT_INVENTORY_MISMATCH")
            for ref in [attempt["stdout_ref"],attempt["stderr_ref"]] + attempt["output_refs"]: verify(ref)
            if attempt.get("process_guard_ref"): verify(attempt["process_guard_ref"])
        if len(receipt["attempts"]) > native["limits"]["max_attempts"]: C.fail("HARNESS_ATTEMPT_INVENTORY_MISMATCH")
        if result["status"] == "completed" and receipt["status"] != "completed": C.fail("HARNESS_RECEIPT_STATUS_MISMATCH")
        if result["status"] == "completed":
            expected = []
            for job in native["jobs"]:
                done = [a for a in receipt["attempts"] if a["trial_id"] == job["trial_id"] and a["status"] == "completed"]
                if len(done) != 1: C.fail("HARNESS_ATTEMPT_INVENTORY_MISMATCH")
                if any(done[0][key] != job[key] for key in ("seed","group","arm_role")):
                    C.fail("HARNESS_ATTEMPT_INVENTORY_MISMATCH")
                for path in job["output_paths"]:
                    relative = C.safe_path(root,done[0]["attempt_path"]+"/workspace/"+path).relative_to(root).as_posix()
                    ref = next((item for item in done[0]["output_refs"] if item["path"] == relative),None)
                    if ref is None: C.fail("HARNESS_DECLARED_OUTPUT_RECEIPT_REQUIRED")
                    expected.append(ref)
            if sorted(expected,key=C.canonical) != sorted(result.get("output_refs",[]),key=C.canonical):
                C.fail("HARNESS_OUTPUT_INVENTORY_MISMATCH")
    elif result["status"] == "completed": C.fail("HARNESS_RECEIPT_REQUIRED")
    for ref in result.get("output_refs",[]): verify(ref)
    if result.get("execution_context_ref"): verify(result["execution_context_ref"])


def _event(batch_root, kind, **fields):
    with (batch_root/"events.jsonl").open("a",encoding="utf-8") as stream:
        stream.write(C.canonical({"observed_at":C.now(),"event":kind,**fields})+"\n")
        stream.flush(); os.fsync(stream.fileno())


def _report(root, plan, state, batch_root, *, boundary=None):
    report = copy.deepcopy(state)
    report.update(observed_at=C.now(),report_kind="window" if boundary else "checkpoint",gate_advanced=False,
        scientific_result_verified=False,human_review_required_for_new_direction=True,
        elapsed_wall_seconds=max(0,time.monotonic()-state["started_monotonic"]),
        remaining_wall_seconds=max(0,state["hard_deadline_monotonic"]-time.monotonic()))
    intervals = {}
    for task in state["tasks"].values():
        start = task.get("started_monotonic")
        if start is None: continue
        end = min(time.monotonic(),task.get("completed_monotonic",time.monotonic()))
        for device in task.get("devices",[]): intervals.setdefault(device,[]).append((start,end))
    busy = 0
    for values in intervals.values():
        merged = []
        for start,end in sorted(values):
            if merged and start <= merged[-1][1]: merged[-1] = (merged[-1][0],max(end,merged[-1][1]))
            else: merged.append((start,end))
        busy += sum(end-start for start,end in merged)
    report["resource_accounting"] = {"gpu_device_occupied_seconds":busy,
        "gpu_task_reserved_seconds_upper_bound":sum(C.load_file(C.verify_ref(root,task["plan_ref"]))["limits"]["wall_time_seconds"]
            * task["resources"]["gpu_count"] for task in plan["tasks"]),
        "billing_verified":False,"llm_calls_by_scheduler":0}
    _atomic(batch_root/"report.json",report)
    if boundary: _atomic(batch_root/"reports"/("window-%04d.json" % boundary),report)
    return report


def run_harness(root, plan, *, authorizer=None, poll_seconds=1.0, stop_after_report=False):
    root = C.root_path(root); validate_plan(root,plan)
    if not callable(authorizer): C.fail("HARNESS_EXECUTION_AUTHORIZATION_REQUIRED")
    scope = {"operation":"bounded_dependency_harness","root":str(root),"plan_digest":plan["plan_digest"],"plan":copy.deepcopy(plan)}
    if authorizer(scope) is not True: C.fail("HARNESS_EXECUTION_NOT_AUTHORIZED")
    if not Path("/proc/sys/kernel/random/boot_id").is_file(): C.fail("HARNESS_LINUX_HOST_REQUIRED")
    pool = C.root_path(plan["pool_dir"]); driver_fd = _lock(pool/"driver.lock")
    if driver_fd is None: C.fail("HARNESS_HOST_DRIVER_ALREADY_ACTIVE")
    execution_fd = None; device_fds = {}; children = {}; verified_results = set()
    batch_root = C.safe_path(root,plan["output_root"])/plan["batch_id"]
    task_map = {task["task_id"]:task for task in plan["tasks"]}
    owner = {"root":str(root),"batch_id":plan["batch_id"],"plan_digest":plan["plan_digest"]}
    runtime_resources = {}
    try:
        for name,task in task_map.items():
            resource = copy.deepcopy(task["resources"])
            if resource["allow_gpu_share"]:
                profile = C.load_file(C.verify_ref(root,resource["memory_profile_ref"]))
                resource.update(_workload_digest=profile["workload_digest"],_co_location=profile["co_location_workload_digests"])
            runtime_resources[name] = resource
        batch_root.mkdir(parents=True,exist_ok=True)
        state_path = batch_root/"state.json"
        if state_path.exists():
            state = C.load_file(state_path)
            if state["plan_digest"] != plan["plan_digest"] or state["root"] != str(root): C.fail("HARNESS_RESUME_IDENTITY_MISMATCH")
            if state["boot_id"] != _self_identity()["boot_id"]: C.fail("HARNESS_HOST_RESTART_RECONCILE_REQUIRED")
        else:
            if any(batch_root.iterdir()): C.fail("HARNESS_STATE_INCOMPLETE_RECONCILE_REQUIRED")
            started = time.time(); mono = time.monotonic()
            state = {"format":"research-harness-state-v1","batch_id":plan["batch_id"],"plan_digest":plan["plan_digest"],
                "root":str(root),"started_at":C.now(),"started_epoch":started,
                "hard_deadline_epoch":started+plan["limits"]["total_wall_seconds"],"reported_windows":0,"status":"running",
                "started_monotonic":mono,"hard_deadline_monotonic":mono+plan["limits"]["total_wall_seconds"],
                "boot_id":_self_identity()["boot_id"],
                "tasks":{name:{"status":"pending","queued_at":C.now()} for name in task_map},"gate_advanced":False}
            _atomic(batch_root/"plan.json",plan); _atomic(state_path,state)
            _event(batch_root,"batch_started",plan_digest=plan["plan_digest"])
        while True:
            now = time.monotonic()
            for name,record in state["tasks"].items():
                task = task_map[name]; result_path = batch_root/"tasks"/name/"result.json"
                if record["status"] in TERMINAL:
                    if name not in verified_results: _verify_result(root,task,record); verified_results.add(name)
                    continue
                if record["status"] in {"running","unknown"}:
                    identity_path = batch_root/"tasks"/name/"process.json"
                    if not record.get("process") and identity_path.is_file(): record["process"] = C.load_file(identity_path)
                    if result_path.is_file():
                        result = C.load_file(result_path); _verify_result(root,task,result)
                        record.update(result,completed_epoch=C.stamp(result.get("completed_at",C.now())).timestamp())
                        verified_results.add(name)
                        _event(batch_root,"task_finished",task_id=name,status=record["status"])
                    elif not _alive(record.get("process")) and not (name in children and children[name].poll() is None):
                        if record["status"] == "running":
                            record.update(status="unknown",reason_code="HARNESS_PROCESS_LOST_RECONCILE_REQUIRED")
                    else: record["status"] = "running"
                if name in children and children[name].poll() is not None: children.pop(name)
            for name,record in state["tasks"].items():
                if record["status"] != "pending": continue
                dependencies = [state["tasks"][dep]["status"] for dep in task_map[name]["depends_on"]]
                if any(status in TERMINAL-{"completed"} for status in dependencies):
                    record.update(status="blocked",reason_code="HARNESS_DEPENDENCY_FAILED")
            elapsed = max(0,now-state["started_monotonic"])
            boundary = int(elapsed/plan["limits"]["window_seconds"])
            if boundary > state["reported_windows"]:
                state["reported_windows"] = boundary
                _atomic(state_path,state); _event(batch_root,"window_report",window=boundary)
                report = _report(root,plan,state,batch_root,boundary=boundary)
                if stop_after_report and state["status"] not in {"completed","failed","budget_exhausted"}:
                    report["status"] = "handoff"; return report
            if state["status"] in {"completed","failed"}:
                _atomic(state_path,state); return _report(root,plan,state,batch_root)
            if now >= state["hard_deadline_monotonic"]:
                for name,record in state["tasks"].items():
                    if record["status"] == "pending": record.update(status="budget_exhausted",reason_code="HARNESS_HARD_DEADLINE")
                    if record["status"] == "running" and now > state["hard_deadline_monotonic"]+2 and not record.get("deadline_signal_sent"):
                        # Workers have their own SIGALRM hard deadline. Allow
                        # receipt cleanup before the driver's fallback signal.
                        try:
                            if name in children: children[name].terminate()
                            elif _alive(record.get("process")): os.kill(record["process"]["pid"],signal.SIGTERM)
                        except ProcessLookupError: pass
                        record["deadline_signal_sent"] = True
                state["status"] = "budget_exhausted"
                if any(record["status"] == "running" for record in state["tasks"].values()):
                    if now > state["hard_deadline_monotonic"]+5:
                        state["status"] = "reconciliation_required"
                        _atomic(state_path,state); return _report(root,plan,state,batch_root)
                    _atomic(state_path,state); time.sleep(min(poll_seconds,0.2)); continue
            elif all(record["status"] in TERMINAL for record in state["tasks"].values()):
                state["status"] = "completed" if all(record["status"] == "completed" for record in state["tasks"].values()) else "failed"
            elif any(record["status"] == "unknown" for record in state["tasks"].values()):
                state["status"] = "reconciliation_required"
                # Lost children can retain the host lock or GPU allocations. Never retry.
                _atomic(state_path,state); return _report(root,plan,state,batch_root)
            else:
                if execution_fd is None: execution_fd = _owned_lock(pool/"execution.lock",owner)
                active = [{"resources":runtime_resources[name],"devices":record.get("devices",[])}
                          for name,record in state["tasks"].items() if record["status"] == "running"]
                identities = [record["process"] for record in state["tasks"].values() if record["status"] == "running" and record.get("process")]
                snapshot = gpu_snapshot(identities) if plan["gpus"]["uuids"] else []
                state["gpu_snapshot"] = snapshot
                ready = sorted((task for task in plan["tasks"] if state["tasks"][task["task_id"]]["status"] == "pending" and
                    all(state["tasks"][dep]["status"] == "completed" for dep in task["depends_on"])),key=lambda task:-task["priority"])
                for task in ready:
                    record = state["tasks"][task["task_id"]]; resource = runtime_resources[task["task_id"]]; limits = plan["limits"]
                    if execution_fd is None: record["reason_code"] = "HARNESS_HOST_ORPHANS_OR_OTHER_OWNER"; continue
                    if len(active) >= limits["max_parallel_tasks"]: record["reason_code"] = "HARNESS_TASK_SLOTS_BUSY"; continue
                    if sum(item["resources"]["cpu_cores"] for item in active)+resource["cpu_cores"] > limits["cpu_cores"]:
                        record["reason_code"] = "HARNESS_CPU_RESERVATION_BUSY"; continue
                    if sum(item["resources"]["ram_mib"] for item in active)+resource["ram_mib"] > min(limits["ram_mib"],_available_ram()):
                        record["reason_code"] = "HARNESS_RAM_RESERVATION_BUSY"; continue
                    keys = {key for item in active for key in item["resources"]["exclusive_keys"]}
                    if keys.intersection(resource["exclusive_keys"]): record["reason_code"] = "HARNESS_MUTABLE_ASSET_BUSY"; continue
                    devices = select_devices(resource,active,snapshot,plan["gpus"])
                    if devices is None: record["reason_code"] = "HARNESS_GPU_ADMISSION_WAIT"; continue
                    new_fds = {}
                    for device in devices:
                        if device not in device_fds:
                            fd = _owned_lock(pool/(C.hashed(device)+".gpu.lock"),owner)
                            if fd is None: break
                            new_fds[device] = fd
                    else:
                        device_fds.update(new_fds)
                        native = C.load_file(C.verify_ref(root,task["plan_ref"])); R.validate_plan(root,native)
                        if resource["memory_profile_ref"]: C.verify_ref(root,resource["memory_profile_ref"])
                        run_root = C.safe_path(root,native["output_root"])/native["run_id"]
                        if run_root.exists():
                            record.update(status="unknown",reason_code="HARNESS_NATIVE_RUN_EXISTS_RECONCILE_REQUIRED"); continue
                        task_dir = batch_root/"tasks"/task["task_id"]; task_dir.mkdir(parents=True,exist_ok=True)
                        _atomic(task_dir/"task.json",task)
                        # Persist dispatch intent before Popen: a crash never makes it retryable.
                        record.update(status="unknown",reason_code="HARNESS_DISPATCH_RECONCILE_REQUIRED",devices=devices,
                                      started_at=C.now(),started_epoch=time.time(),started_monotonic=time.monotonic())
                        _atomic(state_path,state)
                        command = [sys.executable,str(Path(__file__).resolve()),"--worker",str(root),str(batch_root),
                                   task["task_id"],str(min(state["hard_deadline_monotonic"],record["started_monotonic"]+native["limits"]["wall_time_seconds"])),
                                   str(execution_fd),C.canonical(devices),plan["plan_digest"],C.canonical([execution_fd,*device_fds.values()])]
                        try:
                            with (task_dir/"worker.stdout.log").open("ab") as out,(task_dir/"worker.stderr.log").open("ab") as err:
                                child = subprocess.Popen(command,stdout=out,stderr=err,start_new_session=True,
                                    pass_fds=(execution_fd,*device_fds.values()),shell=False)
                        except OSError:
                            C.fail("HARNESS_WORKER_LAUNCH_RECONCILE_REQUIRED")
                        record.update(status="running",process=None,reason_code=None)
                        children[task["task_id"]] = child
                        active.append({"resources":resource,"devices":devices})
                        _atomic(state_path,state); _event(batch_root,"task_started",task_id=task["task_id"],launcher_pid=child.pid,
                                                         devices=devices,native_plan_digest=native["plan_digest"])
                        continue
                    for fd in new_fds.values(): os.close(fd)
                    record["reason_code"] = "HARNESS_GPU_OWNER_BUSY"
            _atomic(state_path,state)
            if state["status"] in {"completed","failed","budget_exhausted"}: return _report(root,plan,state,batch_root)
            time.sleep(max(0.01,poll_seconds))
    except KeyboardInterrupt:
        # The worker owns its process-group timeout and inherits execution locks.
        _atomic(state_path,state); return _report(root,plan,state,batch_root)
    finally:
        for child in children.values():
            if child.poll() is not None: child.wait()
            else: threading.Thread(target=child.wait,daemon=True).start()
        for fd in device_fds.values(): os.close(fd)
        if execution_fd is not None: os.close(execution_fd)
        os.close(driver_fd)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "--worker":
        _,root,batch_root,task_id,deadline,pool_fd,devices,digest,resource_fds = argv
        _worker(C.root_path(root),C.root_path(batch_root),task_id,float(deadline),int(pool_fd),C.parse_json(devices),digest,C.parse_json(resource_fds)); return 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan",nargs="?"); parser.add_argument("--root")
    parser.add_argument("--inspect-host",action="store_true",help="read CPU/RAM/GPU inventory; launches no workload")
    parser.add_argument("--freeze",action="store_true",help="validate draft and print canonical plan with digest; launches nothing")
    parser.add_argument("--execute",action="store_true"); parser.add_argument("--approved-plan-digest")
    parser.add_argument("--status",action="store_true",help="read retained report; launches nothing")
    parser.add_argument("--stop-after-report",action="store_true",help="handoff driver at next window; live jobs retain deadlines")
    args = parser.parse_args(argv)
    try:
        if args.inspect_host:
            if not Path("/proc/sys/kernel/random/boot_id").is_file(): C.fail("HARNESS_LINUX_HOST_REQUIRED")
            print(C.canonical({"platform":"linux","cpu_cores":len(os.sched_getaffinity(0)),"available_ram_mib":_available_ram(),
                               "gpus":gpu_snapshot(),"execution_started":False})); return 0
        if not args.root or not args.plan: parser.error("plan and --root are required")
        if sum(bool(value) for value in (args.freeze,args.execute,args.status)) > 1: parser.error("choose freeze, execute or status")
        root = C.root_path(args.root); plan = C.load_file(Path(args.plan))
        if args.freeze: plan["plan_digest"] = plan_digest(plan)
        validate_plan(root,plan)
        if args.status:
            path = C.safe_path(root,plan["output_root"])/plan["batch_id"]/"state.json"
            result = C.load_file(path)
            if result["plan_digest"] != plan["plan_digest"] or result["root"] != str(root): C.fail("HARNESS_RESUME_IDENTITY_MISMATCH")
            result["observed_at"] = C.now(); result["status_is_retained_observation"] = True
        elif args.freeze: result = plan
        elif not args.execute: result = {"status":"validated","plan_digest":plan["plan_digest"],"execution_started":False}
        else:
            if args.approved_plan_digest != plan["plan_digest"]: C.fail("HARNESS_APPROVED_PLAN_DIGEST_REQUIRED")
            result = run_harness(root,plan,authorizer=lambda scope:scope["plan_digest"] == args.approved_plan_digest,
                                 stop_after_report=args.stop_after_report)
        print(C.canonical(result)); return 0
    except C.Failure as error:
        print(C.canonical({"status":"blocked","reason_code":error.code,"path":error.path})); return 2


if __name__ == "__main__": raise SystemExit(main())
