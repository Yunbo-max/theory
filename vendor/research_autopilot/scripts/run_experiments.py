#!/usr/bin/env python3
"""Bounded local command execution and authorized native scorer replay.

Planning/validation are read-only. Execution always requires a live exact-scope
authorizer, or the CLI's explicit --execute plus the reviewed plan digest. This
runner collects artifacts; it never advances research gates or computes scores.
"""
import argparse
import contextlib
import copy
import fcntl
import hashlib
import os
import signal
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
import _autoresearch as C
import _native_eval as N


def _cwd(root, relative):
    return root if relative == "." else C.safe_path(root, relative)


def plan_digest(plan):
    return C.hashed({k:v for k,v in plan.items() if k != "plan_digest"})


def validate_plan(root, plan):
    root = C.root_path(root); C.validate(plan, "experiment-run-plan")
    if plan["plan_digest"] != plan_digest(plan): C.fail("RUN_PLAN_DIGEST_MISMATCH")
    limits = plan["limits"]
    C.safe_path(root, plan["output_root"])
    trials = {job["trial_id"] for job in plan["jobs"]}
    if len(trials) != len(plan["jobs"]): C.fail("DUPLICATE_RUN_TRIAL")
    mode = plan["evidence_mode"]
    available = limits["max_development_trials"] if mode == "developmental" else limits["max_confirmation_trials"]
    if not available or len(trials) > available: C.fail("RUN_TRIAL_BUDGET_EXCEEDED")
    if len(trials) > limits["max_attempts"]: C.fail("RUN_ATTEMPT_BUDGET_EXCEEDED")
    if mode != "developmental" and limits["max_retries_per_trial"] != 0: C.fail("CONFIRMATION_RETRIES_FORBIDDEN")
    if limits["attempt_timeout_seconds"] > limits["wall_time_seconds"]: C.fail("RUN_TIMEOUT_EXCEEDS_BUDGET")
    if plan["purpose"] == "scientific":
        if not plan["protocol_ref"]: C.fail("RUN_NATIVE_PROTOCOL_REQUIRED")
        protocol = C.load_file(C.verify_ref(root, plan["protocol_ref"]))
        contract = N.verify_protocol(root, protocol)
        if protocol.get("method_discovery"):
            from verify_methods import verify_run_design
            verify_run_design(root, plan, protocol)
        if plan["protocol_digest"] != protocol.get("protocol_digest"): C.fail("RUN_PROTOCOL_DIGEST_MISMATCH")
        if mode != "developmental" and (not protocol.get("frozen_at") or mode != protocol.get("evidence_mode")):
            C.fail("RUN_CONFIRMATION_FREEZE_REQUIRED")
        for job in plan["jobs"]:
            if job["seed"] not in protocol["seed_policy"]["seeds"] or job["group"] not in protocol["required_groups"]:
                C.fail("RUN_OUTSIDE_FROZEN_INVENTORY")
            contract = N.contract_for_group(protocol,job["group"])
            if job["arm_role"] not in contract["arm_requirements"]: C.fail("RUN_NATIVE_ARM_UNDECLARED")
            required = contract["arm_requirements"][job["arm_role"]]["implementation_refs"]
            actual = {(r["path"],r["sha256"]) for r in job["code_refs"]}
            if not {(r["path"],r["sha256"]) for r in required}.issubset(actual): C.fail("RUN_ARM_CODE_PROOF_REQUIRED")
            native_inputs = [contract["sample_manifest_ref"],contract["labels_or_tests_ref"]]
            if contract.get("selection"): native_inputs.append(contract["selection"]["selected_manifest_ref"])
            declared = {(r["path"],r["sha256"]) for r in job["input_refs"]}
            if not {(r["path"],r["sha256"]) for r in native_inputs}.issubset(declared): C.fail("RUN_NATIVE_INPUT_PROOF_REQUIRED")
    elif plan["protocol_ref"] is not None or plan["protocol_digest"] is not None:
        C.fail("ENGINEERING_RUN_CANNOT_CLAIM_PROTOCOL")
    for job in plan["jobs"]:
        _cwd(root, job["cwd"])
        for item in job["input_refs"] + job["code_refs"]: C.verify_ref(root, item)
        pinned = {C.safe_path(root,item["path"]) for item in job["input_refs"] + job["code_refs"]}
        for relative in job["output_paths"]:
            if C.safe_path(root, relative) in pinned: C.fail("RUN_OUTPUT_OVERLAPS_INPUT")
    for name in ("git_refs","model_refs","data_refs","environment_refs"):
        for item in plan["provenance"].get(name, []): C.verify_ref(root, item)
    return plan


def make_plan(root, *, run_id, jobs, provenance, limits, purpose="engineering", evidence_mode="developmental",
              protocol_ref=None, output_root="runs/attempts"):
    protocol = C.load_file(C.verify_ref(C.root_path(root), protocol_ref)) if protocol_ref else None
    plan = C.envelope("experiment-run-plan", run_id=run_id, purpose=purpose, evidence_mode=evidence_mode,
        protocol_ref=protocol_ref, protocol_digest=protocol.get("protocol_digest") if protocol else None,
        output_root=output_root, provenance=copy.deepcopy(provenance), limits=copy.deepcopy(limits), jobs=copy.deepcopy(jobs))
    plan["plan_digest"] = plan_digest(plan)
    return validate_plan(root, plan)


def _authorize(authorizer, scope):
    if not callable(authorizer): C.fail("RUN_EXECUTION_AUTHORIZATION_REQUIRED")
    if authorizer(copy.deepcopy(scope)) is not True: C.fail("RUN_EXECUTION_NOT_AUTHORIZED")


def _execute(command, cwd, timeout, stdout_path, stderr_path, *, environment=None, process_fds=()):
    """Execute argv without a shell, retaining logs and killing its process group."""
    started_at = C.now(); started = time.monotonic(); process = None
    status = "failed"; exit_code = None; reason = None
    guard_output = stdout_path.with_name("process-guard.json")
    actual_command = command
    if process_fds:
        parent_pid = Path("/proc/self/stat").read_text().split()[0]
        actual_command = [sys.executable,str(Path(__file__).with_name("_process_guard.py")),parent_pid,str(timeout),
                          str(guard_output),C.canonical(list(process_fds)),C.canonical(command)]
    def kill_owned():
        if process_fds:
            process.terminate()
            try: process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL); process.wait()
        else:
            os.killpg(process.pid,signal.SIGKILL); process.wait()
    with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
        try:
            process = subprocess.Popen(actual_command, cwd=str(cwd), stdout=stdout, stderr=stderr,
                                       shell=False, start_new_session=True, env=environment,pass_fds=tuple(process_fds))
            try:
                exit_code = process.wait(timeout=timeout)
                status = "completed" if exit_code == 0 else "failed"
                if process_fds:
                    if not guard_output.is_file():
                        status = "interrupted"; reason = "RUN_GUARD_RECEIPT_MISSING"
                    else:
                        guarded = C.load_file(guard_output)
                        if guarded["command"] != command or guarded["cwd"] != str(cwd): C.fail("RUN_GUARD_RECEIPT_MISMATCH")
                        status = guarded["status"]; exit_code = guarded["exit_code"]; reason = guarded["reason_code"]
                # A finished launcher can leave training/evaluation children
                # alive. They must not escape its timeout/resource ownership.
                try:
                    os.killpg(process.pid,0)
                except ProcessLookupError: pass
                else:
                    os.killpg(process.pid,signal.SIGKILL)
                    status = "interrupted"; reason = "RUN_UNOWNED_PROCESS_GROUP_CHILDREN"
            except subprocess.TimeoutExpired:
                kill_owned(); exit_code = process.returncode; status = "timeout"
            except KeyboardInterrupt:
                kill_owned(); exit_code = process.returncode; status = "interrupted"
        except OSError:
            stderr.write(b"COMMAND_EXECUTION_UNAVAILABLE\n")
            status = "failed"
    result = {"status":status,"exit_code":exit_code,"started_at":started_at,"completed_at":C.now(),
              "seconds":time.monotonic()-started}
    if reason: result["reason_code"] = reason
    return result


def _stage(root, job, attempt):
    """Stage pinned project inputs/code into a new per-attempt workspace."""
    workspace = attempt / "workspace"; workspace.mkdir()
    replacements = {}
    for item in job["input_refs"] + job["code_refs"]:
        source = C.safe_path(root,item["path"]); target = C.safe_path(workspace, item["path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if C.filehash(target) != item["sha256"]: C.fail("RUN_STAGED_INPUT_COLLISION")
        else:
            # A reflink is an isolated copy-on-write snapshot; unlike a hard
            # link, an experiment cannot overwrite the pinned source through it.
            try:
                with source.open("rb") as src,target.open("xb") as dst:
                    fcntl.ioctl(dst.fileno(),0x40049409,src.fileno())  # Linux FICLONE
                shutil.copystat(source,target)
            except OSError:
                if target.exists(): target.unlink()
                shutil.copy2(source,target)
            if C.filehash(target) != item["sha256"]: C.fail("RUN_STAGED_INPUT_COLLISION")
        replacements[str(source)] = str(target)
    command = []
    for arg in job["command"]:
        for source, target in sorted(replacements.items(), key=lambda pair:-len(pair[0])):
            arg = arg.replace(source, target)
        command.append(arg)
    cwd = _cwd(workspace, job["cwd"]); cwd.mkdir(parents=True, exist_ok=True)
    return command, cwd, workspace


def run_plan(root, plan, *, authorizer=None, lease_factory=None, process_fds=()):
    """Run a validated finite plan, preserving every attempt and failure.

    lease_factory(scope) may return a context manager owned by the host resource
    pool. Filesystem isolation is not an OS sandbox; authorization must account
    for the full effects of the reviewed existing commands and dependencies.
    """
    root = C.root_path(root); validate_plan(root, plan)
    if not callable(authorizer): C.fail("RUN_EXECUTION_AUTHORIZATION_REQUIRED")
    run_root = C.safe_path(root, plan["output_root"]) / plan["run_id"]
    if run_root.exists(): C.fail("RUN_ID_ALREADY_USED")
    base_scope = {"operation":"bounded_experiment_run","run_id":plan["run_id"],"plan_digest":plan["plan_digest"],
        "purpose":plan["purpose"],"evidence_mode":plan["evidence_mode"],"output_root":str(run_root),
        "budget":copy.deepcopy(plan["limits"]),"provenance":copy.deepcopy(plan["provenance"])}
    # Complete reviewable scope is authorized before any attempt directories exist.
    for job in plan["jobs"]:
        _authorize(authorizer, {**base_scope,"trial_id":job["trial_id"],"command":job["command"],
            "cwd":str(_cwd(root, job["cwd"])),"input_refs":job["input_refs"],"code_refs":job["code_refs"]})
    try: run_root.mkdir(parents=True, exist_ok=False)
    except FileExistsError: C.fail("RUN_ID_ALREADY_USED")
    (run_root / "plan.json").write_text(C.canonical(plan)+"\n")
    started_at = C.now(); start = time.monotonic(); deadline = start + plan["limits"]["wall_time_seconds"]
    attempts = []; successes = set(); status = "completed"
    for job in plan["jobs"]:
        for retry in range(plan["limits"]["max_retries_per_trial"] + 1):
            remaining = deadline - time.monotonic()
            if remaining <= 0 or len(attempts) >= plan["limits"]["max_attempts"]:
                status = "budget_exhausted"; break
            attempt = run_root / (job["trial_id"] + "-a" + str(retry + 1) + "-" + uuid.uuid4().hex)
            attempt.mkdir()
            command, cwd, workspace = _stage(root, job, attempt)
            scope = {**base_scope,"trial_id":job["trial_id"],"attempt_path":str(attempt),"command":command,"cwd":str(cwd),
                "input_refs":job["input_refs"],"code_refs":job["code_refs"],
                "timeout_seconds":min(plan["limits"]["attempt_timeout_seconds"], remaining)}
            try:
                _authorize(authorizer, scope)
                lease = lease_factory(copy.deepcopy(scope)) if callable(lease_factory) else contextlib.nullcontext()
                with lease:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0: C.fail("RUN_WALL_TIME_BUDGET_EXHAUSTED")
                    execution = _execute(command, cwd, min(plan["limits"]["attempt_timeout_seconds"], remaining),
                                         attempt / "stdout.log", attempt / "stderr.log",process_fds=process_fds)
            except C.Failure as error:
                # Expired wall time is a timeout; authority/resource faults interrupt.
                expired = error.code == "RUN_WALL_TIME_BUDGET_EXHAUSTED"
                execution = {"status":"timeout" if expired else "interrupted","exit_code":None,"started_at":C.now(),"completed_at":C.now(),"seconds":0,
                             "reason_code":error.code}
                (attempt / "stdout.log").touch(); (attempt / "stderr.log").touch()
            outputs = []
            for relative in job["output_paths"]:
                path = C.safe_path(workspace, relative)
                if path.is_file(): outputs.append(C.reference(root, path))
            record = {"attempt_id":attempt.name,"attempt_path":attempt.relative_to(root).as_posix(),
                "trial_id":job["trial_id"],"retry_index":retry,"evidence_mode":plan["evidence_mode"],
                "seed":job["seed"],"group":job["group"],"arm_role":job["arm_role"],
                "command":command,"cwd":str(cwd),"provenance":copy.deepcopy(plan["provenance"]),
                "input_refs":job["input_refs"],"code_refs":job["code_refs"],"output_refs":outputs,
                "stdout_ref":C.reference(root, attempt / "stdout.log"),"stderr_ref":C.reference(root, attempt / "stderr.log"),
                **execution}
            guard_path = attempt/"process-guard.json"
            if guard_path.is_file(): record["process_guard_ref"] = C.reference(root,guard_path)
            (attempt / "attempt.json").write_text(C.canonical(record)+"\n"); attempts.append(record)
            if execution["status"] == "completed": successes.add(job["trial_id"]); break
            if execution["status"] == "interrupted": status = "interrupted"; break
            if execution["status"] == "timeout" and deadline-time.monotonic() <= 0: status = "budget_exhausted"; break
        if status in {"budget_exhausted","interrupted"}: break
        if job["trial_id"] not in successes: status = "failed"
    if len(successes) != len(plan["jobs"]) and status == "completed": status = "failed"
    result = C.envelope("experiment-run-receipt", run_id=plan["run_id"],plan_digest=plan["plan_digest"],purpose=plan["purpose"],
        evidence_mode=plan["evidence_mode"],status=status,attempts=attempts,started_at=started_at,completed_at=C.now(),
        resources={"seconds":time.monotonic()-start},provenance=copy.deepcopy(plan["provenance"]),gate_advanced=False)
    C.validate(result,"experiment-run-receipt")
    (run_root / "receipt.json").write_text(C.canonical(result)+"\n")
    return result


def build_official_replay_context(root, authorizer, *, output_root="runs/native-replays", timeout_seconds=60.0, lease_factory=None):
    """Build a live replay callback from an explicitly scoped host authorizer.

    Invocation executes the pinned existing scorer command. JSON is only parsed
    by _native_eval. No persisted approval/scorer boolean is accepted here.
    """
    root = C.root_path(root)
    if not callable(authorizer): C.fail("RUN_EXECUTION_AUTHORIZATION_REQUIRED")
    if N._number(timeout_seconds,"RUN_TIMEOUT_INVALID") <= 0: C.fail("RUN_TIMEOUT_INVALID")
    destination = C.safe_path(root, output_root)
    def replay(request):
        scorer = request["scorer"]; N._scorer(root, scorer)
        for item in request["input_refs"]: C.verify_ref(root,item)
        expected = N._command(root, scorer, request["receipt"])
        if request["command"] != expected or request["cwd"] != str(_cwd(root,scorer["cwd"])):
            C.fail("NATIVE_SCORER_REPLAY_BINDING_MISMATCH")
        if request["request_digest"] != C.hashed({k:v for k,v in request.items() if k != "request_digest"}):
            C.fail("NATIVE_SCORER_REPLAY_BINDING_MISMATCH")
        execution_id = uuid.uuid4().hex; attempt = destination / execution_id
        output = attempt / "native-output.json"
        command = N._command(root,scorer,request["receipt"],output_path=output)
        scope = {"operation":"native_scorer_replay","root":str(root),"execution_id":execution_id,
            "nonce":request["nonce"],"request_digest":request["request_digest"],"command":command,
            "cwd":request["cwd"],"input_refs":request["input_refs"],"code_refs":scorer["code_refs"],
            "scorer_identity":scorer["identity"],"scorer_revision":scorer["revision"],
            "output_root":str(attempt),"budget":request["budget"],"timeout_seconds":timeout_seconds}
        _authorize(authorizer,scope)
        attempt.mkdir(parents=True,exist_ok=False)
        lease = lease_factory(copy.deepcopy(scope)) if callable(lease_factory) else contextlib.nullcontext()
        with lease:
            # Recheck frozen artifacts after acquiring the resource lease.
            for item in request["input_refs"] + scorer["code_refs"]: C.verify_ref(root,item)
            execution = _execute(command,Path(request["cwd"]),timeout_seconds,attempt/"stdout.log",attempt/"stderr.log")
        native_output = attempt / "stdout.log" if scorer["output"]["source"] == "stdout" else output
        result = {"nonce":request["nonce"],"request_digest":request["request_digest"],"execution_id":execution_id,
            "exit_code":execution["exit_code"],"argv":command,"cwd":request["cwd"],"input_refs":request["input_refs"],
            "execution_status":execution["status"],"reason_code":execution.get("reason_code"),
            "code_refs":scorer["code_refs"],"scorer_identity":scorer["identity"],"scorer_revision":scorer["revision"],
            "stdout_ref":C.reference(root,attempt/"stdout.log"),"stderr_ref":C.reference(root,attempt/"stderr.log"),
            "native_output_ref":C.reference(root,native_output) if native_output.is_file() else None,
            "started_at":execution["started_at"],"completed_at":execution["completed_at"]}
        (attempt/"execution.json").write_text(C.canonical(result)+"\n")
        if result["native_output_ref"] is None: C.fail("NATIVE_REPLAY_OUTPUT_REQUIRED")
        if execution["status"] != "completed": C.fail("NATIVE_SCORER_REPLAY_FAILED")
        return result
    return replay


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan"); parser.add_argument("--root",required=True)
    parser.add_argument("--execute",action="store_true",help="execute the reviewed exact finite plan")
    parser.add_argument("--approved-plan-digest",help="digest of the plan explicitly authorized for this invocation")
    args = parser.parse_args(argv)
    try:
        root = C.root_path(args.root); plan = C.load_file(Path(args.plan)); validate_plan(root,plan)
        if not args.execute:
            result = {"status":"validated","plan_digest":plan["plan_digest"],"execution_started":False}
        else:
            if args.approved_plan_digest != plan["plan_digest"]: C.fail("RUN_APPROVED_PLAN_DIGEST_REQUIRED")
            def exact_scope(scope):
                return scope.get("plan_digest") == args.approved_plan_digest and scope.get("run_id") == plan["run_id"]
            result = run_plan(root,plan,authorizer=exact_scope)
        print(C.canonical(result)); return 0
    except C.Failure as error:
        print(C.canonical({"status":"blocked","reason_code":error.code,"path":error.path})); return 2


if __name__ == "__main__": raise SystemExit(main())
