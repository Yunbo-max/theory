"""Project adapter to the unmodified research-autopilot execution owner.

Engineering jobs and scientifically admitted native plans remain different
purposes. This module never synthesizes scientific qualification receipts.
"""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid


ROOT = Path(__file__).resolve().parents[1]


def runtime():
    path = ROOT / "vendor/research_autopilot/scripts"
    if not path.is_dir():
        raise RuntimeError("Pinned research-autopilot runtime is missing")
    manifest = json.loads((path.parent / "VENDOR.json").read_text())
    for rel, expected in manifest["files"].items():
        actual = hashlib.sha256((path.parent / rel).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Research execution runtime changed: {rel}")
    sys.path.insert(0, str(path))
    import _autoresearch as C
    import run_experiments as R
    import run_harness as H
    return C, R, H


def project_refs(root):
    """Stage real project source/config/tests; large runtime assets are explicit."""
    root = Path(root).resolve()
    names = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=root).decode().split("\0")
    retained = []
    for name in sorted(set(names)):
        path = root / name
        if not name or not path.is_file() or path.is_symlink():
            continue
        if name.startswith(("runs/", "artifacts/", "returns/", ".research-autopilot/")):
            continue
        retained.append({"path": name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return retained


def environment():
    return {"python": sys.version, "interpreter": str(Path(sys.executable).resolve()),
            "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}}


def execute_engineering(command, *, label, seconds=180, inputs=(), outputs=(),
                        cpu_cores=2, ram_mib=4096):
    """Run a reviewed finite CPU software/asset check and retain exact attempts."""
    C, R, H = runtime()
    root = ROOT
    run_id = label + "-" + uuid.uuid4().hex[:10]
    destination = root / "runs/controller" / run_id
    destination.mkdir(parents=True)
    env = environment()
    (destination / "environment.json").write_text(C.canonical(env) + "\n")
    refs = project_refs(root)
    job = {"trial_id": "job", "command": list(command), "cwd": ".",
           "input_refs": list(inputs), "code_refs": refs, "output_paths": list(outputs),
           "seed": 0, "group": "engineering", "arm_role": "engineering"}
    native = R.make_plan(root, run_id=run_id, jobs=[job], provenance={
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "model_revision": "no-model-workload", "data_revision": C.hashed(list(inputs)),
        "environment_digest": C.hashed(env),
        "environment_refs": [C.reference(root, destination / "environment.json")]},
        limits={"max_attempts": 1, "max_development_trials": 1, "max_confirmation_trials": 0,
                "max_retries_per_trial": 0, "wall_time_seconds": seconds,
                "attempt_timeout_seconds": seconds})
    native_path = destination / "native-plan.json"
    native_path.write_text(C.canonical(native) + "\n")
    plan = H.make_plan(root, batch_id=run_id, tasks=[{
        "task_id": "job", "idea_id": "engineering", "plan_ref": C.reference(root, native_path),
        "depends_on": [], "priority": 1,
        "resources": {"cpu_cores": cpu_cores, "ram_mib": ram_mib, "gpu_count": 0,
                      "gpu_peak_mib": None, "allow_gpu_share": False,
                      "memory_profile_ref": None, "exclusive_keys": ["theory-engineering"]}}],
        limits={"window_seconds": seconds + 20, "total_wall_seconds": seconds + 20,
                "max_parallel_tasks": 1, "cpu_cores": cpu_cores, "ram_mib": ram_mib,
                "max_gpu_task_seconds": 0})
    (destination / "harness-plan.json").write_text(C.canonical(plan) + "\n")
    result = H.run_harness(root, plan,
        authorizer=lambda scope: scope.get("plan_digest") == plan["plan_digest"])
    receipt_path = root / native["output_root"] / run_id / "receipt.json"
    receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else None
    if receipt:
        for attempt in receipt["attempts"]:
            folder = root / attempt["attempt_path"]
            for stream in ("stdout.log", "stderr.log"):
                path = folder / stream
                if path.exists():
                    content=path.read_text()
                    if len(content)>16000:
                        print(f"Full log: {path}; showing final 16000 characters")
                    print(content[-16000:], end="")
    success = bool(receipt and receipt["status"] == "completed")
    print(json.dumps({"status": "completed" if success else "failed", "run_id": run_id,
                      "receipt": str(receipt_path), "harness": result.get("status"),
                      "scope": "engineering; no scientific gate advanced"}))
    return 0 if success else 1
