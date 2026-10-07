"""Finite dependency controller for the real research-autopilot execution owner.

Catalog compilation is metadata, not admission. Every executable unit, including
live native replay, is a bounded task of the pinned skill harness. Checkpoints
are bound only after real parent outputs exist. No model score is invented here.
"""
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
import uuid

from .harness import ROOT, environment as current_environment, project_refs, runtime
from .io import atomic_json, digest, object_hash, read_json
from .suite_design import ORDER, expected_cells, make_suite, resolve_arm, verify_suite
from . import suite_admission as admission


class QueueError(ValueError):
    pass


def _fail(code, detail=""):
    raise QueueError(code + (": " + str(detail) if detail else ""))


def _path(root, value):
    root = Path(root).resolve()
    path = Path(value)
    path = path if path.is_absolute() else root / path
    try:
        path.resolve().relative_to(root)
    except ValueError:
        _fail("PATH_OUTSIDE_PROJECT", value)
    if path.is_symlink():
        _fail("SYMLINK_FORBIDDEN", value)
    return path


def ref(root, path):
    path = _path(root, path)
    if not path.is_file():
        _fail("REFERENCE_FILE_REQUIRED", path)
    return {"path": path.relative_to(Path(root).resolve()).as_posix(), "sha256": digest(path)}


def verify_ref(root, value):
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        _fail("PINNED_REFERENCE_REQUIRED")
    path = _path(root, value["path"])
    if not path.is_file() or digest(path) != value["sha256"]:
        _fail("REFERENCE_DIGEST_MISMATCH", value["path"])
    return path


def _unique_refs(values):
    found = {}
    for value in values:
        if value["path"] in found and found[value["path"]] != value["sha256"]:
            _fail("CONFLICTING_REFERENCES", value["path"])
        found[value["path"]] = value["sha256"]
    return [{"path": path, "sha256": sha} for path, sha in sorted(found.items())]


def _nested_refs(value):
    if isinstance(value, dict):
        if set(value) == {"path", "sha256"}:
            yield value
        else:
            for child in value.values():
                yield from _nested_refs(child)
    elif isinstance(value, list):
        for child in value:
            yield from _nested_refs(child)


def _rebase_paths(value, old, new):
    old, new = str(old).rstrip("/"), str(new).rstrip("/")
    if isinstance(value, dict):
        return {key: _rebase_paths(child, old, new) for key, child in value.items()}
    if isinstance(value, list):
        return [_rebase_paths(child, old, new) for child in value]
    if isinstance(value, str) and (value == old or value.startswith(old + "/")):
        return new + value[len(old):]
    return value


def source_refs(root):
    root = Path(root).resolve()
    paths = [root / "pyproject.toml"]
    # These small retained source records are runtime dependencies, not caches:
    # training verifies clean IDs from DESIGN_DATA and catalogue reconstruction
    # reads the method specification inside the isolated attempt workspace.
    paths.append(root / "research/design-v2/method-specs.json")
    paths.extend(root / "research/design-v2/data" / name for name in (
        "train_prompts-clean-v2.jsonl", "clean-training-manifest.json",
        "humaneval-manifest.json", "mbpp-plus-audit.json", "lcb-native-row-audit.json"))
    for name in ("recursive_ssd", "scripts", "configs", "vendor", "tests"):
        paths.extend(p for p in (root / name).rglob("*") if p.is_file()
                     and "__pycache__" not in p.parts and p.suffix != ".pyc" and not p.is_symlink())
    return [ref(root, p) for p in sorted(set(paths)) if p.is_file()]


def _manifest_files(root, manifest_ref):
    path = verify_ref(root, manifest_ref)
    manifest = read_json(path)
    files = manifest.get("files", {})
    if isinstance(files, dict):
        return [ref(root, path.parent / name) if digest(_path(root, path.parent / name)) == sha
                else _fail("MANIFEST_FILE_DIGEST_MISMATCH", name) for name, sha in files.items()]
    if isinstance(files, list):
        result = []
        for item in files:
            source = Path(item["path"])
            if not source.is_absolute():
                source = path.parent / source
            value = ref(root, source)
            if value["sha256"] != item["sha256"]:
                _fail("MODEL_FILE_DIGEST_MISMATCH", item["path"])
            result.append(value)
        return result
    _fail("MANIFEST_FILES_INVALID")


def validate_dependencies(nodes):
    ids = {node["node_id"] for node in nodes}
    if len(ids) != len(nodes):
        _fail("DUPLICATE_NODE")
    mapping = {node["node_id"]: node for node in nodes}
    seen, active = set(), set()
    def visit(identifier):
        if identifier in active:
            _fail("DEPENDENCY_CYCLE", identifier)
        if identifier in seen:
            return
        if identifier not in mapping:
            _fail("DEPENDENCY_UNKNOWN", identifier)
        active.add(identifier)
        for parent in mapping[identifier]["depends_on"]:
            visit(parent)
        active.remove(identifier)
        seen.add(identifier)
    for identifier in ids:
        visit(identifier)
    return nodes


def compile_nodes(suite):
    """Expand the complete catalog, without fictional future file references."""
    verify_suite(suite)
    trajectories = {t["trajectory_id"]: t for t in suite["trajectories"]}
    matching = {(t["arm_id"], t["seed"]): t for t in suite["trajectories"]}
    def bundles(arm):
        if suite["stage"] == "tuning":
            if arm == "base":
                return ["M03", "M01"]
            return ["M03" if arm.startswith(("M03__", "arithmetic_anchor__")) else "M01"]
        return [candidate for candidate, spec in suite["method_bundles"].items() if arm in spec["arms"]]
    def train_id(t, round_):
        return t["trajectory_id"] + "-train-r" + str(round_)
    nodes = []
    derivations = {"head_temperature": "M06", "m13_step_scale": "M13",
                   "clock_m13": "M13", "clock_m15": "M15"}
    for trajectory in suite["trajectories"]:
        arm = trajectory["arm_id"]
        if arm == "base":
            continue
        for round_ in range(1, trajectory["rounds"] + 1):
            parent = train_id(trajectory, round_ - 1) if round_ > 1 else None
            deps = [parent] if parent else []
            calibrations = []
            for key in trajectory["arm"].get("required_calibration", []):
                if key not in derivations:
                    continue  # external development selection is required at materialization
                if key == "head_temperature" and suite["stage"] not in {"development", "tuning"}:
                    continue  # this global hyperparameter is frozen on development before confirmation
                source = matching.get((derivations[key], trajectory["seed"]))
                if source is None:
                    _fail("CALIBRATION_SOURCE_NOT_PLANNED", key)
                cid = "cal-" + object_hash({"key": key, "source": source["trajectory_id"], "round": round_})[:24]
                if not any(n["node_id"] == cid for n in nodes):
                    nodes.append({"node_id": cid, "kind": "derive-calibration", "arm_id": arm,
                        "trajectory_id": trajectory["trajectory_id"], "round": round_,
                        "calibration_key": key, "source_node_id": train_id(source, round_),
                        "depends_on": [train_id(source, round_)], "bundles": bundles(arm)})
                calibrations.append(cid)
            deps += calibrations
            nodes.append({"node_id": train_id(trajectory, round_), "kind": "train", "arm_id": arm,
                "trajectory_id": trajectory["trajectory_id"], "round": round_,
                "parent_node_id": parent, "lag_node_id": train_id(trajectory, round_ - 2) if round_ > 2 else None,
                "fixed_node_id": train_id(trajectory, 1) if arm == "fixed_data" and round_ > 1 else None,
                "calibration_nodes": calibrations, "depends_on": deps, "bundles": bundles(arm)})
    for unit in suite["evaluation_units"]:
        parent = train_id(trajectories[unit["trajectory_id"]], unit["round"]) if unit["round"] else None
        nodes.append({"node_id": unit["cell_id"], "kind": "evaluate", "arm_id": unit["arm_id"],
            "trajectory_id": unit["trajectory_id"], "round": unit["round"], "unit": deepcopy(unit),
            "parent_node_id": parent, "depends_on": [parent] if parent else [], "bundles": bundles(unit["arm_id"])})
    return validate_dependencies(nodes)


def ready_nodes(nodes, state, *, bundle):
    result = []
    for node in nodes:
        if bundle not in node["bundles"] or state[node["node_id"]]["status"] != "pending":
            continue
        ready = True
        for parent in node["depends_on"]:
            record = state[parent]
            if record["status"] == "completed":
                if not record.get("receipt_ref") or not record.get("workload_receipt_ref"):
                    _fail("DEPENDENCY_RECEIPT_REQUIRED", parent)
            else:
                ready = False
        if ready:
            result.append(node)
    # Complete initial/native endpoints before later rounds whenever possible.
    return sorted(result, key=lambda n: (n["round"], {"derive-calibration": 0, "evaluate": 1, "train": 2}[n["kind"]], n["node_id"]))


def inventory_status(nodes, state):
    statuses = Counter(state[n["node_id"]]["status"] for n in nodes)
    bundles = {}
    for bundle in sorted({b for n in nodes for b in n["bundles"]}):
        retained = [n for n in nodes if bundle in n["bundles"]]
        counts = Counter(state[n["node_id"]]["status"] for n in retained)
        bundles[bundle] = {"status": "completed" if counts["completed"] == len(retained) else "incomplete",
                           "expected": len(retained), **dict(counts)}
    return {"expected": len(nodes), "counts": dict(statuses), "bundles": bundles}


@contextmanager
def _lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "controller.lock").open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            _fail("QUEUE_CONTROLLER_ALREADY_ACTIVE")
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _queue_digest(queue):
    return object_hash({key: value for key, value in queue.items() if key != "queue_digest"})


def authorization_path(root, directory):
    directory = _path(root, directory)
    location = directory / "budget-location.json"
    if location.exists():
        return _path(root, read_json(location)["path"])
    return directory / "budget.json"


def initialize_authorization(root, directory, *, original_start, cap_seconds=28800,
                             already_used_seconds=0, budget_from=None):
    root, directory = Path(root).resolve(), _path(root, directory)
    stamp = datetime.fromisoformat(original_start.replace("Z", "+00:00"))
    if stamp.tzinfo is None:
        _fail("TIMESTAMP_TIMEZONE_REQUIRED")
    start = stamp.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    shared = (authorization_path(root, budget_from) if budget_from else
              root / "runs/authorizations" / ("window-" + object_hash({"start": start, "cap": float(cap_seconds)})[:20] + ".json"))
    budget = admission.initialize_budget(shared, original_start=start, cap_seconds=cap_seconds,
                                          already_used_seconds=already_used_seconds)
    pointer = {"project_root": str(root), "path": str(shared.relative_to(root))}
    location = directory / "budget-location.json"
    if location.exists() and read_json(location) != pointer:
        _fail("BUDGET_RESTART_FORBIDDEN")
    atomic_json(location, pointer)
    return budget


def build_queue(root, directory, *, suite_ref, benchmark_manifest_ref, model_manifest_ref,
                original_start, cap_seconds=28800, already_used_seconds=0,
                code_refs=None, environment=None, budget_from=None):
    root, directory = Path(root).resolve(), _path(root, directory)
    suite = verify_suite(read_json(verify_ref(root, suite_ref)))
    env = current_environment() if environment is None else environment
    code = source_refs(root) if code_refs is None else _unique_refs(code_refs)
    for item in code:
        verify_ref(root, item)
    data_files = _manifest_files(root, benchmark_manifest_ref)
    model_files = _manifest_files(root, model_manifest_ref)
    budget = initialize_authorization(root, directory, original_start=original_start,
        cap_seconds=cap_seconds, already_used_seconds=already_used_seconds, budget_from=budget_from)
    identity = {"suite_ref": suite_ref, "benchmark_manifest_ref": benchmark_manifest_ref,
        "project_root": str(root),
        "budget_location_ref": ref(root, directory / "budget-location.json"),
        "model_manifest_ref": model_manifest_ref, "data_refs": data_files, "model_refs": model_files,
        "code_refs": code, "source_inventory": "project" if code_refs is None else "explicit",
        "environment_digest": object_hash(env), "original_start": budget["original_start"],
        "absolute_end": budget["absolute_end"], "cap_seconds": budget["cap_seconds"],
        "initial_used_seconds": budget["initial_used_seconds"]}
    queue = {"schema": "recursive-ssd-queue-v3", "identity": identity,
        "suite_digest": suite["suite_digest"], "stage": suite["stage"], "nodes": compile_nodes(suite),
        "retry_policy": {"development_max_attempts_per_node": 2, "confirmation_max_attempts_per_node": 1},
        "scientific_dispatch_ready": False, "admission": "per complete bundle and current canonical state"}
    queue["queue_digest"] = _queue_digest(queue)
    with _lock(directory):
        if (directory / "queue.json").exists():
            if read_json(directory / "queue.json") != queue:
                _fail("QUEUE_IDENTITY_CHANGED_CREATE_CHILD")
        else:
            atomic_json(directory / "environment.json", env)
            atomic_json(directory / "queue.json", queue)
            atomic_json(directory / "state.json", {"queue_digest": queue["queue_digest"], "admissions": {},
                "nodes": {n["node_id"]: {"status": "pending", "attempts": []} for n in queue["nodes"]}})
    return queue


def verify_queue(root, directory, *, environment=None):
    root, directory = Path(root).resolve(), _path(root, directory)
    queue = read_json(directory / "queue.json")
    if queue.get("queue_digest") != _queue_digest(queue):
        _fail("QUEUE_IDENTITY_DIGEST_MISMATCH")
    state = read_json(directory / "state.json")
    if state["queue_digest"] != queue["queue_digest"] or set(state["nodes"]) != {n["node_id"] for n in queue["nodes"]}:
        _fail("QUEUE_STATE_IDENTITY_MISMATCH")
    identity = queue["identity"]
    for item in _nested_refs(identity):
        verify_ref(root, item)
    if identity["source_inventory"] == "project" and source_refs(root) != identity["code_refs"]:
        _fail("SOURCE_INVENTORY_CHANGED")
    env = current_environment() if environment is None else environment
    if object_hash(env) != identity["environment_digest"]:
        _fail("ENVIRONMENT_IDENTITY_CHANGED")
    suite = verify_suite(read_json(verify_ref(root, identity["suite_ref"])))
    if suite["suite_digest"] != queue["suite_digest"] or compile_nodes(suite) != queue["nodes"]:
        _fail("QUEUE_CATALOG_CHANGED")
    budget = read_json(authorization_path(root, directory))
    if any(budget[k] != identity[k] for k in ("original_start", "absolute_end", "cap_seconds", "initial_used_seconds")):
        _fail("ORIGINAL_BUDGET_IDENTITY_CHANGED")
    admission.remaining_budget(authorization_path(root, directory))  # validate all cumulative accounting
    return queue, state, suite


def _time_left(directory):
    path = Path(directory)
    if path.is_file():
        budget = read_json(path)
    else:
        pointer = read_json(path / "budget-location.json") if (path / "budget-location.json").exists() else None
        budget = read_json(Path(pointer["project_root"]) / pointer["path"] if pointer else path / "budget.json")
    end = datetime.fromisoformat(budget["absolute_end"].replace("Z", "+00:00")).timestamp()
    left = min(end - time.time(), budget["cap_seconds"] - budget["cumulative_used_seconds"])
    if left <= 0:
        _fail("ORIGINAL_BUDGET_EXPIRED")
    return left


def _conda():
    prefix = os.environ.get("CONDA_PREFIX")
    if not prefix or not Path(sys.executable).resolve().is_relative_to(Path(prefix).resolve()):
        _fail("ACTIVE_NATIVE_CONDA_REQUIRED", "activate the project Conda interpreter")


def _resources(bindings, *, gpu=True):
    resource = deepcopy(bindings.get("resources", {}))
    resource.setdefault("cpu_cores", 2)
    resource.setdefault("ram_mib", 14000 if gpu else 4096)
    resource.update(gpu_count=1 if gpu else 0, allow_gpu_share=False,
                    exclusive_keys=["recursive-ssd-single-host"])
    resource.setdefault("gpu_peak_mib", None)
    resource.setdefault("memory_profile_ref", None)
    if not gpu:
        resource.update(gpu_peak_mib=None, memory_profile_ref=None)
    elif not resource["gpu_peak_mib"] or not resource["memory_profile_ref"]:
        _fail("MEASURED_GPU_PROFILE_REQUIRED")
    return resource


def _native_inputs(root, protocol_ref, group):
    C, _, _ = runtime()
    import _native_eval as N
    protocol = read_json(verify_ref(root, protocol_ref))
    contract = N.contract_for_group(protocol, group)
    inputs = [contract["sample_manifest_ref"], contract["labels_or_tests_ref"]]
    if contract.get("selection"):
        inputs.append(contract["selection"]["selected_manifest_ref"])
    definition = read_json(verify_ref(root, contract["native_definition_ref"]))
    inputs += definition.get("required_input_refs", [])
    for item in inputs:
        C.verify_ref(root, item)
    return protocol, contract, inputs


def _node_scope(node, bindings, suite):
    unit = node.get("unit", {"benchmark": "humaneval", "split": "confirm" if suite["stage"] in {"confirmation", "boundary", "model_boundary"} else "dev"})
    key = unit["benchmark"] + "/" + unit["split"]
    try:
        return bindings["groups"][key], bindings["arm_roles"][node["arm_id"]]
    except KeyError as error:
        _fail("COMPLETE_NATIVE_BINDING_REQUIRED", str(error))


def _qualification_replay_count(root, protocol):
    import _native_eval as N
    total = 0
    for group in protocol["required_groups"]:
        for source in admission.qualification_sources(protocol, group):
            record = source["record"]
            qualifier = read_json(verify_ref(root, record["protocol_ref"]))
            manifest = read_json(verify_ref(root, record["manifest_ref"]))
            contract = N.contract_for_group(qualifier, manifest["group"])
            total += len(contract["arm_requirements"]) * (2 if contract["scorer"]["kind"] == "faithful_harness" else 1)
    return total


def validate_shared_configuration(root, state, bindings):
    """Shared trajectories cannot acquire a different global training config."""
    for record in state["admissions"].values():
        previous = read_json(verify_ref(root, record["bindings_ref"]))
        if previous.get("config", {}) != bindings.get("config", {}):
            _fail("SHARED_BASELINE_CONFIGURATION_CHANGED_CREATE_CHILD")
        if previous.get("calibration", {}) != bindings.get("calibration", {}):
            _fail("SHARED_BASELINE_CALIBRATION_CHANGED_CREATE_CHILD")


def effective_calibration(suite, bindings):
    """An admitted extension cannot replace a value frozen in the suite."""
    frozen, supplied = suite.get("calibration", {}), bindings.get("calibration", {})
    for key in frozen.keys() & supplied.keys():
        if frozen[key] != supplied[key]:
            _fail("FROZEN_SUITE_CALIBRATION_CHANGED", key)
    return {**deepcopy(frozen), **deepcopy(supplied)}


def validate_qualification_job(job):
    """A baseline label cannot hide candidate training settings."""
    from .suite_design import arm_registry
    kind = job.get("kind")
    if kind not in {"preflight", "train", "evaluate"}:
        _fail("QUALIFICATION_JOB_KIND_FORBIDDEN")
    arm = job.get("trajectory", {}).get("arm_id", job.get("unit", {}).get("arm_id", "hard"))
    if arm not in {"base", "hard", "full_soft"}:
        _fail("QUALIFICATION_CANDIDATE_FORBIDDEN")
    if kind == "train" and job.get("trajectory", {}).get("arm") != arm_registry()[arm]:
        _fail("QUALIFICATION_BASELINE_SETTINGS_MISMATCH")
    if kind == "preflight" and set(job.get("methods", ["hard", "full_soft"])) - {"hard", "full_soft"}:
        _fail("QUALIFICATION_CANDIDATE_FORBIDDEN")
    return arm


def materialize_comparator_qualification(root, request, data, bindings, protocol):
    """Derive an exact comparator job from real, protocol-bound dependencies.

    The caller supplies only a catalogue node ID, never training settings or a
    checkpoint. Selected methods require their own frozen child route;
    measured controls consume only verified, charged child dependencies.
    """
    root, data = Path(root).resolve(), _path(root, data)
    scope = admission._qualification(protocol, root)
    if scope["purpose"] != "comparator-qualification" or set(request) != {"node_id"}:
        _fail("QUALIFICATION_CATALOGUE_NODE_REQUEST_REQUIRED")
    suite = read_json(verify_ref(root, scope["suite_ref"]))
    nodes = {node["node_id"]: node for node in compile_nodes(suite)}
    node = nodes.get(request["node_id"])
    expected_arm = scope["arm_bindings"].get(bindings["arm_role"])
    if (not node or node["kind"] not in {"train", "evaluate"}
            or node["arm_id"] != expected_arm or scope["bundle_id"] not in node["bundles"]):
        _fail("QUALIFICATION_EXACT_CATALOGUE_NODE_REQUIRED")
    from .suite_precursor import validate_node_native_inputs
    native_inputs = validate_node_native_inputs(root, data, node, protocol, bindings["group"])
    if bindings.get("config", {}) != scope["config"] or bindings.get("calibration", {}) != scope["calibration"]:
        _fail("QUALIFICATION_FROZEN_SETTINGS_MISMATCH")
    if bindings.get("tuning_budget") != scope.get("tuning_budget"):
        _fail("QUALIFICATION_FROZEN_TUNING_ALLOWANCE_REQUIRED")
    dependencies = set()
    def visit(identifier):
        if identifier in dependencies:
            return
        parent = nodes[identifier]
        if parent["kind"] not in {"train", "derive-calibration"}:
            _fail("QUALIFICATION_PRECURSOR_DESIGN_REQUIRED", identifier)
        dependencies.add(identifier)
        for predecessor in parent["depends_on"]:
            visit(predecessor)
    for identifier in node["depends_on"]:
        visit(identifier)
    state = {"nodes": {}}
    inputs = [scope["suite_ref"], scope["method_verification_ref"], *native_inputs]
    source = bindings.get("dependency_state_ref")
    if source:
        state = read_json(verify_ref(root, source))
        inputs.append(source)
    if set(state.get("nodes", {})) != dependencies:
        _fail("QUALIFICATION_EXACT_DEPENDENCY_CLOSURE_REQUIRED")
    for identifier in sorted(dependencies):
        entry = state["nodes"][identifier]
        _completed_output(root, state, identifier)
        native = read_json(verify_ref(root, entry["receipt_ref"]))
        provenance = native.get("provenance", {})
        if provenance.get("admission_scope") == "candidate-calibration-child":
            from .suite_precursor import validate_child_dependency
            validate_child_dependency(root, state, identifier, suite=suite, scope=scope,
                                      budget_path=bindings["budget_path"])
            inputs += list(_nested_refs(entry))
            continue
        if nodes[identifier]["arm_id"] != expected_arm or nodes[identifier]["kind"] != "train":
            _fail("QUALIFICATION_CANDIDATE_DEPENDENCY_REQUIRES_CHILD", identifier)
        expected = {"admission_scope": "comparator-qualification",
                    "qualification_protocol_ref": bindings["protocol_ref"],
                    "qualification_arm_role": bindings["arm_role"],
                    "node_id": identifier, "qualification_suite_digest": suite["suite_digest"],
                    **scope["execution_provenance"]}
        if any(provenance.get(key) != value for key, value in expected.items()):
            _fail("QUALIFICATION_DEPENDENCY_SCOPE_MISMATCH", identifier)
        budget = admission._read_budget(_path(root, bindings["budget_path"]))
        if not any(row["receipt_ref"] == entry["receipt_ref"] for row in budget["charges"]):
            _fail("QUALIFICATION_DEPENDENCY_CHARGE_REQUIRED", identifier)
        for previous_attempt in entry["attempts"]:
            plan = read_json(verify_ref(root, previous_attempt["native_plan_ref"]))
            if plan["protocol_ref"] != bindings["protocol_ref"]:
                _fail("QUALIFICATION_DEPENDENCY_PROTOCOL_MISMATCH")
            _, R, _ = runtime()
            R.validate_plan(root, plan)
            validate_bound_comparator_plan(root, plan, protocol)
        inputs += list(_nested_refs(entry))
    # materialize_job also resolves lag and fixed-data ancestry by ordered round.
    state["nodes"][node["node_id"]] = {"status": "pending"}
    queue = {"identity": {"project_root": str(root), "suite_ref": scope["suite_ref"],
        "benchmark_manifest_ref": ref(root, data / "benchmarks-manifest.json"),
        "model_manifest_ref": ref(root, data / "model-manifest.json")}}
    job, dependency_inputs = materialize_job(root, queue, state, suite, node, bindings)
    return job, _unique_refs([*inputs, *dependency_inputs])


def validate_bound_comparator_plan(root, plan, protocol):
    """Prevent direct build_scientific_plan callers from supplying arbitrary argv."""
    provenance = plan["provenance"]
    bindings_ref = provenance.get("qualification_bindings_ref")
    job_ref = provenance.get("qualification_job_ref")
    bindings = read_json(verify_ref(root, bindings_ref))
    actual_job = read_json(verify_ref(root, job_ref))
    if (bindings.get("protocol_ref") != plan["protocol_ref"]
            or len(plan["jobs"]) != 1
            or provenance.get("qualification_protocol_ref") != plan["protocol_ref"]):
        _fail("QUALIFICATION_PROTOCOL_BINDING_MISMATCH")
    entry = plan["jobs"][0]
    if entry["arm_role"] != bindings["arm_role"] or entry["group"] != bindings["group"]:
        _fail("QUALIFICATION_PLAN_ROLE_MISMATCH")
    expected_job, required = materialize_comparator_qualification(
        root, {"node_id": provenance.get("node_id")}, bindings["data"], bindings, protocol)
    if actual_job != expected_job:
        _fail("QUALIFICATION_DERIVED_JOB_MISMATCH")
    scope = protocol["suite_qualification"]
    if any(provenance.get(key) != value for key, value in scope["execution_provenance"].items()):
        _fail("QUALIFICATION_PROVENANCE_MISMATCH")
    if provenance["environment_digest"] != object_hash(current_environment()):
        _fail("QUALIFICATION_CURRENT_ENVIRONMENT_MISMATCH")
    if _unique_refs(entry["code_refs"]) != _unique_refs(source_refs(root)):
        _fail("QUALIFICATION_CURRENT_SOURCE_INVENTORY_MISMATCH")
    data = _path(root, bindings["data"])
    if (read_json(data / "model-manifest.json")["revision"] != provenance["model_revision"]
            or ref(root, data / "benchmarks-manifest.json")["sha256"] != provenance["data_revision"]):
        _fail("QUALIFICATION_DATA_MODEL_BINDING_MISMATCH")
    if provenance.get("qualification_suite_digest") != actual_job["suite_digest"]:
        _fail("QUALIFICATION_SUITE_BINDING_MISMATCH")
    if actual_job["stage"] == "tuning" and actual_job["kind"] == "train":
        state_ref = bindings.get("dependency_state_ref")
        state = read_json(verify_ref(root, state_ref)) if state_ref else {"nodes": {}}
        paid = sum(read_json(verify_ref(root, item["receipt_ref"]))["resources"]["seconds"]
                   for item in state["nodes"].values())
        if paid + plan["limits"]["wall_time_seconds"] > scope["tuning_budget"]["trial_wall_seconds"]:
            _fail("QUALIFICATION_TUNING_TRIAL_ALLOWANCE_EXCEEDED")
    identity = actual_job.get("trajectory", actual_job.get("unit", {}))
    if entry["seed"] != identity["seed"]:
        _fail("QUALIFICATION_PLAN_SEED_MISMATCH")
    argv = entry["command"]
    expected_tail = ["-m", "recursive_ssd.suite", "execute", "--job", str(verify_ref(root, job_ref)),
                     "--data", bindings["data"], "--output", "results", "--seconds",
                     str(plan["limits"]["wall_time_seconds"])]
    if argv[1:] != expected_tail or Path(argv[0]).resolve() != Path(sys.executable).resolve():
        _fail("QUALIFICATION_COMMAND_NOT_DERIVED")
    if entry["output_paths"] != ["results/receipt.json"]:
        _fail("QUALIFICATION_OUTPUT_SCOPE_MISMATCH")
    manifest = ref(root, data / "benchmarks-manifest.json")
    model_manifest = ref(root, data / "model-manifest.json")
    required += [bindings_ref, job_ref, scope["suite_ref"], scope["method_verification_ref"],
                 manifest, model_manifest, *_manifest_files(root, manifest), *_manifest_files(root, model_manifest)]
    declared = {(value["path"], value["sha256"]) for value in entry["input_refs"]}
    if any((value["path"], value["sha256"]) not in declared for value in required):
        _fail("QUALIFICATION_DEPENDENCY_INPUTS_REQUIRED")


def admit_bundle(root, directory, bundle, bindings, *, environment=None):
    root, directory = Path(root).resolve(), _path(root, directory)
    queue, state, suite = verify_queue(root, directory, environment=environment)
    _time_left(directory)
    selected = [n for n in queue["nodes"] if bundle in n["bundles"]]
    if not selected:
        _fail("UNKNOWN_COMPLETE_BUNDLE", bundle)
    validate_shared_configuration(root, state, bindings)
    inherited = suite.get("development_binding")
    if inherited and bindings.get("config", {}) != inherited["config"]:
        _fail("SELECTED_DEVELOPMENT_SETTINGS_CHANGED", "configuration")
    for key in ("protocol_ref", "verification_ref", "calibration_ref", "gpu_uuid", "replay_timeout_seconds", "replay_calls_per_node"):
        if key not in bindings:
            _fail("SCIENTIFIC_BINDING_REQUIRED", key)
    checked = admission.validate_candidate_binding(root, bindings["protocol_ref"], bindings["verification_ref"])
    if checked["candidate_id"] != bundle:
        _fail("CURRENT_CANDIDATE_BUNDLE_MISMATCH")
    protocol = read_json(verify_ref(root, bindings["protocol_ref"]))
    frozen_binding = {"suite_digest": suite["suite_digest"],
        "benchmark_manifest_ref": queue["identity"]["benchmark_manifest_ref"],
        "model_manifest_ref": queue["identity"]["model_manifest_ref"],
        "code_digest": object_hash(queue["identity"]["code_refs"]),
        "environment_digest": queue["identity"]["environment_digest"],
        "config_digest": object_hash(bindings.get("config", {})),
        "calibration_digest": object_hash(effective_calibration(suite, bindings))}
    if suite["stage"] == "tuning":
        allowance = bindings.get("tuning_budget")
        if (not isinstance(allowance, dict) or isinstance(allowance.get("trial_wall_seconds"), bool)
                or not isinstance(allowance.get("trial_wall_seconds"), (int, float))
                or not 0 < allowance["trial_wall_seconds"] <= 28800 or not allowance.get("source_refs")):
            _fail("FROZEN_EQUAL_TUNING_WALL_ALLOWANCE_REQUIRED")
        for prior_admission in state["admissions"].values():
            previous = read_json(verify_ref(root, prior_admission["bindings_ref"]))
            if previous.get("tuning_budget") != allowance:
                _fail("TUNING_FAMILIES_REQUIRE_IDENTICAL_WALL_ALLOWANCE")
        frozen_binding["tuning_budget"] = allowance
    if protocol.get("suite_binding") != frozen_binding:
        _fail("FROZEN_SUITE_CONFIGURATION_BINDING_REQUIRED")
    _resources(bindings)
    for item in _nested_refs(bindings):
        verify_ref(root, item)
    if not isinstance(bindings["gpu_uuid"], str) or not bindings["gpu_uuid"].startswith("GPU-"):
        _fail("EXACT_SINGLE_GPU_UUID_REQUIRED")
    for node in selected:
        group, role = _node_scope(node, bindings, suite)
        _, contract, _ = _native_inputs(root, bindings["protocol_ref"], group)
        expected = "initial" if node["arm_id"] == "base" else node["arm_id"]
        if contract["arm_requirements"].get(role, {}).get("name") != expected:
            _fail("NATIVE_ARM_BINDING_MISMATCH", node["node_id"])
    timeout, calls = bindings["replay_timeout_seconds"], bindings["replay_calls_per_node"]
    if type(calls) is not int or calls < 1 or isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 0 < timeout <= 3600:
        _fail("FINITE_LIVE_REPLAY_BUDGET_REQUIRED")
    if calls != _qualification_replay_count(root, protocol):
        _fail("EXACT_NATIVE_REPLAY_CALL_INVENTORY_REQUIRED")
    costs = read_json(verify_ref(root, bindings["calibration_ref"]))
    ids = [n["node_id"] for n in selected]
    if not set(ids).issubset(costs.get("costs", {})):
        _fail("COMPLETE_BUNDLE_CALIBRATION_REQUIRED")
    pending = [n["node_id"] for n in selected if state["nodes"][n["node_id"]]["status"] != "completed"]
    expected = {"environment_digest": queue["identity"]["environment_digest"],
                "model_revision": suite["model_revision"], "code_digest": object_hash(queue["identity"]["code_refs"]),
                "benchmark_manifest_ref": queue["identity"]["benchmark_manifest_ref"]}
    reserve = float(bindings.get("finalization_reserve_seconds", 900))
    overhead = len(pending) * calls * timeout
    if reserve < 0 or not math.isfinite(reserve):
        _fail("FINITE_FINALIZATION_RESERVE_REQUIRED")
    if not pending:
        return {"status": "completed", "bundle_id": bundle, "gate_advanced": False}
    binding_hash = object_hash(bindings)
    prior = state["admissions"].get(bundle)
    if prior and prior["bindings_digest"] != binding_hash:
        _fail("ADMISSION_BINDINGS_CHANGED_CREATE_CHILD")
    if prior:
        return {"bundle_id": bundle, "status": "previously reserved", "admission": prior,
                "gate_advanced": False, "dispatch_ready": False}
    reservation_id = "bundle-" + queue["queue_digest"][:12] + "-" + bundle
    budget = read_json(authorization_path(root, directory))
    existing = budget["reservations"].get(reservation_id, {})
    needed = sum(costs["costs"][cell]["upper_seconds"] for cell in pending)
    available = admission.remaining_budget(authorization_path(root, directory)) + existing.get("seconds", 0)
    if needed + overhead + reserve > available:
        _fail("COMPLETE_BUNDLE_WITH_REPLAY_EXCEEDS_REMAINING_BUDGET")
    reservation = admission.reserve_bundle(root, authorization_path(root, directory), bundle_id=reservation_id,
        cell_ids=pending, calibration_ref=bindings["calibration_ref"], expected_bindings=expected)
    replay_reservations = []
    for index in range(len(pending) * calls):
        identifier = reservation_id + "-replay-" + str(index)
        admission.reserve_preparation(authorization_path(root, directory), preparation_id=identifier,
            bound_seconds=timeout, purpose="native-reference-replay")
        replay_reservations.append(identifier)
    with _lock(directory):
        state = read_json(directory / "state.json")
        path = directory / "admissions" / (bundle + "-" + binding_hash[:12] + ".json")
        atomic_json(path, bindings)
        state["admissions"][bundle] = {"bindings_ref": ref(root, path), "bindings_digest": binding_hash,
            "reservation_id": reservation_id, "cell_ids": pending, "canonical_check": checked,
            "replay_reservations": replay_reservations, "next_replay_index": 0,
            "remaining_replay_calls": len(pending) * calls, "finalization_reserve_seconds": reserve,
            "status": "reserved; native live replay is required at every dispatch"}
        atomic_json(directory / "state.json", state)
    return {"bundle_id": bundle, "reservation": reservation, "gate_advanced": False,
            "dispatch_ready": False, "scope": "complete resource reservation; per-node live admission remains required"}


def _harness_plan(root, folder, native, *, resources, gpu_uuid=None, hard_seconds=None):
    C, _, H = runtime()
    folder.mkdir(parents=True, exist_ok=True)
    native_path = folder / "native-plan.json"
    if native_path.exists() and read_json(native_path) != native:
        _fail("IMMUTABLE_NATIVE_PLAN_CHANGED")
    atomic_json(native_path, native)
    seconds = native["limits"]["wall_time_seconds"]
    plan = H.make_plan(root, batch_id=native["run_id"], tasks=[{
        "task_id": "job", "idea_id": native["provenance"].get("candidate_id", "native-preparation"),
        "plan_ref": C.reference(root, native_path), "depends_on": [], "priority": 1, "resources": resources}],
        limits={"window_seconds": max(1, min(seconds + 10, hard_seconds or seconds + 10)),
                "total_wall_seconds": max(1, min(seconds + 10, hard_seconds or seconds + 10)),
                "max_parallel_tasks": 1, "cpu_cores": resources["cpu_cores"], "ram_mib": resources["ram_mib"],
                "max_gpu_task_seconds": seconds if resources["gpu_count"] else 0},
        gpus={"uuids": [gpu_uuid] if resources["gpu_count"] else [], "safety_margin_mib": 1024, "max_tasks_per_gpu": 1})
    atomic_json(folder / "harness-plan.json", plan)
    return plan


def _execute_plan(root, folder, plan):
    C, _, H = runtime()
    result = H.run_harness(root, plan, authorizer=lambda scope: scope.get("plan_digest") == plan["plan_digest"])
    task_file = _path(root, plan["output_root"]) / plan["batch_id"] / "tasks/job/result.json"
    if not task_file.exists():
        _fail("HARNESS_RECONCILIATION_REQUIRED", result.get("status"))
    task = read_json(task_file)
    H._verify_result(Path(root), plan["tasks"][0], task)
    atomic_json(folder / "harness-result.json", {"result": result, "task_ref": ref(root, task_file)})
    return task


def _engineering(root, folder, command, *, inputs=(), outputs=(), seconds=180,
                 scope="engineering-preparation", budget_path=None, code=None, reservation_id=None):
    C, R, _ = runtime()
    root, folder = Path(root).resolve(), _path(root, folder)
    run_id = "prep-" + uuid.uuid4().hex[:20]
    if budget_path and reservation_id is None:
        admission.reserve_preparation(budget_path, preparation_id=run_id, bound_seconds=seconds, purpose=scope)
    env = current_environment()
    folder.mkdir(parents=True, exist_ok=True)
    atomic_json(folder / "environment.json", env)
    native = R.make_plan(root, run_id=run_id, jobs=[{"trial_id": "job", "command": list(command), "cwd": ".",
        "input_refs": _unique_refs(inputs), "code_refs": source_refs(root) if code is None else code,
        "output_paths": list(outputs), "seed": 0, "group": "engineering", "arm_role": "engineering"}],
        provenance={"git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
            "model_revision": "no-model-workload", "data_revision": object_hash(list(inputs)),
            "environment_digest": object_hash(env), "environment_refs": [ref(root, folder / "environment.json")],
            "admission_scope": scope},
        limits={"max_attempts": 1, "max_development_trials": 1, "max_confirmation_trials": 0,
                "max_retries_per_trial": 0, "wall_time_seconds": seconds, "attempt_timeout_seconds": seconds})
    plan = _harness_plan(root, folder, native, resources=_resources({}, gpu=False),
                         hard_seconds=_time_left(budget_path) if budget_path else None)
    task = _execute_plan(root, folder, plan)
    if budget_path and task.get("receipt_ref"):
        admission.charge_budget(root, budget_path, reservation_id=reservation_id or run_id, receipt_ref=task["receipt_ref"])
    return task


def _live_replay(root, directory, binding, admission_record, *, state=None):
    """Return N's nonce-bound callback, with the actual harness owning each replay."""
    def replay(request):
        if admission_record["remaining_replay_calls"] <= 0:
            _fail("FROZEN_REPLAY_CALL_LIMIT_REACHED")
        seconds = min(binding["replay_timeout_seconds"], _time_left(directory))
        folder = directory / "replays" / uuid.uuid4().hex
        folder.mkdir(parents=True)
        request_path = folder / "request.json"
        result_path = folder / "execution.json"
        atomic_json(request_path, request)
        reservation_id = None
        if admission_record.get("replay_reservations"):
            index = admission_record["next_replay_index"]
            reservation_id = admission_record["replay_reservations"][index]
            admission_record["next_replay_index"] += 1
        admission_record["remaining_replay_calls"] -= 1
        if state is not None:
            atomic_json(directory / "state.json", state)
        command = [sys.executable, "-m", "recursive_ssd.suite_queue", "_native-replay", "--root", str(root),
                   "--request", str(request_path), "--result", str(result_path), "--seconds", str(seconds)]
        extra = []
        for item in request["input_refs"]:
            path = verify_ref(root, item)
            if path.suffix == ".json" and path.stat().st_size < 16 * 1024 * 1024:
                payload = read_json(path)
                if isinstance(payload, dict) and payload.get("version") == "native-labels-shards-v1":
                    extra.extend(ref(root, verify_ref(root, value)) for value in _nested_refs(payload))
        task = _engineering(root, folder, command,
            inputs=_unique_refs([ref(root, request_path), *request["input_refs"], *request["scorer"]["code_refs"], *extra]),
            outputs=["results/replay-pointer.json"], seconds=seconds, scope="native-reference-replay",
            budget_path=authorization_path(root, directory), reservation_id=reservation_id)
        if task["status"] != "completed" or not result_path.exists():
            _fail("LIVE_NATIVE_REPLAY_FAILED")
        return read_json(result_path)
    return replay


def _completed_output(root, state, identifier):
    record = state["nodes"][identifier]
    if record["status"] != "completed" or not record.get("receipt_ref") or not record.get("workload_receipt_ref"):
        _fail("DEPENDENCY_RECEIPT_REQUIRED", identifier)
    native = read_json(verify_ref(root, record["receipt_ref"]))
    if native["status"] != "completed" or not native.get("attempts"):
        _fail("DEPENDENCY_NATIVE_ATTEMPT_REQUIRED", identifier)
    attempts = [a for a in record.get("attempts", []) if a.get("receipt_ref") == record["receipt_ref"]]
    if len(attempts) != 1 or not attempts[0].get("native_plan_ref"):
        _fail("DEPENDENCY_NATIVE_PLAN_REQUIRED", identifier)
    plan = read_json(verify_ref(root, attempts[0]["native_plan_ref"]))
    if (native["plan_digest"] != plan["plan_digest"] or native["run_id"] != plan["run_id"]
            or native.get("provenance", {}).get("node_id") != identifier):
        _fail("DEPENDENCY_NATIVE_IDENTITY_MISMATCH", identifier)
    if record["workload_receipt_ref"] not in [v for attempt in native["attempts"] for v in attempt["output_refs"]]:
        _fail("DEPENDENCY_OUTPUT_NOT_NATIVE_BOUND", identifier)
    for attempt in native["attempts"]:
        retained = read_json(_path(root, attempt["attempt_path"]) / "attempt.json")
        if retained != attempt:
            _fail("DEPENDENCY_ATTEMPT_CHANGED", identifier)
        for key in ("stdout_ref", "stderr_ref"):
            verify_ref(root, attempt[key])
    for item in record.get("output_refs", []):
        verify_ref(root, item)
    path = verify_ref(root, record["workload_receipt_ref"])
    receipt = read_json(path)
    if receipt.get("status") not in {"complete", "completed"}:
        _fail("DEPENDENCY_WORKLOAD_NOT_COMPLETED", identifier)
    return receipt, path.parent


def _output_ref(root, receipt, folder, key):
    value = receipt.get(key)
    if not isinstance(value, dict):
        _fail("DEPENDENCY_OUTPUT_REQUIRED", key)
    path = _path(root, folder / value["path"])
    bound = ref(root, path)
    if bound["sha256"] != value["sha256"]:
        _fail("DEPENDENCY_OUTPUT_DIGEST_MISMATCH", key)
    return bound


def validate_frozen_head_calibration(root, record, source_suite, *, round_index):
    """Verify the measured development origin, including its actual native attempt."""
    expected = {"stage": "development", "seed": 17, "round": round_index,
                **{key: source_suite[key] for key in ("model", "model_revision", "training_lineage")}}
    if any(record.get(key) != value for key, value in expected.items()):
        _fail("HEAD_CALIBRATION_IDENTITY_MISMATCH")
    origin = record.get("derivation")
    if not isinstance(origin, dict) or set(origin) != {"queue_ref", "state_ref", "node_id"}:
        _fail("HEAD_CALIBRATION_NATIVE_DERIVATION_REQUIRED")
    queue = read_json(verify_ref(root, origin["queue_ref"]))
    state = read_json(verify_ref(root, origin["state_ref"]))
    if queue.get("queue_digest") != _queue_digest(queue) or state.get("queue_digest") != queue["queue_digest"]:
        _fail("HEAD_CALIBRATION_SOURCE_QUEUE_CHANGED")
    development = verify_suite(read_json(verify_ref(root, queue["identity"]["suite_ref"])))
    if (development["stage"] != "development" or queue["suite_digest"] != development["suite_digest"]
            or any(development[key] != source_suite[key] for key in ("model", "model_revision", "training_lineage"))
            or source_suite.get("suite_digest", development["suite_digest"]) != development["suite_digest"]):
        _fail("HEAD_CALIBRATION_SOURCE_DEVELOPMENT_MISMATCH")
    node = next((n for n in queue["nodes"] if n["node_id"] == origin["node_id"]), None)
    if not node or node.get("calibration_key") != "head_temperature" or node["round"] != round_index:
        _fail("HEAD_CALIBRATION_SOURCE_NODE_MISMATCH")
    receipt, folder = _completed_output(root, state, node["node_id"])
    native = read_json(verify_ref(root, state["nodes"][node["node_id"]]["receipt_ref"]))
    if native.get("purpose") != "scientific":
        _fail("HEAD_CALIBRATION_NATIVE_DERIVATION_REQUIRED")
    calibration = read_json(verify_ref(root, _output_ref(root, receipt, folder, "calibration_ref")))
    raw = calibration.get("head_temperature", {}).get("rounds", {}).get(str(round_index))
    raw = _rebase_paths(raw, queue["identity"]["project_root"], root)
    if raw != {key: value for key, value in record.items() if key != "derivation"}:
        _fail("HEAD_CALIBRATION_MEASURED_VALUE_CHANGED")
    parent, _ = _completed_output(root, state, node["source_node_id"])
    if (parent.get("arm_id") != "M06" or any(parent.get(key) != value for key, value in expected.items())):
        _fail("HEAD_CALIBRATION_SOURCE_PARENT_MISMATCH")
    hashes = {item["sha256"] for item in record.get("source_refs", [])}
    if not {state["nodes"][node["source_node_id"]]["workload_receipt_ref"]["sha256"],
            queue["identity"]["model_manifest_ref"]["sha256"]}.issubset(hashes):
        _fail("HEAD_CALIBRATION_PARENT_MODEL_PROVENANCE_REQUIRED")
    for item in _nested_refs(record):
        verify_ref(root, item)
    return record


def materialize_job(root, queue, state, suite, node, bindings):
    """Bind an executable child only to present, verified parent bytes."""
    trajectory = deepcopy(next(t for t in suite["trajectories"] if t["trajectory_id"] == node["trajectory_id"]))
    job = {"kind": node["kind"], "node_id": node["node_id"], "suite_digest": suite["suite_digest"],
           "stage": suite["stage"],
           "project_root": str(Path(root).resolve()),
           "model_manifest": str(verify_ref(root, queue["identity"]["model_manifest_ref"]))}
    inputs = []
    parents = []
    for parent in node["depends_on"]:
        receipt, folder = _completed_output(root, state, parent)
        parents.append((parent, receipt, folder))
        inputs += [state["nodes"][parent]["receipt_ref"], state["nodes"][parent]["workload_receipt_ref"],
                   *state["nodes"][parent].get("output_refs", [])]
    calibration = effective_calibration(suite, bindings)
    if ("head_temperature" in trajectory["arm"].get("required_calibration", [])
            and suite["stage"] not in {"development", "tuning"}):
        frozen = calibration.get("head_temperature", {})
        frozen = frozen.get("rounds", {}).get(str(node["round"]), frozen)
        inherited = suite.get("development_binding")
        if not inherited:
            _fail("HEAD_TEMPERATURE_REQUIRES_FROZEN_DEVELOPMENT_CALIBRATION")
        source_suite = verify_suite(read_json(verify_ref(root, inherited["suite_ref"])))
        policy = suite.get("calibration_policy", {}).get("head_temperature", {})
        source_round = policy.get("round_source_map", {}).get(str(node["round"]), node["round"])
        validate_frozen_head_calibration(root, frozen, source_suite, round_index=source_round)
    if node["kind"] == "derive-calibration":
        _, receipt, folder = parents[0]
        source_ref = state["nodes"][node["source_node_id"]]["workload_receipt_ref"]
        request = {"key": node["calibration_key"], "round": node["round"],
                   "source_refs": [{"path": str(verify_ref(root, source_ref)), "sha256": source_ref["sha256"]}]}
        if node["calibration_key"] == "head_temperature":
            request["records_ref"] = _output_ref(root, receipt, folder, "records_ref")
            request["records_ref"]["path"] = str(verify_ref(root, request["records_ref"]))
            request["checkpoint_ref"] = None
            source_trajectory = next(t for t in suite["trajectories"] if t["trajectory_id"] == receipt["trajectory_id"])
            if node["round"] > 1:
                predecessor = receipt["trajectory_id"] + "-train-r" + str(node["round"] - 1)
                prior, prior_folder = _completed_output(root, state, predecessor)
                checkpoint = _output_ref(root, prior, prior_folder, "checkpoint_ref")
                inputs += [checkpoint, state["nodes"][predecessor]["workload_receipt_ref"]]
                request["checkpoint_ref"] = {"path": str(verify_ref(root, checkpoint)), "sha256": checkpoint["sha256"]}
            manifest_path = verify_ref(root, queue["identity"]["model_manifest_ref"])
            request["source_refs"].append({"path": str(manifest_path), "sha256": digest(manifest_path)})
            request["model_path"] = str(manifest_path.parent / read_json(manifest_path)["snapshot_path"])
            config = read_json(Path(root) / "configs/2080ti_8h.json")
            config.update(bindings.get("config", {}))
            request.update(config=config, decode=source_trajectory["arm"]["decode"])
        job.update(stage=suite["stage"], seed=trajectory["seed"], training_lineage="clean-v2",
                   model=trajectory["model"], model_revision=trajectory["model_revision"], requests=[request])
        return job, _unique_refs(inputs)
    for identifier in node.get("calibration_nodes", []):
        receipt, _ = _completed_output(root, state, identifier)
        values = read_json(verify_ref(root, _output_ref(root, receipt, _completed_output(root, state, identifier)[1], "calibration_ref")))
        calibration.update(values)
    if node["kind"] == "train":
        resolve_arm(trajectory["arm"], calibration, node["round"])
        job.update(trajectory=trajectory, round=node["round"], calibration=calibration,
                   config=deepcopy(bindings.get("config", {})), parent_receipts=[])
        config = read_json(Path(root) / "configs/2080ti_8h.json")
        config.update(job["config"])
        budget_fields = ("train_prompts", "samples_per_prompt", "grad_accum", "epochs", "max_prompt_tokens",
                         "max_new_tokens", "learning_rate", "lora_rank", "max_grad_norm")
        job["training_budget_digest"] = object_hash({"config": {k: config[k] for k in budget_fields},
            "decode": trajectory["arm"]["decode"], "train": trajectory["arm"]["train"],
            "rounds": trajectory["rounds"], "seed": trajectory["seed"], "lineage": trajectory["training_lineage"],
            "tuning_budget": bindings.get("tuning_budget")})
        for field, target, key in (("parent_node_id", "parent_checkpoint", "checkpoint_ref"),
                                    ("lag_node_id", "lag_checkpoint", "checkpoint_ref"),
                                    ("fixed_node_id", "fixed_records", "records_ref")):
            if node.get(field):
                receipt, folder = _completed_output(root, state, node[field])
                value = _output_ref(root, receipt, folder, key)
                inputs += [value, state["nodes"][node[field]]["workload_receipt_ref"]]
                job[target] = str(verify_ref(root, value))
            else:
                job[target] = None
    else:
        job.update(unit=deepcopy(node["unit"]), checkpoint=None, training_receipts=[])
        if node["parent_node_id"]:
            receipt, folder = _completed_output(root, state, node["parent_node_id"])
            value = _output_ref(root, receipt, folder, "checkpoint_ref")
            inputs.append(value)
            job["checkpoint"] = str(verify_ref(root, value))
    for round_ in range(1, node["round"] + (0 if node["kind"] == "train" else 1)):
        identifier = node["trajectory_id"] + "-train-r" + str(round_)
        receipt, folder = _completed_output(root, state, identifier)
        value = state["nodes"][identifier]["workload_receipt_ref"]
        inputs += [value, state["nodes"][identifier]["receipt_ref"], *state["nodes"][identifier].get("output_refs", [])]
        job["parent_receipts" if node["kind"] == "train" else "training_receipts"].append(str(verify_ref(root, value)))
    if state["nodes"][node["node_id"]].get("resume_from"):
        resume = _path(root, state["nodes"][node["node_id"]]["resume_from"])
        job["resume_from"] = str(resume)
        inputs += [ref(root, p) for p in resume.rglob("*") if p.is_file() and not p.is_symlink()]
    for value in _nested_refs(calibration):
        inputs.append(ref(root, verify_ref(root, value)))
    return job, _unique_refs(inputs)


def _reconcile(root, directory, queue, state, bundle, bindings, *, dispatch=True):
    record = state["admissions"][bundle]
    for node in queue["nodes"]:
        entry = state["nodes"][node["node_id"]]
        if entry["status"] != "running":
            continue
        attempt = entry["attempts"][-1]
        folder = _path(root, attempt["controller_path"])
        plan = read_json(verify_ref(root, attempt["harness_plan_ref"]))
        # Resuming a live harness joins the exact existing batch; it never creates a retry.
        result_path = _path(root, plan["output_root"]) / plan["batch_id"] / "tasks/job/result.json"
        if dispatch:
            task = _execute_plan(root, folder, plan)
        elif result_path.exists():
            _, _, H = runtime()
            task = read_json(result_path)
            H._verify_result(root, plan["tasks"][0], task)
        else:
            continue
        if not task.get("receipt_ref"):
            _fail("HARNESS_RECONCILIATION_REQUIRED", node["node_id"])
        receipt = read_json(verify_ref(root, task["receipt_ref"]))
        entry.update(receipt_ref=task["receipt_ref"], status="completed" if task["status"] == "completed" else "failed")
        attempt.update(receipt_ref=task["receipt_ref"], status=entry["status"])
        outputs = []
        for native_attempt in receipt["attempts"]:
            result_folder = _path(root, native_attempt["attempt_path"]) / "workspace/results"
            if result_folder.exists():
                outputs += [ref(root, path) for path in result_folder.rglob("*") if path.is_file() and not path.is_symlink()]
                entry["resume_from"] = str(result_folder)
            for value in native_attempt["output_refs"]:
                if value["path"].endswith("/results/receipt.json"):
                    entry["workload_receipt_ref"] = value
        entry["output_refs"] = _unique_refs(outputs)
        if entry["status"] == "completed":
            _completed_output(root, state, node["node_id"])
        if node["node_id"] in record["cell_ids"]:
            admission.charge_budget(root, authorization_path(root, directory),
                reservation_id=entry.get("retry_reservation_id", record["reservation_id"]),
                receipt_ref=task["receipt_ref"], cell_id=node["node_id"], final=False)
        atomic_json(directory / "state.json", state)
    return state


def _reserve_failed_retries(root, directory, queue, state, suite, bundle, bindings):
    if suite["stage"] not in {"development", "tuning"}:
        _fail("CONFIRMATION_RETRIES_FORBIDDEN")
    admitted = state["admissions"][bundle]
    nodes = [n for n in queue["nodes"] if bundle in n["bundles"] and state["nodes"][n["node_id"]]["status"] == "failed"]
    costs = read_json(verify_ref(root, bindings["calibration_ref"]))
    expected = {"environment_digest": queue["identity"]["environment_digest"], "model_revision": suite["model_revision"],
        "code_digest": object_hash(queue["identity"]["code_refs"]), "benchmark_manifest_ref": queue["identity"]["benchmark_manifest_ref"]}
    for node in nodes:
        entry = state["nodes"][node["node_id"]]
        if len(entry["attempts"]) >= queue["retry_policy"]["development_max_attempts_per_node"]:
            _fail("FINITE_DEVELOPMENT_RETRY_LIMIT", node["node_id"])
        replay = bindings["replay_calls_per_node"] * bindings["replay_timeout_seconds"]
        if costs["costs"][node["node_id"]]["upper_seconds"] + replay + admitted["finalization_reserve_seconds"] > admission.remaining_budget(authorization_path(root, directory)):
            _fail("COMPLETE_RETRY_WITH_PENDING_BUNDLE_EXCEEDS_REMAINING_BUDGET")
        identifier = admitted["reservation_id"] + "-retry-" + object_hash(node["node_id"])[:10] + "-" + str(len(entry["attempts"]))
        admission.reserve_bundle(root, authorization_path(root, directory), bundle_id=identifier, cell_ids=[node["node_id"]],
                                 calibration_ref=bindings["calibration_ref"], expected_bindings=expected)
        for index in range(bindings["replay_calls_per_node"]):
            replay_id = identifier + "-replay-" + str(index)
            admission.reserve_preparation(authorization_path(root, directory), preparation_id=replay_id,
                bound_seconds=bindings["replay_timeout_seconds"], purpose="native-reference-replay")
            admitted["replay_reservations"].append(replay_id)
            admitted["remaining_replay_calls"] += 1
        entry.update(status="pending", retry_reservation_id=identifier)
    atomic_json(directory / "state.json", state)


def _trajectory_paid_seconds(root, queue, state, trajectory_id):
    seconds = 0.
    for node in queue["nodes"]:
        if node["kind"] != "train" or node["trajectory_id"] != trajectory_id:
            continue
        for attempt in state["nodes"][node["node_id"]]["attempts"]:
            if not attempt.get("receipt_ref"):
                _fail("TUNING_ATTEMPT_COST_RECONCILIATION_REQUIRED")
            native = read_json(verify_ref(root, attempt["receipt_ref"]))
            seconds += native["resources"]["seconds"]
    return seconds


def run_queue(root, directory, *, bundle, max_nodes=None, retry_failed=False):
    root, directory = Path(root).resolve(), _path(root, directory)
    _conda()
    with _lock(directory):
        queue, state, suite = verify_queue(root, directory)
        if bundle not in state["admissions"]:
            _fail("COMPLETE_BUNDLE_NOT_ADMITTED", bundle)
        admitted = state["admissions"][bundle]
        bindings = read_json(verify_ref(root, admitted["bindings_ref"]))
        state = _reconcile(root, directory, queue, state, bundle, bindings, dispatch=False)
        _time_left(directory)
        if retry_failed:
            _reserve_failed_retries(root, directory, queue, state, suite, bundle, bindings)
        state = _reconcile(root, directory, queue, state, bundle, bindings)
        completed = 0
        while max_nodes is None or completed < max_nodes:
            left = _time_left(directory) - admitted["finalization_reserve_seconds"]
            if left <= 0:
                break
            available = ready_nodes(queue["nodes"], state["nodes"], bundle=bundle)
            if not available:
                break
            node = available[0]
            costs = read_json(verify_ref(root, bindings["calibration_ref"]))
            seconds = float(costs["costs"][node["node_id"]]["upper_seconds"])
            if suite["stage"] == "tuning" and node["kind"] == "train":
                allowance = bindings["tuning_budget"]["trial_wall_seconds"]
                remaining_trial = allowance - _trajectory_paid_seconds(root, queue, state, node["trajectory_id"])
                if remaining_trial <= 15:
                    state["nodes"][node["node_id"]].update(status="blocked", failure="FROZEN_TUNING_TRIAL_ALLOWANCE_EXHAUSTED")
                    atomic_json(directory / "state.json", state)
                    break
                seconds = min(seconds, remaining_trial)
            if seconds + bindings["replay_timeout_seconds"] * bindings["replay_calls_per_node"] > left:
                break
            job, parent_inputs = materialize_job(root, queue, state, suite, node, bindings)
            folder = directory / "nodes" / node["node_id"] / ("attempt-" + str(len(state["nodes"][node["node_id"]]["attempts"]) + 1))
            folder.mkdir(parents=True, exist_ok=False)
            group, role = _node_scope(node, bindings, suite)
            _, _, native_inputs = _native_inputs(root, bindings["protocol_ref"], group)
            identity = queue["identity"]
            inputs = _unique_refs([identity["benchmark_manifest_ref"],
                identity["model_manifest_ref"], *identity["data_refs"], *identity["model_refs"], *native_inputs, *parent_inputs])
            job["input_refs"] = inputs
            atomic_json(folder / "job.json", job)
            inputs = _unique_refs([ref(root, folder / "job.json"), *inputs])
            command = [sys.executable, "-m", "recursive_ssd.suite", "execute", "--job", str(folder / "job.json"),
                "--data", str(verify_ref(root, identity["benchmark_manifest_ref"]).parent.relative_to(root)),
                "--output", "results", "--seconds", str(seconds)]
            C, _, _ = runtime()
            try:
                native = admission.build_scientific_plan(root,
                    run_id="suite-" + queue["queue_digest"][:10] + "-" + uuid.uuid4().hex[:12], command=command,
                    code_refs=identity["code_refs"], input_refs=inputs, output_paths=["results/receipt.json"],
                    protocol_ref=bindings["protocol_ref"], verification_ref=bindings["verification_ref"],
                    seed=next(t["seed"] for t in suite["trajectories"] if t["trajectory_id"] == node["trajectory_id"]),
                    group=group, arm_role=role, seconds=seconds,
                    provenance={"git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
                        "model_revision": suite["model_revision"], "data_revision": identity["benchmark_manifest_ref"]["sha256"],
                        "environment_digest": identity["environment_digest"], "environment_refs": [ref(root, directory / "environment.json")],
                        "config_digest": object_hash(bindings.get("config", {})),
                        "calibration_digest": object_hash(effective_calibration(suite, bindings)),
                        "model_refs": identity["model_refs"], "candidate_id": bundle, "queue_digest": queue["queue_digest"],
                        "node_id": node["node_id"]},
                    replay_context=_live_replay(root, directory, bindings, admitted, state=state))
            finally:
                atomic_json(directory / "state.json", state)
            plan = _harness_plan(root, folder, native, resources=_resources(bindings, gpu=node["kind"] != "derive-calibration" or node.get("calibration_key") == "head_temperature"),
                                 gpu_uuid=bindings["gpu_uuid"], hard_seconds=_time_left(directory))
            entry = state["nodes"][node["node_id"]]
            entry["status"] = "running"
            entry["attempts"].append({"controller_path": str(folder.relative_to(root)), "job_ref": ref(root, folder / "job.json"),
                "harness_plan_ref": ref(root, folder / "harness-plan.json"), "native_plan_ref": ref(root, folder / "native-plan.json"),
                "status": "running"})
            atomic_json(directory / "state.json", state)
            state = _reconcile(root, directory, queue, state, bundle, bindings)
            completed += 1
            if state["nodes"][node["node_id"]]["status"] == "failed":
                break
        return {**inventory_status(queue["nodes"], state["nodes"]), "queue_digest": queue["queue_digest"],
                "remaining_unreserved_seconds": admission.remaining_budget(authorization_path(root, directory)), "gate_advanced": False}


def select_finalists(suite, report, *, limit=2):
    """Pure development ranking; the CLI additionally verifies native report receipts."""
    verify_suite(suite)
    if suite["stage"] not in {"development", "tuning"}:
        _fail("SELECTION_DEVELOPMENT_ONLY")
    rows = report.get("observations", [])
    expected = {u["cell_id"]: u for u in suite["evaluation_units"]}
    if (report.get("suite_digest") != suite["suite_digest"] or report.get("native_evidence_validated") is not True
            or report.get("inventory", {}).get("complete") is not True or len(rows) != len(expected)
            or {r.get("cell_id") for r in rows} != set(expected)):
        _fail("COMPLETE_NATIVE_DEVELOPMENT_REPORT_REQUIRED")
    endpoint = {}
    from .evaluation import pass_at_k
    for row in rows:
        unit = expected[row["cell_id"]]
        if any(row.get(k) != unit[k] for k in ("arm_id", "model", "benchmark", "split", "seed", "round")):
            _fail("COMPLETE_NATIVE_DEVELOPMENT_IDENTITY_REQUIRED")
        if row.get("status") != "completed" or row.get("optimization_valid") is not True or not row.get("per_task"):
            _fail("COMPLETE_NATIVE_DEVELOPMENT_VALIDITY_REQUIRED")
        if row["benchmark"] != "humaneval" or row["split"] != "dev" or row["seed"] != 17:
            _fail("SELECTION_DEVELOPMENT_ONLY")
        if row["round"] not in {0, suite["rounds"]}:
            continue
        counts = row["per_task"]
        if any(v.get("n") != 10 or type(v.get("correct", v.get("plus_correct"))) is not int for v in counts.values()):
            _fail("COMPLETE_NATIVE_DEVELOPMENT_COUNTS_REQUIRED")
        def score(k):
            return sum(pass_at_k(v["n"], v.get("correct", v.get("plus_correct")), k) for v in counts.values()) / len(counts)
        endpoint[row["arm_id"]] = {"pass@1": score(1), "pass@10": score(10), "training_seconds": row["training_seconds"]}
    if suite["stage"] == "tuning":
        selections = {}
        for family, arms in suite["tuning_groups"].items():
            ranked = sorted(arms, key=lambda a: (-endpoint[a]["pass@10"], -endpoint[a]["pass@1"], endpoint[a]["training_seconds"], a))
            selections[family] = ranked[0]
        settings = {t["arm_id"]: t["arm"] for t in suite["trajectories"] if t["arm_id"] in selections.values()}
        return {"schema": "recursive-ssd-development-selection-v1", "suite_digest": suite["suite_digest"],
                "selection_scope": "development_only; no scientific verdict", "hyperparameters": selections,
                "arm_settings": settings, "tuning_parameters": {"selected_values": {
                    family: settings[arm]["parameters"] for family, arm in selections.items()}, "source_refs": []}}
    ranked = []
    for candidate in suite["candidate_ids"]:
        controls = [a for a in suite["method_bundles"][candidate]["arms"] if a != candidate and not a.startswith("M")]
        strongest = max(controls, key=lambda a: (endpoint[a]["pass@10"], endpoint[a]["pass@1"], a))
        score, control = endpoint[candidate], endpoint[strongest]
        delta = score["pass@10"] - control["pass@10"]
        p1 = score["pass@1"] - control["pass@1"]
        ranked.append({"candidate_id": candidate, "strongest_simple_baseline": strongest,
            "pass10_delta": delta, "pass1_delta": p1, "eligible": p1 >= -.01,
            "training_seconds": score["training_seconds"]})
    ranked.sort(key=lambda row: (-row["pass10_delta"], row["training_seconds"], ORDER.index(row["candidate_id"])))
    return {"schema": "recursive-ssd-development-selection-v1", "suite_digest": suite["suite_digest"],
            "selection_scope": "development_only; no scientific verdict", "ranked": ranked,
            "finalists": [r["candidate_id"] for r in ranked if r["eligible"]][:min(2, limit)]}


def queue_report(root, directory, *, evidence_context=None):
    from .analysis import comparison_report, g01_criteria, validate_inventory
    root, directory = Path(root).resolve(), _path(root, directory)
    queue, state, suite = verify_queue(root, directory)
    manifest = read_json(verify_ref(root, queue["identity"]["benchmark_manifest_ref"]))
    expected = expected_cells(suite, manifest)
    observations, retained = [], []
    native_valid = True
    for node in queue["nodes"]:
        if node["kind"] != "evaluate":
            continue
        entry = state["nodes"][node["node_id"]]
        if entry["status"] == "completed":
            receipt, folder = _completed_output(root, state, node["node_id"])
            observation = receipt.get("observation")
            if not isinstance(observation, dict):
                _fail("NATIVE_OBSERVATION_REQUIRED", node["node_id"])
            observation = _rebase_paths(deepcopy(observation), queue["identity"]["project_root"], root)
            observation["source_refs"] = [entry["receipt_ref"], entry["workload_receipt_ref"]]
            if suite["stage"] == "tuning" and node["round"]:
                admitted = next(state["admissions"][b] for b in node["bundles"] if b in state["admissions"])
                binding = read_json(verify_ref(root, admitted["bindings_ref"]))
                cap = binding["tuning_budget"]["trial_wall_seconds"]
                paid = _trajectory_paid_seconds(root, queue, state, node["trajectory_id"])
                observation.update(trial_wall_seconds=paid, trial_wall_allowance_seconds=cap)
                if paid > cap:
                    observation["optimization_valid"] = False
                    observation["failure"] = "FROZEN_TUNING_TRIAL_ALLOWANCE_EXCEEDED"
            observations.append(observation)
            retained.extend([entry["receipt_ref"], entry["workload_receipt_ref"], *entry["output_refs"]])
            native = read_json(verify_ref(root, entry["receipt_ref"]))
            native_valid &= native.get("purpose") == "scientific" and native["status"] == "completed"
        else:
            observations.append({**node["unit"], "status": entry["status"], "failure": entry.get("failure"),
                                 "attempts": entry["attempts"]})
            native_valid = False
    inventory = validate_inventory(expected, observations)
    report = {"schema": "recursive-ssd-queue-report-v3", "queue_digest": queue["queue_digest"],
        "queue_ref": ref(root, directory / "queue.json"), "suite_digest": suite["suite_digest"],
        "suite_ref": queue["identity"]["suite_ref"], "stage": suite["stage"], "expected_cells": expected,
        "observations": observations, "inventory": inventory,
        "native_evidence_validated": bool(native_valid and inventory["complete"]),
        "evidence_refs": _unique_refs(retained), "node_inventory": inventory_status(queue["nodes"], state["nodes"]),
        "budget": read_json(authorization_path(root, directory)), "analyses": {}, "decision": "INCONCLUSIVE", "gate_advanced": False}
    context = _rebase_paths(evidence_context or {}, queue["identity"]["project_root"], root)
    for candidate in suite["candidate_ids"] if suite["comparisons"] else []:
        comparisons = [c for c in suite["comparisons"] if c["candidate_id"] == candidate]
        candidate_context = dict(context.get(candidate, {}))
        if suite["stage"] in {"boundary", "model_boundary"}:
            candidate_context["phase"] = suite["stage"]
        report["analyses"][candidate] = comparison_report(expected, observations, comparisons,
            family_size=suite["analysis"]["family_size"], criteria=g01_criteria(comparisons, method_id=candidate),
            evidence_context=candidate_context, repeats=suite["analysis"]["bootstrap_repeats"],
            seed=suite["analysis"]["bootstrap_seed"], alpha=suite["analysis"]["alpha"])
    report["calibration_policy"] = deepcopy(suite.get("calibration_policy", {}))
    report = _rebase_paths(report, root, queue["identity"]["project_root"])
    atomic_json(directory / "report.json", report)
    lines = ["# Multi-benchmark experiment report", "", "Scientific status: INCONCLUSIVE until complete reviewed evidence.", "",
             f"Stage: {suite['stage']}; complete native inventory: {inventory['complete']}.", "",
             "| Bundle | Status | Expected nodes | Completed | Failed |", "|---|---|---:|---:|---:|"]
    for name, value in report["node_inventory"]["bundles"].items():
        lines.append(f"| {name} | {value['status']} | {value['expected']} | {value.get('completed', 0)} | {value.get('failed', 0)} |")
    lines += ["", "Full native counts, missing cells, attempt lineage, cost and fixed-family analyses are in report.json.",
              "No engineering check is a scientific result; no gate is advanced by this report."]
    (directory / "REPORT.md").write_text("\n".join(lines) + "\n")
    return report


def _selection_settings(root, directory, queue, state, suite):
    """Retain actual development settings, including completed head diagnostics."""
    config, calibration = {}, deepcopy(suite.get("calibration", {}))
    for admitted in state["admissions"].values():
        binding = read_json(verify_ref(root, admitted["bindings_ref"]))
        validate_shared_configuration(root, state, binding)
        config = deepcopy(binding.get("config", {}))
        calibration.update(effective_calibration(suite, binding))
    # Clocks and step matching are measured again from the corresponding seed.
    calibration = {key: value for key, value in calibration.items() if key in {"arithmetic_floor", "head_temperature"}}
    for node in queue["nodes"]:
        if node.get("calibration_key") != "head_temperature":
            continue
        receipt, folder = _completed_output(root, state, node["node_id"])
        values = read_json(verify_ref(root, _output_ref(root, receipt, folder, "calibration_ref")))
        record = _rebase_paths(values["head_temperature"]["rounds"][str(node["round"])],
                               queue["identity"]["project_root"], root)
        record["derivation"] = {"queue_ref": ref(root, directory / "queue.json"),
                                "state_ref": ref(root, directory / "state.json"), "node_id": node["node_id"]}
        validate_frozen_head_calibration(root, record, suite, round_index=node["round"])
        calibration.setdefault("head_temperature", {"rounds": {}})["rounds"][str(node["round"])] = record
    return config, _rebase_paths(calibration, root, queue["identity"]["project_root"])


def compile_selected_suite(root, stage, selected, *, tuning_override=None):
    """Compile from an already native-validated selection without retuning it."""
    if stage not in {"confirmation", "boundary", "model_boundary"}:
        _fail("SELECTED_SUITE_STAGE_REQUIRED")
    queue = read_json(verify_ref(root, selected["queue_ref"]))
    if queue.get("queue_digest") != _queue_digest(queue):
        _fail("SELECTED_QUEUE_IDENTITY_CHANGED")
    source_ref = queue["identity"]["suite_ref"]
    source = verify_suite(read_json(verify_ref(root, source_ref)))
    if (source["stage"] != "development" or selected.get("suite_digest") != source["suite_digest"]
            or queue["suite_digest"] != source["suite_digest"]):
        _fail("COMPLETE_NATIVE_DEVELOPMENT_SELECTION_REQUIRED")
    tuning = deepcopy(source.get("tuning_selection"))
    if tuning_override is not None and tuning_override != tuning:
        _fail("SELECTED_DEVELOPMENT_SETTINGS_CHANGED", "tuning selection")
    calibration = deepcopy(selected.get("frozen_calibration", source.get("calibration", {})))
    inherited_config = deepcopy(selected.get("frozen_config", {}))
    for item in _nested_refs({"tuning": tuning, "calibration": calibration}):
        verify_ref(root, item)
    policy = {}
    if "head_temperature" in calibration:
        records = calibration["head_temperature"].get("rounds", {})
        for round_ in (1, 2, 3):
            validate_frozen_head_calibration(root, records.get(str(round_), {}), source, round_index=round_)
        mapping = {str(round_): round_ for round_ in (1, 2, 3)}
        if stage == "boundary":
            for round_ in (4, 5):
                records[str(round_)] = deepcopy(records["3"])
                mapping[str(round_)] = 3
        policy["head_temperature"] = {"origin_stage": "development", "origin_model": source["model"],
            "origin_model_revision": source["model_revision"], "round_source_map": mapping,
            "same_model_distribution_match": stage != "model_boundary",
            "scope": "frozen development-model parameter transfer; no same-model matched-head claim" if stage == "model_boundary"
                     else "development R1-R3 frozen; R4-R5 prospectively carry forward R3" if stage == "boundary"
                     else "same-model development calibration frozen before independent confirmation"}
    suite = make_suite(stage, finalists=selected["finalists"], model=source["model"],
                       calibration=calibration, tuning_selection=tuning)
    suite["development_binding"] = {"queue_ref": selected["queue_ref"], "suite_ref": source_ref,
                                     "config": inherited_config}
    suite["calibration_policy"] = policy
    suite["suite_digest"] = object_hash({key: value for key, value in suite.items() if key != "suite_digest"})
    return verify_suite(suite)


def verified_selection(root, directory, output):
    directory = _path(root, directory)
    queue, state, suite = verify_queue(root, directory)
    report = queue_report(root, directory)
    selected = select_finalists(suite, report)
    output = _path(root, output)
    frozen_report = output.with_name(output.stem + "-native-report.json")
    atomic_json(frozen_report, report)
    selected.update(report_ref=ref(root, frozen_report), queue_ref=ref(root, directory / "queue.json"),
                    evidence_refs=report["evidence_refs"])
    if suite["stage"] == "development":
        selected["frozen_config"], selected["frozen_calibration"] = _selection_settings(root, directory, queue, state, suite)
    if suite["stage"] == "tuning":
        rows = {row["arm_id"]: row for row in report["observations"] if row["round"] == suite["rounds"]}
        trials = []
        for family in ("M03", "arithmetic_anchor"):
            for arm in suite["tuning_groups"][family]:
                row = rows[arm]
                budget = row.get("training_budget_digest")
                if not isinstance(budget, str) or len(budget) != 64:
                    _fail("MEASURED_EQUAL_TUNING_BUDGET_REQUIRED", arm)
                spec = next(t["arm"] for t in suite["trajectories"] if t["arm_id"] == arm)
                trials.append({"family": family, "arm_id": arm, "parameters": spec["parameters"],
                    "status": "completed", "optimization_valid": row["optimization_valid"],
                    "budget_digest": budget, "source_refs": row["source_refs"],
                    "trial_wall_seconds": row["trial_wall_seconds"],
                    "trial_wall_allowance_seconds": row["trial_wall_allowance_seconds"]})
        if len({trial["budget_digest"] for trial in trials}) != 1:
            _fail("UNEQUAL_TUNING_BUDGETS")
        chosen = selected["hyperparameters"]["arithmetic_anchor"]
        selected["tuning_selection"] = {"stage": "tuning", "seed": 17, "training_lineage": "clean-v2",
            "model": suite["model"], "model_revision": suite["model_revision"],
            "inventory": {"complete": True, "expected": 6, "observed": 6}, "trials": trials,
            "selected_control": {"arm_id": chosen, "parameters": selected["arm_settings"][chosen]["parameters"]}}
        selected["tuning_parameters"]["source_refs"] = [selected["report_ref"], *report["evidence_refs"]]
    atomic_json(output, selected)
    return selected


def validate_selection(root, path, *, tuning=False):
    """Verify the retained native inputs without rerunning statistical workloads."""
    selected = read_json(_path(root, path))
    queue_path = verify_ref(root, selected["queue_ref"])
    queue, state, suite = verify_queue(root, queue_path.parent)
    report = read_json(verify_ref(root, selected["report_ref"]))
    for item in selected.get("evidence_refs", []):
        verify_ref(root, item)
    for node in queue["nodes"]:
        if node["kind"] == "evaluate":
            _completed_output(root, state, node["node_id"])
    expected = select_finalists(suite, report)
    key = "hyperparameters" if tuning else "finalists"
    if key not in expected or selected.get(key) != expected[key]:
        _fail("DEVELOPMENT_SELECTION_IDENTITY_CHANGED")
    if tuning:
        packet = selected.get("tuning_parameters")
        if not isinstance(packet, dict) or packet.get("selected_values") != expected["tuning_parameters"]["selected_values"]:
            _fail("TUNING_PARAMETER_IDENTITY_CHANGED")
    else:
        config, calibration = _selection_settings(root, queue_path.parent, queue, state, suite)
        if selected.get("frozen_config") != config or selected.get("frozen_calibration") != calibration:
            _fail("SELECTED_DEVELOPMENT_SETTINGS_CHANGED", "frozen configuration/calibration")
    return selected


def collect_queue(root, directory, output, *, include_checkpoints=False):
    root, directory, output = Path(root).resolve(), _path(root, directory), _path(root, output)
    queue, state, _ = verify_queue(root, directory)
    files = {p for p in directory.rglob("*") if p.is_file() and not p.is_symlink() and p != output}
    files.add(authorization_path(root, directory))
    for value in _nested_refs({"queue": queue, "state": state}):
        files.add(verify_ref(root, value))
    for node in state["nodes"].values():
        for attempt in node["attempts"]:
            if attempt.get("receipt_ref"):
                native = read_json(verify_ref(root, attempt["receipt_ref"]))
                for item in native["attempts"]:
                    folder = _path(root, item["attempt_path"])
                    files.update(p for p in folder.rglob("*") if p.is_file() and not p.is_symlink() and "__pycache__" not in p.parts)
    weights = {".pt", ".bin", ".safetensors"}
    retained = sorted(p for p in files if include_checkpoints or p.suffix not in weights)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w:gz") as archive:
        for path in retained:
            archive.add(path, arcname=str(path.relative_to(root)), recursive=False)
    return {"bundle": str(output), "sha256": digest(output), "files": len(retained),
            "checkpoint_files_included": include_checkpoints, "scope": "actual retained artifacts; absent cells remain absent"}


def _native_replay_worker(args):
    C, R, _ = runtime()
    root = Path(args.root).resolve()
    request = read_json(args.request)
    callback = R.build_official_replay_context(root,
        lambda scope: scope.get("request_digest") == request["request_digest"] and scope.get("nonce") == request["nonce"],
        output_root=str(Path(args.result).parent.relative_to(root) / "native"), timeout_seconds=max(1, args.seconds - 5))
    result = callback(request)
    atomic_json(args.result, result)
    atomic_json("results/replay-pointer.json", {"execution_ref": ref(root, args.result)})


def _prepare_worker(spec, output):
    """Preparation worker, reached only through a staged native harness job."""
    kind, data = spec["kind"], Path(spec["data"])
    data.mkdir(parents=True, exist_ok=True)
    if kind == "assets":
        from .benchmarks import prepare_benchmarks, acquire_lcb_references
        prepare_benchmarks(data, include_lcb=True)
        acquire_lcb_references(data)
        files = [ref(spec["project_root"], p) for p in data.rglob("*") if p.is_file() and not p.is_symlink()]
        result = {"kind": kind, "status": "completed", "files": files,
                  "manifest_ref": ref(spec["project_root"], data / "benchmarks-manifest.json")}
    elif kind == "model":
        from .suite import execute
        model_output = Path("results/model").resolve()
        execute({"kind": "prepare-model", "model": spec["model"]}, data, model_output, spec["seconds"])
        manifest = read_json(model_output / "model-manifest.json")
        for item in manifest["files"]:
            source, target = model_output / item["path"], data / item["path"]
            if digest(source) != item["sha256"]:
                _fail("MODEL_DOWNLOAD_DIGEST_CHANGED")
            if target.exists() and digest(target) != item["sha256"]:
                _fail("MODEL_PROMOTION_REQUIRES_NEW_DIRECTORY")
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copy2(source, target)
        previous = data / "model-manifest.json"
        if previous.exists() and read_json(previous) != manifest:
            _fail("MODEL_MANIFEST_CHANGED_CREATE_CHILD")
        atomic_json(previous, manifest)
        result = {"kind": kind, "status": "completed", "model_manifest_ref": ref(spec["project_root"], previous),
                  "model_files": _manifest_files(spec["project_root"], ref(spec["project_root"], previous))}
    elif kind == "qualify":
        from .benchmarks import prepare_reference_qualification
        from .suite_score import score_benchmark
        from .io import Deadline
        work = Path("results/qualification").resolve()
        packet = prepare_reference_qualification(data, work)
        selected = next(job for job in packet["jobs"] if job["benchmark"] == spec["benchmark"])
        metrics = score_benchmark(selected["raw_path"], selected["tasks_path"], work / "scoring",
            spec["benchmark"], selected["expected_samples"],
            Deadline(time.time() + spec["seconds"]), qualify=True, data_root=data)
        result = {"kind": kind, "status": "completed", "benchmark": spec["benchmark"], "metrics": metrics,
                  "scope": "released native scorer replay only; not a qualified model or scientific result"}
    else:
        _fail("PREPARATION_KIND_UNSUPPORTED", kind)
    atomic_json(output, result)
    print(json.dumps(result))


def prepare_operation(root, *, kind, data, seconds, model=None, benchmark="humaneval", queue=None):
    _conda()
    root, data = Path(root).resolve(), _path(root, data)
    folder = root / "runs/controller" / (kind + "-" + uuid.uuid4().hex[:12])
    folder.mkdir(parents=True)
    spec = {"kind": kind, "data": str(data), "project_root": str(root), "seconds": seconds,
            "model": model or "Qwen/Qwen2.5-Coder-1.5B-Instruct", "benchmark": benchmark}
    atomic_json(folder / "preparation.json", spec)
    inputs = [ref(root, folder / "preparation.json")]
    # Source-backed retained release/split records are required by asset preparation.
    inputs += [ref(root, p) for p in (root / "research/design-v2/data").glob("*") if p.is_file()]
    if kind == "qualify":
        manifest_ref = ref(root, data / "benchmarks-manifest.json")
        inputs += [manifest_ref, *_manifest_files(root, manifest_ref)]
    task = _engineering(root, folder,
        [sys.executable, "-m", "recursive_ssd.suite_queue", "_prepare", "--spec", str(folder / "preparation.json"),
         "--output", "results/receipt.json"], inputs=inputs, outputs=["results/receipt.json"], seconds=seconds,
        budget_path=authorization_path(root, queue) if queue else None)
    if task["status"] == "completed" and kind in {"assets", "model"}:
        native = read_json(verify_ref(root, task["receipt_ref"]))
        attempt = next(a for a in native["attempts"] if a["status"] == "completed")
        workspace = _path(root, attempt["attempt_path"]) / "workspace"
        receipt = read_json(workspace / "results/receipt.json")
        outputs = (receipt["files"] if kind == "assets" else
                   [receipt["model_manifest_ref"], *receipt["model_files"]])
        for item in _unique_refs(outputs):
            source = verify_ref(workspace, item)
            target = _path(root, item["path"])
            if target.exists() and digest(target) != item["sha256"]:
                _fail("PREPARATION_DESTINATION_CHANGED_CREATE_CHILD", target)
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
            if digest(target) != item["sha256"]:
                _fail("PREPARATION_PROMOTION_DIGEST_MISMATCH", target)
        atomic_json(folder / "promoted-outputs.json", {"native_receipt_ref": task["receipt_ref"], "outputs": outputs})
        task["promoted_outputs_ref"] = ref(root, folder / "promoted-outputs.json")
    return task


def _snapshot_inputs(root, directory, *, exclude=()):
    """Freeze the exact report inputs, including retained native attempts."""
    queue, state, _ = verify_queue(root, directory)
    excluded = {_path(root, value) for value in exclude}
    files = set()
    files.add(authorization_path(root, directory))
    for path in Path(directory).rglob("*"):
        if (path.is_file() and not path.is_symlink() and path not in excluded
                and path.name not in {"report.json", "REPORT.md", "controller.lock"}):
            files.add(path)
    for value in _nested_refs(queue):
        files.add(verify_ref(root, value))
    for value in _nested_refs(state):
        files.add(verify_ref(root, value))
    for entry in state["nodes"].values():
        for attempt in entry["attempts"]:
            if not attempt.get("receipt_ref"):
                continue
            native = read_json(verify_ref(root, attempt["receipt_ref"]))
            for value in _nested_refs(native):
                files.add(verify_ref(root, value))
            for record in native["attempts"]:
                files.add(_path(root, record["attempt_path"]) / "attempt.json")
    # Native plan input snapshots may refer to source-backed protocol metadata.
    for entry in state["nodes"].values():
        for attempt in entry["attempts"]:
            plan = read_json(verify_ref(root, attempt["native_plan_ref"]))
            for value in _nested_refs(plan):
                files.add(verify_ref(root, value))
    return _unique_refs([ref(root, path) for path in files if path not in excluded])


def metadata_operation(root, *, operation, directory, seconds, output=None, evidence_context=None,
                       include_checkpoints=False):
    """Charge bounded analysis/export to the original, still-live authorization.

    Admission's finalization margin is headroom, not a second authorization.
    Reserve before hashing the snapshot because its inputs include the ledger.
    A later file transfer may copy retained outputs, but it must not rerun this
    worker after the original deadline or silently create another budget.
    """
    if operation not in {"report", "select", "collect"}:
        _fail("METADATA_OPERATION_UNSUPPORTED", operation)
    root, directory = Path(root).resolve(), _path(root, directory)
    output_path = _path(root, output) if output else None
    exclude = [output_path, output_path.with_name(output_path.stem + "-native-report.json")] if output_path else []
    with _lock(directory):
        budget_path = authorization_path(root, directory)
        _time_left(budget_path)
        reservation_id = "finalize-" + operation + "-" + uuid.uuid4().hex[:20]
        admission.reserve_preparation(budget_path, preparation_id=reservation_id,
            bound_seconds=seconds, purpose="engineering-preparation")
        inputs = _snapshot_inputs(root, directory, exclude=exclude)
        command = [sys.executable, "-m", "recursive_ssd.suite_queue", "_" + operation,
                   "--root", ".", "--queue", str(directory.relative_to(root))]
        if output_path:
            command += ["--output", str(output_path.relative_to(root))]
        if evidence_context:
            value = ref(root, evidence_context)
            inputs.append(value)
            inputs += [ref(root, verify_ref(root, item)) for item in _nested_refs(read_json(verify_ref(root, value)))]
            command += ["--evidence-context", value["path"]]
        if include_checkpoints:
            command.append("--include-checkpoints")
        folder = root / "runs/controller" / (operation + "-" + uuid.uuid4().hex[:12])
        task = _engineering(root, folder, command, inputs=inputs, outputs=["results/export.json"], seconds=seconds,
                            budget_path=budget_path, reservation_id=reservation_id)
        if task["status"] != "completed":
            return {**task, "budget_ref": ref(root, budget_path)}
        native = read_json(verify_ref(root, task["receipt_ref"]))
        attempt = next(a for a in native["attempts"] if a["status"] == "completed")
        workspace = _path(root, attempt["attempt_path"]) / "workspace"
        export = read_json(workspace / "results/export.json")
        for item in export["files"]:
            source = verify_ref(workspace, item)
            target = _path(root, item["path"])
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            if digest(target) != item["sha256"]:
                _fail("REPORT_PROMOTION_DIGEST_MISMATCH")
        atomic_json(folder / "promoted-outputs.json", {"native_receipt_ref": task["receipt_ref"], "outputs": export["files"]})
        return {"status": "completed", "receipt_ref": task["receipt_ref"], "outputs": export["files"],
                "budget_ref": ref(root, budget_path),
                "accounting_scope": "export uses pre-settlement inputs; current budget_ref includes this metadata attempt"}


def qualification_job(root, *, job, data, bindings, seconds, budget_path):
    """Execute bounded baseline or exact-catalogue comparator qualification."""
    _conda()
    root, data = Path(root).resolve(), _path(root, data)
    seconds = float(seconds)
    protocol, contract, required = _native_inputs(root, bindings["protocol_ref"], bindings["group"])
    child = bool(protocol.get("suite_calibration_child"))
    if child:
        from .suite_precursor import child_scope, prepare_child_job
        scope, _, _ = child_scope(root, protocol)
        admission.validate_candidate_binding(root, bindings["protocol_ref"], bindings.get("verification_ref"))
    else:
        if bindings.get("verification_ref") is not None:
            _fail("QUALIFICATION_CANDIDATE_SCOPE_CONFLICT")
        scope = admission._qualification(protocol, root)
    comparator = not child and scope["purpose"] == "comparator-qualification"
    bindings = deepcopy(bindings)
    bound_inputs = []
    if comparator or child:
        bindings["data"] = str(data.relative_to(root))
        bindings["budget_path"] = str(_path(root, budget_path).relative_to(root))
        if child:
            job, bound_inputs = prepare_child_job(root, job, data, bindings, protocol, budget_path=budget_path)
            if seconds != scope["node_seconds"][job["node_id"]]:
                _fail("CALIBRATION_CHILD_FROZEN_NODE_BOUND_REQUIRED")
            arm = contract["arm_requirements"][bindings["arm_role"]]["name"]
        else:
            job, bound_inputs = materialize_comparator_qualification(root, job, data, bindings, protocol)
            arm = scope["arm_bindings"][bindings["arm_role"]]
    else:
        arm = validate_qualification_job(job)
    folder = root / "runs/controller" / ("qualification-" + uuid.uuid4().hex[:14])
    folder.mkdir(parents=True)
    job = deepcopy(job)
    job.update(project_root=str(root), model_manifest=str(data / "model-manifest.json"))
    atomic_json(folder / "job.json", job)
    manifest, model = ref(root, data / "benchmarks-manifest.json"), ref(root, data / "model-manifest.json")
    expected = "initial" if arm == "base" else arm
    if contract["arm_requirements"][bindings["arm_role"]]["name"] != expected:
        _fail("QUALIFICATION_ARM_BINDING_MISMATCH")
    inputs = _unique_refs([ref(root, folder / "job.json"), manifest, model,
                          *_manifest_files(root, manifest), *_manifest_files(root, model), *required,
                          *bindings.get("input_refs", []), *bound_inputs])
    env = current_environment()
    atomic_json(folder / "environment.json", env)
    run_id = "qualification-" + uuid.uuid4().hex[:16]
    reservation_id = run_id
    replay_context = None
    provenance = {"git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "model_revision": read_json(data / "model-manifest.json")["revision"], "data_revision": manifest["sha256"],
        "config_digest": object_hash(job.get("config", {})),
        "environment_digest": object_hash(env), "environment_refs": [ref(root, folder / "environment.json")]}
    if comparator or child:
        atomic_json(folder / "bindings.json", bindings)
        binding_ref = ref(root, folder / "bindings.json")
        inputs = _unique_refs([*inputs, binding_ref])
        provenance.update(node_id=job["node_id"], qualification_protocol_ref=bindings["protocol_ref"],
            qualification_suite_digest=job["suite_digest"], qualification_arm_role=bindings["arm_role"],
            qualification_bindings_ref=binding_ref, qualification_job_ref=ref(root, folder / "job.json"),
            qualification_budget_path=bindings["budget_path"])
        if child:
            reservation_id = "calibration-child-" + bindings["protocol_ref"]["sha256"]
            calls = _qualification_replay_count(root, protocol)
            timeout = bindings.get("replay_timeout_seconds")
            if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 0 < timeout <= 3600:
                _fail("FINITE_LIVE_REPLAY_BUDGET_REQUIRED")
            if calls != bindings.get("replay_calls_per_node"):
                _fail("EXACT_NATIVE_REPLAY_CALL_INVENTORY_REQUIRED")
            budget = admission._read_budget(budget_path)
            charged_ids = {entry["reservation_id"] for entry in budget["charges"]}
            child_reservation = budget["reservations"].get(reservation_id)
            needed = 0 if child_reservation else sum(scope["node_seconds"].values())
            replay_inventory = {node: [reservation_id + "-" + node + "-replay-" + str(index)
                                      for index in range(calls)] for node in scope["node_ids"]}
            missing_replays = [identifier for values in replay_inventory.values() for identifier in values
                               if identifier not in budget["reservations"] and identifier not in charged_ids]
            if needed + timeout * len(missing_replays) > admission.remaining_budget(budget_path):
                _fail("COMPLETE_CALIBRATION_CHILD_WITH_REPLAY_EXCEEDS_REMAINING_BUDGET")
            admission.reserve_calibration_child(root, budget_path, protocol_ref=bindings["protocol_ref"],
                verification_ref=bindings["verification_ref"], node_id=job["node_id"], reservation_id=reservation_id)
            atomic_json(folder / "budget-location.json", {"project_root": str(root), "path": bindings["budget_path"]})
            for identifier in missing_replays:
                admission.reserve_preparation(budget_path, preparation_id=identifier,
                    bound_seconds=timeout, purpose="native-reference-replay")
            replay_context = _live_replay(root, folder, bindings, {"remaining_replay_calls": calls,
                "next_replay_index": 0, "replay_reservations": replay_inventory[job["node_id"]]})
        else:
            admission.reserve_comparator_qualification(root, budget_path, calibration_id=run_id,
                bound_seconds=seconds, protocol_ref=bindings["protocol_ref"], arm_role=bindings["arm_role"])
    else:
        admission.reserve_calibration(budget_path, calibration_id=run_id, bound_seconds=seconds,
                                      arm_ids=[expected])
    native = admission.build_scientific_plan(root, run_id=run_id,
        command=[sys.executable, "-m", "recursive_ssd.suite", "execute", "--job", str(folder / "job.json"),
                 "--data", str(data.relative_to(root)), "--output", "results", "--seconds", str(seconds)],
        code_refs=source_refs(root), input_refs=inputs, output_paths=["results/receipt.json"],
        protocol_ref=bindings["protocol_ref"], verification_ref=bindings.get("verification_ref") if child else None,
        seed=job.get("trajectory", job.get("unit", {})).get("seed", 17), group=bindings["group"],
        arm_role=bindings["arm_role"], seconds=seconds,
        provenance=provenance, replay_context=replay_context,
        evidence_mode="developmental" if child else None)
    profile = bindings.get("resources", {})
    measured = child and (profile.get("gpu_peak_mib") is not None or profile.get("memory_profile_ref") is not None)
    resource = _resources(bindings, gpu=True) if measured else _resources(bindings, gpu=False)
    if not measured:
        resource.update(gpu_count=1, ram_mib=bindings.get("resources", {}).get("ram_mib", 14000))
    # A finite reviewed measurement child may discover its peak on one exclusive
    # GPU. Main full-suite admission still requires the real measured profile.
    plan = _harness_plan(root, folder, native, resources=resource, gpu_uuid=bindings["gpu_uuid"],
                         hard_seconds=_time_left(budget_path))
    task = _execute_plan(root, folder, plan)
    if task.get("receipt_ref"):
        admission.charge_budget(root, budget_path, reservation_id=reservation_id, receipt_ref=task["receipt_ref"],
                                cell_id=job["node_id"] if child else None, final=not child)
    if (comparator or child) and task.get("receipt_ref"):
        native_receipt = read_json(verify_ref(root, task["receipt_ref"]))
        outputs = []
        entry = {"status": "completed" if task["status"] == "completed" else "failed",
            "receipt_ref": task["receipt_ref"], "attempts": [{"receipt_ref": task["receipt_ref"],
            "native_plan_ref": ref(root, folder / "native-plan.json")}]}
        for attempt in native_receipt["attempts"]:
            result_folder = _path(root, attempt["attempt_path"]) / "workspace/results"
            if result_folder.is_dir():
                outputs += [ref(root, path) for path in result_folder.rglob("*")
                            if path.is_file() and not path.is_symlink()]
            for output in attempt["output_refs"]:
                if output["path"].endswith("/results/receipt.json"):
                    entry["workload_receipt_ref"] = output
        entry["output_refs"] = _unique_refs(outputs)
        atomic_json(folder / "dependency-record.json", {"nodes": {job["node_id"]: entry}})
        task = {**task, "dependency_state_ref": ref(root, folder / "dependency-record.json")}
    return task


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    replay = sub.add_parser("_native-replay")
    replay.add_argument("--root", required=True)
    replay.add_argument("--request", required=True)
    replay.add_argument("--result", required=True)
    replay.add_argument("--seconds", type=float, required=True)
    prepare = sub.add_parser("_prepare")
    prepare.add_argument("--spec", required=True)
    prepare.add_argument("--output", required=True)
    for name in ("_report", "_select", "_collect"):
        p = sub.add_parser(name)
        p.add_argument("--root", required=True)
        p.add_argument("--queue", required=True)
        p.add_argument("--output")
        p.add_argument("--include-checkpoints", action="store_true")
        p.add_argument("--evidence-context")
    args = parser.parse_args(argv)
    if args.operation == "_native-replay":
        _native_replay_worker(args)
        return
    if args.operation == "_prepare":
        from .suite import stage_paths
        _prepare_worker(stage_paths(read_json(args.spec), ROOT), args.output)
        return
    if args.operation == "_report":
        result = queue_report(args.root, args.queue,
                              evidence_context=read_json(args.evidence_context) if args.evidence_context else None)
        print(json.dumps({"report": str(Path(args.queue) / "report.json"), "inventory_complete": result["inventory"]["complete"]}))
        exports = [Path(args.queue) / "report.json", Path(args.queue) / "REPORT.md"]
    elif args.operation == "_select":
        print(json.dumps(verified_selection(args.root, args.queue, args.output)))
        selected = Path(args.output)
        exports = [Path(args.queue) / "report.json", Path(args.queue) / "REPORT.md", selected,
                   selected.with_name(selected.stem + "-native-report.json")]
    else:
        print(json.dumps(collect_queue(args.root, args.queue, args.output, include_checkpoints=args.include_checkpoints)))
        exports = [Path(args.output)]
    atomic_json("results/export.json", {"files": [ref(args.root, path) for path in exports]})


if __name__ == "__main__":
    main()
