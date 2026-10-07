"""Source-bound, finite G01 children for candidate comparator/calibration work.

Generated source, not executed in the Web authoring environment. These helpers
materialize and validate metadata only. Execution remains owned by the pinned
project harness; this route does not make a candidate a baseline or close the
full comparison. Current canonical/method/native replay admission is performed
by suite_admission for every newly dispatched child.
"""
from copy import deepcopy
from pathlib import Path
import sys

from . import suite_admission as admission
from .io import object_hash, read_json


def _queue():
    # Avoid an import cycle: qualification_job calls this module lazily.
    from . import suite_queue
    return suite_queue


def _fail(code, detail=""):
    raise admission.AdmissionError(code, str(detail))


def _ancestors(nodes, identifier):
    result = set()
    def visit(current):
        for parent in nodes[current]["depends_on"]:
            if parent not in result:
                result.add(parent)
                visit(parent)
    visit(identifier)
    return result


def _effective_arm(nodes, node):
    return nodes[node["source_node_id"]]["arm_id"] if node["kind"] == "derive-calibration" else node["arm_id"]


def child_scope(root, protocol):
    """Validate an exact catalog child without granting scientific admission.

    Every node is pinned in the protocol, with its transitive dependencies,
    native group/role and finite cost bound. A selected candidate used as a
    comparator keeps its own method_discovery ID, including aliased ablations.
    A calibration derivation has its source candidate's native role.
    """
    Q = _queue()
    from .suite_design import ORDER, arm_registry, make_suite, verify_suite
    scope = protocol.get("suite_calibration_child")
    fields = {"purpose", "suite_ref", "bundle_id", "node_ids", "node_roles",
              "node_seconds", "config", "calibration", "tuning_budget", "execution_provenance"}
    if not isinstance(scope, dict) or set(scope) != fields:
        _fail("CALIBRATION_CHILD_SCOPE_REQUIRED")
    if (scope["purpose"] != "native-comparator-calibration" or protocol.get("suite_qualification")
            or protocol.get("evidence_mode") != "prospective_confirmatory"):
        _fail("CALIBRATION_CHILD_SCIENTIFIC_SCOPE_REQUIRED")
    candidate = protocol.get("method_discovery", {}).get("candidate_id")
    if candidate not in ORDER:
        _fail("CALIBRATION_CHILD_OWN_METHOD_DESIGN_REQUIRED")
    suite = verify_suite(read_json(Q.verify_ref(root, scope["suite_ref"])))
    if suite["stage"] not in {"development", "tuning"} or scope["bundle_id"] not in suite["candidate_ids"]:
        _fail("CALIBRATION_CHILD_DEVELOPMENT_CATALOGUE_REQUIRED")
    expected = make_suite(suite["stage"], model=suite["model"], calibration=suite.get("calibration"),
                          tuning_selection=suite.get("tuning_selection"))
    if suite != expected:
        _fail("CALIBRATION_CHILD_EXACT_CATALOGUE_REQUIRED")
    nodes = {node["node_id"]: node for node in Q.compile_nodes(suite)}
    selected = scope["node_ids"]
    if (not isinstance(selected, list) or not selected
            or any(not isinstance(identifier, str) or identifier not in nodes for identifier in selected)
            or len(set(selected)) != len(selected)):
        _fail("CALIBRATION_CHILD_FINITE_NODE_INVENTORY_REQUIRED")
    if not isinstance(scope["node_roles"], dict) or set(scope["node_roles"]) != set(selected):
        _fail("CALIBRATION_CHILD_EXACT_NODE_ROLES_REQUIRED")
    if not isinstance(scope["node_seconds"], dict) or set(scope["node_seconds"]) != set(selected):
        _fail("CALIBRATION_CHILD_COMPLETE_COST_BOUNDS_REQUIRED")
    seconds = sum(admission._number(value, "CALIBRATION_CHILD_FINITE_COST_REQUIRED", positive=True)
                  for value in scope["node_seconds"].values())
    if seconds > admission.MAX_TOTAL_SECONDS:
        _fail("CALIBRATION_CHILD_AUTHORIZED_CAP_EXCEEDED")
    if not isinstance(scope["config"], dict) or not isinstance(scope["calibration"], dict):
        _fail("CALIBRATION_CHILD_FROZEN_SETTINGS_REQUIRED")
    if scope["calibration"] != suite["calibration"]:
        _fail("CALIBRATION_CHILD_CATALOGUE_CALIBRATION_REQUIRED")
    tuning = scope["tuning_budget"]
    if suite["stage"] == "tuning":
        if (not isinstance(tuning, dict) or set(tuning) != {"trial_wall_seconds", "source_refs"}
                or not isinstance(tuning["source_refs"], list) or not tuning["source_refs"]):
            _fail("CALIBRATION_CHILD_FROZEN_TUNING_ALLOWANCE_REQUIRED")
        allowance = admission._number(tuning["trial_wall_seconds"], "CALIBRATION_CHILD_FINITE_COST_REQUIRED", positive=True)
        if allowance > admission.MAX_TOTAL_SECONDS:
            _fail("CALIBRATION_CHILD_AUTHORIZED_CAP_EXCEEDED")
        for trajectory in {nodes[identifier]["trajectory_id"] for identifier in selected}:
            training_seconds = sum(scope["node_seconds"][identifier] for identifier in selected
                                   if nodes[identifier]["trajectory_id"] == trajectory and nodes[identifier]["kind"] == "train")
            if training_seconds > allowance:
                _fail("CALIBRATION_CHILD_TUNING_TRIAL_BOUND_EXCEEDED", trajectory)
    elif tuning is not None:
        _fail("CALIBRATION_CHILD_TUNING_SCOPE_MISMATCH")
    provenance = scope["execution_provenance"]
    if (not isinstance(provenance, dict)
            or not {"model_revision", "environment_digest", "data_revision"}.issubset(provenance)
            or any(not isinstance(value, str) or not value for value in provenance.values())
            or provenance["model_revision"] != suite["model_revision"]):
        _fail("CALIBRATION_CHILD_EXECUTION_PROVENANCE_REQUIRED")
    _, _, N, _ = admission._modules()
    registry = arm_registry()
    for identifier in selected:
        node = nodes[identifier]
        arm = _effective_arm(nodes, node)
        if scope["bundle_id"] not in node["bundles"]:
            _fail("CALIBRATION_CHILD_NODE_OUTSIDE_BUNDLE", identifier)
        if registry[arm]["method"] != candidate:
            _fail("CALIBRATION_CHILD_OWN_METHOD_DESIGN_REQUIRED", arm)
        if not _ancestors(nodes, identifier).issubset(selected):
            _fail("CALIBRATION_CHILD_COMPLETE_DEPENDENCY_SCOPE_REQUIRED", identifier)
        role = scope["node_roles"][identifier]
        if (not isinstance(role, dict) or set(role) != {"group", "arm_role"}
                or role["group"] not in protocol.get("required_groups", [])):
            _fail("CALIBRATION_CHILD_NATIVE_ROLE_REQUIRED", identifier)
        contract = N.contract_for_group(protocol, role["group"])
        if contract["arm_requirements"].get(role["arm_role"], {}).get("name") != arm:
            _fail("CALIBRATION_CHILD_NATIVE_ARM_MISMATCH", identifier)
    Q.effective_calibration(suite, scope)
    for value in Q._nested_refs(scope):
        Q.verify_ref(root, value)
    return scope, suite, nodes


def validate_child_dependency(root, state, identifier, *, suite, scope, budget_path):
    """Verify a historical candidate child without requiring it to be current.

    Its immutable method design and native plan remain verified. The canonical
    current candidate may legitimately have moved since it ran. Retained native
    attempts, bound outputs and a charge in the same original authorization are
    required; a fabricated completed flag cannot provide calibration.
    """
    Q = _queue()
    root = Path(root).resolve()
    receipt, folder = Q._completed_output(root, state, identifier)
    entry = state["nodes"][identifier]
    native = read_json(Q.verify_ref(root, entry["receipt_ref"]))
    provenance = native.get("provenance", {})
    if (native.get("purpose") != "scientific"
            or provenance.get("admission_scope") != "candidate-calibration-child"
            or provenance.get("qualification_suite_digest") != suite["suite_digest"]
            or provenance.get("node_id") != identifier):
        _fail("CALIBRATION_CHILD_NATIVE_DEPENDENCY_REQUIRED", identifier)
    if any(provenance.get(key) != value for key, value in scope["execution_provenance"].items()):
        _fail("CALIBRATION_CHILD_DEPENDENCY_PROVENANCE_MISMATCH", identifier)
    ledger_path = Q._path(root, budget_path)
    if Q._path(root, provenance.get("qualification_budget_path", "")) != ledger_path:
        _fail("CALIBRATION_CHILD_ORIGINAL_BUDGET_REQUIRED", identifier)
    ledger = admission._read_budget(ledger_path)
    charges = [item for item in ledger["charges"] if item["receipt_ref"] == entry["receipt_ref"]]
    if (len(charges) != 1 or charges[0]["seconds"] != native["resources"]["seconds"]
            or charges[0]["status"] != native["status"]):
        _fail("CALIBRATION_CHILD_CHARGED_DEPENDENCY_REQUIRED", identifier)
    if (admission._timestamp(native["started_at"]) < admission._timestamp(ledger["original_start"])
            or admission._timestamp(native["completed_at"]) > admission._timestamp(ledger["absolute_end"])):
        _fail("CALIBRATION_CHILD_DEPENDENCY_OUTSIDE_AUTHORIZATION", identifier)
    matched = [attempt for attempt in entry["attempts"] if attempt.get("receipt_ref") == entry["receipt_ref"]]
    plan = read_json(Q.verify_ref(root, matched[0]["native_plan_ref"]))
    protocol = read_json(Q.verify_ref(root, plan["protocol_ref"]))
    previous, previous_suite, _ = child_scope(root, protocol)
    if (previous_suite["suite_digest"] != suite["suite_digest"]
            or previous["config"] != scope["config"] or previous["calibration"] != scope["calibration"]):
        _fail("CALIBRATION_CHILD_DEPENDENCY_SETTINGS_MISMATCH", identifier)
    if previous.get("tuning_budget") != scope.get("tuning_budget"):
        _fail("CALIBRATION_CHILD_DEPENDENCY_SETTINGS_MISMATCH", "tuning_budget")
    _, R, _, M = admission._modules()
    M.verify_run_design(root, plan, protocol)
    R.validate_plan(root, plan)
    validate_child_plan(root, plan, protocol)
    if plan["provenance"] != provenance:
        _fail("CALIBRATION_CHILD_NATIVE_PLAN_PROVENANCE_MISMATCH", identifier)
    return receipt, folder


def validate_node_native_inputs(root, data, node, protocol, group):
    """Bind a catalog endpoint to exact native bytes, task IDs and decoder.

    Benchmark names are not heuristics. Prepared asset paths/IDs and the native
    contract must agree, including the project HumanEval split adapter.
    """
    Q = _queue()
    from .suite_design import BENCHMARK_SETTINGS
    data = Q._path(root, data)
    manifest_ref = Q.ref(root, data / "benchmarks-manifest.json")
    manifest = read_json(Q.verify_ref(root, manifest_ref))
    unit = node.get("unit", {"benchmark": "humaneval", "split": "dev",
                             "sampling": BENCHMARK_SETTINGS["humaneval"], "expected_samples": 10})
    info = manifest["benchmarks"][unit["benchmark"]]["splits"][unit["split"]]
    tests = Q.ref(root, data / info["path"])
    if manifest["files"][info["path"]] != tests["sha256"]:
        _fail("CALIBRATION_CHILD_PREPARED_NATIVE_BYTES_CHANGED")
    C, _, N, _ = admission._modules()
    contract = N.contract_for_group(protocol, group)
    samples = read_json(Q.verify_ref(root, contract["sample_manifest_ref"]))
    definition = read_json(Q.verify_ref(root, contract["native_definition_ref"]))
    if (samples["sample_ids"] != info["task_ids"] or samples["denominator"] != info["count"]
            or samples["predictions_per_sample"] != unit["expected_samples"]
            or samples["split"] != contract["split"] or samples.get("labels_or_tests_ref") != contract["labels_or_tests_ref"]
            or contract.get("selection") is not None):
        _fail("CALIBRATION_CHILD_EXACT_NATIVE_INVENTORY_REQUIRED")
    parameters = {key: value for key, value in unit["sampling"].items() if key not in {"samples", "coverage"}}
    parameters["n"] = unit["expected_samples"]
    if (contract["sampling"]["parameters"] != parameters
            or samples["sampling"] != contract["sampling"] or samples["budget"] != contract["budget"]):
        _fail("CALIBRATION_CHILD_NATIVE_DECODER_MISMATCH")
    if unit["benchmark"] == "livecodebench":
        labels = read_json(Q.verify_ref(root, contract["labels_or_tests_ref"]))
        if (labels.get("version") != "native-labels-shards-v1" or labels.get("index_ref") != tests
                or labels.get("asset_manifest_ref") != manifest_ref or labels.get("sample_ids") != info["task_ids"]):
            _fail("CALIBRATION_CHILD_NATIVE_LABEL_BINDING_MISMATCH")
    elif contract["labels_or_tests_ref"] != tests:
        _fail("CALIBRATION_CHILD_NATIVE_LABEL_BINDING_MISMATCH")
    if unit["split"] != "full":
        adapter = read_json(Q.verify_ref(root, definition.get("project_split_adapter_ref")))
        if (adapter.get("asset_manifest_ref") != manifest_ref or adapter.get("project_split") != unit["split"]
                or adapter.get("selected_tests_ref") != tests):
            _fail("CALIBRATION_CHILD_NATIVE_SPLIT_MISMATCH")
    admission._verify_split_adapter(C, Path(root).resolve(), definition, samples)
    return [contract["sample_manifest_ref"], contract["labels_or_tests_ref"],
            contract["native_definition_ref"], *admission._definition_inputs(C, root, definition)]


def prepare_child_job(root, request, data, bindings, protocol, *, budget_path=None):
    """Derive one exact child node; callers cannot inject jobs/checkpoints/argv."""
    Q = _queue()
    root, data = Path(root).resolve(), Q._path(root, data)
    scope, suite, nodes = child_scope(root, protocol)
    if not isinstance(request, dict) or set(request) != {"node_id"} or request["node_id"] not in scope["node_ids"]:
        _fail("CALIBRATION_CHILD_EXACT_NODE_REQUEST_REQUIRED")
    node = nodes[request["node_id"]]
    role = scope["node_roles"][node["node_id"]]
    if any(bindings.get(key) != value for key, value in role.items()):
        _fail("CALIBRATION_CHILD_PLAN_ROLE_MISMATCH")
    if any(bindings.get(key, {}) != scope[key] for key in ("config", "calibration")):
        _fail("CALIBRATION_CHILD_FROZEN_SETTINGS_MISMATCH")
    if bindings.get("tuning_budget") != scope["tuning_budget"]:
        _fail("CALIBRATION_CHILD_FROZEN_TUNING_ALLOWANCE_REQUIRED")
    if bindings.get("verification_ref") is None:
        _fail("RUN_METHOD_DESIGN_EVIDENCE_REQUIRED")
    budget_path = budget_path or bindings.get("budget_path")
    if not budget_path or not bindings.get("budget_path"):
        _fail("CALIBRATION_CHILD_ORIGINAL_BUDGET_REQUIRED")
    if Q._path(root, budget_path) != Q._path(root, bindings["budget_path"]):
        _fail("CALIBRATION_CHILD_ORIGINAL_BUDGET_REQUIRED")
    admission._read_budget(Q._path(root, budget_path))
    dependencies = _ancestors(nodes, node["node_id"])
    state = {"nodes": {}}
    inputs = [scope["suite_ref"], bindings["verification_ref"]]
    source = bindings.get("dependency_state_ref")
    if source:
        state = read_json(Q.verify_ref(root, source))
        inputs.append(source)
    if set(state.get("nodes", {})) != dependencies:
        _fail("CALIBRATION_CHILD_EXACT_DEPENDENCY_CLOSURE_REQUIRED")
    for identifier in sorted(dependencies):
        validate_child_dependency(root, state, identifier, suite=suite, scope=scope, budget_path=budget_path)
        inputs.extend(Q._nested_refs(state["nodes"][identifier]))
    state = deepcopy(state)
    state["nodes"][node["node_id"]] = {"status": "pending"}
    queue = {"identity": {"project_root": str(root), "suite_ref": scope["suite_ref"],
        "benchmark_manifest_ref": Q.ref(root, data / "benchmarks-manifest.json"),
        "model_manifest_ref": Q.ref(root, data / "model-manifest.json")}}
    job, retained = Q.materialize_job(root, queue, state, suite, node, bindings)
    native_inputs = validate_node_native_inputs(root, data, node, protocol, role["group"])
    return job, Q._unique_refs([*inputs, *retained, *native_inputs])


def validate_child_plan(root, plan, protocol):
    """Bind a prospective or historical native plan to its exact generated job.

    This helper intentionally does not call the current-canonical verifier; the
    caller must do that for dispatch, or verify retained method/native evidence
    for historical dependency reuse. No resource reservation occurs here.
    """
    Q = _queue()
    scope, suite, nodes = child_scope(root, protocol)
    provenance = plan.get("provenance", {})
    if (plan.get("purpose") != "scientific" or plan.get("evidence_mode") != "developmental"
            or provenance.get("admission_scope") != "candidate-calibration-child"
            or provenance.get("qualification_protocol_ref") != plan.get("protocol_ref")
            or len(plan.get("jobs", [])) != 1):
        _fail("CALIBRATION_CHILD_NATIVE_PLAN_REQUIRED")
    binding_ref, job_ref = provenance.get("qualification_bindings_ref"), provenance.get("qualification_job_ref")
    bindings = read_json(Q.verify_ref(root, binding_ref))
    job = read_json(Q.verify_ref(root, job_ref))
    node_id = provenance.get("node_id")
    if (node_id not in scope["node_ids"] or bindings.get("protocol_ref") != plan["protocol_ref"]
            or bindings.get("verification_ref") != provenance.get("method_verification_ref")
            or bindings.get("budget_path") != provenance.get("qualification_budget_path")
            or provenance.get("qualification_suite_digest") != suite["suite_digest"]):
        _fail("CALIBRATION_CHILD_PROTOCOL_BINDING_MISMATCH")
    entry = plan["jobs"][0]
    role = scope["node_roles"][node_id]
    if (any(entry.get(key) != value for key, value in role.items())
            or provenance.get("qualification_arm_role") != role["arm_role"]):
        _fail("CALIBRATION_CHILD_PLAN_ROLE_MISMATCH")
    expected, required = prepare_child_job(root, {"node_id": node_id}, bindings["data"], bindings, protocol)
    if job != expected:
        _fail("CALIBRATION_CHILD_DERIVED_JOB_MISMATCH")
    if any(provenance.get(key) != value for key, value in scope["execution_provenance"].items()):
        _fail("CALIBRATION_CHILD_PROVENANCE_MISMATCH")
    if provenance["environment_digest"] != object_hash(Q.current_environment()):
        _fail("CALIBRATION_CHILD_CURRENT_ENVIRONMENT_MISMATCH")
    if Q._unique_refs(entry["code_refs"]) != Q._unique_refs(Q.source_refs(root)):
        _fail("CALIBRATION_CHILD_CURRENT_SOURCE_INVENTORY_MISMATCH")
    data = Q._path(root, bindings["data"])
    manifest = Q.ref(root, data / "benchmarks-manifest.json")
    model = Q.ref(root, data / "model-manifest.json")
    if (read_json(Q.verify_ref(root, model))["revision"] != provenance["model_revision"]
            or manifest["sha256"] != provenance["data_revision"]):
        _fail("CALIBRATION_CHILD_DATA_MODEL_BINDING_MISMATCH")
    trajectory = next(item for item in suite["trajectories"] if item["trajectory_id"] == nodes[node_id]["trajectory_id"])
    if entry["seed"] != trajectory["seed"]:
        _fail("CALIBRATION_CHILD_PLAN_SEED_MISMATCH")
    seconds = float(scope["node_seconds"][node_id])
    limits = plan["limits"]
    if (limits["wall_time_seconds"] != seconds or limits["attempt_timeout_seconds"] != seconds
            or limits["max_attempts"] != 1 or limits["max_retries_per_trial"] != 0
            or limits["max_development_trials"] != 1 or limits["max_confirmation_trials"] != 0):
        _fail("CALIBRATION_CHILD_FINITE_RUN_LIMITS_REQUIRED")
    argv = entry["command"]
    expected_tail = ["-m", "recursive_ssd.suite", "execute", "--job", str(Q.verify_ref(root, job_ref)),
                     "--data", bindings["data"], "--output", "results", "--seconds", str(seconds)]
    if (not argv or argv[1:] != expected_tail or Path(argv[0]).resolve() != Path(sys.executable).resolve()
            or entry.get("cwd") != "."):
        _fail("CALIBRATION_CHILD_COMMAND_NOT_DERIVED")
    if entry["output_paths"] != ["results/receipt.json"]:
        _fail("CALIBRATION_CHILD_OUTPUT_SCOPE_MISMATCH")
    required += [binding_ref, job_ref, manifest, model]
    required += Q._manifest_files(root, manifest) + Q._manifest_files(root, model)
    declared = {(value["path"], value["sha256"]) for value in entry["input_refs"]}
    if any((value["path"], value["sha256"]) not in declared for value in required):
        _fail("CALIBRATION_CHILD_DEPENDENCY_INPUTS_REQUIRED")
    return scope
