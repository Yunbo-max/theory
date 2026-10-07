"""Generated, unexecuted semantic tests for Local's pinned software harness.

Artificial fixtures exercise admission invariants. They are not actual method
design verification, target-host qualification or scientific evidence.
"""
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import pytest

from recursive_ssd import suite_admission as admission
from recursive_ssd import suite_precursor as child
from recursive_ssd import suite_queue as queue
from recursive_ssd.io import object_hash
from recursive_ssd.suite_design import BENCHMARK_SETTINGS, make_suite


def put(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    return {"path": name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


@pytest.fixture
def scoped(tmp_path, monkeypatch):
    suite = make_suite("development")
    suite_ref = put(tmp_path, "suite.json", suite)
    nodes = queue.compile_nodes(suite)
    # M06's measured-head derivation has an ordinary consumer arm. Its actual
    # native role must identify source M06, not temperature_matched_head.
    train = next(node for node in nodes if node["arm_id"] == "M06"
                 and node["kind"] == "train" and node["round"] == 1)
    derive = next(node for node in nodes if node["kind"] == "derive-calibration"
                  and node["calibration_key"] == "head_temperature" and node["round"] == 1)
    ids = [train["node_id"], derive["node_id"]]
    protocol = {"evidence_mode": "prospective_confirmatory", "method_discovery": {"candidate_id": "M06"},
        "required_groups": ["human-dev"], "native_eval_contracts": {"human-dev": {
            "arm_requirements": {"treatment": {"name": "M06"}}, "benchmark_id": "humaneval-plus"}},
        "suite_calibration_child": {"purpose": "native-comparator-calibration", "suite_ref": suite_ref,
            "bundle_id": "M06", "node_ids": ids,
            "node_roles": {identifier: {"group": "human-dev", "arm_role": "treatment"} for identifier in ids},
            "node_seconds": {identifier: 60. for identifier in ids}, "config": {}, "calibration": {}, "tuning_budget": None,
            "execution_provenance": {"model_revision": suite["model_revision"],
                "environment_digest": "engineering-fixture", "data_revision": "engineering-fixture"}}}
    native = SimpleNamespace(contract_for_group=lambda p, group: p["native_eval_contracts"][group])
    monkeypatch.setattr(admission, "_modules", lambda: (None, None, native, None))
    return protocol, suite, train, derive


def test_source_candidate_and_ordinary_derive_consumer_use_same_own_method(tmp_path, scoped):
    protocol, suite, train, derive = scoped
    scope, actual, nodes = child.child_scope(tmp_path, protocol)
    assert actual["suite_digest"] == suite["suite_digest"]
    assert scope["node_ids"] == [train["node_id"], derive["node_id"]]
    assert derive["arm_id"] == "temperature_matched_head"
    assert child._effective_arm(nodes, derive) == "M06"


@pytest.mark.parametrize("mutation,expected", [
    (lambda p: p.update(evidence_mode="confirmation"), "SCIENTIFIC_SCOPE"),
    (lambda p: p.update(suite_qualification={"purpose": "baseline-calibration"}), "SCIENTIFIC_SCOPE"),
    (lambda p: p["method_discovery"].update(candidate_id="M03"), "OWN_METHOD_DESIGN"),
    (lambda p: p["suite_calibration_child"]["node_seconds"].clear(), "COMPLETE_COST_BOUNDS"),
    (lambda p: p["suite_calibration_child"]["node_roles"].clear(), "EXACT_NODE_ROLES"),
    (lambda p: p["suite_calibration_child"].update(allow_any_checkpoint=True), "SCOPE_REQUIRED"),
])
def test_scope_rejects_shortcuts(tmp_path, scoped, mutation, expected):
    protocol = deepcopy(scoped[0])
    mutation(protocol)
    with pytest.raises(admission.AdmissionError, match=expected):
        child.child_scope(tmp_path, protocol)


@pytest.mark.parametrize("value", [0, -1, True, float("inf"), float("nan")])
def test_costs_are_positive_finite_not_flags(tmp_path, scoped, value):
    protocol, _, train, _ = scoped
    protocol["suite_calibration_child"]["node_seconds"][train["node_id"]] = value
    with pytest.raises(admission.AdmissionError, match="FINITE_COST"):
        child.child_scope(tmp_path, protocol)


def test_declared_nodes_cannot_drop_source_dependency(tmp_path, scoped):
    protocol, _, train, _ = scoped
    scope = protocol["suite_calibration_child"]
    scope["node_ids"].remove(train["node_id"])
    scope["node_roles"].pop(train["node_id"])
    scope["node_seconds"].pop(train["node_id"])
    with pytest.raises(admission.AdmissionError, match="COMPLETE_DEPENDENCY_SCOPE"):
        child.child_scope(tmp_path, protocol)


def test_derive_cannot_claim_consumer_as_native_source(tmp_path, scoped):
    protocol, _, _, _ = scoped
    protocol["native_eval_contracts"]["human-dev"]["arm_requirements"]["treatment"]["name"] = "temperature_matched_head"
    with pytest.raises(admission.AdmissionError, match="NATIVE_ARM_MISMATCH"):
        child.child_scope(tmp_path, protocol)


def test_protected_alias_requires_own_candidate_not_consuming_bundle(tmp_path, scoped):
    protocol, suite, _, _ = scoped
    nodes = queue.compile_nodes(suite)
    alias = next(node for node in nodes if node["arm_id"] == "floor_zero"
                 and node["kind"] == "train" and node["round"] == 1)
    scope = protocol["suite_calibration_child"]
    scope.update(bundle_id="M03", node_ids=[alias["node_id"]],
        node_roles={alias["node_id"]: {"group": "human-dev", "arm_role": "treatment"}},
        node_seconds={alias["node_id"]: 60.})
    protocol["native_eval_contracts"]["human-dev"]["arm_requirements"]["treatment"]["name"] = "floor_zero"
    protocol["method_discovery"]["candidate_id"] = "M03"
    child.child_scope(tmp_path, protocol)
    protocol["method_discovery"]["candidate_id"] = "M06"
    with pytest.raises(admission.AdmissionError, match="OWN_METHOD_DESIGN"):
        child.child_scope(tmp_path, protocol)


def test_candidate_as_control_retains_its_own_method_design(tmp_path, scoped):
    protocol, suite, _, _ = scoped
    node = next(node for node in queue.compile_nodes(suite) if node["arm_id"] == "M01"
                and node["kind"] == "train" and node["round"] == 1)
    assert "M02" in node["bundles"]
    scope = protocol["suite_calibration_child"]
    scope.update(bundle_id="M02", node_ids=[node["node_id"]],
        node_roles={node["node_id"]: {"group": "human-dev", "arm_role": "treatment"}},
        node_seconds={node["node_id"]: 60.})
    protocol["native_eval_contracts"]["human-dev"]["arm_requirements"]["treatment"]["name"] = "M01"
    protocol["method_discovery"]["candidate_id"] = "M01"
    child.child_scope(tmp_path, protocol)
    protocol["method_discovery"]["candidate_id"] = "M02"
    with pytest.raises(admission.AdmissionError, match="OWN_METHOD_DESIGN"):
        child.child_scope(tmp_path, protocol)


def test_recomputed_digest_cannot_hide_changed_catalogue_arm(tmp_path, scoped):
    protocol, suite, _, _ = scoped
    altered = deepcopy(suite)
    trajectory = next(item for item in altered["trajectories"] if item["arm_id"] == "M06")
    trajectory["arm"]["method"] = "M15"
    altered["suite_digest"] = object_hash({key: value for key, value in altered.items() if key != "suite_digest"})
    protocol["suite_calibration_child"]["suite_ref"] = put(tmp_path, "altered.json", altered)
    with pytest.raises(admission.AdmissionError, match="EXACT_CATALOGUE"):
        child.child_scope(tmp_path, protocol)


def test_tuning_uses_frozen_equal_trial_allowance(tmp_path, scoped):
    protocol = scoped[0]
    suite = make_suite("tuning")
    node = next(item for item in queue.compile_nodes(suite)
                if item["arm_id"].startswith("M03__") and item["kind"] == "train" and item["round"] == 1)
    scope = protocol["suite_calibration_child"]
    scope.update(bundle_id="M03", suite_ref=put(tmp_path, "tuning.json", suite), node_ids=[node["node_id"]],
        node_roles={node["node_id"]: {"group": "human-dev", "arm_role": "treatment"}},
        node_seconds={node["node_id"]: 60.}, tuning_budget={"trial_wall_seconds": 59.,
            "source_refs": [put(tmp_path, "allowance-evidence.json", {"kind": "engineering-fixture"})]})
    protocol["method_discovery"]["candidate_id"] = "M03"
    protocol["native_eval_contracts"]["human-dev"]["arm_requirements"]["treatment"]["name"] = node["arm_id"]
    with pytest.raises(admission.AdmissionError, match="TUNING_TRIAL_BOUND"):
        child.child_scope(tmp_path, protocol)
    scope["tuning_budget"]["trial_wall_seconds"] = 60.
    child.child_scope(tmp_path, protocol)


def test_caller_cannot_supply_checkpoint_or_settings(tmp_path, scoped):
    protocol, _, train, _ = scoped
    with pytest.raises(admission.AdmissionError, match="EXACT_NODE_REQUEST"):
        child.prepare_child_job(tmp_path, {"node_id": train["node_id"], "parent_checkpoint": "arbitrary"},
                                "data", {}, protocol)


def test_receipt_flag_without_native_attempts_is_not_calibration(tmp_path):
    state = {"nodes": {"source": {"status": "completed"}}}
    with pytest.raises(ValueError, match="DEPENDENCY_RECEIPT_REQUIRED"):
        child.validate_child_dependency(tmp_path, state, "source", suite={}, scope={}, budget_path="budget.json")


def test_plan_cannot_skip_candidate_admission(tmp_path, scoped):
    protocol = scoped[0]
    with pytest.raises(admission.AdmissionError, match="NATIVE_PLAN_REQUIRED"):
        child.validate_child_plan(tmp_path, {"purpose": "engineering", "jobs": []}, protocol)


@pytest.fixture
def native_inputs(tmp_path, monkeypatch):
    tests = put(tmp_path, "data/tests.json", [{"task_id": "fixture-1"}])
    put(tmp_path, "data/benchmarks-manifest.json", {"files": {"tests.json": tests["sha256"]},
        "benchmarks": {"mbpp": {"splits": {"full": {"path": "tests.json", "count": 1,
                                                          "task_ids": ["fixture-1"]}}}}})
    setting = deepcopy(BENCHMARK_SETTINGS["mbpp"])
    parameters = {key: value for key, value in setting.items() if key not in {"samples", "coverage"}}
    parameters["n"] = 10
    sampling = {"policy": "fixture", "parameters": parameters}
    samples = {"sample_ids": ["fixture-1"], "denominator": 1, "predictions_per_sample": 10,
        "split": "test", "labels_or_tests_ref": tests, "sampling": sampling, "budget": {"fixture": True}}
    contract = {"sample_manifest_ref": put(tmp_path, "native/samples.json", samples),
        "native_definition_ref": put(tmp_path, "native/definition.json", {}),
        "labels_or_tests_ref": tests, "split": "test", "sampling": sampling, "budget": {"fixture": True}}
    protocol = {"native_eval_contracts": {"g": contract}}
    N = SimpleNamespace(contract_for_group=lambda p, group: p["native_eval_contracts"][group])
    monkeypatch.setattr(admission, "_modules", lambda: (None, None, N, None))
    node = {"unit": {"benchmark": "mbpp", "split": "full", "expected_samples": 10, "sampling": setting}}
    return protocol, node, samples


def test_native_binding_uses_actual_tests_and_ids_not_name_substrings(tmp_path, native_inputs):
    protocol, node, _ = native_inputs
    refs = child.validate_node_native_inputs(tmp_path, "data", node, protocol, "g")
    assert protocol["native_eval_contracts"]["g"]["labels_or_tests_ref"] in refs


def test_wrong_native_ids_are_rejected_even_with_matching_count(tmp_path, native_inputs):
    protocol, node, samples = native_inputs
    samples["sample_ids"] = ["other-fixture"]
    protocol["native_eval_contracts"]["g"]["sample_manifest_ref"] = put(tmp_path, "native/wrong.json", samples)
    with pytest.raises(admission.AdmissionError, match="EXACT_NATIVE_INVENTORY"):
        child.validate_node_native_inputs(tmp_path, "data", node, protocol, "g")


def test_native_decoder_change_is_rejected(tmp_path, native_inputs):
    protocol, node, samples = native_inputs
    protocol["native_eval_contracts"]["g"]["sampling"]["parameters"]["temperature"] = 0.
    with pytest.raises(admission.AdmissionError, match="NATIVE_DECODER_MISMATCH"):
        child.validate_node_native_inputs(tmp_path, "data", node, protocol, "g")


def test_native_byte_identity_is_required_not_just_same_ids(tmp_path, native_inputs):
    protocol, node, samples = native_inputs
    changed = put(tmp_path, "native/other-tests.json", [{"task_id": "fixture-1", "changed": True}])
    protocol["native_eval_contracts"]["g"]["labels_or_tests_ref"] = changed
    samples["labels_or_tests_ref"] = changed
    protocol["native_eval_contracts"]["g"]["sample_manifest_ref"] = put(tmp_path, "native/changed.json", samples)
    with pytest.raises(admission.AdmissionError, match="NATIVE_LABEL_BINDING_MISMATCH"):
        child.validate_node_native_inputs(tmp_path, "data", node, protocol, "g")
