"""Planned Local metadata tests; generated_unexecuted in the Web handoff.

These checks never qualify an algorithm or a benchmark. They keep the reported
bootstrap gap tied to the actual complete comparison catalogue and preserve the
existing candidate execution boundary.
"""
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import pytest

from recursive_ssd.io import object_hash
from recursive_ssd.suite_admission import (
    AdmissionError, QUALIFICATION_ARMS, qualification_dependency_report, qualification_sources,
)
from recursive_ssd.suite_design import ORDER, make_suite


def comparator(report, candidate, arm):
    return next(row for row in report["bundles"][candidate]["required_comparators"]
                if row["arm_id"] == arm)


def test_report_retains_every_required_comparator_without_granting_dispatch():
    suite = make_suite("development")
    report = qualification_dependency_report(suite)
    assert set(report["bundles"]) == set(ORDER)
    assert report["scientific_dispatch_ready"] is False
    assert report["gate_advanced"] is False
    for candidate, bundle in suite["method_bundles"].items():
        rows = report["bundles"][candidate]["required_comparators"]
        assert [row["arm_id"] for row in rows] == [arm for arm in bundle["arms"] if arm != candidate]
        assert report["bundles"][candidate]["full_comparator_bootstrap_implemented"] is False
    assert set(report["bootstrap_arm_allowlist"]) == QUALIFICATION_ARMS


def test_candidate_as_control_keeps_its_candidate_identity():
    report = qualification_dependency_report(make_suite("development"))
    row = comparator(report, "M02", "M01")
    assert row["selected_candidate_as_control"] == "M01"
    assert row["selected_method_implementation"] == "M01"
    assert row["existing_bootstrap_eligible"] is False
    assert "CANDIDATE_METHOD_REQUIRES_OWN_VERIFIED_DESIGN" in row["blockers"]


def test_ordinary_scope_does_not_use_the_old_four_name_allowlist_as_qualification():
    report = qualification_dependency_report(make_suite("development"))
    for arm in ("arithmetic_anchor", "lower_lr", "fixed_data", "gkd_fkl_update"):
        row = comparator(report, "M03", arm)
        assert row["existing_bootstrap_eligible"] is False
        assert row["ordinary_comparator_scope_eligible"] is True
    assert report["scientific_dispatch_ready"] is False


@pytest.mark.parametrize("candidate,arm,method", [
    ("M03", "floor_zero", "M03"),
    ("M06", "head_retention_zero", "M06"),
    ("M14", "allocation_without_weight_correction", "M14"),
])
def test_control_label_does_not_hide_selected_method(candidate, arm, method):
    row = comparator(qualification_dependency_report(make_suite("development")), candidate, arm)
    assert row["selected_candidate_as_control"] is None
    assert row["selected_method_implementation"] == method
    assert row["existing_bootstrap_eligible"] is False


@pytest.mark.parametrize("candidate,arm,key", [
    ("M06", "temperature_matched_head", "head_temperature"),
    ("M13", "full_soft_lr_matched", "m13_step_scale"),
    ("M13", "full_soft_equal_clock", "clock_m13"),
    ("M15", "hard_equal_clock", "clock_m15"),
])
def test_measured_controls_expose_the_exact_candidate_dependency(candidate, arm, key):
    row = comparator(qualification_dependency_report(make_suite("development")), candidate, arm)
    assert row["calibration_dependencies"] == [{
        "key": key, "source_candidate_id": candidate,
        "source_kind": "candidate_training_receipt", "requires_real_source_refs": True,
    }]
    assert "VERIFIED_CALIBRATION_CHILD_REQUIRED" in row["blockers"]
    assert row["requires_verified_child_dependencies"] is True
    assert row["ordinary_comparator_scope_eligible"] is True


def test_tuned_control_does_not_invent_an_external_selection():
    row = comparator(qualification_dependency_report(make_suite("development")),
                     "M03", "arithmetic_anchor_tuned")
    assert row["calibration_dependencies"][0]["source_candidate_id"] is None
    assert "ACTUAL_COMPLETE_DEVELOPMENT_SELECTION_REQUIRED" in row["blockers"]


def test_self_consistent_digest_cannot_remove_a_required_control():
    suite = deepcopy(make_suite("development"))
    suite["method_bundles"]["M03"]["arms"].remove("floor_zero")
    suite["suite_digest"] = object_hash({key: value for key, value in suite.items() if key != "suite_digest"})
    with pytest.raises(AdmissionError, match="QUALIFICATION_CATALOGUE_BINDING_MISMATCH"):
        qualification_dependency_report(suite)


def test_per_role_qualification_reuses_one_source_once_and_requires_every_control():
    pair = {"protocol_ref": {"path": "q.json", "sha256": "1" * 64},
            "manifest_ref": {"path": "m.json", "sha256": "2" * 64}}
    protocol = {"required_groups": ["dev"], "native_eval_contract": {
        "arm_requirements": {role: {} for role in ("treatment", "baseline", "arithmetic_anchor")}},
        "suite_qualification_evidence": {"dev": {"comparators": {
            "baseline": {**pair, "source_arm_role": "treatment"},
            "arithmetic_anchor": {**pair, "source_arm_role": "arithmetic_anchor"},
        }}}}
    sources = qualification_sources(protocol, "dev")
    assert len(sources) == 1
    assert sources[0]["roles"] == {"baseline": "treatment", "arithmetic_anchor": "arithmetic_anchor"}
    del protocol["suite_qualification_evidence"]["dev"]["comparators"]["arithmetic_anchor"]
    with pytest.raises(AdmissionError, match="QUALIFICATION_COMPARATOR_INVENTORY_MISMATCH"):
        qualification_sources(protocol, "dev")


def test_per_role_source_cannot_introduce_main_treatment_or_untyped_role():
    pair = {"protocol_ref": {"path": "q.json", "sha256": "1" * 64},
            "manifest_ref": {"path": "m.json", "sha256": "2" * 64}}
    protocol = {"required_groups": ["dev"], "native_eval_contract": {
        "arm_requirements": {role: {} for role in ("treatment", "baseline", "full_soft")}},
        "suite_qualification_evidence": {"dev": {"comparators": {
            "baseline": {**pair, "source_arm_role": "baseline"},
            "full_soft": {**pair, "source_arm_role": None},
        }}}}
    with pytest.raises(AdmissionError, match="QUALIFICATION_SOURCE_BINDING_INVALID"):
        qualification_sources(protocol, "dev")
    protocol["suite_qualification_evidence"]["dev"]["comparators"]["full_soft"]["source_arm_role"] = "full_soft"
    protocol["suite_qualification_evidence"]["dev"]["comparators"]["treatment"] = {**pair, "source_arm_role": "treatment"}
    with pytest.raises(AdmissionError, match="QUALIFICATION_COMPARATOR_INVENTORY_MISMATCH"):
        qualification_sources(protocol, "dev")


def test_legacy_whole_manifest_form_retains_exact_comparator_inventory():
    pair = {"protocol_ref": {"path": "q.json", "sha256": "1" * 64},
            "manifest_ref": {"path": "m.json", "sha256": "2" * 64}}
    protocol = {"required_groups": ["dev"], "native_eval_contract": {
        "arm_requirements": {role: {} for role in ("treatment", "baseline", "full_soft")}},
        "suite_qualification_evidence": {"dev": pair}}
    assert qualification_sources(protocol, "dev") == [{
        "record": pair, "roles": {"baseline": None, "full_soft": None}}]


@pytest.fixture
def comparator_scope_fixture(tmp_path, monkeypatch):
    """Metadata logic only; the fake verifier never supplies scientific evidence."""
    from recursive_ssd import suite_admission as admission
    C, R, N, _ = admission._modules()
    def put(name, value):
        path = tmp_path / name
        path.write_text(json.dumps(value) + "\n")
        return {"path": name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    code = put("code.json", {"engineering_fixture": True})
    implementation = put("implementation.json", {"code_refs": [code]})
    batch = {"candidates": [{"candidate_id": "M03", "implementation_ref": implementation}]}
    verifier = SimpleNamespace(read=lambda *args: batch,
        before_action=lambda *args: {"workflow_boundary": {"ready": True, "reason_codes": []}})
    monkeypatch.setattr(admission, "_modules", lambda: (C, R, N, verifier))
    suite = make_suite("development")
    arms = {"treatment": "hard", "baseline": "base", "arithmetic_anchor": "arithmetic_anchor"}
    protocol = {"seed_policy": {"seeds": [17]}, "native_eval_contract": {"arm_requirements": {
        role: {"name": "initial" if arm == "base" else arm, "implementation_refs": [code]}
        for role, arm in arms.items()}},
        "suite_qualification": {"purpose": "comparator-qualification", "allowed_arm_roles": list(arms),
            "suite_ref": put("suite.json", suite), "bundle_id": "M03", "arm_bindings": arms,
            "method_verification_ref": put("batch.json", batch), "config": {}, "calibration": {},
            "execution_provenance": {"model_revision": suite["model_revision"],
                                     "data_revision": "fixture", "environment_digest": "fixture"}}}
    return admission, tmp_path, protocol, verifier


def test_catalogue_scope_accepts_required_ordinary_comparator_with_code_boundary(comparator_scope_fixture):
    admission, root, protocol, _ = comparator_scope_fixture
    scope = admission._qualification(protocol, root)
    assert scope["arm_bindings"]["arithmetic_anchor"] == "arithmetic_anchor"
    assert "method_discovery" not in protocol


def test_catalogue_scope_rejects_candidate_alias_despite_control_label(comparator_scope_fixture):
    admission, root, protocol, _ = comparator_scope_fixture
    protocol["suite_qualification"]["arm_bindings"]["arithmetic_anchor"] = "floor_zero"
    protocol["native_eval_contract"]["arm_requirements"]["arithmetic_anchor"]["name"] = "floor_zero"
    with pytest.raises(AdmissionError, match="QUALIFICATION_SELECTED_METHOD_REQUIRES_OWN_DESIGN"):
        admission._qualification(protocol, root)


def test_catalogue_scope_rejects_saved_or_stale_code_readiness(comparator_scope_fixture):
    admission, root, protocol, verifier = comparator_scope_fixture
    verifier.before_action = lambda *args: {"workflow_boundary": {"ready": False, "reason_codes": ["STALE_CODE"]}}
    protocol["code_verified"] = True
    with pytest.raises(AdmissionError, match="QUALIFICATION_CURRENT_CODE_EVIDENCE_REQUIRED"):
        admission._qualification(protocol, root)


def test_catalogue_scope_rejects_unrequired_arm_and_implementation_substitution(comparator_scope_fixture):
    admission, root, protocol, _ = comparator_scope_fixture
    changed = deepcopy(protocol)
    changed["suite_qualification"]["arm_bindings"]["arithmetic_anchor"] = "head_shape_only"
    with pytest.raises(AdmissionError, match="QUALIFICATION_NOT_A_REQUIRED_COMPARATOR"):
        admission._qualification(changed, root)
    protocol["native_eval_contract"]["arm_requirements"]["arithmetic_anchor"]["implementation_refs"] = []
    with pytest.raises(AdmissionError, match="QUALIFICATION_CURRENT_CODE_BINDING_MISMATCH"):
        admission._qualification(protocol, root)


def test_reviewed_child_reserves_whole_inventory_before_first_node(tmp_path, monkeypatch):
    from recursive_ssd import suite_admission as admission
    from recursive_ssd import suite_precursor
    path = tmp_path / "budget.json"
    admission.initialize_budget(path, original_start="2026-10-06T15:00:00Z", cap_seconds=28800,
                                already_used_seconds=0)
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps({"fixture": "not an executable scientific protocol"}))
    protocol_ref = {"path": "protocol.json", "sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest()}
    observed = []
    monkeypatch.setattr(admission, "validate_candidate_binding", lambda *args: observed.append(args))
    monkeypatch.setattr(suite_precursor, "child_scope", lambda *args: ({
        "node_ids": ["train", "derive"], "node_seconds": {"train": 90., "derive": 30.}}, {}, {}))
    result = admission.reserve_calibration_child(tmp_path, path, protocol_ref=protocol_ref,
        verification_ref={"fixture": "current design verified by mocked boundary"}, node_id="train",
        reservation_id="finite-child", now="2026-10-06T15:01:00Z")
    assert observed
    assert result["seconds"] == 120.
    assert result["cell_ids"] == ["train", "derive"]
    assert result["cell_seconds"] == {"train": 90., "derive": 30.}
    assert result["purpose"] == "candidate-calibration-child"
    with pytest.raises(AdmissionError, match="ORIGINAL_BUDGET_EXHAUSTED"):
        admission.reserve_calibration_child(tmp_path, path, protocol_ref=protocol_ref,
            verification_ref={"fixture": "same mocked boundary"}, node_id="derive",
            reservation_id="finite-child", now="2026-10-07T00:00:00Z")


def test_tuning_arithmetic_control_family_has_exact_scope_without_candidate_escape(comparator_scope_fixture):
    admission, root, protocol, _ = comparator_scope_fixture
    suite = make_suite("tuning")
    path = root / "tuning-suite.json"
    path.write_text(json.dumps(suite))
    scope = protocol["suite_qualification"]
    scope["suite_ref"] = {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    family = suite["tuning_groups"]["arithmetic_anchor"]
    code_refs = protocol["native_eval_contract"]["arm_requirements"]["baseline"]["implementation_refs"]
    arms = {"treatment": family[0], "baseline": "base", family[1]: family[1]}
    scope.update(allowed_arm_roles=list(arms), arm_bindings=arms,
                 tuning_budget={"trial_wall_seconds": 180., "source_refs": code_refs})
    protocol["native_eval_contract"]["arm_requirements"] = {
        role: {"name": "initial" if arm == "base" else arm, "implementation_refs": code_refs}
        for role, arm in arms.items()}
    assert admission._qualification(protocol, root)["arm_bindings"]["treatment"] == family[0]
    scope["arm_bindings"]["treatment"] = suite["tuning_groups"]["M03"][0]
    with pytest.raises(AdmissionError, match="QUALIFICATION_NOT_A_REQUIRED_COMPARATOR"):
        admission._qualification(protocol, root)
    scope["arm_bindings"]["treatment"] = family[0]
    scope["tuning_budget"] = None
    with pytest.raises(AdmissionError, match="QUALIFICATION_FROZEN_TUNING_ALLOWANCE_REQUIRED"):
        admission._qualification(protocol, root)
