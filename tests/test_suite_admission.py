"""Admission metadata checks only: no generated code, scorer or GPU is executed.

The temporary definitions below are engineering contract fixtures. They are not
benchmark qualification, parent-problem evidence, or a scientific gate record.
"""
import copy
import hashlib
import json
import sys

import pytest

from recursive_ssd.suite_admission import (
    AdmissionError,
    prepare_native_assets,
    build_native_contract,
    build_scientific_plan,
    charge_budget,
    initialize_budget,
    prepare_native_protocol,
    remaining_budget,
    reserve_bundle,
    reserve_calibration,
)


def put(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    return {"path": name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


@pytest.fixture
def native_fixture(tmp_path):
    source = put(tmp_path, "engineering/source.json", {"scope": "metadata fixture only"})
    code = put(tmp_path, "engineering/code.json", {"scope": "never executed"})
    tests = put(tmp_path, "engineering/tests.json", {"released_ids_fixture": ["case-1", "case-2"]})
    sampling = {"policy": "fixed samples", "parameters": {"n": 10}}
    budget = {"seconds": 60}
    samples = put(tmp_path, "engineering/samples.json", {
        "benchmark_id": "engineering-contract-fixture", "benchmark_revision": "metadata-v1",
        "split": "test", "sample_ids": ["case-1", "case-2"], "denominator": 2,
        "predictions_per_sample": 10, "labels_or_tests_ref": tests,
        "sampling": sampling, "budget": budget,
    })
    scorer = {"kind": "official", "identity": "metadata only; never executed", "revision": "v1",
              "source_refs": [source], "code_refs": [code],
              "command": [sys.executable, "engineering/not-an-executable.py", "{predictions}"],
              "cwd": ".", "output": {"format": "json", "source": "stdout"},
              "denominator_path": ["task_count"]}
    definition = put(tmp_path, "engineering/definition.json", {
        "benchmark_id": "engineering-contract-fixture", "benchmark_revision": "metadata-v1",
        "published_at": "2020-01-01T00:00:00Z", "source_url": "https://example.org/engineering-fixture",
        "primary_metric": "pass@1", "metrics": [{"name": "pass@1", "output_path": ["pass@1"]},
                                                     {"name": "pass@10", "output_path": ["pass@10"]}],
        "splits": [{"name": "test", "sample_manifest_ref": samples, "labels_or_tests_ref": tests}],
        "prediction_format": {"format": "jsonl", "records_path": [], "id_path": ["task_id"]},
        "sampling": sampling, "budget": budget, "scorer": scorer, "publication_refs": [source],
    })
    arms = {role: {"name": name, "revision": "engineering-v1", "implementation_refs": [code]}
            for role, name in (("treatment", "hard"), ("baseline", "initial"), ("full_soft", "full_soft"))}
    qualification = {"metric": "pass@1", "operator": "ge", "threshold": 0,
                     "reference_ref": source}
    spec = {"native_definition_ref": definition, "baseline_qualification": qualification,
            "control_qualifications": [{**qualification, "role": "full_soft"}]}
    snapshot = put(tmp_path, "engineering/snapshot.json", {"scope": "engineering only"})
    bindings = {"protocol": {
        "schema_id": "gate-a-protocol", "schema_version": "1.0.0", "protocol_id": "metadata-fixture",
        "evidence_mode": "prospective_confirmatory", "evidence_snapshot_ref": snapshot,
        "min_valid_runs": 1, "guardrails": {}, "criteria": [{
            "id": "metadata", "metric": "pass@1", "direction": "maximize", "min_effect": 0,
            "inclusive": True, "indispensable": True, "min_runs": 1,
            "uncertainty": "none", "confidence": 0.95}],
    }, "qualification": {"purpose": "baseline-calibration", "allowed_arm_roles": list(arms)}}
    return {"root": tmp_path, "samples": samples, "tests": tests, "code": code,
            "arms": arms, "spec": spec, "bindings": bindings}


def compile_fixture(fixture):
    return prepare_native_protocol(
        fixture["root"], {"main": fixture["samples"]}, fixture["arms"], [17], ["main"],
        {"main": fixture["spec"]}, fixture["bindings"], "engineering/protocol.json")


def plan_kwargs(f):
    return dict(run_id="metadata-test", command=[sys.executable, "-m", "recursive_ssd.suite", "qualify"],
                code_refs=[f["code"]], input_refs=[f["samples"], f["tests"]],
                output_paths=["engineering/result.json"], seed=17, group="main", arm_role="baseline",
                provenance={"git_revision": "engineering", "model_revision": "engineering-no-model",
                            "data_revision": "engineering", "environment_digest": "a" * 64}, seconds=30)


def test_contract_derives_identity_and_exact_metrics_from_pinned_definition(native_fixture):
    f = native_fixture
    result = build_native_contract(f["root"], f["samples"], f["arms"], f["spec"])
    assert result["metrics"][-1] == {"name": "pass@10", "output_path": ["pass@10"]}
    assert result["sample_manifest_ref"] == f["samples"]
    assert result["labels_or_tests_ref"] == f["tests"]
    assert result["contrasts"] == {"treatment": "hard", "baseline": "initial", "controls": ["full_soft"]}


def test_compiler_pins_inputs_and_never_certifies_design(native_fixture):
    f = native_fixture
    protocol = compile_fixture(f)
    assert protocol["required_groups"] == ["main"]
    assert protocol["seed_policy"]["seeds"] == [17]
    assert "design_verified" not in json.dumps(protocol)
    assert "frozen_at" not in protocol
    assert "protocol_digest" in protocol
    altered = copy.deepcopy(f["arms"])
    altered["treatment"]["revision"] = "changed"
    with pytest.raises(AdmissionError, match="IMMUTABLE"):
        prepare_native_protocol(f["root"], {"main": f["samples"]}, altered, [17], ["main"],
                                {"main": f["spec"]}, f["bindings"], "engineering/protocol.json")


def test_changed_native_samples_or_denominator_are_rejected(native_fixture):
    f = native_fixture
    path = f["root"] / f["samples"]["path"]
    body = json.loads(path.read_text())
    body["denominator"] = 1
    f["samples"] = put(f["root"], f["samples"]["path"], body)
    with pytest.raises(AdmissionError, match="DENOMINATOR|MANIFEST"):
        build_native_contract(f["root"], f["samples"], f["arms"], f["spec"])


def test_stale_scorer_ref_is_rejected(native_fixture):
    f = native_fixture
    (f["root"] / f["code"]["path"]).write_text("changed source\n")
    with pytest.raises(AdmissionError, match="DIGEST_MISMATCH"):
        compile_fixture(f)


def test_asset_adapter_preserves_full_release_and_marks_project_subset(native_fixture):
    f = native_fixture
    # Metadata fixture checks exact selection; no evaluator or solution is used.
    rows = [{"task_id": "HumanEval/" + str(i), "prompt": "metadata"} for i in range(164)]
    ranked = sorted(rows, key=lambda row: int(hashlib.sha256(
        ("eval-split-v1|" + row["task_id"]).encode()).hexdigest()[:15], 16))
    full_path = f["root"] / "assets/full.jsonl"
    full_path.parent.mkdir(parents=True)
    full_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    selected_path = f["root"] / "assets/dev.jsonl"
    selected_path.write_text("".join(json.dumps(row) + "\n" for row in ranked[:32]))
    split_ref = put(f["root"], "engineering/split.json", {
        "dev_ids": [r["task_id"] for r in ranked[:32]],
        "confirm_ids": [r["task_id"] for r in ranked[32:]],
    })
    files = {str(p.relative_to(full_path.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in (full_path, selected_path)}
    assets_ref = put(f["root"], "assets/benchmarks-manifest.json", {
        "version": "clean-v2-native-benchmarks-1", "files": files,
        "benchmarks": {"humaneval": {"splits": {
            "full": {"path": "full.jsonl", "count": 164, "task_ids": [r["task_id"] for r in rows]},
            "dev": {"path": "dev.jsonl", "count": 32, "task_ids": [r["task_id"] for r in ranked[:32]]},
        }}},
    })
    definition = json.loads((f["root"] / f["spec"]["native_definition_ref"]["path"]).read_text())
    definition.update(benchmark_id="humaneval", benchmark_revision="v0.1.10")
    definition.pop("splits")
    definition["upstream_split"] = "test"
    result = prepare_native_assets(f["root"], assets_ref,
        groups={"he-dev": {"benchmark": "humaneval", "split": "dev"}},
        definitions={"he-dev": definition}, output_dir="engineering/native",
        human_eval_split_ref=split_ref, subset_support_ref=f["code"])
    samples = json.loads((f["root"] / result["benchmark_manifest"]["he-dev"]["path"]).read_text())
    assert samples["split"] == "test/project-dev"
    assert samples["denominator"] == 32
    assert samples["sample_ids"] == [r["task_id"] for r in ranked[:32]]
    adapter = json.loads((f["root"] / result["split_adapter_refs"]["he-dev"]["path"]).read_text())
    assert adapter["source_count"] == 164
    assert adapter["upstream_split"] == "test"
    assert adapter["project_split"] == "dev"
    assert adapter["scope"] == "project-selected published tasks; native tests and scoring retained"
    # Substituting any row, even with the same ID, must fail original-test parity.
    ranked[0]["prompt"] = "changed"
    selected_path.write_text("".join(json.dumps(row) + "\n" for row in ranked[:32]))
    with pytest.raises(AdmissionError, match="DIGEST_MISMATCH"):
        prepare_native_assets(f["root"], assets_ref,
            groups={"he-dev": {"benchmark": "humaneval", "split": "dev"}},
            definitions={"he-dev": definition}, output_dir="engineering/other",
            human_eval_split_ref=split_ref, subset_support_ref=f["code"])


def test_qualification_is_a_scientific_native_plan_not_an_engineering_escape(native_fixture):
    f = native_fixture
    compile_fixture(f)
    ref = {"path": "engineering/protocol.json", "sha256": hashlib.sha256(
        (f["root"] / "engineering/protocol.json").read_bytes()).hexdigest()}
    plan = build_scientific_plan(f["root"], protocol_ref=ref, verification_ref=None, **plan_kwargs(f))
    assert plan["purpose"] == "scientific"
    assert plan["evidence_mode"] == "developmental"
    assert plan["limits"]["wall_time_seconds"] == 30
    assert plan["provenance"]["admission_scope"] == "baseline-calibration"


def test_candidate_cannot_hide_in_qualification_protocol(native_fixture):
    f = native_fixture
    f["arms"]["treatment"]["name"] = "M03"
    with pytest.raises(AdmissionError, match="QUALIFICATION_ARM"):
        compile_fixture(f)


def test_candidate_requires_current_batch_and_canonical_ledger(native_fixture):
    f = native_fixture
    protocol = compile_fixture(f)
    # A stored boolean is never accepted as a method discovery or ledger proof.
    protocol.pop("suite_qualification")
    protocol["design_verified"] = True
    ref = put(f["root"], "engineering/not-a-candidate-proof.json", protocol)
    with pytest.raises(AdmissionError, match="METHOD_DISCOVERY_REQUIRED"):
        build_scientific_plan(f["root"], protocol_ref=ref, verification_ref=None, **plan_kwargs(f))


def test_cannot_run_an_undeclared_seed_or_native_arm(native_fixture):
    f = native_fixture
    compile_fixture(f)
    ref = {"path": "engineering/protocol.json", "sha256": hashlib.sha256(
        (f["root"] / "engineering/protocol.json").read_bytes()).hexdigest()}
    args = plan_kwargs(f)
    args["seed"] = 23
    with pytest.raises(AdmissionError, match="FROZEN_INVENTORY"):
        build_scientific_plan(f["root"], protocol_ref=ref, verification_ref=None, **args)


def test_budget_resume_preserves_original_deadline_and_cumulative_use(tmp_path):
    path = tmp_path / "budget.json"
    original = "2026-10-06T16:00:00+01:00"
    first = initialize_budget(path, original_start=original, cap_seconds=28800, already_used_seconds=123)
    again = initialize_budget(path, original_start="2026-10-06T15:00:00Z", cap_seconds=28800)
    assert again == first
    assert again["absolute_end"] == "2026-10-06T23:00:00Z"
    assert again["cumulative_used_seconds"] == 123
    assert remaining_budget(again, now="2026-10-06T22:59:30Z") == 30
    with pytest.raises(AdmissionError, match="BUDGET_RESTART"):
        initialize_budget(path, original_start="2026-10-06T16:00:00Z", cap_seconds=28800)


@pytest.mark.parametrize("start,cap", [("2026-10-06T15:00:00", 28800),
                                       ("2026-10-06T15:00:00Z", 28801),
                                       ("2026-10-06T15:00:00Z", float("nan"))])
def test_budget_requires_aware_original_start_and_finite_authorized_cap(tmp_path, start, cap):
    with pytest.raises(AdmissionError):
        initialize_budget(tmp_path / "budget.json", original_start=start, cap_seconds=cap)


def test_unmeasured_candidate_bundle_rejected_but_finite_baseline_calibration_allowed(tmp_path):
    path = tmp_path / "budget.json"
    initialize_budget(path, original_start="2026-10-06T15:00:00Z")
    with pytest.raises(AdmissionError, match="CALIBRATION_REQUIRED"):
        reserve_bundle(tmp_path, path, bundle_id="M03-complete", cell_ids=["a", "b"],
                       calibration_ref=None, expected_bindings={}, now="2026-10-06T15:01:00Z")
    reservation = reserve_calibration(path, calibration_id="initial", bound_seconds=300,
                                      arm_ids=["initial", "hard", "full_soft"],
                                      now="2026-10-06T15:01:00Z")
    assert reservation["seconds"] == 300
    with pytest.raises(AdmissionError, match="QUALIFICATION_ARM"):
        reserve_calibration(path, calibration_id="bad", bound_seconds=300, arm_ids=["M03"],
                            now="2026-10-06T15:01:00Z")


def test_budget_accounts_failed_attempts_once_and_requires_real_receipt_ref(tmp_path):
    path = tmp_path / "budget.json"
    initialize_budget(path, original_start="2026-10-06T15:00:00Z")
    reserve_calibration(path, calibration_id="initial", bound_seconds=300, arm_ids=["initial"],
                        now="2026-10-06T15:01:00Z")
    # Metadata fixture; validates accounting only and never enters scientific evidence.
    receipt = put(tmp_path, "engineering/failed-receipt.json", {
        "schema_id": "experiment-run-receipt", "schema_version": "1.0.0", "run_id": "engineering",
        "plan_digest": "a" * 64, "purpose": "scientific", "evidence_mode": "developmental",
        "status": "failed", "attempts": [], "started_at": "2026-10-06T15:01:00Z",
        "completed_at": "2026-10-06T15:03:00Z", "resources": {"seconds": 120},
        "provenance": {"git_revision": "engineering", "model_revision": "engineering",
                       "data_revision": "engineering", "environment_digest": "b" * 64,
                       "admission_scope": "baseline-calibration"},
        "gate_advanced": False,
    })
    first = charge_budget(tmp_path, path, reservation_id="initial", receipt_ref=receipt)
    assert first["cumulative_used_seconds"] == 120
    assert charge_budget(tmp_path, path, reservation_id="initial", receipt_ref=receipt) == first
    assert not first["reservations"]


def test_complete_bundle_reserves_remaining_cells_across_incremental_charges(tmp_path):
    # All receipts in this test are engineering metadata doubles. No scientific
    # plan is created/admitted, and no observed GPU performance is asserted.
    path = tmp_path / "budget.json"
    initialize_budget(path, original_start="2026-10-06T15:00:00Z")
    code = put(tmp_path, "engineering/never-executed.json", {"engineering_fixture": True})
    raw = put(tmp_path, "engineering/not-host-output.json", {"engineering_fixture": True})
    assets = put(tmp_path, "engineering/not-dataset.json", {"engineering_fixture": True})
    host = put(tmp_path, "engineering/host-metadata-fixture.json", {
        "gpu_count": 1, "gpu_model": "RTX 2080 Ti", "observed_at": "2026-10-06T15:00:00Z",
        "evidence_refs": [raw], "engineering_fixture": True,
    })
    bindings = {"model_revision": "engineering-no-model", "environment_digest": "b" * 64,
                "benchmark_manifest_ref": assets, "code_digest": hashlib.sha256(json.dumps(
                    [code], sort_keys=True, separators=(",", ":")).encode()).hexdigest()}

    def receipt(name, seconds, scope):
        return put(tmp_path, "engineering/" + name + ".json", {
            "schema_id": "experiment-run-receipt", "schema_version": "1.0.0", "run_id": name,
            "plan_digest": "a" * 64, "purpose": "scientific", "evidence_mode": "developmental",
            "status": "completed", "attempts": [{"seconds": seconds,
                "stdout_ref": raw, "stderr_ref": raw, "code_refs": [code]}],
            "started_at": "2026-10-06T15:01:00Z", "completed_at": "2026-10-06T15:03:00Z",
            "resources": {"seconds": seconds}, "gate_advanced": False,
            "provenance": {"git_revision": "engineering", "data_revision": "engineering",
                "model_revision": bindings["model_revision"], "environment_digest": bindings["environment_digest"],
                "admission_scope": scope},
        })

    measured = receipt("calibration-metadata", 10, "baseline-calibration")
    calibration = put(tmp_path, "engineering/calibration.json", {
        "version": "suite-resource-calibration-v1", "host_ref": host, "bindings": bindings,
        "costs": {"cell-a": {"upper_seconds": 40, "receipt_ref": measured},
                  "cell-b": {"upper_seconds": 60, "receipt_ref": measured}},
    })
    bundle = reserve_bundle(tmp_path, path, bundle_id="complete", cell_ids=["cell-a", "cell-b"],
                            calibration_ref=calibration, expected_bindings=bindings,
                            now="2026-10-06T15:01:00Z")
    assert bundle["seconds"] == 100
    first_receipt = receipt("child-a-metadata", 20, "candidate")
    with pytest.raises(AdmissionError, match="UNCHARGED_CELLS"):
        charge_budget(tmp_path, path, reservation_id="complete", cell_id="cell-a", receipt_ref=first_receipt)
    first = charge_budget(tmp_path, path, reservation_id="complete", cell_id="cell-a",
                          receipt_ref=first_receipt, final=False)
    assert first["cumulative_used_seconds"] == 20
    assert first["reservations"]["complete"]["seconds"] == 60
    resumed = reserve_bundle(tmp_path, path, bundle_id="complete", cell_ids=["cell-a", "cell-b"],
                             calibration_ref=calibration, expected_bindings=bindings,
                             now="2026-10-06T15:02:00Z")
    assert resumed["seconds"] == 60
    final = charge_budget(tmp_path, path, reservation_id="complete", cell_id="cell-b",
                          receipt_ref=receipt("child-b-metadata", 30, "candidate"))
    assert final["cumulative_used_seconds"] == 50
    assert not final["reservations"]
    with pytest.raises(AdmissionError, match="COMPLETED_RESERVATION_ID_REUSED"):
        reserve_bundle(tmp_path, path, bundle_id="complete", cell_ids=["cell-a", "cell-b"],
                       calibration_ref=calibration, expected_bindings=bindings,
                       now="2026-10-06T15:04:00Z")
