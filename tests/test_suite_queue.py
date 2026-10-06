"""Queue metadata checks; fixtures are not GPU/scientific evidence."""
import copy
import hashlib
import json
from datetime import datetime, timedelta, timezone

import pytest


def api():
    from recursive_ssd import suite_queue
    return suite_queue


def put(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    return {"path": name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def test_complete_catalog_keeps_all_methods_controls_and_recursive_dependencies():
    from recursive_ssd.suite_design import ORDER, make_suite
    suite = make_suite("development")
    nodes = api().compile_nodes(suite)
    assert set().union(*(set(node["bundles"]) for node in nodes)) == set(ORDER)
    assert len([n for n in nodes if n["kind"] == "evaluate"]) == len(suite["evaluation_units"])
    by_id = {node["node_id"]: node for node in nodes}
    for node in nodes:
        if node["kind"] == "train" and node["round"] > 1:
            parent = by_id[node["parent_node_id"]]
            assert parent["round"] == node["round"] - 1
            assert parent["trajectory_id"] == node["trajectory_id"]
            assert parent["node_id"] in node["depends_on"]
        if node["kind"] == "evaluate" and node["round"]:
            assert by_id[node["parent_node_id"]]["round"] == node["round"]
    assert all("checkpoint" not in node for node in nodes)
    assert any(n["arm_id"] == "full_soft_equal_clock" for n in nodes)
    assert any(n["arm_id"] == "gkd_rkl_update" for n in nodes)


def test_confirmation_and_boundary_keep_complete_benchmark_inventory():
    from recursive_ssd.suite_design import make_suite
    nodes = api().compile_nodes(make_suite("confirmation", finalists=["M03", "M15"]))
    evaluations = [n for n in nodes if n["kind"] == "evaluate"]
    assert {n["unit"]["benchmark"] for n in evaluations} == {"humaneval", "mbpp", "livecodebench"}
    assert {n["unit"]["seed"] for n in evaluations} == {23, 47, 71, 101, 131}
    assert {n["round"] for n in evaluations} == {0, 3}
    boundary = api().compile_nodes(make_suite("boundary", finalists=["M03"]))
    assert max(n["round"] for n in boundary) == 5


def test_dependency_failures_remain_pending_and_do_not_shrink_bundle():
    from recursive_ssd.suite_design import make_suite
    nodes = api().compile_nodes(make_suite("development"))
    state = {n["node_id"]: {"status": "pending", "attempts": []} for n in nodes}
    m03 = [n for n in nodes if n["arm_id"] == "M03" and n["kind"] == "train"]
    state[m03[0]["node_id"]]["status"] = "failed"
    available = api().ready_nodes(nodes, state, bundle="M03")
    assert m03[1]["node_id"] not in {n["node_id"] for n in available}
    summary = api().inventory_status(nodes, state)
    assert summary["bundles"]["M03"]["status"] == "incomplete"
    assert summary["bundles"]["M03"]["failed"] == 1
    assert summary["expected"] == len(nodes)


def test_dependency_cycle_and_fictional_completed_parent_are_rejected():
    nodes = [{"node_id": "a", "depends_on": ["b"]}, {"node_id": "b", "depends_on": ["a"]}]
    with pytest.raises(ValueError, match="DEPENDENCY_CYCLE"):
        api().validate_dependencies(nodes)
    with pytest.raises(ValueError, match="DEPENDENCY_RECEIPT_REQUIRED"):
        api().ready_nodes([{ "node_id": "b", "depends_on": ["a"], "bundles": ["M03"]}],
                          {"a": {"status": "completed"}, "b": {"status": "pending"}}, bundle="M03")


def test_queue_identity_rejects_changed_inputs_and_never_resets_original_deadline(tmp_path):
    from recursive_ssd.suite_design import make_suite
    suite = put(tmp_path, "inputs/suite.json", make_suite("development"))
    source = put(tmp_path, "source.py", {"version": 1})
    data = put(tmp_path, "inputs/benchmarks-manifest.json", {"files": {}})
    model = put(tmp_path, "inputs/model-manifest.json", {"model_revision": "engineering-fixture"})
    start = "2026-10-06T15:00:00Z"
    kw = dict(suite_ref=suite, benchmark_manifest_ref=data, model_manifest_ref=model,
              original_start=start, code_refs=[source], environment={"scope": "engineering fixture"})
    first = api().build_queue(tmp_path, "runs/q", **kw)
    assert first["identity"]["original_start"] == start
    again = api().build_queue(tmp_path, "runs/q", **kw)
    assert first["queue_digest"] == again["queue_digest"]
    with pytest.raises(ValueError, match="IDENTITY|RESTART"):
        api().build_queue(tmp_path, "runs/q", **{**kw, "original_start": "2026-10-06T16:00:00Z"})
    (tmp_path / "source.py").write_text("changed source")
    with pytest.raises(ValueError, match="DIGEST|IDENTITY"):
        api().verify_queue(tmp_path, "runs/q", environment=kw["environment"])


def test_expired_queue_cannot_materialize_even_a_ready_dependency(tmp_path):
    from recursive_ssd.suite_design import make_suite
    now = datetime.now(timezone.utc)
    suite = put(tmp_path, "inputs/suite.json", make_suite("development"))
    source = put(tmp_path, "source.py", {"version": 1})
    data = put(tmp_path, "inputs/benchmarks-manifest.json", {"files": {}})
    model = put(tmp_path, "inputs/model-manifest.json", {"version": "fixture"})
    env = {"scope": "engineering fixture"}
    api().build_queue(tmp_path, "runs/q", suite_ref=suite, benchmark_manifest_ref=data,
        model_manifest_ref=model, original_start=(now - timedelta(hours=9)).isoformat(),
        code_refs=[source], environment=env)
    with pytest.raises(ValueError, match="ORIGINAL_BUDGET_EXPIRED"):
        api().admit_bundle(tmp_path, "runs/q", "M03", {}, environment=env)


def test_selection_refuses_partial_or_unqualified_development_report():
    from recursive_ssd.suite_design import make_suite
    suite = make_suite("development")
    with pytest.raises(ValueError, match="COMPLETE_NATIVE_DEVELOPMENT"):
        api().select_finalists(suite, {"suite_digest": suite["suite_digest"],
                                     "inventory": {"complete": False}})
    confirmation = make_suite("confirmation", finalists=["M03"])
    with pytest.raises(ValueError, match="DEVELOPMENT_ONLY"):
        api().select_finalists(confirmation, {})


def test_finalist_selection_uses_strongest_baseline_noninferiority_and_cost_tie_break():
    from recursive_ssd.suite_design import make_suite
    suite = make_suite("development")
    report = {"suite_digest": suite["suite_digest"], "native_evidence_validated": True,
              "inventory": {"complete": True}, "observations": []}
    # Counts are artificial engineering fixtures; only decision arithmetic is tested.
    for unit in suite["evaluation_units"]:
        score = 4 if unit["arm_id"].startswith("M") else 2
        cost = 1. if unit["arm_id"] == "M04" else 2.
        report["observations"].append({**unit, "status": "completed", "optimization_valid": True,
            "training_seconds": cost, "per_task": {"task": {"n": 10, "correct": score}}})
    selected = api().select_finalists(suite, report)
    assert selected["finalists"] == ["M04", "M03"]
    assert selected["selection_scope"] == "development_only; no scientific verdict"
    broken = copy.deepcopy(report)
    broken["observations"] = broken["observations"][:-1]
    with pytest.raises(ValueError, match="COMPLETE_NATIVE_DEVELOPMENT"):
        api().select_finalists(suite, broken)


def test_duplicate_queue_state_or_fake_attempt_success_cannot_unlock_children(tmp_path):
    node = {"node_id": "child", "depends_on": ["parent"], "bundles": ["M03"]}
    with pytest.raises(ValueError, match="DEPENDENCY_RECEIPT_REQUIRED"):
        api().ready_nodes([node], {"parent": {"status": "completed", "attempts": []},
                                  "child": {"status": "pending"}}, bundle="M03")


def test_tuning_is_admitted_under_each_actual_candidate_and_head_is_not_retuned_on_confirmation():
    from recursive_ssd.suite_design import make_suite
    nodes = api().compile_nodes(make_suite("tuning"))
    assert set().union(*(set(n["bundles"]) for n in nodes)) == {"M03", "M01"}
    for node in nodes:
        if node["arm_id"].startswith(("M03__", "arithmetic_anchor__")):
            assert node["bundles"] == ["M03"]
        if node["arm_id"].startswith(("M01__", "arithmetic_same_smoothing__")):
            assert node["bundles"] == ["M01"]
    confirmation = api().compile_nodes(make_suite("confirmation", finalists=["M06"]))
    assert not any(n.get("calibration_key") == "head_temperature" for n in confirmation)
    assert any(n["arm_id"] == "temperature_matched_head" for n in confirmation)


def test_completed_flag_requires_exact_native_plan_and_recorded_output(tmp_path):
    native = put(tmp_path, "runs/native.json", {"status": "completed", "attempts": [{"output_refs": []}]})
    workload = put(tmp_path, "runs/output/receipt.json", {"status": "complete", "kind": "train"})
    state = {"nodes": {"train-r1": {"status": "completed", "receipt_ref": native,
             "workload_receipt_ref": workload, "attempts": [], "output_refs": []}}}
    with pytest.raises(ValueError, match="DEPENDENCY_NATIVE_PLAN_REQUIRED"):
        api()._completed_output(tmp_path, state, "train-r1")


def test_report_preserves_every_pending_native_cell_instead_of_reporting_empty_success(tmp_path):
    from recursive_ssd.suite_design import make_suite
    from recursive_ssd.harness import environment
    suite = put(tmp_path, "inputs/suite.json", make_suite("development"))
    source = put(tmp_path, "source.py", {"version": 1})
    data = put(tmp_path, "inputs/benchmarks-manifest.json", {"files": {}, "benchmarks": {
        "humaneval": {"splits": {"dev": {"task_ids": ["engineering-fixture-task"]}}}}})
    model = put(tmp_path, "inputs/model-manifest.json", {"files": {}})
    api().build_queue(tmp_path, "runs/q", suite_ref=suite, benchmark_manifest_ref=data,
        model_manifest_ref=model, original_start="2026-10-06T15:00:00Z", code_refs=[source], environment=environment())
    report = api().queue_report(tmp_path, "runs/q")
    expected = len(make_suite("development")["evaluation_units"])
    assert report["inventory"]["counts"]["expected"] == expected
    assert report["inventory"]["counts"]["pending"] == expected
    assert report["native_evidence_validated"] is False
    assert set(report["analyses"]) == set(make_suite("development")["candidate_ids"])
    assert all(result["decision"] == "INCONCLUSIVE" for result in report["analyses"].values())


def test_path_rebase_maps_only_the_frozen_project_and_keeps_receipt_hashes():
    value = {"project_root": "/old/project", "checkpoint": "/old/project/runs/a.pt",
             "unrelated": "/old/project-other/file", "nested": [{"path": "/old/project/data/a", "sha256": "f" * 64}]}
    result = api()._rebase_paths(value, "/old/project", "/attempt/workspace")
    assert result["checkpoint"] == "/attempt/workspace/runs/a.pt"
    assert result["unrelated"] == value["unrelated"]
    assert result["nested"][0]["sha256"] == value["nested"][0]["sha256"]


def test_new_stage_shares_original_authorization_and_cannot_erase_prior_reservations(tmp_path):
    from recursive_ssd import suite_admission
    start = datetime.now(timezone.utc).isoformat()
    api().initialize_authorization(tmp_path, "runs/tuning", original_start=start, already_used_seconds=37)
    path = api().authorization_path(tmp_path, "runs/tuning")
    suite_admission.reserve_calibration(path, calibration_id="real-host-probe", bound_seconds=100, arm_ids=["hard"])
    api().initialize_authorization(tmp_path, "runs/development", original_start=start, budget_from="runs/tuning")
    assert api().authorization_path(tmp_path, "runs/development") == path
    budget = json.loads(path.read_text())
    assert budget["initial_used_seconds"] == 37
    assert "real-host-probe" in budget["reservations"]
    assert budget["cumulative_used_seconds"] == 37
    with pytest.raises(ValueError, match="BUDGET_RESTART"):
        api().initialize_authorization(tmp_path, "runs/confirmation", budget_from="runs/tuning",
                                       original_start=(datetime.now(timezone.utc) + timedelta(hours=1)).isoformat())


def test_shared_baseline_cannot_be_reused_after_global_configuration_changes(tmp_path):
    binding = put(tmp_path, "inputs/m03-binding.json", {"config": {"learning_rate": .0002}})
    state = {"admissions": {"M03": {"bindings_ref": binding}}}
    api().validate_shared_configuration(tmp_path, state, {"config": {"learning_rate": .0002}})
    with pytest.raises(ValueError, match="SHARED_BASELINE_CONFIGURATION_CHANGED"):
        api().validate_shared_configuration(tmp_path, state, {"config": {"learning_rate": .0001}})


def test_candidate_settings_cannot_hide_under_a_qualification_baseline_name():
    from recursive_ssd.suite_design import arm_registry
    hard = arm_registry()["hard"]
    valid = {"kind": "train", "trajectory": {"arm_id": "hard", "arm": hard}}
    assert api().validate_qualification_job(valid) == "hard"
    altered = copy.deepcopy(valid)
    altered["trajectory"]["arm"]["method"] = "M03"
    with pytest.raises(ValueError, match="QUALIFICATION_BASELINE_SETTINGS_MISMATCH"):
        api().validate_qualification_job(altered)
    with pytest.raises(ValueError, match="QUALIFICATION_CANDIDATE_FORBIDDEN"):
        api().validate_qualification_job({"kind": "preflight", "methods": ["M03"]})


def test_selected_stages_inherit_exact_development_hyperparameters_and_calibration(tmp_path):
    from recursive_ssd.suite_design import make_suite
    evidence = put(tmp_path, "inputs/tuning-evidence.json", {"scope": "engineering fixture"})
    tuning = {"selected_values": {"M03": {"floor": .3}, "arithmetic_anchor": {"floor": .03},
              "M01": {"alpha": .75}, "arithmetic_same_smoothing": {"alpha": .25}}, "source_refs": [evidence]}
    calibration = {"arithmetic_floor": {"value": .03, "source_refs": [evidence]}}
    development = make_suite("development", tuning_selection=tuning, calibration=calibration)
    source = put(tmp_path, "inputs/development.json", development)
    queue = {"identity": {"suite_ref": source}, "suite_digest": development["suite_digest"]}
    queue["queue_digest"] = api()._queue_digest(queue)
    selected = {"queue_ref": put(tmp_path, "runs/development/queue.json", queue),
                "suite_digest": development["suite_digest"], "finalists": ["M03", "M01"]}
    for stage in ("confirmation", "boundary", "model_boundary"):
        result = api().compile_selected_suite(tmp_path, stage, selected)
        assert result["tuning_selection"] == tuning
        assert result["calibration"] == calibration
        settings = {row["arm_id"]: row["arm"]["parameters"] for row in result["trajectories"]}
        assert settings["M03"]["floor"] == .3 and settings["M01"]["alpha"] == .75
        assert settings["arithmetic_anchor"]["floor"] == .03
        assert result["rounds"] == (5 if stage == "boundary" else 3)
        assert result["model"] == ("Qwen/Qwen2.5-Coder-0.5B-Instruct" if stage == "model_boundary" else development["model"])
    changed = copy.deepcopy(tuning)
    changed["selected_values"]["M03"]["floor"] = .1
    with pytest.raises(ValueError, match="SELECTED_DEVELOPMENT_SETTINGS_CHANGED"):
        api().compile_selected_suite(tmp_path, "confirmation", selected, tuning_override=changed)


def test_shared_baseline_cannot_reuse_different_bound_calibration(tmp_path):
    binding = put(tmp_path, "inputs/first.json", {"calibration": {"arithmetic_floor": {"value": .03}}})
    state = {"admissions": {"M03": {"bindings_ref": binding}}}
    with pytest.raises(ValueError, match="SHARED_BASELINE_CALIBRATION_CHANGED"):
        api().validate_shared_configuration(tmp_path, state, {"calibration": {"arithmetic_floor": {"value": .3}}})


def test_head_calibration_rejects_wrong_development_model_and_metadata_only_provenance(tmp_path):
    from recursive_ssd.suite_design import make_suite
    suite = make_suite("development")
    record = {"stage": "development", "seed": 17, "training_lineage": "clean-v2", "round": 1,
              "model": suite["model"], "model_revision": suite["model_revision"], "value": .5, "source_refs": []}
    changed = dict(record, model_revision="changed-model")
    with pytest.raises(ValueError, match="HEAD_CALIBRATION_IDENTITY_MISMATCH"):
        api().validate_frozen_head_calibration(tmp_path, changed, suite, round_index=1)
    with pytest.raises(ValueError, match="HEAD_CALIBRATION_NATIVE_DERIVATION_REQUIRED"):
        api().validate_frozen_head_calibration(tmp_path, record, suite, round_index=1)
