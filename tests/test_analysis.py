"""Scalar/count engineering fixtures; these are never benchmark evidence."""
import copy
import hashlib
import importlib
import json

import pytest


def api():
    try:
        return importlib.import_module("recursive_ssd.analysis")
    except ModuleNotFoundError as exc:
        if exc.name == "recursive_ssd.analysis":
            pytest.fail("the analysis API has not been implemented")
        raise


SEEDS = [23, 47, 71, 101, 131]
PREREQUISITES = ["protocol_freeze", "native_scoring", "baseline_qualification",
                 "implementation_verification", "optimization_diagnostics",
                 "cost_accounting", "independent_confirmation", "e04_review"]


def cell(arm="candidate", seed=23, counts=(0, 10), round_=3, **changes):
    value = {"cell_id": f"{arm}-{seed}-r{round_}", "arm_id": arm,
             "model": "engineering-model", "benchmark": "humaneval",
             "split": "confirmation", "seed": seed, "round": round_,
             "status": "completed", "optimization_valid": True,
             "training_seconds": 100., "identity_refs": {},
             "per_task": {f"task-{i}": {"n": 10, "correct": c}
                          for i, c in enumerate(counts)}}
    value.update(changes)
    return value


def expected(observations):
    return [{**{k: row[k] for k in ("cell_id", "arm_id", "model", "benchmark",
                                     "split", "seed", "round")},
             "task_ids": list(row["per_task"]), "expected_samples": 10,
             "identity_refs": row["identity_refs"]} for row in observations]


def context(tmp_path):
    path = tmp_path / "engineering-review-fixture.json"
    path.write_text(json.dumps({"kind": "engineering_fixture_only"}))
    ref = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return {"phase": "confirmation", "expected_seeds": SEEDS, "endpoint_round": 3,
            "prerequisites": {key: {"assessment": "supported", "evidence_refs": [ref]}
                              for key in PREREQUISITES},
            "required_mechanisms": ["fixture_mechanism"],
            "mechanisms": {"fixture_mechanism": {"assessment": "supported",
                                                   "evidence_refs": [ref]}}}


def estimate(delta=.04, interval=(.03, .05), unit="absolute_probability", **changes):
    value = {"mean_delta": delta, "interval": list(interval), "unit": unit,
             "valid": True, "flags": [], "seeds": SEEDS, "n_seeds": 5,
             "candidate_round": 3, "control_round": 3}
    value.update(changes)
    return value


def criterion(**changes):
    value = {"criterion_id": "coverage", "estimate_id": "baseline:pass@10",
             "relation": "ge", "threshold": .02,
             "unit": "absolute_probability", "kind": "task"}
    value.update(changes)
    return value


def test_inventory_does_not_drop_failed_missing_or_duplicate_cells():
    rows = [cell(seed=s) for s in SEEDS]
    planned = expected(rows)
    rows[1]["status"] = "failed"
    rows[1]["failure"] = "OOM retained as execution failure"
    observed = rows[:4] + [copy.deepcopy(rows[0])]
    result = api().validate_inventory(planned, observed)
    assert not result["complete"]
    assert result["counts"]["expected"] == 5
    assert result["counts"]["observed"] == 5
    assert result["counts"]["missing"] == 1
    assert result["counts"]["failed"] == 1
    assert result["counts"]["duplicate_observations"] == 1
    assert len(result["eligible_cells"]) == 2
    assert {x["code"] for x in result["issues"]} >= {"missing_cell", "failed_cell", "duplicate_cell"}


@pytest.mark.parametrize("change,code", [
    ({"arm_id": "other"}, "cell_identity_mismatch"),
    ({"benchmark": "other"}, "cell_identity_mismatch"),
    ({"round": 2}, "cell_identity_mismatch"),
    ({"optimization_valid": False}, "optimization_invalid"),
    ({"training_seconds": float("nan")}, "invalid_training_seconds"),
    ({"per_task": {"task-0": {"n": 9, "correct": 1}, "task-1": {"n": 10, "correct": 1}}}, "draw_count_mismatch"),
    ({"per_task": {"task-0": {"n": 10, "correct": 11}, "task-1": {"n": 10, "correct": 1}}}, "invalid_success_count"),
    ({"per_task": {"wrong-task": {"n": 10, "correct": 1}}}, "task_inventory_mismatch"),
])
def test_inventory_rejects_denominator_identity_and_invalid_measurements(change, code):
    row = cell()
    planned = expected([row])
    row.update(change)
    result = api().validate_inventory(planned, [row])
    assert not result["complete"]
    assert code in {x["code"] for x in result["issues"]}
    assert not result["eligible_cells"]


def test_inventory_accepts_native_plus_counts_but_not_conflicting_counts():
    row = cell()
    planned = expected([row])
    for counts in row["per_task"].values():
        counts["plus_correct"] = counts.pop("correct")
    assert api().validate_inventory(planned, [row])["complete"]
    row["per_task"]["task-0"]["correct"] = 1
    assert not api().validate_inventory(planned, [row])["complete"]


def test_inventory_preserves_unexpected_cells_and_checks_frozen_identity_refs():
    row = cell(identity_refs={"scorer": "frozen-sha"})
    planned = expected([row])
    row["identity_refs"] = {"scorer": "changed-sha"}
    extra = cell(arm="unexpected")
    report = api().validate_inventory(planned, [row, extra])
    assert report["counts"]["unexpected"] == 1
    assert len(report["ineligible_cells"]) == 2
    assert {x["code"] for x in report["issues"]} >= {"identity_ref_mismatch", "unexpected_cell"}


def test_inventory_rejects_duplicate_expected_ids_and_semantic_keys():
    row = cell()
    plan = expected([row])
    assert not api().validate_inventory(plan + copy.deepcopy(plan), [row])["complete"]
    duplicate = {**plan[0], "cell_id": "alias-for-same-run"}
    result = api().validate_inventory(plan + [duplicate], [row])
    assert "duplicate_expected_key" in {issue["code"] for issue in result["issues"]}


def test_crossed_resampling_keeps_tasks_and_training_seeds_as_two_clusters():
    candidate = [cell(seed=23, counts=(0, 2)), cell(seed=47, counts=(8, 10))]
    control = [cell(arm="control", seed=s, counts=(0, 0)) for s in (23, 47)]
    report = api().crossed_interval(candidate, control, "pass@1", repeats=4000)
    assert report["mean_delta"] == pytest.approx(.5)
    assert report["interval"] == pytest.approx([0., 1.])
    assert report["task_interval"] == pytest.approx([.4, .6])
    assert report["seed_interval"] == pytest.approx([.1, .9])
    assert report["n_tasks"] == 2 and report["n_seeds"] == 2
    assert report["raw_draw_summary"]["candidate"]["draws"] == 40
    assert report["raw_draw_summary"]["candidate"]["correct"] == 20
    assert report["unit"] == "absolute_probability"


def test_shared_resampling_preserves_exact_pairing_and_input_order_invariance():
    candidate = [cell(seed=23, counts=(0, 9)), cell(seed=47, counts=(7, 2))]
    control = [cell(arm="control", seed=47, counts=(7, 2)),
               cell(arm="control", seed=23, counts=(0, 9))]
    first = api().crossed_interval(candidate, control, "pass@1", repeats=1000)
    second = api().crossed_interval(list(reversed(candidate)), control, "pass@1", repeats=1000)
    assert first["interval"] == [0., 0.]
    assert first == second


@pytest.mark.parametrize("metric,want", [("pass@1", .1), ("pass@5", .5), ("pass@10", 1.)])
def test_native_success_counts_are_used_for_each_pass_at_k(metric, want):
    candidate = [cell(seed=s, counts=(1, 1)) for s in (23, 47)]
    control = [cell(arm="control", seed=s, counts=(0, 0)) for s in (23, 47)]
    result = api().crossed_interval(candidate, control, metric, repeats=1000)
    assert result["mean_delta"] == pytest.approx(want)
    assert result["interval"] == pytest.approx([want, want])


@pytest.mark.parametrize("change", ["seed", "task", "benchmark", "draws", "duplicate", "nan"])
def test_crossed_interval_rejects_nonpaired_or_invalid_inputs(change):
    candidate = [cell(seed=s) for s in (23, 47)]
    control = [cell(arm="control", seed=s) for s in (23, 47)]
    if change == "seed":
        control[0]["seed"] = 17
    elif change == "task":
        control[0]["per_task"]["other"] = control[0]["per_task"].pop("task-0")
    elif change == "benchmark":
        control[0]["benchmark"] = "mbpp"
    elif change == "draws":
        control[0]["per_task"]["task-0"]["n"] = 11
    elif change == "duplicate":
        control.append(copy.deepcopy(control[0]))
    else:
        control[0]["per_task"]["task-0"]["correct"] = float("nan")
    with pytest.raises(ValueError):
        api().crossed_interval(candidate, control, "pass@1", repeats=1000)


def test_timing_relative_effect_uses_seed_means_not_task_replication():
    candidate = [cell(seed=s, training_seconds=t) for s, t in ((23, 80.), (47, 160.))]
    control = [cell(arm="control", seed=s, training_seconds=t) for s, t in ((23, 100.), (47, 200.))]
    result = api().crossed_interval(candidate, control, "training_seconds_relative", repeats=1000)
    assert result["mean_delta"] == pytest.approx(-.2)
    assert result["interval"] == pytest.approx([-.2, -.2])
    assert result["independent_unit"] == "training_seed"
    assert result["unit"] == "relative_fraction"


def test_low_monte_carlo_resolution_and_one_seed_are_reported_not_certified():
    result = api().crossed_interval([cell()], [cell(arm="control")], "pass@1", repeats=100, alpha=.001)
    assert not result["valid"]
    assert set(result["flags"]) >= {"insufficient_training_seed_replication", "monte_carlo_tail_resolution_low"}


@pytest.mark.parametrize("kwargs", [{"repeats": True}, {"repeats": 0}, {"alpha": float("nan")}, {"alpha": 1.}, {"seed": True}])
def test_bootstrap_configuration_rejects_invalid_scalars(kwargs):
    with pytest.raises(ValueError):
        api().crossed_interval([cell()], [cell(arm="control")], "pass@1", **kwargs)


@pytest.mark.parametrize("interval,want", [((.03, .05), "PASS"), ((-.04, .01), "KILL"), ((.01, .04), "INCONCLUSIVE")])
def test_criteria_use_simultaneous_bounds_not_point_estimates(tmp_path, interval, want):
    output = api().evaluate_criteria({"baseline:pass@10": estimate(interval=interval)},
                                     [criterion()], context(tmp_path))
    assert output["decision"] == want
    assert output["gate_advanced"] is False
    assert output["scientific_result_verified"] is False


def test_mechanism_contradiction_revises_only_when_task_effects_are_supported(tmp_path):
    evidence = context(tmp_path)
    evidence["mechanisms"]["fixture_mechanism"]["assessment"] = "contradicted"
    result = api().evaluate_criteria({"baseline:pass@10": estimate()}, [criterion()], evidence)
    assert result["decision"] == "REVISE"
    result = api().evaluate_criteria({"baseline:pass@10": estimate(interval=(-.04, .01))}, [criterion()], evidence)
    assert result["decision"] == "KILL"


def test_missing_mechanism_measurements_cannot_support_pass(tmp_path):
    evidence = context(tmp_path)
    evidence["mechanisms"] = {}
    output = api().evaluate_criteria({"baseline:pass@10": estimate()}, [criterion()], evidence)
    assert output["decision"] == "INCONCLUSIVE"
    assert "mechanism_evidence_missing" in {x["code"] for x in output["issues"]}


@pytest.mark.parametrize("change", ["development", "missing_seed", "round", "missing_receipt", "changed_receipt", "naked_boolean", "nan"])
def test_unqualified_evidence_cannot_produce_a_pass_or_scientific_kill(tmp_path, change):
    evidence = context(tmp_path)
    value = estimate(interval=(-.1, -.05))
    if change == "development":
        evidence["phase"] = "development"
    elif change == "missing_seed":
        value["seeds"] = SEEDS[:-1]
    elif change == "round":
        value["candidate_round"] = 2
    elif change == "missing_receipt":
        evidence["prerequisites"].pop("native_scoring")
    elif change == "changed_receipt":
        evidence["prerequisites"]["native_scoring"]["evidence_refs"][0]["sha256"] = "0" * 64
    elif change == "naked_boolean":
        evidence["prerequisites"]["native_scoring"] = True
    else:
        value["interval"] = [float("nan"), .05]
    result = api().evaluate_criteria({"baseline:pass@10": value}, [criterion()], evidence)
    assert result["decision"] == "INCONCLUSIVE"


def test_g01_retention_and_efficiency_thresholds_are_in_native_units():
    comparisons = [dict(comparison_id="base", candidate_arm="M03", control_arm="initial_model",
                        control_role="initial_model", benchmark="humaneval", metrics=["pass@1", "pass@10"]),
                   dict(comparison_id="soft", candidate_arm="M03", control_arm="full_soft",
                        control_role="baseline", benchmark="humaneval", metrics=["pass@1", "pass@10"])]
    rules = api().g01_criteria(comparisons, method_id="M03")
    assert {(x["estimate_id"], x["relation"], x["threshold"]) for x in rules} == {
        ("base:pass@1", "ge", .02), ("base:pass@10", "ge", -.01),
        ("soft:pass@1", "ge", -.01), ("soft:pass@10", "ge", .02)}
    comparisons[1]["metrics"].append("training_seconds_relative")
    rules = api().g01_criteria(comparisons[1:], method_id="M07")
    assert {(x["estimate_id"], x["relation"], x["threshold"]) for x in rules} == {
        ("soft:pass@1", "ge", -.01), ("soft:pass@10", "ge", -.01),
        ("soft:training_seconds_relative", "le", -.10)}


def test_report_retains_failed_comparisons_in_the_frozen_family(tmp_path):
    rows = [cell(arm=arm, seed=s, counts=(10, 10) if arm == "candidate" else (0, 0))
            for arm in ("candidate", "control") for s in SEEDS]
    plan = expected(rows)
    comparisons = [{"comparison_id": "baseline", "candidate_arm": "candidate", "control_arm": "control",
                    "model": "engineering-model", "benchmark": "humaneval", "split": "confirmation",
                    "candidate_round": 3, "control_round": 3, "metrics": ["pass@1", "pass@10"]}]
    report = api().comparison_report(plan, rows[:-1], comparisons, family_size=40,
                                     criteria=[criterion()], evidence_context=context(tmp_path), repeats=2000)
    assert report["family_size"] == 40
    assert report["per_interval_alpha"] == pytest.approx(.05 / 40)
    assert report["decision"] == "INCONCLUSIVE"
    assert len(report["comparisons"]) == 1
    assert report["comparisons"][0]["status"] == "incomplete"
    assert set(report["estimates"]) == {"baseline:pass@1", "baseline:pass@10"}
    assert not any(x["valid"] for x in report["estimates"].values())


def test_complete_report_carries_native_counts_and_conditional_scoped_decision(tmp_path):
    rows = [cell(arm=arm, seed=s, counts=(10, 10) if arm == "candidate" else (0, 0))
            for arm in ("candidate", "control") for s in SEEDS]
    comparison = {"comparison_id": "baseline", "candidate_arm": "candidate", "control_arm": "control",
                  "model": "engineering-model", "benchmark": "humaneval", "split": "confirmation",
                  "candidate_round": 3, "control_round": 3, "metrics": ["pass@1", "pass@10"]}
    report = api().comparison_report(expected(rows), rows, [comparison], family_size=2,
                                     criteria=[criterion()], evidence_context=context(tmp_path), repeats=2000)
    assert report["decision"] == "PASS"
    assert report["inventory"]["counts"]["eligible"] == 10
    assert report["estimates"]["baseline:pass@10"]["raw_draw_summary"]["candidate"]["draws"] == 100
    assert report["scientific_result_verified"] is False


def test_family_size_cannot_shrink_to_available_estimates():
    rows = [cell(), cell(arm="control")]
    comparison = {"comparison_id": "baseline", "candidate_arm": "candidate", "control_arm": "control",
                  "model": "engineering-model", "benchmark": "humaneval", "split": "confirmation",
                  "candidate_round": 3, "control_round": 3, "metrics": ["pass@1", "pass@10"]}
    with pytest.raises(ValueError, match="family"):
        api().comparison_report(expected(rows), rows, [comparison], family_size=1,
                                 criteria=[], evidence_context={})


def test_recommended_repeats_resolves_frozen_bonferroni_tails_before_results():
    assert api().recommended_repeats(1) == 20000
    assert api().recommended_repeats(400) == 320000
    assert api().recommended_repeats(7, min_tail_draws=11, alpha=.01) == 20000
    with pytest.raises(ValueError):
        api().recommended_repeats(True)


def test_factorial_interval_estimates_paired_interaction_not_a_pairwise_gain():
    arms = [[cell(arm=arm, seed=s, counts=counts) for s in (23, 47)]
            for arm, counts in (("a00", (0, 2)), ("a01", (1, 3)),
                                ("a10", (2, 4)), ("a11", (5, 7)))]
    result = api().factorial_interval(*arms, "pass@1", repeats=1000)
    assert result["mean_delta"] == pytest.approx(.2)
    assert result["interval"] == pytest.approx([.2, .2])
    assert result["contrast"] == "11-10-01+00"
    assert result["raw_draw_summary"]["11"]["draws"] == 40
    arms[2][0]["per_task"]["wrong"] = arms[2][0]["per_task"].pop("task-0")
    with pytest.raises(ValueError):
        api().factorial_interval(*arms, "pass@1", repeats=1000)


def test_factorial_report_keeps_all_four_arms_and_frozen_family_slots():
    rows = [cell(arm=arm, seed=s, counts=counts) for arm, counts in
            (("a00", (0, 2)), ("a01", (1, 3)), ("a10", (2, 4)), ("a11", (5, 7)))
            for s in SEEDS]
    comparison = {"comparison_id": "interaction", "factorial_arms": ["a00", "a01", "a10", "a11"],
                  "round": 3, "model": "engineering-model", "benchmark": "humaneval",
                  "split": "confirmation", "metrics": ["pass@1", "pass@10"]}
    report = api().comparison_report(expected(rows), rows, [comparison], family_size=4,
                                     criteria=[], evidence_context={}, repeats=2000)
    assert report["decision"] == "INCONCLUSIVE"
    assert report["planned_endpoint_slots"] == 2
    assert report["estimates"]["interaction:pass@1"]["mean_delta"] == pytest.approx(.2)
    assert set(report["comparisons"][0]["factorial_cells"]) == {"00", "01", "10", "11"}


def test_known_benchmarks_cannot_silently_swap_their_native_coverage_endpoint():
    rows = [cell(), cell(arm="control")]
    comparison = {"comparison_id": "baseline", "candidate_arm": "candidate", "control_arm": "control",
                  "model": "engineering-model", "benchmark": "humaneval", "split": "confirmation",
                  "candidate_round": 3, "control_round": 3, "metrics": ["pass@1", "pass@5"]}
    with pytest.raises(ValueError, match="native"):
        api().comparison_report(expected(rows), rows, [comparison], family_size=2,
                                 criteria=[], evidence_context={})


def test_invalid_status_types_are_accounted_for_without_crashing_inventory():
    row = cell()
    planned = expected([row])
    row["status"] = ["completed"]
    report = api().validate_inventory(planned, [row])
    assert not report["complete"]
    assert "invalid_status" in {issue["code"] for issue in report["issues"]}


def test_training_rounds_require_positive_cumulative_time_but_base_can_have_zero():
    row = cell(training_seconds=0.)
    assert not api().validate_inventory(expected([row]), [row])["complete"]
    row = cell(arm="initial_model", round_=0, training_seconds=0.)
    assert api().validate_inventory(expected([row]), [row])["complete"]


def test_semantic_duplicate_observations_get_an_explicit_count():
    row = cell()
    alias = {**copy.deepcopy(row), "cell_id": "same-run-another-id"}
    report = api().validate_inventory(expected([row]), [row, alias])
    assert report["counts"]["duplicate_semantic_observations"] == 1
    assert not report["eligible_cells"]


def test_malformed_identity_references_reject_direct_pairing_with_value_error():
    bad = cell(identity_refs=None)
    with pytest.raises(ValueError, match="identity"):
        api().crossed_interval([bad], [cell(arm="control")], "pass@1", repeats=1000)


@pytest.mark.parametrize("factorial", [False, True])
def test_shared_identity_compares_verified_bytes_across_different_paths(tmp_path, factorial):
    groups = []
    for index in range(4 if factorial else 2):
        path = tmp_path / f"identity-{index}.json"
        path.write_text('{"decoder": "same protocol"}')
        ref = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        groups.append([cell(arm=f"arm-{index}", seed=s, identity_refs={"decoder": ref}) for s in (23, 47)])
    call = api().factorial_interval if factorial else api().crossed_interval
    assert call(*groups, "pass@1", repeats=1000)["interval"] == [0., 0.]
    changed = tmp_path / "identity-0.json"
    changed.write_text('{"decoder": "different protocol"}')
    with pytest.raises(ValueError, match="identity"):
        call(*groups, "pass@1", repeats=1000)
    for row in groups[0]:
        row["identity_refs"]["decoder"]["sha256"] = hashlib.sha256(changed.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="decoder"):
        call(*groups, "pass@1", repeats=1000)


@pytest.mark.parametrize("flag,code", [("cost_matching_valid", "cost_matching_invalid"),
                                      ("budget_valid", "budget_invalid")])
def test_unmatched_clock_or_exceeded_budget_cannot_become_an_eligible_comparison(flag, code):
    row = cell(**{flag: False})
    result = api().validate_inventory(expected([row]), [row])
    assert not result["complete"]
    assert not result["eligible_cells"]
    assert code in {issue["code"] for issue in result["issues"]}
    with pytest.raises(ValueError, match="cost|budget"):
        api().crossed_interval([row], [cell(arm="control")], "pass@1", repeats=1000)
