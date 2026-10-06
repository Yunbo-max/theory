"""Calibration arithmetic/provenance fixtures; not benchmark/model evidence."""
import copy
import importlib
import json
import time

import pytest

from recursive_ssd.io import Deadline, digest, object_hash, read_json


def api():
    try:
        return importlib.import_module("recursive_ssd.suite_calibration")
    except ModuleNotFoundError as exc:
        if exc.name == "recursive_ssd.suite_calibration":
            pytest.fail("the measured calibration adapter is not implemented")
        raise


def put(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return {"path": str(path), "sha256": digest(path)}


def job(requests, **changes):
    value = {"stage": "development", "seed": 17, "training_lineage": "clean-v2",
             "model": "engineering-model", "model_revision": "engineering-revision",
             "requests": requests}
    value.update(changes)
    return value


def training_receipt(arm="M13", **changes):
    value = {"status": "complete", "arm_id": arm, "trajectory_id": "engineering-trajectory",
             "round": 2, "model": "engineering-model", "model_revision": "engineering-revision",
             "seed": 17, "stage": "development", "training_lineage": "clean-v2",
             "training_seconds": 30., "cumulative_training_seconds": 90.,
             "costs": {"generation_seconds": 6., "proxy_seconds": 2., "training_seconds": 30.,
                       "save_seconds": 2., "total_round_seconds": 40.}}
    value.update(changes)
    return value


def derive(tmp_path, requests, **changes):
    return api().derive_calibration(job(requests, **changes), tmp_path,
                                     tmp_path / "out", Deadline(time.time() + 60))


def test_clock_calibrations_use_actual_per_round_and_all_round_costs(tmp_path):
    m13 = put(tmp_path, "m13.json", training_receipt())
    m15 = put(tmp_path, "m15.json", training_receipt("M15"))
    receipt = derive(tmp_path, [{"key": "clock_m13", "round": 2, "source_refs": [m13]},
                                {"key": "clock_m15", "round": 2, "source_refs": [m15]}])
    calibrated = read_json(tmp_path / "out/calibration.json")
    assert calibrated["clock_m13"]["rounds"]["2"]["value"] == 30.
    assert calibrated["clock_m15"]["rounds"]["2"]["value"] == 40.
    assert receipt["status"] == "complete"
    assert receipt["calibration_ref"]["sha256"] == digest(tmp_path / "out/calibration.json")
    assert calibrated["clock_m15"]["rounds"]["2"]["source_refs"]


@pytest.mark.parametrize("changes", [{"status": "failed"}, {"seed": 23}, {"round": 1},
                                     {"training_lineage": "old-v1"}, {"arm_id": "hard"},
                                     {"model_revision": "other-revision"}, {"training_seconds": float("nan")}])
def test_clock_derivation_rejects_wrong_or_unfinished_measurements(tmp_path, changes):
    ref = put(tmp_path, "source.json", training_receipt(**changes))
    with pytest.raises(ValueError):
        derive(tmp_path, [{"key": "clock_m13", "round": 2, "source_refs": [ref]}])
    assert not (tmp_path / "out/calibration.json").exists()


def test_clock_rejects_inconsistent_cost_sum_and_changed_source_bytes(tmp_path):
    source = training_receipt("M15")
    source["costs"]["total_round_seconds"] = 30.
    ref = put(tmp_path, "source.json", source)
    with pytest.raises(ValueError, match="cost"):
        derive(tmp_path, [{"key": "clock_m15", "round": 2, "source_refs": [ref]}])
    source["costs"]["total_round_seconds"] = 40.
    (tmp_path / "source.json").write_text(json.dumps(source))
    with pytest.raises(ValueError, match="hash"):
        derive(tmp_path, [{"key": "clock_m15", "round": 2, "source_refs": [ref]}])


def step_sources(tmp_path, events, attempted=3):
    events_path = tmp_path / "events.jsonl"
    events_path.write_text("".join(json.dumps(event) + "\n" for event in events))
    event_ref = {"path": str(events_path), "sha256": digest(events_path)}
    summary = put(tmp_path, "training.json", {"attempted_updates": attempted})
    receipt = training_receipt(events_ref=event_ref, training_ref=summary)
    return put(tmp_path, "receipt.json", receipt)


def test_step_scale_includes_rejected_steps_and_checks_complete_optimizer_groups(tmp_path):
    events = [{"kind": "optimizer_step", "group": i, "accepted": accepted, "backtrack_gamma": gamma}
              for i, accepted, gamma in ((0, True, 1.), (1, False, 0.), (2, True, .5))]
    ref = step_sources(tmp_path, events)
    derive(tmp_path, [{"key": "m13_step_scale", "round": 2, "source_refs": [ref]}])
    record = read_json(tmp_path / "out/calibration.json")["m13_step_scale"]["rounds"]["2"]
    assert record["value"] == .5
    assert record["attempted_updates"] == 3
    assert record["rejected_updates"] == 1
    assert len(record["source_refs"]) == 3
    ref = step_sources(tmp_path, events[:-1])
    with pytest.raises(ValueError, match="optimizer"):
        api().derive_calibration(job([{"key": "m13_step_scale", "round": 2, "source_refs": [ref]}]),
                                 tmp_path, tmp_path / "missing", Deadline(time.time() + 60))


def test_zero_step_scale_cannot_become_a_positive_learning_rate_calibration(tmp_path):
    ref = step_sources(tmp_path, [{"kind": "optimizer_step", "group": 0,
                                  "accepted": False, "backtrack_gamma": 0.}], attempted=1)
    with pytest.raises(ValueError, match="zero"):
        derive(tmp_path, [{"key": "m13_step_scale", "round": 2, "source_refs": [ref]}])


def test_head_temperature_matches_mean_gini_without_changing_zero_support():
    result = api().match_head_temperature([[.9, .1, 0.], [.8, .2, 0.]], .4097222222222222)
    assert result["value"] == pytest.approx(.5, abs=1e-4)
    assert result["achieved"] == pytest.approx(.4097222222222222, abs=1e-5)
    assert result["support_sizes"] == [2, 2]
    assert api().match_head_temperature([[.9, .1, 0.]], .5)["value"] == 0.
    with pytest.raises(ValueError, match="reach"):
        api().match_head_temperature([[.9, .1, 0.]], .6)


@pytest.mark.parametrize("rows,target", [([], .2), ([[.7, .1]], .2), ([[float("nan"), .1]], .2),
                                        ([[.9, .1]], .1), ([[.9, .1]], float("nan"))])
def test_head_temperature_rejects_invalid_or_unreachable_probability_data(rows, target):
    with pytest.raises(ValueError):
        api().match_head_temperature(rows, target)


def test_head_calibration_binds_actual_probability_and_target_diagnostics(tmp_path):
    mu = [[.9, .1, 0.], [.8, .2, 0.]]
    targets = [.375, 4. / 9.]
    diagnostics = {**training_receipt("M06"), "mu_rows": mu, "target_gini_rows": targets,
                   "mu_sha256": object_hash(mu), "target_gini_sha256": object_hash(targets), "prefix_count": 2}
    ref = put(tmp_path, "head.json", diagnostics)
    derive(tmp_path, [{"key": "head_temperature", "round": 2, "source_refs": [ref]}])
    record = read_json(tmp_path / "out/calibration.json")["head_temperature"]["rounds"]["2"]
    assert record["value"] == pytest.approx(.5, abs=1e-4)
    assert record["achieved"] == pytest.approx(record["target"], abs=1e-5)


def fair_tuning(tmp_path):
    evidence = put(tmp_path, "engineering-native-reference.json", {"kind": "engineering-only"})
    trials = [{"family": family, "arm_id": f"{family}-{floor}", "parameters": {"floor": floor},
               "status": "completed", "optimization_valid": True, "budget_digest": "same-budget",
               "source_refs": [evidence]} for family in ("M03", "arithmetic_anchor") for floor in (.03, .1, .3)]
    return {"tuning_selection": {"stage": "tuning", "seed": 17, "training_lineage": "clean-v2",
             "model": "engineering-model", "model_revision": "engineering-revision",
             "inventory": {"complete": True, "expected": 6, "observed": 6}, "trials": trials,
             "selected_control": {"arm_id": "arithmetic_anchor-0.1", "parameters": {"floor": .1}}}}


def test_arithmetic_calibration_requires_complete_fair_development_grid(tmp_path):
    packet = fair_tuning(tmp_path)
    ref = put(tmp_path, "tuning.json", packet)
    derive(tmp_path, [{"key": "arithmetic_floor", "source_refs": [ref]}])
    assert read_json(tmp_path / "out/calibration.json")["arithmetic_floor"]["value"] == .1
    for i, mutation in enumerate(("confirmation", "missing", "unequal_budget")):
        bad = copy.deepcopy(packet)
        if mutation == "confirmation":
            bad["tuning_selection"]["stage"] = "confirmation"
        elif mutation == "missing":
            bad["tuning_selection"]["trials"].pop()
        else:
            bad["tuning_selection"]["trials"][0]["budget_digest"] = "more-training"
        ref = put(tmp_path, f"bad-{i}.json", bad)
        with pytest.raises(ValueError):
            api().derive_calibration(job([{"key": "arithmetic_floor", "source_refs": [ref]}]), tmp_path,
                                     tmp_path / f"bad-out-{i}", Deadline(time.time() + 60))


def test_confirmation_clocks_require_same_actual_confirmation_seed(tmp_path):
    ref = put(tmp_path, "confirm.json", training_receipt(seed=23, stage="confirmation"))
    derive(tmp_path, [{"key": "clock_m13", "round": 2, "source_refs": [ref]}],
           stage="confirmation", seed=23)
    with pytest.raises(ValueError, match="seed"):
        api().derive_calibration(job([{"key": "clock_m13", "round": 2, "source_refs": [ref]}],
                                    stage="confirmation", seed=47), tmp_path,
                                 tmp_path / "wrong-seed", Deadline(time.time() + 60))


def test_real_tiny_policy_head_diagnostics_count_every_self_generated_prefix():
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from recursive_ssd.model import Policy
    from recursive_ssd.methods import Decode
    torch.set_num_threads(1)
    torch.manual_seed(17)
    base = Qwen2ForCausalLM(Qwen2Config(vocab_size=31, hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
        attention_dropout=0., max_position_embeddings=128))
    policy = Policy(base, rank=2, device="cpu")
    records = [{"prompt_ids": [1, 2], "completion_ids": [3, 4, 5]},
               {"prompt_ids": [2, 3], "completion_ids": [6, 7]}]
    value = api().collect_head_diagnostics(policy, records, Decode(1.5, 5, .8),
                                          {"diversity_retention": .9}, Deadline(time.time() + 60), chunk=2)
    assert value["prefix_count"] == 5
    assert len(value["mu_rows"]) == len(value["target_gini_rows"]) == 5
    assert all(1 <= len(row) <= 5 and sum(row) == pytest.approx(1., abs=1e-6) for row in value["mu_rows"])
    assert value["mu_sha256"] == object_hash(value["mu_rows"])
