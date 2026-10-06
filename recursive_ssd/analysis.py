"""Strict native-count comparison analysis for the retained G01 protocol.

This module computes evidence-bound, scoped analytical decisions. It neither
authorizes execution nor certifies scientific gates. Engineering fixtures used
to test it are not benchmark evidence. See docs/ANALYSIS.md for the contracts.
"""
from collections import Counter
from collections.abc import Mapping
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .evaluation import pass_at_k


CELL_FIELDS = ("cell_id", "arm_id", "model", "benchmark", "split", "seed", "round")
SCOPE_FIELDS = ("model", "benchmark", "split")
CONFIRMATION_SEEDS = (23, 47, 71, 101, 131)
PREREQUISITES = ("protocol_freeze", "native_scoring", "baseline_qualification",
                 "implementation_verification", "optimization_diagnostics",
                 "cost_accounting", "independent_confirmation", "e04_review")
FAILED_STATUSES = {"failed", "timeout", "oom", "interrupted", "cancelled"}
PENDING_STATUSES = {"pending", "running", "planned", "blocked"}
MATCHED_REFERENCES = ("scorer", "decoder", "benchmark_inventory", "evaluation_seeds",
                      "model_base", "training_data", "evaluation_protocol")
NATIVE_COVERAGE = {"humaneval": "pass@10", "mbpp": "pass@10", "livecodebench": "pass@5"}
ANALYSIS_SCOPE = ("Scoped analysis conditional on cited reviewed evidence; citation identity "
                  "checks do not authenticate scientific claims or advance research gates.")


def _integer(value, minimum=0):
    return type(value) is int and value >= minimum


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _issue(code, **details):
    return {"code": code, **details}


def _identity_errors(row):
    if not isinstance(row, Mapping):
        return ["cell_is_not_mapping"]
    errors = [f"invalid_{key}" for key in CELL_FIELDS[:5]
              if not isinstance(row.get(key), str) or not row[key]]
    errors.extend(f"invalid_{key}" for key in ("seed", "round")
                  if not _integer(row.get(key)))
    return errors


def _semantic_key(row):
    return tuple(row.get(key) for key in CELL_FIELDS[1:])


def _counts(value):
    if not isinstance(value, Mapping) or not _integer(value.get("n"), 1):
        raise ValueError("invalid_draw_count")
    n = value["n"]
    correct = value.get("correct", value.get("plus_correct"))
    if not _integer(correct) or correct > n:
        raise ValueError("invalid_success_count")
    if "plus_correct" in value and (not _integer(value["plus_correct"])
                                     or value["plus_correct"] != correct):
        raise ValueError("conflicting_success_counts")
    return n, correct


def _task_plan(row):
    task_ids = row.get("task_ids")
    if (not isinstance(task_ids, (list, tuple)) or not task_ids
            or any(not isinstance(task, str) or not task for task in task_ids)
            or len(set(task_ids)) != len(task_ids)):
        raise ValueError("invalid_expected_tasks")
    samples = row.get("expected_samples")
    if _integer(samples, 1):
        return {task: samples for task in task_ids}
    if (isinstance(samples, Mapping) and set(samples) == set(task_ids)
            and all(_integer(n, 1) for n in samples.values())):
        return dict(samples)
    raise ValueError("invalid_expected_draw_counts")


def validate_inventory(expected_cells, observations):
    """Account for every planned cell and observed row without dropping failures.

    Invalid plans are reported alongside invalid observations. Duplicate rows are
    all ineligible, even if one of their copies appears otherwise complete.
    """
    if (not isinstance(expected_cells, (list, tuple))
            or not isinstance(observations, (list, tuple))):
        raise ValueError("cell inventories must be lists or tuples")
    issues, expected_by_id, plan_tasks, invalid_plans = [], {}, {}, set()
    expected_ids, expected_keys = Counter(), Counter()
    for index, row in enumerate(expected_cells):
        errors = _identity_errors(row)
        if errors:
            issues.append(_issue("invalid_expected_cell", expected_index=index, reasons=errors))
            continue
        cid = row["cell_id"]
        expected_ids[cid] += 1
        expected_keys[_semantic_key(row)] += 1
        expected_by_id[cid] = row
        try:
            plan_tasks[cid] = _task_plan(row)
        except ValueError as exc:
            invalid_plans.add(cid)
            issues.append(_issue(str(exc), cell_id=cid, expected_index=index))
        if not isinstance(row.get("identity_refs", {}), Mapping):
            invalid_plans.add(cid)
            issues.append(_issue("invalid_expected_identity_refs", cell_id=cid))
    if not expected_cells:
        issues.append(_issue("empty_expected_inventory"))
    for cid, count in expected_ids.items():
        if count > 1:
            invalid_plans.add(cid)
            issues.append(_issue("duplicate_expected_cell", cell_id=cid, count=count))
    for key, count in expected_keys.items():
        if count > 1:
            invalid_plans.update(cid for cid, row in expected_by_id.items() if _semantic_key(row) == key)
            issues.append(_issue("duplicate_expected_key", key=list(key), count=count))

    observed_ids, observed_keys = Counter(), Counter()
    for row in observations:
        if not _identity_errors(row):
            observed_ids[row["cell_id"]] += 1
            observed_keys[_semantic_key(row)] += 1
    eligible, ineligible = [], []
    counts = {"expected": len(expected_cells), "observed": len(observations),
              "completed": 0, "failed": 0, "pending": 0, "missing": 0,
              "unexpected": 0, "duplicate_observations": sum(n - 1 for n in observed_ids.values()),
              "duplicate_semantic_observations": sum(n - 1 for n in observed_keys.values()),
              "invalid": 0, "eligible": 0, "ineligible": 0}
    for index, row in enumerate(observations):
        reasons = []
        identity_errors = _identity_errors(row)
        cid = row.get("cell_id") if isinstance(row, Mapping) else None
        if identity_errors:
            reasons.append(_issue("invalid_observation", reasons=identity_errors))
        else:
            if observed_ids[cid] > 1 or observed_keys[_semantic_key(row)] > 1:
                reasons.append(_issue("duplicate_cell"))
            plan = expected_by_id.get(cid)
            if plan is None:
                counts["unexpected"] += 1
                reasons.append(_issue("unexpected_cell"))
            else:
                mismatch = [key for key in CELL_FIELDS if row[key] != plan[key]]
                if mismatch:
                    reasons.append(_issue("cell_identity_mismatch", fields=mismatch))
                if cid in invalid_plans:
                    reasons.append(_issue("invalid_expected_cell"))
                refs = row.get("identity_refs", {})
                expected_refs = plan.get("identity_refs", {})
                if (not isinstance(refs, Mapping) or not isinstance(expected_refs, Mapping)
                        or any(refs.get(key) != value for key, value in expected_refs.items())):
                    reasons.append(_issue("identity_ref_mismatch"))

            status = row.get("status")
            if status == "completed":
                counts["completed"] += 1
            elif isinstance(status, str) and status in FAILED_STATUSES:
                counts["failed"] += 1
                reasons.append(_issue("failed_cell", status=status, failure=row.get("failure")))
            elif isinstance(status, str) and status in PENDING_STATUSES:
                counts["pending"] += 1
                reasons.append(_issue("pending_cell", status=status))
            else:
                reasons.append(_issue("invalid_status"))
            if status == "completed":
                if row.get("optimization_valid") is not True:
                    reasons.append(_issue("optimization_invalid"))
                if row.get("cost_matching_valid") is False:
                    reasons.append(_issue("cost_matching_invalid"))
                if row.get("budget_valid") is False:
                    reasons.append(_issue("budget_invalid"))
                seconds = row.get("training_seconds")
                if not _finite(seconds) or seconds < 0 or (row["round"] > 0 and seconds == 0):
                    reasons.append(_issue("invalid_training_seconds"))
                per_task = row.get("per_task")
                expected_tasks = plan_tasks.get(cid, {})
                if not isinstance(per_task, Mapping) or not per_task:
                    reasons.append(_issue("task_inventory_mismatch"))
                else:
                    if expected_tasks and set(per_task) != set(expected_tasks):
                        reasons.append(_issue("task_inventory_mismatch",
                                              missing_tasks=sorted(set(expected_tasks) - set(per_task)),
                                              unexpected_tasks=sorted(set(per_task) - set(expected_tasks), key=str)))
                    for task, value in per_task.items():
                        try:
                            n, _ = _counts(value)
                            if task in expected_tasks and n != expected_tasks[task]:
                                reasons.append(_issue("draw_count_mismatch", task_id=task,
                                                      expected=expected_tasks[task], observed=n))
                        except ValueError as exc:
                            reasons.append(_issue(str(exc), task_id=task))
        if reasons:
            counts["invalid"] += 1
            detail = {"cell_id": cid, "observation_index": index, "reasons": reasons}
            ineligible.append(detail)
            issues.extend({**reason, "cell_id": cid, "observation_index": index} for reason in reasons)
        else:
            eligible.append(dict(row))
    for cid in expected_by_id:
        if cid not in observed_ids:
            counts["missing"] += 1
            detail = _issue("missing_cell", cell_id=cid)
            issues.append(detail)
            ineligible.append({"cell_id": cid, "observation_index": None, "reasons": [detail]})
    counts["eligible"], counts["ineligible"] = len(eligible), len(ineligible)
    return {"complete": not issues and len(eligible) == len(expected_cells),
            "valid": not issues, "counts": counts, "issues": issues,
            "eligible_cells": eligible, "ineligible_cells": ineligible}


def _arm(cells):
    if not isinstance(cells, (list, tuple)) or not cells:
        raise ValueError("a comparison arm requires nonempty seed cells")
    rows, citation_cache = {}, {}
    for row in cells:
        if _identity_errors(row) or row.get("status") != "completed" or row.get("optimization_valid") is not True:
            raise ValueError("comparison requires valid completed cells")
        if row.get("cost_matching_valid") is False or row.get("budget_valid") is False:
            raise ValueError("comparison requires valid cost matching and training budget")
        if row["seed"] in rows:
            raise ValueError("duplicate training seed")
        if (not _finite(row.get("training_seconds")) or row["training_seconds"] < 0
                or (row["round"] > 0 and row["training_seconds"] == 0)):
            raise ValueError("invalid training time")
        if not isinstance(row.get("identity_refs", {}), Mapping):
            raise ValueError("invalid identity references")
        for value in row.get("identity_refs", {}).values():
            if isinstance(value, Mapping) and not _cited_assessment({"evidence_refs": [value]}, citation_cache):
                raise ValueError("identity reference bytes are missing or changed")
        tasks = row.get("per_task")
        if not isinstance(tasks, Mapping) or not tasks or any(not isinstance(t, str) for t in tasks):
            raise ValueError("comparison requires official per-task counts")
        for value in tasks.values():
            _counts(value)
        rows[row["seed"]] = row
    for field in ("arm_id", *SCOPE_FIELDS, "round"):
        if len({row[field] for row in rows.values()}) != 1:
            raise ValueError(f"comparison arm mixes {field}")
    return rows


def _ref_identity(value):
    # Paths locate cited bytes; equal frozen content may live in distinct cells.
    return value.get("sha256") if isinstance(value, Mapping) else value


def _bootstrap_config(repeats, seed, alpha):
    if not _integer(repeats, 1) or not _integer(seed) or not _finite(alpha) or not 0 < alpha < 1:
        raise ValueError("invalid bootstrap repeats, seed or alpha")


def recommended_repeats(family_size, min_tail_draws=20, alpha=.05):
    """Choose a Monte Carlo budget before freezing the family/results.

    This resolves numerical tail sampling, not limited benchmark/seed precision.
    The returned budget still needs finite host-resource admission.
    """
    if (not _integer(family_size, 1) or not _integer(min_tail_draws, 1)
            or not _finite(alpha) or not 0 < alpha < 1):
        raise ValueError("invalid frozen family or Monte Carlo tail requirement")
    count = 2 * family_size * min_tail_draws / alpha
    if not math.isfinite(count):
        raise ValueError("unrepresentable bootstrap budget")
    return max(20000, math.ceil(count))


def _metric_k(metric):
    if metric == "training_seconds_relative":
        return None
    if not isinstance(metric, str) or not metric.startswith("pass@"):
        raise ValueError("metric must be pass@k or training_seconds_relative")
    try:
        k = int(metric[5:])
    except ValueError as exc:
        raise ValueError("invalid pass@k endpoint") from exc
    if k < 1 or metric != f"pass@{k}":
        raise ValueError("invalid pass@k endpoint")
    return k


def _crossed_samples(delta, *, ns, nt, repeats, seed, timing=None):
    """One schedule for paired and factorial contrasts; bounded temporaries."""
    crossed, task_only, seed_only = (np.empty(repeats) for _ in range(3))
    rng = np.random.default_rng(seed)
    for start in range(0, repeats, 256):
        stop = min(start + 256, repeats)
        width = stop - start
        si = rng.integers(0, ns, (width, ns))
        ti = rng.integers(0, nt, (width, nt))
        if timing is not None:
            av, bv = timing
            values = av[si].mean(axis=1) / bv[si].mean(axis=1) - 1.
            crossed[start:stop] = seed_only[start:stop] = values
            task_only[start:stop] = av.mean() / bv.mean() - 1.
        else:
            crossed[start:stop] = delta[si[:, :, None], ti[:, None, :]].mean(axis=2).mean(axis=1)
            task_only[start:stop] = delta[:, ti].mean(axis=2).mean(axis=0)
            seed_only[start:stop] = delta[si, :].mean(axis=2).mean(axis=1)
    return crossed, task_only, seed_only


def _bootstrap_summary(distributions, *, seeds, tasks, repeats, seed, alpha):
    intervals = [np.quantile(values, [alpha / 2., 1. - alpha / 2.], method="linear").tolist()
                 for values in distributions]
    flags = []
    if len(seeds) < 2:
        flags.append("insufficient_training_seed_replication")
    tail_draws = repeats * alpha / 2.
    if tail_draws < 10:
        flags.append("monte_carlo_tail_resolution_low")
    schedule = {"seed": seed, "repeats": repeats, "tasks": tasks, "seeds": seeds,
                "generator": "numpy.default_rng.PCG64", "batch_size": 256}
    return {"interval": intervals[0], "task_interval": intervals[1], "seed_interval": intervals[2],
            "n_tasks": len(tasks), "n_seeds": len(seeds), "tasks": tasks, "seeds": seeds,
            "repeats": repeats, "analysis_seed": seed, "alpha": alpha,
            "confidence_level": 1. - alpha, "quantile_method": "linear_percentile",
            "monte_carlo": {"expected_draws_per_tail": tail_draws,
                            "minimum_expected_draws_per_tail": 10},
            "paired_index_schedule_sha256": hashlib.sha256(
                json.dumps(schedule, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "valid": not flags, "flags": flags}


def crossed_interval(candidate, control, metric, repeats=20000, seed=20261006, alpha=.05):
    """Paired crossed task/seed percentile bootstrap, retaining native counts.

    This is not a bootstrap over independent decoding draws. The deterministic
    schedule is shared across arms/rounds with the same sorted tasks and seeds.
    """
    _bootstrap_config(repeats, seed, alpha)
    k = _metric_k(metric)
    a, b = _arm(candidate), _arm(control)
    if set(a) != set(b):
        raise ValueError("paired comparison requires identical training seeds")
    seeds = sorted(a)
    first_a, first_b = a[seeds[0]], b[seeds[0]]
    if any(first_a[field] != first_b[field] for field in SCOPE_FIELDS):
        raise ValueError("paired comparison requires identical model, benchmark and split")
    tasks = sorted(first_a["per_task"])
    count_template = [_counts(first_a["per_task"][task])[0] for task in tasks]
    if k is not None and any(n < k for n in count_template):
        raise ValueError("native draw count cannot support requested pass@k")
    for row in [*a.values(), *b.values()]:
        if set(row["per_task"]) != set(tasks):
            raise ValueError("paired comparison requires identical task inventories")
        if [_counts(row["per_task"][task])[0] for task in tasks] != count_template:
            raise ValueError("paired comparison requires frozen draw counts")
    for ref in MATCHED_REFERENCES:
        supplied = [_ref_identity(row.get("identity_refs", {}).get(ref)) for row in [*a.values(), *b.values()]]
        if any(value is not None for value in supplied) and any(value != supplied[0] for value in supplied):
            raise ValueError(f"paired comparison has mismatched {ref} identity")

    summary = {}
    for label, rows in (("candidate", a), ("control", b)):
        per_seed = []
        for training_seed in seeds:
            counts = [_counts(rows[training_seed]["per_task"][task]) for task in tasks]
            per_seed.append({"seed": training_seed, "tasks": len(tasks),
                             "draws": sum(n for n, _ in counts),
                             "correct": sum(c for _, c in counts)})
        summary[label] = {"draws": sum(row["draws"] for row in per_seed),
                          "correct": sum(row["correct"] for row in per_seed), "by_seed": per_seed}
    ns, nt = len(seeds), len(tasks)
    if k is None:
        av = np.array([a[s]["training_seconds"] for s in seeds], dtype=float)
        bv = np.array([b[s]["training_seconds"] for s in seeds], dtype=float)
        if np.any(bv <= 0):
            raise ValueError("relative training time requires positive control times")
        mean_delta = float(av.mean() / bv.mean() - 1.)
    else:
        av = np.array([[pass_at_k(*_counts(a[s]["per_task"][t]), k) for t in tasks] for s in seeds])
        bv = np.array([[pass_at_k(*_counts(b[s]["per_task"][t]), k) for t in tasks] for s in seeds])
        delta = av - bv
        mean_delta = float(delta.mean(axis=1).mean())
    distributions = _crossed_samples(None if k is None else delta, ns=ns, nt=nt,
                                     repeats=repeats, seed=seed, timing=(av, bv) if k is None else None)
    output = {"metric": metric, "mean_delta": mean_delta,
              **_bootstrap_summary(distributions, seeds=seeds, tasks=tasks, repeats=repeats, seed=seed, alpha=alpha),
              "candidate_mean": float(av.mean()), "control_mean": float(bv.mean()),
              "per_seed_means": [{"seed": s, "candidate": float(av[i].mean()),
                                   "control": float(bv[i].mean())} for i, s in enumerate(seeds)],
              "unit": "relative_fraction" if k is None else "absolute_probability",
              "independent_unit": "training_seed" if k is None else "task_and_training_seed",
              "candidate_round": first_a["round"], "control_round": first_b["round"],
              **{field: first_a[field] for field in SCOPE_FIELDS},
              "raw_draw_summary": summary}
    if k is not None:
        output["mean_delta_pp"] = 100. * mean_delta
        output["interval_pp"] = [100. * value for value in output["interval"]]
    return output


def factorial_interval(cells00, cells01, cells10, cells11, metric, repeats=20000, seed=20261006, alpha=.05):
    """Paired 2×2 interaction (11 − 10 − 01 + 00) in native task units.

    All four factors require the same endpoint round, seed set, task inventory,
    draw counts and matched data/scoring identities. Relative time ratios do not
    have this additive interaction contract and are intentionally rejected.
    """
    _bootstrap_config(repeats, seed, alpha)
    k = _metric_k(metric)
    if k is None:
        raise ValueError("factorial interaction requires a native pass@k endpoint")
    arms = {key: _arm(value) for key, value in zip(("00", "01", "10", "11"),
                                                 (cells00, cells01, cells10, cells11))}
    seeds = sorted(arms["00"])
    first = arms["00"][seeds[0]]
    tasks = sorted(first["per_task"])
    expected_n = [_counts(first["per_task"][task])[0] for task in tasks]
    if any(n < k for n in expected_n):
        raise ValueError("native draw count cannot support requested pass@k")
    all_rows = []
    for rows in arms.values():
        if sorted(rows) != seeds:
            raise ValueError("factorial comparison requires identical training seeds")
        for row in rows.values():
            if (any(row[field] != first[field] for field in (*SCOPE_FIELDS, "round"))
                    or set(row["per_task"]) != set(tasks)
                    or [_counts(row["per_task"][task])[0] for task in tasks] != expected_n):
                raise ValueError("factorial scope, round, task or draw inventory mismatch")
            all_rows.append(row)
    if len({rows[seeds[0]]["arm_id"] for rows in arms.values()}) != 4:
        raise ValueError("factorial comparison requires four distinct frozen arms")
    for ref in MATCHED_REFERENCES:
        values = [_ref_identity(row.get("identity_refs", {}).get(ref)) for row in all_rows]
        if any(value is not None for value in values) and any(value != values[0] for value in values):
            raise ValueError(f"factorial comparison has mismatched {ref} identity")
    matrices = {key: np.array([[pass_at_k(*_counts(rows[s]["per_task"][t]), k) for t in tasks]
                               for s in seeds]) for key, rows in arms.items()}
    delta = matrices["11"] - matrices["10"] - matrices["01"] + matrices["00"]
    distributions = _crossed_samples(delta, ns=len(seeds), nt=len(tasks), repeats=repeats, seed=seed)
    mean_delta = float(delta.mean(axis=1).mean())
    raw_summary = {}
    for key, rows in arms.items():
        by_seed = [{"seed": s, "tasks": len(tasks),
                    "draws": sum(_counts(rows[s]["per_task"][t])[0] for t in tasks),
                    "correct": sum(_counts(rows[s]["per_task"][t])[1] for t in tasks)} for s in seeds]
        raw_summary[key] = {"draws": sum(row["draws"] for row in by_seed),
                            "correct": sum(row["correct"] for row in by_seed), "by_seed": by_seed}
    output = {"metric": metric, "contrast": "11-10-01+00", "mean_delta": mean_delta,
              "mean_delta_pp": 100. * mean_delta, "unit": "absolute_probability",
              "independent_unit": "task_and_training_seed",
              **_bootstrap_summary(distributions, seeds=seeds, tasks=tasks, repeats=repeats, seed=seed, alpha=alpha),
              "candidate_round": first["round"], "control_round": first["round"],
              **{field: first[field] for field in SCOPE_FIELDS},
              "arm_ids": {key: rows[seeds[0]]["arm_id"] for key, rows in arms.items()},
              "arm_means": {key: float(values.mean()) for key, values in matrices.items()},
              "per_seed_means": [{"seed": s, **{key: float(values[i].mean()) for key, values in matrices.items()}}
                                 for i, s in enumerate(seeds)], "raw_draw_summary": raw_summary}
    output["interval_pp"] = [100. * value for value in output["interval"]]
    return output


def _cited_assessment(record, cache):
    """Check citation bytes, not reviewer authority or empirical truth."""
    if not isinstance(record, Mapping):
        return False
    refs = record.get("evidence_refs")
    if not isinstance(refs, list) or not refs:
        return False
    for ref in refs:
        if not isinstance(ref, Mapping) or not isinstance(ref.get("path"), str):
            return False
        sha = ref.get("sha256")
        if not isinstance(sha, str) or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
            return False
        key = (ref["path"], sha)
        if key not in cache:
            try:
                path = Path(ref["path"])
                if not path.is_file():
                    cache[key] = False
                else:
                    digest = hashlib.sha256()
                    with path.open("rb") as source:
                        for block in iter(lambda: source.read(1024 * 1024), b""):
                            digest.update(block)
                    cache[key] = digest.hexdigest() == sha
            except (OSError, ValueError):
                cache[key] = False
        if not cache[key]:
            return False
    return True


def evaluate_criteria(estimates, criteria, evidence_context):
    """Evaluate frozen interval rules; never infer empirical qualification.

    Missing prerequisites or invalid measurements precede adverse/positive
    outcomes. Mechanism contradiction matters after task effects are supported.
    """
    if not isinstance(estimates, Mapping) or not isinstance(criteria, (list, tuple)) or not isinstance(evidence_context, Mapping):
        raise ValueError("invalid criteria evaluation contract")
    issues = list(evidence_context.get("analysis_issues", []))
    cache, results = {}, []
    if evidence_context.get("phase") != "confirmation":
        issues.append(_issue("independent_confirmation_required"))
    expected_seeds = evidence_context.get("expected_seeds")
    if (not isinstance(expected_seeds, (list, tuple))
            or any(not _integer(s) for s in expected_seeds)
            or sorted(expected_seeds) != list(CONFIRMATION_SEEDS)):
        issues.append(_issue("confirmation_seed_plan_mismatch"))
    if evidence_context.get("endpoint_round") != 3:
        issues.append(_issue("fixed_endpoint_round_required"))
    prerequisites = evidence_context.get("prerequisites", {})
    for key in PREREQUISITES:
        record = prerequisites.get(key) if isinstance(prerequisites, Mapping) else None
        if (not isinstance(record, Mapping) or record.get("assessment") != "supported"
                or not _cited_assessment(record, cache)):
            issues.append(_issue("prerequisite_evidence_missing_or_invalid", prerequisite=key))
    if not criteria:
        issues.append(_issue("no_frozen_criteria"))
    seen = set()
    for rule in criteria:
        if (not isinstance(rule, Mapping) or not isinstance(rule.get("criterion_id"), str)
                or not rule["criterion_id"] or rule["criterion_id"] in seen
                or not isinstance(rule.get("estimate_id"), str)
                or rule.get("relation") not in {"ge", "le"}
                or not _finite(rule.get("threshold"))
                or rule.get("unit") not in {"absolute_probability", "relative_fraction"}
                or rule.get("kind", "task") not in {"task", "resource"}):
            raise ValueError("invalid or duplicate frozen criterion")
        seen.add(rule["criterion_id"])
        result = {**rule, "status": "inconclusive"}
        value = estimates.get(rule["estimate_id"])
        valid = isinstance(value, Mapping)
        interval = value.get("interval") if valid else None
        valid = (valid and value.get("valid") is True and not value.get("flags")
                 and value.get("unit") == rule["unit"] and _finite(value.get("mean_delta"))
                 and isinstance(interval, (list, tuple)) and len(interval) == 2
                 and all(_finite(x) for x in interval) and interval[0] <= interval[1])
        if not valid:
            issues.append(_issue("invalid_or_missing_interval", estimate_id=rule["estimate_id"]))
        else:
            observed_seeds = value.get("seeds")
            if (not isinstance(observed_seeds, (list, tuple))
                    or any(not _integer(s) for s in observed_seeds)
                    or sorted(observed_seeds) != list(CONFIRMATION_SEEDS)
                    or value.get("n_seeds") != 5):
                issues.append(_issue("incomplete_confirmation_seeds", estimate_id=rule["estimate_id"]))
            if value.get("candidate_round") != 3 or value.get("control_round") not in {0, 3}:
                issues.append(_issue("endpoint_round_mismatch", estimate_id=rule["estimate_id"]))
            lower, upper = interval
            threshold = rule["threshold"]
            if rule["relation"] == "ge":
                state = "supported" if lower >= threshold else "excluded" if upper < threshold else "inconclusive"
            else:
                state = "supported" if upper <= threshold else "excluded" if lower > threshold else "inconclusive"
            result.update(status=state, mean_delta=value["mean_delta"], interval=list(interval))
        results.append(result)
    # Invalid inventory/scoring/optimization is not negative scientific evidence.
    decision = "INCONCLUSIVE"
    if not issues and any(row["status"] == "excluded" for row in results):
        decision = "KILL"
    elif not issues and all(row["status"] == "supported" for row in results):
        required = evidence_context.get("required_mechanisms")
        mechanisms = evidence_context.get("mechanisms", {})
        if (not isinstance(required, (list, tuple)) or not required
                or any(not isinstance(key, str) or not key for key in required)
                or len(set(required)) != len(required)):
            issues.append(_issue("mechanism_requirements_missing"))
        else:
            mechanism_states = []
            for key in required:
                record = mechanisms.get(key) if isinstance(mechanisms, Mapping) else None
                if (not isinstance(record, Mapping) or not _cited_assessment(record, cache)
                        or record.get("assessment") not in {"supported", "contradicted", "unresolved"}):
                    issues.append(_issue("mechanism_evidence_missing", mechanism_id=key))
                else:
                    mechanism_states.append(record["assessment"])
            if not issues:
                if "contradicted" in mechanism_states:
                    decision = "REVISE"
                elif all(state == "supported" for state in mechanism_states):
                    decision = "PASS"
    return {"decision": decision, "criteria": results, "issues": issues,
            "scope": ANALYSIS_SCOPE, "gate_advanced": False, "scientific_result_verified": False}


def _coverage_metric(comparison):
    metrics = comparison.get("metrics", [])
    coverage = [metric for metric in metrics if metric in {"pass@5", "pass@10"}]
    if "pass@1" not in metrics or len(coverage) != 1:
        raise ValueError("G01 requires pass@1 and exactly one native coverage endpoint")
    return coverage[0]


def g01_criteria(comparisons, *, method_id, self_improvement_claim=True):
    """Construct explicit retained G01 thresholds, not data-selected cutoffs."""
    rules = []
    for comparison in comparisons:
        role = comparison.get("control_role", "baseline")
        if "factorial_arms" in comparison or role in {"mechanism", "distinctive", "ablation"}:
            continue
        coverage = _coverage_metric(comparison)
        cid = comparison["comparison_id"]

        def add(metric, threshold, relation="ge", kind="task"):
            rules.append({"criterion_id": f"{cid}:{metric}:{relation}:{threshold:g}",
                          "estimate_id": f"{cid}:{metric}", "relation": relation,
                          "threshold": threshold, "kind": kind,
                          "unit": "relative_fraction" if kind == "resource" else "absolute_probability"})

        if method_id == "M07":
            add("pass@1", -.01)
            add(coverage, -.01)
            if comparison.get("control_arm") == "full_soft" or role == "efficiency_reference":
                if "training_seconds_relative" not in comparison["metrics"]:
                    raise ValueError("M07 full-soft comparison requires training time endpoint")
                add("training_seconds_relative", -.10, "le", "resource")
        elif role == "initial_model":
            if self_improvement_claim:
                add("pass@1", .02)
            add(coverage, -.01)
        else:
            add("pass@1", -.01)
            add(coverage, .02)
    return rules


def comparison_report(expected_cells, observations, comparisons, *, family_size,
                      criteria, evidence_context, repeats=20000, seed=20261006, alpha=.05):
    """Analyze every frozen comparison, including unavailable endpoint slots."""
    _bootstrap_config(repeats, seed, alpha)
    if not isinstance(comparisons, (list, tuple)) or not isinstance(evidence_context, Mapping):
        raise ValueError("invalid comparison report contract")
    ids, slots = set(), 0
    for comparison in comparisons:
        factorial = isinstance(comparison, Mapping) and "factorial_arms" in comparison
        required = ("comparison_id", *SCOPE_FIELDS) if factorial else ("comparison_id", "candidate_arm", "control_arm", *SCOPE_FIELDS)
        if (not isinstance(comparison, Mapping)
                or any(not isinstance(comparison.get(key), str) or not comparison[key] for key in required)
                or comparison["comparison_id"] in ids):
            raise ValueError("invalid or duplicate comparison identity")
        ids.add(comparison["comparison_id"])
        if factorial:
            arms = comparison["factorial_arms"]
            if (not isinstance(arms, (list, tuple)) or len(arms) != 4
                    or any(not isinstance(arm, str) or not arm for arm in arms)
                    or len(set(arms)) != 4 or not _integer(comparison.get("round", 3))
                    or "candidate_arm" in comparison or "control_arm" in comparison):
                raise ValueError("factorial comparison requires four distinct ordered arms and one round")
        metrics = comparison.get("metrics")
        if (not isinstance(metrics, (list, tuple)) or not metrics
                or any(not isinstance(m, str) for m in metrics) or len(set(metrics)) != len(metrics)):
            raise ValueError("invalid or duplicate comparison endpoints")
        for metric in metrics:
            _metric_k(metric)
            coverage = NATIVE_COVERAGE.get(comparison["benchmark"])
            if coverage is None or metric not in {"pass@1", coverage, "training_seconds_relative"}:
                raise ValueError("comparison requires the frozen native benchmark endpoints")
            if factorial and metric == "training_seconds_relative":
                raise ValueError("factorial comparison requires native pass@k endpoints")
        for key in ("candidate_round", "control_round"):
            if not _integer(comparison.get(key, 3)):
                raise ValueError("invalid comparison round")
        if comparison.get("control_round", 3) == 0 and comparison.get("control_role") != "initial_model":
            raise ValueError("round-zero control requires initial_model role")
        slots += len(metrics)
    if not _integer(family_size, 1) or family_size < slots or family_size < len(criteria):
        raise ValueError("frozen family size must cover every planned endpoint and criterion")
    inventory = validate_inventory(expected_cells, observations)
    adjusted_alpha = alpha / family_size
    estimates, comparison_rows, analysis_issues = {}, [], []
    if not inventory["complete"]:
        analysis_issues.append(_issue("comparison_inventory_incomplete"))
    eligible_by_id = {row["cell_id"]: row for row in inventory["eligible_cells"]}
    for comparison in comparisons:
        cid = comparison["comparison_id"]
        factorial = "factorial_arms" in comparison
        selected = {}
        missing = []
        arm_names = (dict(zip(("00", "01", "10", "11"), comparison["factorial_arms"])) if factorial
                     else {label: comparison[f"{label}_arm"] for label in ("candidate", "control")})
        for label, arm_name in arm_names.items():
            round_ = comparison.get("round", 3) if factorial else comparison.get(f"{label}_round", 3)
            planned = [row for row in expected_cells if isinstance(row, Mapping)
                       and row.get("arm_id") == arm_name
                       and row.get("round") == round_
                       and all(row.get(field) == comparison[field] for field in SCOPE_FIELDS)]
            selected[label] = [eligible_by_id[row["cell_id"]] for row in planned if row.get("cell_id") in eligible_by_id]
            missing.extend(row.get("cell_id") for row in planned if row.get("cell_id") not in eligible_by_id)
            if not planned:
                missing.append(f"unplanned:{label}")
        row_report = {"comparison_id": cid, "specification": dict(comparison),
                      "status": "incomplete" if missing else "complete", "missing_or_ineligible_cells": missing,
                      "estimate_ids": []}
        if factorial:
            row_report["factorial_cells"] = {key: [row["cell_id"] for row in cells] for key, cells in selected.items()}
        else:
            row_report.update({f"{key}_cells": [row["cell_id"] for row in cells] for key, cells in selected.items()})
        for metric in comparison["metrics"]:
            eid = f"{cid}:{metric}"
            row_report["estimate_ids"].append(eid)
            if missing:
                estimates[eid] = {"valid": False, "flags": ["incomplete_comparison"],
                                  "metric": metric, "interval": None, "mean_delta": None,
                                  "alpha": adjusted_alpha, "missing_or_ineligible_cells": list(missing)}
                continue
            try:
                if factorial:
                    estimates[eid] = factorial_interval(*(selected[key] for key in ("00", "01", "10", "11")),
                                                         metric, repeats=repeats, seed=seed, alpha=adjusted_alpha)
                else:
                    estimates[eid] = crossed_interval(selected["candidate"], selected["control"], metric,
                                                       repeats=repeats, seed=seed, alpha=adjusted_alpha)
                estimates[eid]["control_role"] = comparison.get("control_role", "baseline")
                if not estimates[eid]["valid"]:
                    row_report["status"] = "invalid"
            except ValueError as exc:
                row_report["status"] = "invalid"
                estimates[eid] = {"valid": False, "flags": ["invalid_paired_comparison"],
                                  "metric": metric, "interval": None, "mean_delta": None,
                                  "alpha": adjusted_alpha, "reason": str(exc)}
        if row_report["status"] != "complete":
            analysis_issues.append(_issue("required_comparison_unavailable", comparison_id=cid,
                                          status=row_report["status"]))
        comparison_rows.append(row_report)
    evaluated = evaluate_criteria(estimates, criteria, {**evidence_context, "analysis_issues": analysis_issues})
    return {"schema_version": "native-comparison-analysis-v1", "inventory": inventory,
            "comparisons": comparison_rows, "estimates": estimates,
            "family_size": family_size, "planned_endpoint_slots": slots,
            "family_size_source": "externally frozen; includes unavailable comparisons",
            "family_alpha": alpha, "per_interval_alpha": adjusted_alpha,
            "multiplicity": "Bonferroni", "repeats": repeats, "analysis_seed": seed, **evaluated}


compare_report = comparison_report
