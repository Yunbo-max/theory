"""Measured calibration for matched controls, with retained source identities.

These are outcome-independent arithmetic/provenance checks. They do not certify
the scientific validity of the source training, native evaluation, or tuning.
"""
from collections import Counter
from collections.abc import Mapping
import math
from pathlib import Path

from .io import atomic_json, digest, object_hash, read_json, read_jsonl


KEYS = {"clock_m13", "clock_m15", "m13_step_scale", "arithmetic_floor", "head_temperature"}
CONFIRMATION_SEEDS = {23, 47, 71, 101, 131}


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _positive(value, label):
    if not _finite(value) or value <= 0:
        raise ValueError(f"invalid measured {label}")
    return float(value)


def match_head_temperature(mu_rows, target_gini):
    """One beta in [0,1] matching mean Gini on unchanged positive support."""
    if not isinstance(mu_rows, (list, tuple)) or not mu_rows or not _finite(target_gini):
        raise ValueError("invalid head probability/target data")
    rows = []
    for row in mu_rows:
        if (not isinstance(row, (list, tuple)) or not row
                or any(not _finite(p) or p < 0 for p in row)
                or not math.isclose(math.fsum(row), 1., rel_tol=0., abs_tol=1e-6)):
            raise ValueError("head rows must contain normalized finite probabilities")
        support = [float(p) for p in row if p > 0]
        if not support:
            raise ValueError("head support is empty")
        total = math.fsum(support)
        rows.append([math.log(p / total) for p in support])

    def at(beta):
        ginis = []
        for logs in rows:
            maximum = beta * max(logs)
            weights = [math.exp(beta * p - maximum) for p in logs]
            total = math.fsum(weights)
            ginis.append(1. - math.fsum((p / total) ** 2 for p in weights))
        return math.fsum(ginis) / len(ginis)

    low_gini, high_gini = at(1.), at(0.)
    if not low_gini - 1e-8 <= target_gini <= high_gini + 1e-8:
        raise ValueError("target mean Gini cannot be reached on the frozen support with beta in [0,1]")
    if abs(target_gini - low_gini) <= 1e-8:
        beta = 1.
    elif abs(target_gini - high_gini) <= 1e-8:
        beta = 0.
    else:
        lo, hi = 0., 1.
        for _ in range(60):
            beta = (lo + hi) / 2.
            achieved = at(beta)
            if abs(achieved - target_gini) <= 1e-9:
                break
            if achieved > target_gini:
                lo = beta
            else:
                hi = beta
    achieved = at(beta)
    if abs(achieved - target_gini) > 1e-5:
        raise ValueError("head temperature failed its declared Gini tolerance")
    return {"value": beta, "achieved": achieved, "target": float(target_gini),
            "tolerance": 1e-5, "support_sizes": [len(row) for row in rows],
            "procedure": "single_beta_bisection_mean_prefix_gini_unchanged_support",
            "attainable_gini": [low_gini, high_gini]}


def collect_head_diagnostics(policy, records, decode, parameters, deadline, *, chunk=32):
    """Actual frozen-teacher/initial-anchor forwards on self-generated prefixes.

    Uses every completion prefix supplied, stores only positive head support,
    and never reads native answers/tests. Small-policy tests are engineering only.
    """
    import torch
    from .methods import decoder, gini, target
    from .train import completion_inputs
    if not isinstance(records, (list, tuple)) or not records or type(chunk) is not int or chunk < 1:
        raise ValueError("invalid frozen prefix records or chunk size")
    mu_rows, target_rows, support_ids = [], [], []
    forbidden = {"canonical_solution", "test_list", "tests", "answer", "reference_solution"}
    with torch.no_grad():
        for record in records:
            deadline.check()
            if not isinstance(record, Mapping) or forbidden.intersection(record):
                raise ValueError("head calibration accepts self-generated records without answers/tests")
            ids, labels, start = completion_inputs(policy, record)
            teacher = policy.hidden(ids, "teacher")[0, start:].detach()
            anchor = policy.hidden(ids, "anchor")[0, start:].detach()
            for offset in range(0, len(labels), chunk):
                deadline.check()
                logits = policy.logits(teacher[offset:offset + chunk])
                p = logits.float().softmax(-1)
                mu = decoder(logits, decode)
                ap = policy.logits(anchor[offset:offset + chunk]).float().softmax(-1)
                q = target("M06", p, mu, labels[offset:offset + chunk], anchor=ap, **parameters)
                goals = gini(q)
                for probability, goal in zip(mu, goals):
                    support = torch.nonzero(probability > 0, as_tuple=False).flatten()
                    mu_rows.append(probability[support].cpu().tolist())
                    support_ids.append(support.cpu().tolist())
                    target_rows.append(float(goal))
    if not mu_rows:
        raise ValueError("no frozen prefixes were measured")
    return {"status": "complete", "prefix_count": len(mu_rows), "record_count": len(records),
            "mu_rows": mu_rows, "target_gini_rows": target_rows, "support_token_ids": support_ids,
            "target_gini": math.fsum(target_rows) / len(target_rows),
            "mu_sha256": object_hash(mu_rows), "target_gini_sha256": object_hash(target_rows),
            "procedure": "all_self_generated_prefixes_frozen_teacher_and_initial_anchor_M06_target",
            "scope": "distribution diagnostic; no task correctness evidence"}


def _reference(ref, base, retained):
    if (not isinstance(ref, Mapping) or not isinstance(ref.get("path"), str)
            or not isinstance(ref.get("sha256"), str) or len(ref["sha256"]) != 64):
        raise ValueError("calibration requires a path and SHA-256 for every source")
    path = Path(ref["path"])
    path = path if path.is_absolute() else Path(base) / path
    path = path.resolve()
    if not path.is_file() or digest(path) != ref["sha256"]:
        raise ValueError(f"calibration source hash mismatch: {path}")
    actual = {"path": str(path), "sha256": ref["sha256"]}
    if actual not in retained:
        retained.append(actual)
    return path


def _scope(source, job, request, arm):
    if source.get("status") not in {"complete", "completed"}:
        raise ValueError("calibration source is not complete")
    if source.get("arm_id") != arm:
        raise ValueError(f"calibration requires the measured {arm} source arm")
    for field in ("model", "model_revision", "training_lineage", "seed"):
        if source.get(field) != job[field]:
            raise ValueError(f"calibration source {field} mismatch")
    if source.get("stage") not in ({"development", "tuning"} if job["seed"] == 17
                                    else {"confirmation", "boundary", "model_boundary"}):
        raise ValueError("calibration source stage does not match development/confirmation scope")
    round_ = request.get("round", source.get("round"))
    if type(round_) is not int or round_ < 1 or source.get("round") != round_:
        raise ValueError("calibration source round mismatch")
    return round_


def _one_source(documents, predicate, label):
    selected = [(path, source) for path, source in documents if isinstance(source, Mapping) and predicate(source)]
    if len(selected) != 1:
        raise ValueError(f"exactly one {label} source receipt is required")
    return selected[0]


def _clock(source, key):
    seconds = _positive(source.get("training_seconds"), "training_seconds")
    costs = source.get("costs")
    if not isinstance(costs, Mapping) or not _finite(costs.get("training_seconds")):
        raise ValueError("measured training cost accounting is missing")
    if not math.isclose(costs["training_seconds"], seconds, rel_tol=1e-9, abs_tol=1e-6):
        raise ValueError("training cost accounting disagrees with the round receipt")
    if key == "clock_m13":
        return {"value": seconds, "procedure": "actual_M13_per_round_training_wall_seconds"}
    parts = {name: costs.get(name) for name in ("generation_seconds", "proxy_seconds", "training_seconds", "save_seconds")}
    if any(not _finite(value) or value < 0 for value in parts.values()):
        raise ValueError("incomplete or invalid measured M15 cost components")
    total = _positive(costs.get("total_round_seconds"), "total round cost")
    if not math.isclose(total, math.fsum(parts.values()), rel_tol=1e-9, abs_tol=1e-6):
        raise ValueError("M15 total round cost does not equal all measured phase costs")
    return {"value": total, "procedure": "actual_M15_generation_proxy_training_save_wall_seconds", "components": parts}


def _step_scale(source_path, source, retained):
    training_path = _reference(source.get("training_ref"), source_path.parent, retained)
    events_path = _reference(source.get("events_ref"), source_path.parent, retained)
    count = read_json(training_path).get("attempted_updates")
    if type(count) is not int or count < 1:
        raise ValueError("complete optimizer attempt count is required")
    events = [event for event in read_jsonl(events_path) if event.get("kind") == "optimizer_step"]
    groups = [event.get("group") for event in events]
    if (len(events) != count or any(type(group) is not int for group in groups)
            or sorted(groups) != list(range(count))):
        raise ValueError("optimizer events are missing, duplicated, or inconsistent with attempted updates")
    for event in events:
        gamma, accepted = event.get("backtrack_gamma"), event.get("accepted")
        if (type(accepted) is not bool or not _finite(gamma) or not 0 <= gamma <= 1
                or (not accepted and gamma != 0) or (accepted and gamma <= 0)):
            raise ValueError("optimizer accepted/rejected gamma accounting is inconsistent")
    value = math.fsum(event["backtrack_gamma"] for event in events) / count
    if value == 0:
        raise ValueError("zero effective M13 step scale is unqualified for positive learning-rate training")
    return {"value": value, "attempted_updates": count,
            "rejected_updates": sum(not event["accepted"] for event in events),
            "procedure": "mean_all_actual_optimizer_gammas_including_rejected_zero_steps"}


def _arithmetic(source_path, packet, job, retained):
    selection = packet.get("tuning_selection")
    if not isinstance(selection, Mapping) or selection.get("stage") != "tuning" or selection.get("seed") != 17:
        raise ValueError("arithmetic calibration requires a development-only tuning report")
    for field in ("model", "model_revision", "training_lineage"):
        if selection.get(field) != job[field]:
            raise ValueError(f"arithmetic tuning {field} mismatch")
    inventory, trials = selection.get("inventory", {}), selection.get("trials")
    if (not isinstance(inventory, Mapping) or inventory.get("complete") is not True
            or inventory.get("expected") != 6 or inventory.get("observed") != 6
            or not isinstance(trials, list) or len(trials) != 6):
        raise ValueError("complete fair three-grid development tuning inventory is required")
    ids, budgets = set(), set()
    grid = {"M03": [], "arithmetic_anchor": []}
    for trial in trials:
        if not isinstance(trial, Mapping) or trial.get("family") not in grid:
            raise ValueError("unexpected arithmetic tuning trial family")
        family, arm = trial["family"], trial.get("arm_id")
        parameters = trial.get("parameters", {})
        floor = parameters.get("floor") if isinstance(parameters, Mapping) else None
        if (not isinstance(arm, str) or not arm or arm in ids
                or trial.get("status") not in {"complete", "completed"}
                or trial.get("optimization_valid") is not True or floor not in {.03, .1, .3}
                or not isinstance(trial.get("budget_digest"), str) or not trial["budget_digest"]
                or not isinstance(trial.get("source_refs"), list) or not trial["source_refs"]):
            raise ValueError("unqualified or incomplete fair arithmetic tuning trial")
        ids.add(arm)
        budgets.add(trial["budget_digest"])
        grid[family].append(floor)
        for ref in trial["source_refs"]:
            _reference(ref, source_path.parent, retained)
    if len(budgets) != 1 or any(sorted(values) != [.03, .1, .3] for values in grid.values()):
        raise ValueError("arithmetic tuning grids or compute budgets are not matched")
    selected = selection.get("selected_control", {})
    candidates = [trial for trial in trials if trial["family"] == "arithmetic_anchor"
                  and trial["arm_id"] == selected.get("arm_id") and trial["parameters"] == selected.get("parameters")]
    if len(candidates) != 1:
        raise ValueError("explicit selected arithmetic control is absent from the complete grid")
    return {"value": float(candidates[0]["parameters"]["floor"]), "selected_control": dict(selected),
            "procedure": "explicit_selected_control_from_complete_equal_budget_three_grid_development_tuning",
            "budget_digest": next(iter(budgets))}


def _generate_head_diagnostics(request, job, data, output, deadline, retained):
    """Native policy path; supplied records must cover the clean training set."""
    import torch
    from transformers import AutoModelForCausalLM
    from .methods import Decode
    from .model import Policy
    data = Path(data)
    manifest_path = data / "benchmarks-manifest.json"
    manifest = read_json(manifest_path)
    training_path = _reference({"path": "train_prompts.jsonl", "sha256": manifest["files"]["train_prompts.jsonl"]}, data, retained)
    prompts = read_jsonl(training_path)
    if (manifest.get("training", {}).get("lineage") != "clean-v2" or len(prompts) != 32
            or any(set(row) != {"task_id", "text"} for row in prompts)):
        raise ValueError("head diagnostics require the complete clean-v2 prompt-only training inventory")
    retained.append({"path": str(manifest_path.resolve()), "sha256": digest(manifest_path)})
    records_path = _reference(request.get("records_ref"), Path.cwd(), retained)
    records = read_jsonl(records_path)
    config = request.get("config")
    if not isinstance(config, Mapping) or config.get("samples_per_prompt") != 2:
        raise ValueError("head diagnostics require the frozen two self-responses per training prompt")
    expected_ids = {row["task_id"] for row in prompts}
    counts = Counter(row.get("task_id") for row in records)
    keys = [(row.get("task_id"), row.get("sample_id")) for row in records]
    if set(counts) != expected_ids or any(count != 2 for count in counts.values()) or len(set(keys)) != len(keys):
        raise ValueError("head diagnostic self-response inventory is incomplete or duplicated")
    round_ = request.get("round")
    if type(round_) is not int or round_ < 1 or any(row.get("round") != round_ for row in records):
        raise ValueError("head diagnostic frozen-prefix round mismatch")
    checkpoint = request.get("checkpoint_ref", request.get("checkpoint"))
    checkpoint_path = _reference(checkpoint, Path.cwd(), retained) if checkpoint is not None else None
    parent_sha = digest(checkpoint_path) if checkpoint_path else job["model_revision"]
    if (checkpoint_path is None and round_ != 1) or any(row.get("parent_sha") != parent_sha for row in records):
        raise ValueError("head diagnostics require the actual frozen teacher of the cited self-responses")
    model_path = Path(request.get("model_path", ""))
    if not model_path.is_dir():
        raise ValueError("head diagnostics require the pinned local model directory")
    deadline.check()
    device = request.get("device", "cuda")
    base = AutoModelForCausalLM.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.float16 if str(device).startswith("cuda") else torch.float32, attn_implementation="sdpa")
    policy = Policy(base, rank=config["lora_rank"], device=device)
    if checkpoint_path is not None:
        policy.load_checkpoint(checkpoint_path, "teacher")
    decode_config = request.get("decode", {key: config[key] for key in ("temperature", "top_k", "top_p")})
    result = collect_head_diagnostics(policy, records, Decode(**decode_config), config.get("method_parameters", {}),
                                      deadline, chunk=config["vocab_chunk"])
    result.update({key: job[key] for key in ("model", "model_revision", "seed", "stage", "training_lineage")})
    result.update(arm_id="M06", round=round_, source_refs=list(retained),
                  procedure_ref={"path": str(Path(__file__).resolve()), "sha256": digest(__file__)})
    path = Path(output) / f"head-diagnostics-r{round_}.json"
    atomic_json(path, result)
    _reference({"path": str(path.resolve()), "sha256": digest(path)}, Path.cwd(), retained)
    return path, result


def derive_calibration(job, data, output, deadline):
    """Derive and persist measured values; missing evidence never gets defaults.

    Requests use key, optional round, and source_refs of actual hashed receipts.
    Source relative paths resolve from the current controller directory; nested
    training/events references resolve from their owning receipt directory.
    Head requests can instead supply records_ref/checkpoint_ref/model_path/config
    to execute actual frozen-prefix diagnostics inside the admitted worker.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if not isinstance(job, Mapping) or job.get("stage") not in {"development", "tuning", "confirmation", "boundary", "model_boundary"}:
        raise ValueError("calibration requires an explicit development/confirmation stage")
    development = job["stage"] in {"development", "tuning"}
    if (type(job.get("seed")) is not int or (job["seed"] != 17 if development else job["seed"] not in CONFIRMATION_SEEDS)
            or job.get("training_lineage") != "clean-v2"
            or any(not isinstance(job.get(key), str) or not job[key] for key in ("model", "model_revision"))):
        raise ValueError("invalid calibration seed, model or clean training lineage")
    requests = job.get("requests")
    if not isinstance(requests, list) or not requests:
        raise ValueError("calibration requires measured requests")
    calibration, all_refs, seen = {}, [], set()
    for request in requests:
        deadline.check()
        if not isinstance(request, Mapping) or request.get("key") not in KEYS or "value" in request:
            raise ValueError("unknown calibration or unsupported convenience value")
        key, retained, documents = request["key"], [], []
        refs = request.get("source_refs", [])
        if not isinstance(refs, list):
            raise ValueError("calibration source_refs must be a list")
        for ref in refs:
            path = _reference(ref, Path.cwd(), retained)
            if path.suffix == ".json":
                documents.append((path, read_json(path)))
        if key == "arithmetic_floor":
            path, packet = _one_source(documents, lambda value: "tuning_selection" in value, "development tuning")
            record, round_ = _arithmetic(path, packet, job, retained), None
        elif key == "head_temperature":
            if "records_ref" in request:
                path, source = _generate_head_diagnostics(request, job, data, output, deadline, retained)
            else:
                path, source = _one_source(documents, lambda value: "mu_rows" in value, "head diagnostic")
            round_ = _scope(source, job, request, "M06")
            mu, targets = source.get("mu_rows"), source.get("target_gini_rows")
            if (not isinstance(mu, list) or not isinstance(targets, list) or not mu or len(mu) != len(targets)
                    or source.get("prefix_count") != len(mu) or any(not _finite(value) for value in targets)
                    or source.get("mu_sha256") != object_hash(mu) or source.get("target_gini_sha256") != object_hash(targets)):
                raise ValueError("head probability/target diagnostic hashes or inventories are invalid")
            record = match_head_temperature(mu, math.fsum(targets) / len(targets))
        else:
            arm = "M15" if key == "clock_m15" else "M13"
            path, source = _one_source(documents, lambda value: value.get("arm_id") == arm, arm)
            round_ = _scope(source, job, request, arm)
            record = _step_scale(path, source, retained) if key == "m13_step_scale" else _clock(source, key)
        identity = (key, round_)
        if identity in seen:
            raise ValueError("duplicate calibration request for the same key/round")
        seen.add(identity)
        record.update(source_refs=list(retained), stage=job["stage"], seed=job["seed"],
                      model=job["model"], model_revision=job["model_revision"], training_lineage="clean-v2")
        if round_ is None:
            calibration[key] = record
        else:
            record["round"] = round_
            calibration.setdefault(key, {"rounds": {}})["rounds"][str(round_)] = record
        all_refs.extend(ref for ref in retained if ref not in all_refs)
    identity = object_hash({"job": dict(job), "source_refs": all_refs})
    receipt_path, calibration_path = output / "receipt.json", output / "calibration.json"
    if receipt_path.exists() or calibration_path.exists():
        if (not receipt_path.is_file() or not calibration_path.is_file()
                or read_json(receipt_path).get("identity") != identity
                or read_json(receipt_path).get("calibration_ref", {}).get("sha256") != digest(calibration_path)):
            raise ValueError("existing calibration identity changed; retain it and use a new output directory")
        return read_json(receipt_path)
    atomic_json(calibration_path, calibration)
    receipt = {"schema": "measured-control-calibration-v1", "status": "complete", "identity": identity,
               "calibration_ref": {"path": "calibration.json", "sha256": digest(calibration_path)},
               "source_refs": all_refs, "records": len(seen), "scientific_result_verified": False,
               "scope": "measured control calibration; no empirical gate certification"}
    atomic_json(receipt_path, receipt)
    return receipt
