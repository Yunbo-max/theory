"""Compile native contracts and admit finite, evidence-bound suite work.

This is an adapter to the pinned, unmodified research-autopilot runtime. It
neither generates qualification/review evidence nor turns stored PASS flags
into evidence. Nothing in this module launches a workload. The outer project
harness remains the only execution owner.
"""
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import fcntl
from functools import wraps
import hashlib
import json
import math
import operator
from pathlib import Path
from types import SimpleNamespace

from .harness import runtime


MAX_TOTAL_SECONDS = 8 * 60 * 60
QUALIFICATION_ARMS = frozenset({"initial", "hard", "full_soft", "native_reference"})
QUALIFICATION_CALIBRATION_SOURCES = {
    "head_temperature": "M06", "m13_step_scale": "M13",
    "clock_m13": "M13", "clock_m15": "M15",
}
_CONTAINERS = frozenset({"docker", "podman", "singularity", "apptainer"})


class AdmissionError(ValueError):
    """A precise admission blocker; never an automatic scientific verdict."""

    def __init__(self, code, detail=""):
        self.code = code
        self.detail = detail
        super().__init__(code + (": " + detail if detail else ""))


def qualification_dependency_report(suite):
    """Describe the real bootstrap gap without authorizing new execution.

    This is an authoring diagnostic, not a qualification packet. In particular,
    renaming a selected method as an ablation/control cannot remove the method
    evidence required to execute it. The current four-arm bootstrap is retained.
    No qualification result, gate flag, resource measurement or approval is
    created here. Local may use the report to author a reviewed child protocol.
    """
    from .suite_design import ORDER, arm_registry, comparison_bundles, verify_suite
    from .io import object_hash

    verify_suite(suite)
    registry = arm_registry()
    actual_arms = {row["arm_id"]: row["arm"] for row in suite["trajectories"]}
    catalogue = comparison_bundles()
    candidates = suite.get("candidate_ids", [])
    if not candidates or any(candidate not in catalogue for candidate in candidates):
        raise AdmissionError("QUALIFICATION_CATALOGUE_BINDING_REQUIRED")
    for candidate in candidates:
        if suite.get("method_bundles", {}).get(candidate) != catalogue[candidate]:
            raise AdmissionError("QUALIFICATION_CATALOGUE_BINDING_MISMATCH", candidate)
    bundles = {}
    for candidate in candidates:
        comparisons = catalogue[candidate]
        controls = [arm for arm in comparisons["arms"] if arm != candidate]
        scoped = []
        for arm_id in controls:
            if arm_id not in registry:
                raise AdmissionError("QUALIFICATION_CATALOGUE_ARM_UNKNOWN", arm_id)
            arm = actual_arms.get(arm_id, registry[arm_id])
            native_name = "initial" if arm_id == "base" else arm_id
            prerequisites = []
            for key in arm["required_calibration"]:
                source = QUALIFICATION_CALIBRATION_SOURCES.get(key)
                prerequisites.append({
                    "key": key,
                    "source_candidate_id": source,
                    "source_kind": "candidate_training_receipt" if source else "development_selection",
                    "requires_real_source_refs": True,
                })
            selected_method = arm["method"] if arm["method"] in ORDER else None
            allowed = native_name in QUALIFICATION_ARMS and not selected_method and not prerequisites
            comparator_scope = (not selected_method and suite["stage"] == "development"
                                and not any(item["source_kind"] == "development_selection" for item in prerequisites))
            blockers = []
            if selected_method:
                blockers.append("CANDIDATE_METHOD_REQUIRES_OWN_VERIFIED_DESIGN")
            if any(item["source_candidate_id"] == candidate for item in prerequisites):
                blockers.append("VERIFIED_CALIBRATION_CHILD_REQUIRED")
            if any(item["source_kind"] == "development_selection" for item in prerequisites):
                blockers.append("ACTUAL_COMPLETE_DEVELOPMENT_SELECTION_REQUIRED")
            scoped.append({
                "arm_id": arm_id, "native_arm_name": native_name,
                "definition_digest": object_hash(arm),
                "existing_bootstrap_eligible": allowed,
                "ordinary_comparator_scope_eligible": comparator_scope,
                "selected_candidate_as_control": arm_id if arm_id in ORDER else None,
                "selected_method_implementation": selected_method,
                "calibration_dependencies": prerequisites,
                "requires_verified_child_dependencies": any(item["source_candidate_id"] for item in prerequisites),
                "blockers": blockers,
            })
        bundles[candidate] = {
            "treatment_arm": candidate,
            "required_comparators": scoped,
            "comparator_count": len(scoped),
            "unsupported_bootstrap_arms": [row["arm_id"] for row in scoped
                                            if not row["existing_bootstrap_eligible"]],
            "full_comparator_bootstrap_implemented": all(row["existing_bootstrap_eligible"] for row in scoped),
            "requires_reviewed_child_design": [row["arm_id"] for row in scoped
                                                if not row["ordinary_comparator_scope_eligible"]],
        }
    return {
        "schema": "recursive-ssd-qualification-dependency-report-v1",
        "scope": "source-derived authoring diagnostic; no execution or qualification authority",
        "suite_digest": suite["suite_digest"], "stage": suite["stage"],
        "catalogue_digest": object_hash(catalogue),
        "bootstrap_arm_allowlist": sorted(QUALIFICATION_ARMS),
        "scientific_dispatch_ready": False, "gate_advanced": False,
        "bundles": bundles,
    }


def _checked(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except AdmissionError:
            raise
        except Exception as error:
            if hasattr(error, "code") and hasattr(error, "path"):
                raise AdmissionError(error.code, str(error.path)) from error
            if isinstance(error, (FileNotFoundError, PermissionError)):
                raise AdmissionError("EVIDENCE_UNAVAILABLE", str(error)) from error
            if isinstance(error, KeyError):
                raise AdmissionError("BINDING_REQUIRED", str(error)) from error
            raise
    return wrapped


def _modules():
    C, R, _ = runtime()
    import _native_eval as N
    import verify_methods as M
    return C, R, N, M


def _number(value, code, *, positive=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0 or (positive and value == 0)):
        raise AdmissionError(code, "a finite positive number is required" if positive
                             else "a finite nonnegative number is required")
    return float(value)


def _timestamp(value):
    try:
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value
        if not isinstance(stamp, datetime) or stamp.tzinfo is None or stamp.utcoffset() is None:
            raise AdmissionError("TIMESTAMP_TIMEZONE_REQUIRED")
        return stamp.astimezone(timezone.utc)
    except ValueError as error:
        raise AdmissionError("INVALID_TIMESTAMP", str(value)) from error


def _iso(stamp):
    return stamp.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _now(value):
    return datetime.now(timezone.utc) if value is None else _timestamp(value)


def _refs(C, root, values, code="EVIDENCE_REFS_REQUIRED"):
    if not isinstance(values, list) or not values:
        raise AdmissionError(code)
    seen = set()
    for ref in values:
        if not isinstance(ref, dict) or set(ref) != {"path", "sha256"}:
            raise AdmissionError("INVALID_REFERENCE")
        C.verify_ref(root, ref)
        key = (ref["path"], ref["sha256"])
        if key in seen:
            raise AdmissionError("DUPLICATE_REFERENCE", ref["path"])
        seen.add(key)
    return values


def _native_command(command):
    if not isinstance(command, list) or not command or any(not isinstance(x, str) or not x for x in command):
        raise AdmissionError("COMMAND_ARGV_REQUIRED")
    if any(Path(arg).name.lower() in _CONTAINERS for arg in command):
        raise AdmissionError("NATIVE_EXECUTION_REQUIRED", "container execution is not authorized")
    if Path(command[0]).name in {"sh", "bash", "zsh", "fish"}:
        raise AdmissionError("NATIVE_ARGV_REQUIRED", "use the pinned Python or Conda entry point")


def _contrasts(arms):
    if not isinstance(arms, dict) or not {"treatment", "baseline"}.issubset(arms) or len(arms) < 3:
        raise AdmissionError("NATIVE_ARM_REQUIREMENTS_INCOMPLETE")
    controls = [role for role in arms if role not in {"treatment", "baseline"}]
    for role in controls:
        if arms[role].get("name") != role:
            raise AdmissionError("NATIVE_CONTROL_ROLE_NAME_MISMATCH", role)
    return {"treatment": arms["treatment"]["name"], "baseline": arms["baseline"]["name"],
            "controls": controls}


def _rows(C, root, ref):
    path = C.verify_ref(root, ref)
    return [C.parse_json(line) for line in path.read_text().splitlines() if line.strip()]


def _immutable(C, root, relative, value):
    path = C.safe_path(root, relative)
    if path.exists():
        if C.load_file(path) != value:
            raise AdmissionError("NATIVE_ADAPTER_IMMUTABLE_CREATE_CHILD", relative)
    else:
        C.atomic(path, C.canonical(value) + "\n")
    return C.reference(root, path)


def _verify_split_adapter(C, root, definition, samples):
    ref = definition.get("project_split_adapter_ref")
    if not ref:
        if "/project-" in samples["split"]:
            raise AdmissionError("PROJECT_SPLIT_ADAPTER_REQUIRED")
        return
    adapter = C.load_file(C.verify_ref(root, ref))
    if adapter.get("version") != "humaneval-project-split-v1":
        raise AdmissionError("PROJECT_SPLIT_ADAPTER_UNSUPPORTED")
    for key in ("asset_manifest_ref", "human_eval_split_ref", "subset_support_ref"):
        C.verify_ref(root, adapter[key])
    original = _rows(C, root, adapter["source_tests_ref"])
    selected = _rows(C, root, adapter["selected_tests_ref"])
    ids = [row["task_id"] for row in original]
    if len(ids) != 164 or len(set(ids)) != 164 or adapter["source_count"] != 164:
        raise AdmissionError("ORIGINAL_HUMANEVAL_INVENTORY_REQUIRED")
    ranked = sorted(ids, key=lambda task: int(hashlib.sha256(
        ("eval-split-v1|" + task).encode()).hexdigest()[:15], 16))
    split = adapter["project_split"]
    if split not in {"dev", "confirm"}:
        raise AdmissionError("PROJECT_SPLIT_UNSUPPORTED")
    expected = ranked[:32] if split == "dev" else ranked[32:]
    retained = C.load_file(C.verify_ref(root, adapter["human_eval_split_ref"]))
    if retained.get("dev_ids") != ranked[:32] or retained.get("confirm_ids") != ranked[32:]:
        raise AdmissionError("RETAINED_HUMANEVAL_SPLIT_MISMATCH")
    original_by_id = {row["task_id"]: row for row in original}
    if (adapter["sample_ids"] != expected or samples["sample_ids"] != expected
            or [row["task_id"] for row in selected] != expected
            or any(row != original_by_id[row["task_id"]] for row in selected)):
        raise AdmissionError("PROJECT_SUBSET_NATIVE_ROW_MISMATCH")
    if (samples["split"] != adapter["upstream_split"] + "/project-" + split
            or samples["labels_or_tests_ref"] != adapter["selected_tests_ref"]):
        raise AdmissionError("PROJECT_SPLIT_IDENTITY_MISMATCH")


def _definition_inputs(C, root, definition):
    """Close indexed native labels over every original immutable test shard."""
    refs = definition.get("required_input_refs", [])
    if refs:
        _refs(C, root, refs, "NATIVE_TEST_SOURCE_REFS_REQUIRED")
    return refs


@_checked
def prepare_native_assets(root, asset_manifest_ref, *, groups, definitions, output_dir,
                          human_eval_split_ref=None, subset_support_ref=None):
    """Bridge prepared releases to pinned native sample/definition sidecars.

    ``groups`` maps group IDs to {benchmark, split}. ``definitions`` supplies
    exact source-backed upstream identity/metric/scorer/sampling/budget metadata
    for each group and an ``upstream_split``. This function never infers upstream
    publication dates, qualification thresholds, or scorer identities.

    HumanEval dev/confirm use the retained, score-independent original split.
    They are explicitly called test/project-dev and test/project-confirm (for
    upstream_split=test), never described as released upstream splits. Every
    selected row is checked byte-for-value against the full original release.
    The versioned adapter and native subset-support source remain pinned.
    """
    C, _, N, _ = _modules()
    root = C.root_path(root)
    asset_path = C.verify_ref(root, asset_manifest_ref)
    asset = C.load_file(asset_path)
    if asset.get("version") != "clean-v2-native-benchmarks-1":
        raise AdmissionError("BENCHMARK_ASSET_VERSION_UNSUPPORTED")
    if not isinstance(groups, dict) or not groups or set(groups) != set(definitions):
        raise AdmissionError("NATIVE_GROUP_CONTRACTS_INCOMPLETE")
    asset_root = asset_path.parent
    output = C.safe_path(root, str(output_dir))
    full_counts = {"humaneval": 164, "mbpp": 378, "livecodebench": 880}
    result = {"benchmark_manifest": {}, "native_definition_refs": {}, "split_adapter_refs": {}}
    for group, scope in groups.items():
        benchmark, split = scope["benchmark"], scope["split"]
        if benchmark not in full_counts or (benchmark != "humaneval" and split != "full"):
            raise AdmissionError("SUITE_BENCHMARK_SCOPE_UNSUPPORTED", str(scope))
        if split not in {"full", "dev", "confirm"}:
            raise AdmissionError("SUITE_BENCHMARK_SCOPE_UNSUPPORTED", str(scope))
        info = asset["benchmarks"][benchmark]["splits"][split]
        full = asset["benchmarks"][benchmark]["splits"]["full"]

        def asset_ref(relative):
            source = C.safe_path(asset_root, relative)
            ref = {"path": source.relative_to(root).as_posix(), "sha256": asset["files"][relative]}
            C.verify_ref(root, ref)
            return ref

        tests_ref = asset_ref(info["path"])
        full_ref = asset_ref(full["path"])
        if benchmark == "livecodebench" and (root / tests_ref["path"]).stat().st_size > 128 * 1024 * 1024:
            raise AdmissionError("INDEXED_NATIVE_STORAGE_REQUIRED", "prepare the complete native shard index first")
        rows, full_rows = _rows(C, root, tests_ref), _rows(C, root, full_ref)
        id_key = "question_id" if benchmark == "livecodebench" else "task_id"
        ids, full_ids = [row[id_key] for row in rows], [row[id_key] for row in full_rows]
        if (full["count"] != full_counts[benchmark] or len(full_ids) != full_counts[benchmark]
                or len(set(full_ids)) != len(full_ids) or full_ids != full["task_ids"]):
            raise AdmissionError("ORIGINAL_BENCHMARK_INVENTORY_REQUIRED", benchmark)
        if ids != info["task_ids"] or len(ids) != info["count"] or len(ids) != len(set(ids)):
            raise AdmissionError("PREPARED_SAMPLE_INVENTORY_MISMATCH", group)
        definition = deepcopy(definitions[group])
        if "splits" in definition or "project_split_adapter_ref" in definition:
            raise AdmissionError("ADAPTER_DERIVES_SPLITS_FROM_RELEASED_ASSETS")
        upstream = definition.pop("upstream_split")
        if not isinstance(upstream, str) or not upstream:
            raise AdmissionError("UPSTREAM_NATIVE_SPLIT_REQUIRED")
        for key in ("benchmark_id", "benchmark_revision", "published_at", "source_url", "primary_metric",
                    "metrics", "prediction_format", "sampling", "budget", "scorer", "publication_refs"):
            if key not in definition:
                raise AdmissionError("NATIVE_PUBLISHED_DEFINITION_REQUIRED", key)
        _refs(C, root, definition["publication_refs"])
        N._scorer(root, definition["scorer"])
        _native_command(definition["scorer"]["command"])
        if definition["scorer"]["kind"] != "official":
            raise AdmissionError("NATIVE_DEFINITION_REQUIRES_OFFICIAL_SCORER")
        n = definition["sampling"]["parameters"].get("n")
        if type(n) is not int or n != 10:
            raise AdmissionError("SUITE_TEN_SAMPLES_REQUIRED")
        metric_names = {m["name"] for m in definition["metrics"]}
        required = {"pass@1", "pass@5" if benchmark == "livecodebench" else "pass@10"}
        if not required.issubset(metric_names) or definition["primary_metric"] != "pass@1":
            raise AdmissionError("SUITE_NATIVE_METRIC_REQUIRED", group)
        suffix = C.hashed(group)[:16]
        folder = (output / suffix).relative_to(root).as_posix()
        if benchmark == "livecodebench" and any("_native_ref" in row for row in rows):
            # The 880-row index is deliberately small. Native private-test
            # shards stay on disk; hash verification streams instead of loading
            # their multi-gigabyte contents into Python objects.
            if not all(isinstance(row.get("_native_ref"), dict) for row in rows):
                raise AdmissionError("NATIVE_TEST_INDEX_INCOMPLETE")
            sidecar_ref = asset_ref("livecodebench-native-sources.json")
            sidecar = C.load_file(C.verify_ref(root, sidecar_ref))
            if (sidecar.get("storage") != "indexed-native-jsonl" or sidecar.get("total_count") != 880
                    or sidecar.get("release") != "release_v5"
                    or sidecar.get("index") != {"path": info["path"], "sha256": tests_ref["sha256"]}):
                raise AdmissionError("NATIVE_TEST_INDEX_SOURCE_MISMATCH")
            names = {row["_native_ref"]["path"] for row in rows}
            if names != set(sidecar["files"]) or len(names) != 5:
                raise AdmissionError("NATIVE_TEST_SHARD_INVENTORY_MISMATCH")
            shard_refs = [asset_ref(name) for name in sorted(names)]
            sizes = {}
            for ref in shard_refs:
                relative = str((root / ref["path"]).relative_to(asset_root))
                source = sidecar["files"][relative]
                actual_size = (root / ref["path"]).stat().st_size
                if source["sha256"] != ref["sha256"] or source["bytes"] != actual_size:
                    raise AdmissionError("NATIVE_TEST_SHARD_IDENTITY_MISMATCH", relative)
                sizes[relative] = actual_size
            for row in rows:
                pointer = row["_native_ref"]
                offset, length = pointer["offset_bytes"], pointer["length_bytes"]
                if (type(offset) is not int or type(length) is not int or offset < 0 or length <= 0
                        or offset + length > sizes[pointer["path"]]
                        or not isinstance(pointer.get("sha256"), str) or len(pointer["sha256"]) != 64):
                    raise AdmissionError("NATIVE_TEST_INDEX_OFFSET_INVALID")
            label_manifest = {"version": "native-labels-shards-v1", "benchmark": "livecodebench",
                              "release": "release_v5", "asset_manifest_ref": deepcopy(asset_manifest_ref),
                              "native_sources_ref": sidecar_ref, "index_ref": tests_ref,
                              "source_refs": shard_refs, "sample_ids": ids,
                              "labels": "unaltered original native JSONL source lines"}
            labels_ref = _immutable(C, root, folder + "/native-labels.json", label_manifest)
            definition["required_input_refs"] = [deepcopy(asset_manifest_ref), sidecar_ref, tests_ref, *shard_refs]
            definition["native_test_storage"] = "indexed-native-jsonl"
            tests_ref = labels_ref
        native_split = upstream
        if split != "full":
            if human_eval_split_ref is None or subset_support_ref is None:
                raise AdmissionError("PROJECT_SPLIT_SOURCE_BINDINGS_REQUIRED")
            native_split = upstream + "/project-" + split
            adapter = {"version": "humaneval-project-split-v1", "asset_manifest_ref": deepcopy(asset_manifest_ref),
                "upstream_split": upstream, "project_split": split,
                "selection_rule": "ascending sha256('eval-split-v1|' + task_id)[:15]; first32/rest132",
                "human_eval_split_ref": deepcopy(human_eval_split_ref),
                "subset_support_ref": deepcopy(subset_support_ref), "source_tests_ref": full_ref,
                "selected_tests_ref": tests_ref, "source_count": len(full_ids), "sample_ids": ids,
                "scope": "project-selected published tasks; native tests and scoring retained"}
            adapter_ref = _immutable(C, root, folder + "/split-adapter.json", adapter)
            definition["project_split_adapter_ref"] = adapter_ref
            result["split_adapter_refs"][group] = adapter_ref
        sample = {"benchmark_id": definition["benchmark_id"], "benchmark_revision": definition["benchmark_revision"],
                  "split": native_split, "sample_ids": ids, "denominator": len(ids),
                  "predictions_per_sample": n, "labels_or_tests_ref": tests_ref,
                  "sampling": deepcopy(definition["sampling"]), "budget": deepcopy(definition["budget"])}
        _verify_split_adapter(C, root, definition, sample)
        sample_ref = _immutable(C, root, folder + "/samples.json", sample)
        definition["splits"] = [{"name": native_split, "sample_manifest_ref": sample_ref,
                                 "labels_or_tests_ref": tests_ref}]
        definition["asset_manifest_ref"] = deepcopy(asset_manifest_ref)
        definition["upstream_split"] = upstream
        definition["released_full_tests_ref"] = tests_ref if definition.get("native_test_storage") else full_ref
        definition["scope"] = "full published release" if split == "full" else "project-selected published tasks"
        definition_ref = _immutable(C, root, folder + "/native-definition.json", definition)
        result["benchmark_manifest"][group] = sample_ref
        result["native_definition_refs"][group] = definition_ref
    return result


@_checked
def build_native_contract(root, sample_manifest_ref, arms, scorer_spec):
    """Compile one contract from an already captured native definition.

    ``scorer_spec`` has native_definition_ref, baseline_qualification and
    control_qualifications, plus optional scorer/selection. The native definition
    is an authenticated upstream adapter supplied by the caller, not generated
    from remembered facts here. Its metric paths and scorer are copied exactly.
    """
    C, _, N, _ = _modules()
    root = C.root_path(root)
    samples = C.load_file(C.verify_ref(root, sample_manifest_ref))
    definition = C.load_file(C.verify_ref(root, scorer_spec["native_definition_ref"]))
    _verify_split_adapter(C, root, definition, samples)
    _definition_inputs(C, root, definition)
    ids = samples.get("sample_ids")
    if (not isinstance(ids, list) or not ids or any(not isinstance(x, str) or not x for x in ids)
            or len(set(ids)) != len(ids)):
        raise AdmissionError("NATIVE_SAMPLE_IDENTITIES_INVALID")
    if type(samples.get("denominator")) is not int or samples["denominator"] != len(ids):
        raise AdmissionError("NATIVE_DENOMINATOR_MISMATCH", "this suite counts native tasks")
    count = samples.get("predictions_per_sample")
    if type(count) is not int or count <= 0:
        raise AdmissionError("NATIVE_SAMPLING_INVALID")
    parameters = samples.get("sampling", {}).get("parameters", {})
    if "n" in parameters and parameters["n"] != count:
        raise AdmissionError("NATIVE_SAMPLING_COUNT_MISMATCH")
    contract = C.envelope("native-eval-contract",
        benchmark_id=samples["benchmark_id"], benchmark_revision=samples["benchmark_revision"],
        split=samples["split"], sample_manifest_ref=deepcopy(sample_manifest_ref),
        labels_or_tests_ref=deepcopy(samples["labels_or_tests_ref"]),
        native_definition_ref=deepcopy(scorer_spec["native_definition_ref"]),
        published_source_refs=deepcopy(definition["publication_refs"]),
        primary_metric=definition["primary_metric"], metrics=deepcopy(definition["metrics"]),
        prediction_format=deepcopy(definition["prediction_format"]),
        sampling=deepcopy(samples["sampling"]), budget=deepcopy(samples["budget"]),
        scorer=deepcopy(scorer_spec.get("scorer", definition["scorer"])),
        contrasts=_contrasts(arms), arm_requirements=deepcopy(arms),
        baseline_qualification=deepcopy(scorer_spec["baseline_qualification"]),
        control_qualifications=deepcopy(scorer_spec["control_qualifications"]))
    if "selection" in scorer_spec:
        contract["selection"] = deepcopy(scorer_spec["selection"])
    _native_command(contract["scorer"]["command"])
    N.verify_protocol(root, {"native_eval_contract": contract, "contrasts": contract["contrasts"]})
    effective = C.load_file(C.verify_ref(root, N.sample_manifest_ref(contract)))
    if effective["denominator"] != len(effective["sample_ids"]):
        raise AdmissionError("NATIVE_DENOMINATOR_MISMATCH")
    return contract


def _comparator_qualification(root, protocol):
    """Verify a finite exact-catalogue scope; never admit a selected method."""
    from .suite_design import ORDER, make_suite
    C, _, _, M = _modules()
    scope = protocol["suite_qualification"]
    suite = C.load_file(C.verify_ref(root, scope["suite_ref"]))
    if suite.get("stage") not in {"development", "tuning"}:
        raise AdmissionError("QUALIFICATION_DEVELOPMENT_SUITE_REQUIRED")
    expected = make_suite(suite["stage"], model=suite.get("model"),
                          calibration=suite.get("calibration"),
                          tuning_selection=suite.get("tuning_selection"))
    if suite != expected:
        raise AdmissionError("QUALIFICATION_EXACT_CATALOGUE_REQUIRED")
    bundle = scope.get("bundle_id")
    if bundle not in suite["method_bundles"]:
        raise AdmissionError("QUALIFICATION_CATALOGUE_BINDING_REQUIRED")
    if not isinstance(scope.get("config"), dict) or scope.get("calibration") != {}:
        raise AdmissionError("QUALIFICATION_EXACT_SETTINGS_REQUIRED")
    tuning = scope.get("tuning_budget")
    if suite["stage"] == "tuning":
        if not isinstance(tuning, dict) or set(tuning) != {"trial_wall_seconds", "source_refs"}:
            raise AdmissionError("QUALIFICATION_FROZEN_TUNING_ALLOWANCE_REQUIRED")
        if _number(tuning["trial_wall_seconds"], "QUALIFICATION_FINITE_TUNING_ALLOWANCE_REQUIRED", positive=True) > MAX_TOTAL_SECONDS:
            raise AdmissionError("AUTHORIZED_CAP_EXCEEDED")
        _refs(C, root, tuning["source_refs"])
    elif tuning is not None:
        raise AdmissionError("QUALIFICATION_TUNING_SCOPE_MISMATCH")
    provenance = scope.get("execution_provenance")
    if (not isinstance(provenance, dict)
            or set(provenance) != {"model_revision", "data_revision", "environment_digest"}
            or any(not isinstance(value, str) or not value for value in provenance.values())
            or provenance["model_revision"] != suite["model_revision"]):
        raise AdmissionError("QUALIFICATION_PROVENANCE_BINDINGS_REQUIRED")
    batch = M.read(root, scope["method_verification_ref"], "method-verification-batch")
    boundary = M.before_action(root, batch, "experiment-design", bundle)["workflow_boundary"]
    if not boundary["ready"]:
        raise AdmissionError("QUALIFICATION_CURRENT_CODE_EVIDENCE_REQUIRED",
                             ", ".join(boundary["reason_codes"]))
    entry = next(row for row in batch["candidates"] if row["candidate_id"] == bundle)
    implementation = C.load_file(C.verify_ref(root, entry["implementation_ref"]))
    arms = {row["arm_id"]: row["arm"] for row in suite["trajectories"]}
    bindings = scope.get("arm_bindings")
    roles = scope["allowed_arm_roles"]
    if not isinstance(bindings, dict) or set(bindings) != set(roles):
        raise AdmissionError("QUALIFICATION_EXACT_ARM_BINDINGS_REQUIRED")
    if len(set(bindings.values())) != len(bindings):
        raise AdmissionError("QUALIFICATION_DUPLICATE_ARM_BINDING")
    allowed = set(suite["method_bundles"][bundle]["arms"]) - {bundle}
    if suite["stage"] == "tuning":
        family = "arithmetic_anchor" if bundle == "M03" else "arithmetic_same_smoothing"
        allowed = {"base", *suite["tuning_groups"][family]}
    contracts = protocol.get("native_eval_contracts", {"main": protocol["native_eval_contract"]})
    for role, arm_id in bindings.items():
        if arm_id not in allowed or arm_id not in arms:
            raise AdmissionError("QUALIFICATION_NOT_A_REQUIRED_COMPARATOR", str(arm_id))
        arm = arms[arm_id]
        if arm_id in ORDER or arm["method"] in ORDER:
            raise AdmissionError("QUALIFICATION_SELECTED_METHOD_REQUIRES_OWN_DESIGN", arm_id)
        if set(arm["required_calibration"]) - set(QUALIFICATION_CALIBRATION_SOURCES):
            raise AdmissionError("QUALIFICATION_PRECURSOR_DESIGN_REQUIRED", arm_id)
        expected_name = "initial" if arm_id == "base" else arm_id
        for contract in contracts.values():
            if contract["arm_requirements"][role]["name"] != expected_name:
                raise AdmissionError("QUALIFICATION_CATALOGUE_ARM_NAME_MISMATCH", role)
            if contract["arm_requirements"][role]["implementation_refs"] != implementation["code_refs"]:
                raise AdmissionError("QUALIFICATION_CURRENT_CODE_BINDING_MISMATCH", role)
    if protocol["seed_policy"]["seeds"] != suite["seeds"]:
        raise AdmissionError("QUALIFICATION_CATALOGUE_SEED_MISMATCH")
    return suite


def _qualification(protocol, root=None):
    scope = protocol.get("suite_qualification")
    if not isinstance(scope, dict) or scope.get("purpose") not in {
            "baseline-calibration", "native-evaluator-qualification", "comparator-qualification"}:
        raise AdmissionError("METHOD_DISCOVERY_REQUIRED",
                             "candidate plans require the actual current discovery batch")
    roles = scope.get("allowed_arm_roles")
    if not isinstance(roles, list) or not roles or len(set(roles)) != len(roles):
        raise AdmissionError("QUALIFICATION_ARM_SCOPE_REQUIRED")
    contracts = protocol.get("native_eval_contracts", {"main": protocol["native_eval_contract"]})
    for contract in contracts.values():
        if set(roles) != set(contract["arm_requirements"]):
            raise AdmissionError("QUALIFICATION_ARM_SCOPE_MISMATCH")
        if scope["purpose"] != "comparator-qualification":
            for arm in contract["arm_requirements"].values():
                if arm["name"] not in QUALIFICATION_ARMS:
                    raise AdmissionError("QUALIFICATION_ARM_FORBIDDEN", arm["name"])
    if protocol.get("method_discovery"):
        raise AdmissionError("QUALIFICATION_CANDIDATE_SCOPE_CONFLICT")
    if scope["purpose"] == "comparator-qualification":
        if root is None:
            raise AdmissionError("QUALIFICATION_SOURCE_ROOT_REQUIRED")
        _comparator_qualification(root, protocol)
    return scope


def _canonical_ready(C, root, protocol=None, protocol_ref=None):
    missing = [name for name in ("event-ledger.jsonl", "ledger-anchor.json", "research-state.json")
               if not (root / name).is_file()]
    if missing:
        raise AdmissionError("CANONICAL_LEDGER_REQUIRED", ", ".join(missing))
    state, events, anchor = C.project(root)
    from _importance import require_ready
    require_ready(root, state)
    if protocol is not None:
        for key in ("parent_problem_ref", "natural_gate_0_ref", "importance_decision_ref"):
            if protocol.get(key) != state.get(key) or not state.get(key):
                raise AdmissionError("PROTOCOL_IMPORTANCE_BINDING_MISMATCH", key)
        from _evidence import verify_problem_eval_binding
        verify_problem_eval_binding(root, protocol, state["parent_problem_ref"])
        candidate = C.verify_ref(root, state["candidate_ref"], "idea-atom")
        if candidate["candidate_id"] != protocol["method_discovery"]["candidate_id"]:
            raise AdmissionError("CURRENT_CANDIDATE_BINDING_MISMATCH")
    if protocol_ref is not None:
        key = "gate_a_protocol_ref" if protocol["schema_id"] == "gate-a-protocol" else "full_validation_protocol_ref"
        if state.get(key) != protocol_ref:
            raise AdmissionError("CURRENT_FROZEN_PROTOCOL_REQUIRED", key)
        if not any(e["event_type"] == "freeze" and e["payload"].get("protocol_ref") == protocol_ref
                   for e in events):
            raise AdmissionError("CANONICAL_PROTOCOL_FREEZE_REQUIRED")
    return state, anchor


@_checked
def prepare_native_protocol(root, benchmark_manifest, arms, seeds, groups, scorer_spec, bindings, output):
    """Write an immutable native protocol draft; never fabricate a freeze/review.

    ``benchmark_manifest`` and ``scorer_spec`` are keyed by exactly ``groups``.
    Each manifest value is a pinned native *sample manifest*, not the asset
    acquisition manifest. ``bindings['protocol']`` supplies reviewed G01 fields.
    Candidate bindings require ``method_discovery``; baseline-only qualification
    requires an explicit ``qualification`` scope. See SCIENTIFIC_ADMISSION.md.
    """
    C, _, N, M = _modules()
    root = C.root_path(root)
    if not isinstance(groups, list) or not groups or len(set(groups)) != len(groups):
        raise AdmissionError("NATIVE_GROUPS_REQUIRED")
    if set(benchmark_manifest) != set(groups) or set(scorer_spec) != set(groups):
        raise AdmissionError("NATIVE_GROUP_CONTRACTS_INCOMPLETE")
    if not isinstance(seeds, list) or not seeds or any(type(s) is not int for s in seeds) or len(set(seeds)) != len(seeds):
        raise AdmissionError("SEED_INVENTORY_REQUIRED")
    if not isinstance(bindings, dict) or not isinstance(bindings.get("protocol"), dict):
        raise AdmissionError("REVIEWED_PROTOCOL_FIELDS_REQUIRED")
    protocol = deepcopy(bindings["protocol"])
    if any(k in protocol for k in ("frozen_at", "design_verified", "baseline_qualified", "protocol_digest")):
        raise AdmissionError("PROTOCOL_CANNOT_SYNTHESIZE_APPROVAL")
    if bindings.get("method_discovery"):
        link = bindings["method_discovery"]
        batch = M.read(root, link["batch_ref"], "method-verification-batch")
        report = M.before_action(root, batch, "experiment-design", link["candidate_id"])
        if not report["workflow_boundary"]["ready"]:
            raise AdmissionError("METHOD_CODE_NOT_VERIFIED", ", ".join(report["workflow_boundary"]["reason_codes"]))
        _canonical_ready(C, root)
        protocol["method_discovery"] = deepcopy(link)
    elif bindings.get("qualification"):
        protocol["suite_qualification"] = deepcopy(bindings["qualification"])
    else:
        raise AdmissionError("METHOD_DISCOVERY_REQUIRED")
    contracts = {group: build_native_contract(root, benchmark_manifest[group], arms, scorer_spec[group])
                 for group in groups}
    protocol.update(native_eval_contracts=contracts, native_eval_contract=deepcopy(contracts[groups[0]]),
                    seed_policy={"seeds": list(seeds), "allow_extra": False},
                    required_groups=list(groups), contrasts=_contrasts(arms))
    if "suite_qualification" in protocol:
        _qualification(protocol, root)
    if "suite_calibration_child" in protocol:
        from .suite_precursor import child_scope
        child_scope(root, protocol)
    C.validate(protocol)
    C.verify_ref(root, protocol["evidence_snapshot_ref"])
    N.verify_protocol(root, protocol)
    protocol["protocol_digest"] = C.protocol_hash(protocol)
    destination = C.safe_path(root, str(output))
    if destination.exists():
        if C.load_file(destination) != protocol:
            raise AdmissionError("PROTOCOL_DRAFT_IMMUTABLE_CREATE_CHILD", str(output))
    else:
        C.atomic(destination, C.canonical(protocol) + "\n")
    return protocol


@_checked
def freeze_native_protocol(root, protocol_path, *, gate="gate-a", replay_context=None):
    """Invoke the actual canonical freeze transaction after all existing gates.

    This is deliberately separate from preparation. The runtime verifies the
    ledger, parent, Gate 0, IPCG, snapshot, analysis, route, and native contract.
    No direct writes of frozen_at or design_verified occur in this adapter.
    """
    C, _, N, _ = _modules()
    root = C.root_path(root)
    if gate not in {"gate-a", "full-validation"}:
        raise AdmissionError("UNSUPPORTED_GATE")
    path = C.safe_path(root, str(protocol_path))
    protocol = C.load_file(path)
    if protocol.get("suite_qualification") or not protocol.get("method_discovery"):
        raise AdmissionError("METHOD_DISCOVERY_REQUIRED")
    _canonical_ready(C, root)
    N.verify_protocol(root, protocol)
    _live_qualification(C, N, root, protocol, replay_context)
    with C.locked(root):
        return C.freeze_gate(SimpleNamespace(gate=gate, protocol=str(path)), root)


def qualification_sources(protocol, group):
    """Normalize whole-manifest or exact per-comparator source bindings."""
    evidence = protocol.get("suite_qualification_evidence", {})
    if set(evidence) != set(protocol["required_groups"]):
        raise AdmissionError("NATIVE_QUALIFICATION_EVIDENCE_REQUIRED")
    record = evidence[group]
    contract = (protocol["native_eval_contracts"][group] if protocol.get("native_eval_contracts")
                else protocol["native_eval_contract"])
    roles = set(contract["arm_requirements"]) - {"treatment"}
    if "comparators" not in record:
        if set(record) != {"protocol_ref", "manifest_ref"}:
            raise AdmissionError("QUALIFICATION_SOURCE_BINDING_INVALID", group)
        return [{"record": record, "roles": {role: None for role in sorted(roles)}}]
    if set(record) != {"comparators"} or set(record["comparators"]) != roles:
        raise AdmissionError("QUALIFICATION_COMPARATOR_INVENTORY_MISMATCH", group)
    retained = {}
    for role, source in record["comparators"].items():
        if (set(source) != {"protocol_ref", "manifest_ref", "source_arm_role"}
                or not isinstance(source["source_arm_role"], str) or not source["source_arm_role"]):
            raise AdmissionError("QUALIFICATION_SOURCE_BINDING_INVALID", role)
        pair = {key: source[key] for key in ("protocol_ref", "manifest_ref")}
        key = json.dumps(pair, sort_keys=True)
        retained.setdefault(key, {"record": pair, "roles": {}})["roles"][role] = source["source_arm_role"]
    return [retained[key] for key in sorted(retained)]


def _live_qualification(C, N, root, protocol, replay_context):
    """Recheck real qualification runs with the native live replay contract."""
    if not callable(replay_context):
        raise AdmissionError("NATIVE_SCORER_REPLAY_REQUIRED")
    retained = []
    identity = ("benchmark_id", "benchmark_revision", "split", "sample_manifest_ref", "labels_or_tests_ref",
                "metrics", "primary_metric", "prediction_format", "sampling", "budget", "scorer",
                "native_definition_ref", "published_source_refs")
    for group in protocol["required_groups"]:
        expected_provenance = protocol.get("suite_qualification_provenance", {}).get(group)
        if not isinstance(expected_provenance, dict) or not {
                "model_revision", "environment_digest", "data_revision"}.issubset(expected_provenance):
            raise AdmissionError("QUALIFICATION_PROVENANCE_BINDINGS_REQUIRED", group)
        current = N.contract_for_group(protocol, group)
        definition = C.load_file(C.verify_ref(root, current["native_definition_ref"]))
        _definition_inputs(C, root, definition)
        rules = {"baseline": current["baseline_qualification"],
                 **{rule["role"]: rule for rule in current["control_qualifications"]}}
        operations = {"ge": operator.ge, "gt": operator.gt, "le": operator.le,
                      "lt": operator.lt, "eq": operator.eq}
        for source in qualification_sources(protocol, group):
            record = source["record"]
            qualifier = C.load_file(C.verify_ref(root, record["protocol_ref"]))
            manifest = C.load_file(C.verify_ref(root, record["manifest_ref"]))
            C.validate(manifest, "run-manifest")
            if any(manifest["provenance"].get(key) != expected for key, expected in expected_provenance.items()):
                raise AdmissionError("QUALIFICATION_PROVENANCE_MISMATCH", group)
            N.verify_protocol(root, qualifier)
            if qualifier.get("method_discovery"):
                _, _, _, M = _modules()
                M.verify_run_design(root, {"protocol_ref": record["protocol_ref"],
                                          "provenance": manifest["provenance"]}, qualifier)
            else:
                _qualification(qualifier, root)
            previous = N.contract_for_group(qualifier, manifest["group"])
            if any(current[key] != previous[key] for key in identity):
                raise AdmissionError("QUALIFICATION_NATIVE_IDENTITY_MISMATCH", group)
            if current.get("selection") != previous.get("selection"):
                raise AdmissionError("QUALIFICATION_NATIVE_SELECTION_MISMATCH", group)
            by_name = {arm["name"]: role for role, arm in previous["arm_requirements"].items()}
            resolved = {}
            for role, old_role in source["roles"].items():
                arm = current["arm_requirements"][role]
                old_role = old_role or by_name.get(arm["name"])
                if previous["arm_requirements"].get(old_role) != arm:
                    raise AdmissionError("QUALIFICATION_COMPARATOR_IDENTITY_MISMATCH", role)
                resolved[role] = old_role
            result = N.verify_run(root, qualifier, manifest, replay_context=replay_context)
            _definition_inputs(C, root, definition)
            if not result["baseline_qualified"] or not all(result["control_results"].values()):
                raise AdmissionError("NATIVE_COMPARATOR_NOT_QUALIFIED", group)
            for role, old_role in resolved.items():
                rule = rules[role]
                value = result["arm_metrics"][old_role][rule["metric"]]
                if not operations[rule["operator"]](value, rule["threshold"]):
                    raise AdmissionError("CURRENT_NATIVE_COMPARATOR_NOT_QUALIFIED", group + "/" + role)
            retained.append({"group": group, **deepcopy(record), "arm_roles": resolved,
                             "replay_audit": result["replay_audit"]})
    return retained


@_checked
def validate_candidate_binding(root, protocol_ref, verification_ref):
    """Read-only current canonical/method check before reserving or replaying.

    This proves only that the declared evidence chain is current. It does not
    authorize dispatch or substitute for native live replay/resource admission.
    """
    C, _, N, M = _modules()
    root = C.root_path(root)
    protocol = C.load_file(C.verify_ref(root, protocol_ref))
    if not protocol.get("method_discovery") or protocol.get("suite_qualification"):
        raise AdmissionError("METHOD_DISCOVERY_REQUIRED")
    if verification_ref is None:
        raise AdmissionError("RUN_METHOD_DESIGN_EVIDENCE_REQUIRED")
    if not protocol.get("frozen_at") or C.protocol_hash(protocol) != protocol.get("protocol_digest"):
        raise AdmissionError("PROTOCOL_NOT_FROZEN")
    C.stamp(protocol["frozen_at"])
    C.validate(protocol)
    _, anchor = _canonical_ready(C, root, protocol, protocol_ref)
    report = M.verify_run_design(root, {"protocol_ref": protocol_ref,
        "provenance": {"method_verification_ref": verification_ref}}, protocol)
    N.verify_protocol(root, protocol)
    return {"candidate_id": protocol["method_discovery"]["candidate_id"],
            "ledger_head_digest": anchor["head_digest"], "method_batch_digest": report["batch_digest"],
            "protocol_ref": deepcopy(protocol_ref), "verification_ref": deepcopy(verification_ref),
            "scope": "current canonical and method bindings; live qualification and resources still required"}


@_checked
def validate_scientific_plan(root, plan, *, replay_context=None):
    """Revalidate immediately before harness dispatch; no saved READY override."""
    C, R, N, _ = _modules()
    root = C.root_path(root)
    if plan.get("purpose") != "scientific" or not plan.get("protocol_ref"):
        raise AdmissionError("SCIENTIFIC_NATIVE_PLAN_REQUIRED")
    protocol = C.load_file(C.verify_ref(root, plan["protocol_ref"]))
    if protocol.get("method_discovery"):
        binding = validate_candidate_binding(root, plan["protocol_ref"],
                                             plan["provenance"].get("method_verification_ref"))
        qualification = _live_qualification(C, N, root, protocol, replay_context)
        if protocol.get("suite_calibration_child"):
            from .suite_precursor import validate_child_plan
            validate_child_plan(root, plan, protocol)
        result = {"scope": "candidate", "ledger_head_digest": binding["ledger_head_digest"],
                  "method_batch_digest": binding["method_batch_digest"], "qualification_replays": qualification}
    else:
        scope = _qualification(protocol, root)
        if plan["evidence_mode"] != "developmental":
            raise AdmissionError("QUALIFICATION_IS_DEVELOPMENTAL_ONLY")
        if scope["purpose"] == "comparator-qualification":
            from .suite_queue import validate_bound_comparator_plan
            validate_bound_comparator_plan(root, plan, protocol)
        for job in plan["jobs"]:
            if job["arm_role"] not in scope["allowed_arm_roles"]:
                raise AdmissionError("QUALIFICATION_ARM_FORBIDDEN", job["arm_role"])
            contract = N.contract_for_group(protocol, job["group"])
            expected = contract["arm_requirements"][job["arm_role"]]["name"]
            for flag in ("--arm", "--arm-id"):
                if flag in job["command"]:
                    index = job["command"].index(flag) + 1
                    if index == len(job["command"]) or job["command"][index] != expected:
                        raise AdmissionError("QUALIFICATION_COMMAND_ARM_MISMATCH")
        result = {"scope": scope["purpose"], "gate_advanced": False}
    for job in plan["jobs"]:
        _native_command(job["command"])
        if protocol.get("method_discovery"):
            expected = protocol["suite_qualification_provenance"][job["group"]]
            for key in ("model_revision", "environment_digest", "data_revision"):
                if plan["provenance"].get(key) != expected[key]:
                    raise AdmissionError("RUN_QUALIFICATION_PROVENANCE_MISMATCH", key)
        contract = N.contract_for_group(protocol, job["group"])
        definition = C.load_file(C.verify_ref(root, contract["native_definition_ref"]))
        required = _definition_inputs(C, root, definition)
        declared = {(r["path"], r["sha256"]) for r in job["input_refs"]}
        if any((r["path"], r["sha256"]) not in declared for r in required):
            raise AdmissionError("NATIVE_ORIGINAL_TEST_SHARD_INPUTS_REQUIRED", job["group"])
    R.validate_plan(root, plan)
    return result


@_checked
def build_scientific_plan(root, *, run_id, command, code_refs, input_refs, output_paths,
                          protocol_ref, verification_ref, seed, group, arm_role, provenance,
                          seconds, replay_context=None, evidence_mode=None, trial_id="job"):
    """Return the vendor's validated native plan, after project admission.

    ``verification_ref`` is the complete current method-verification *batch*,
    never a saved status report. Candidate qualification replay must be supplied
    by the authorized live host. Baseline calibration accepts no candidate arms.
    Finite resource reservation is separate and must precede harness dispatch.
    """
    C, R, _, _ = _modules()
    root = C.root_path(root)
    _number(seconds, "FINITE_RUN_BUDGET_REQUIRED", positive=True)
    if seconds > MAX_TOTAL_SECONDS:
        raise AdmissionError("AUTHORIZED_CAP_EXCEEDED")
    _native_command(command)
    protocol = C.load_file(C.verify_ref(root, protocol_ref))
    supplied = deepcopy(provenance)
    candidate = bool(protocol.get("method_discovery"))
    if candidate:
        if verification_ref is None:
            raise AdmissionError("RUN_METHOD_DESIGN_EVIDENCE_REQUIRED")
        C.verify_ref(root, verification_ref)
        if supplied.get("method_verification_ref", verification_ref) != verification_ref:
            raise AdmissionError("RUN_METHOD_DISCOVERY_BINDING_MISMATCH")
        supplied["method_verification_ref"] = deepcopy(verification_ref)
        supplied["admission_scope"] = ("candidate-calibration-child" if protocol.get("suite_calibration_child")
                                       else "candidate")
        mode = evidence_mode or protocol["evidence_mode"]
    else:
        scope = _qualification(protocol, root)
        supplied["admission_scope"] = scope["purpose"]
        if verification_ref is not None:
            raise AdmissionError("QUALIFICATION_CANDIDATE_SCOPE_CONFLICT")
        mode = evidence_mode or "developmental"
    limits = {"max_attempts": 1, "max_development_trials": 1 if mode == "developmental" else 0,
              "max_confirmation_trials": 0 if mode == "developmental" else 1,
              "max_retries_per_trial": 0, "wall_time_seconds": seconds,
              "attempt_timeout_seconds": seconds}
    plan = R.make_plan(root, run_id=run_id, purpose="scientific", evidence_mode=mode,
        protocol_ref=protocol_ref, jobs=[{"trial_id": trial_id, "command": list(command), "cwd": ".",
            "code_refs": deepcopy(code_refs), "input_refs": deepcopy(input_refs),
            "output_paths": list(output_paths), "seed": seed, "group": group, "arm_role": arm_role}],
        provenance=supplied, limits=limits)
    validate_scientific_plan(root, plan, replay_context=replay_context)
    return plan


@contextmanager
def _budget_lock(path):
    path = Path(path)
    if path.is_symlink():
        raise AdmissionError("SYMLINK_FORBIDDEN", str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_name(path.name + ".lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield path
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _read_budget(path):
    C, _, _, _ = _modules()
    budget = C.load_file(Path(path))
    if budget.get("version") != "suite-budget-v1":
        raise AdmissionError("BUDGET_VERSION_UNSUPPORTED")
    cap = _number(budget["cap_seconds"], "FINITE_CAP_REQUIRED", positive=True)
    if cap > MAX_TOTAL_SECONDS:
        raise AdmissionError("AUTHORIZED_CAP_EXCEEDED")
    if _timestamp(budget["absolute_end"]) != _timestamp(budget["original_start"]) + timedelta(seconds=cap):
        raise AdmissionError("BUDGET_DEADLINE_CHANGED")
    used = _number(budget["cumulative_used_seconds"], "CUMULATIVE_USAGE_INVALID")
    initial = _number(budget["initial_used_seconds"], "CUMULATIVE_USAGE_INVALID")
    charges = budget["charges"]
    if used != initial + sum(_number(x["seconds"], "USAGE_INVALID") for x in charges):
        raise AdmissionError("BUDGET_ACCOUNTING_MISMATCH")
    if len({x["receipt_ref"]["sha256"] for x in charges}) != len(charges):
        raise AdmissionError("DUPLICATE_BUDGET_CHARGE")
    for identifier, reservation in budget["reservations"].items():
        if identifier != reservation["reservation_id"]:
            raise AdmissionError("RESERVATION_ID_MISMATCH")
        _number(reservation["seconds"], "FINITE_RESERVATION_REQUIRED", positive=True)
    return budget


def _write_budget(path, budget):
    C, _, _, _ = _modules()
    C.atomic(Path(path), C.canonical(budget) + "\n")


@_checked
def initialize_budget(path, *, original_start, cap_seconds=MAX_TOTAL_SECONDS, already_used_seconds=0):
    """Create once, or resume the same absolute eight-hour authorization.

    A missing actual original start is a blocker. This API never substitutes the
    current time or silently extends the total cap on resume.
    """
    start = _timestamp(original_start)
    cap = _number(cap_seconds, "FINITE_CAP_REQUIRED", positive=True)
    used = _number(already_used_seconds, "CUMULATIVE_USAGE_INVALID")
    if cap > MAX_TOTAL_SECONDS:
        raise AdmissionError("AUTHORIZED_CAP_EXCEEDED")
    with _budget_lock(path) as path:
        if path.exists():
            existing = _read_budget(path)
            if _timestamp(existing["original_start"]) != start or existing["cap_seconds"] != cap:
                raise AdmissionError("BUDGET_RESTART_FORBIDDEN", "use the original start and cumulative authorization")
            if used and used != existing["initial_used_seconds"]:
                raise AdmissionError("INITIAL_USAGE_CHANGED")
            return existing
        budget = {"version": "suite-budget-v1", "original_start": _iso(start),
                  "absolute_end": _iso(start + timedelta(seconds=cap)), "cap_seconds": cap,
                  "initial_used_seconds": used, "cumulative_used_seconds": used,
                  "charges": [], "reservations": {}}
        _write_budget(path, budget)
        return budget


@_checked
def remaining_budget(budget, *, now=None):
    """Return unreserved seconds under both the original end and total cap."""
    if not isinstance(budget, dict):
        budget = _read_budget(budget)
    reserved = sum(_number(x["seconds"], "FINITE_RESERVATION_REQUIRED", positive=True)
                   for x in budget["reservations"].values())
    wall = (_timestamp(budget["absolute_end"]) - _now(now)).total_seconds()
    cumulative = budget["cap_seconds"] - budget["cumulative_used_seconds"]
    return max(0.0, min(wall, cumulative) - reserved)


def _reserve(path, reservation, now):
    with _budget_lock(path) as path:
        budget = _read_budget(path)
        identifier = reservation["reservation_id"]
        if identifier in budget["reservations"]:
            stored = budget["reservations"][identifier]
            expected = deepcopy(reservation)
            if stored.get("purpose") in {"candidate", "candidate-calibration-child"}:
                expected["charged_cell_ids"] = list(stored["charged_cell_ids"])
                expected["seconds"] = sum(bound for cell, bound in expected["cell_seconds"].items()
                                          if cell not in stored["charged_cell_ids"])
            if stored != expected:
                raise AdmissionError("RESERVATION_ID_REUSED")
            if (_timestamp(budget["absolute_end"]) <= _now(now)
                    or budget["cumulative_used_seconds"] >= budget["cap_seconds"]):
                raise AdmissionError("ORIGINAL_BUDGET_EXHAUSTED")
            return deepcopy(stored)
        if any(x["reservation_id"] == identifier for x in budget["charges"]):
            raise AdmissionError("COMPLETED_RESERVATION_ID_REUSED")
        if reservation["seconds"] > remaining_budget(budget, now=now):
            raise AdmissionError("COMPLETE_BUNDLE_EXCEEDS_REMAINING_BUDGET")
        budget["reservations"][identifier] = deepcopy(reservation)
        _write_budget(path, budget)
        return deepcopy(reservation)


@_checked
def reserve_calibration(path, *, calibration_id, bound_seconds, arm_ids, now=None):
    """Reserve a finite initial baseline calibration before costs are known."""
    seconds = _number(bound_seconds, "FINITE_CALIBRATION_BOUND_REQUIRED", positive=True)
    if not isinstance(arm_ids, list) or not arm_ids or not set(arm_ids).issubset(QUALIFICATION_ARMS):
        raise AdmissionError("QUALIFICATION_ARM_FORBIDDEN")
    if not isinstance(calibration_id, str) or not calibration_id:
        raise AdmissionError("RESERVATION_ID_REQUIRED")
    return _reserve(path, {"reservation_id": calibration_id, "purpose": "baseline-calibration",
                          "seconds": seconds, "arm_ids": list(arm_ids)}, now)


@_checked
def reserve_comparator_qualification(root, path, *, calibration_id, bound_seconds,
                                     protocol_ref, arm_role, now=None):
    """Reserve only the exact independently checked comparator scope."""
    C, _, _, _ = _modules()
    root = C.root_path(root)
    protocol = C.load_file(C.verify_ref(root, protocol_ref))
    scope = _qualification(protocol, root)
    if scope["purpose"] != "comparator-qualification" or arm_role not in scope["arm_bindings"]:
        raise AdmissionError("QUALIFICATION_EXACT_ARM_BINDINGS_REQUIRED")
    if not isinstance(calibration_id, str) or not calibration_id:
        raise AdmissionError("RESERVATION_ID_REQUIRED")
    return _reserve(path, {"reservation_id": calibration_id, "purpose": "comparator-qualification",
                          "seconds": _number(bound_seconds, "FINITE_CALIBRATION_BOUND_REQUIRED", positive=True),
                          "protocol_ref": deepcopy(protocol_ref), "arm_role": arm_role}, now)


@_checked
def reserve_calibration_child(root, path, *, protocol_ref, verification_ref, node_id,
                              reservation_id, now=None):
    """Reserve a finite reviewed scientific child's node, preserving all gates."""
    C, _, _, _ = _modules()
    root = C.root_path(root)
    validate_candidate_binding(root, protocol_ref, verification_ref)
    protocol = C.load_file(C.verify_ref(root, protocol_ref))
    from .suite_precursor import child_scope
    scope, _, _ = child_scope(root, protocol)
    if node_id not in scope["node_ids"]:
        raise AdmissionError("CALIBRATION_CHILD_NODE_OUTSIDE_SCOPE")
    costs = {key: _number(value, "FINITE_CALIBRATION_BOUND_REQUIRED", positive=True)
             for key, value in scope["node_seconds"].items()}
    reservation = _reserve(path, {"reservation_id": reservation_id, "purpose": "candidate-calibration-child",
        "seconds": sum(costs.values()), "cell_ids": list(scope["node_ids"]), "cell_seconds": costs,
        "charged_cell_ids": [], "protocol_ref": deepcopy(protocol_ref)}, now)
    if node_id in reservation["charged_cell_ids"]:
        raise AdmissionError("CALIBRATION_CHILD_NODE_ALREADY_CHARGED")
    return reservation


@_checked
def reserve_preparation(path, *, preparation_id, bound_seconds,
                        purpose="engineering-preparation", now=None):
    """Charge finite CPU preparation/replay to the same original authorization.

    This reservation does not authorize any execution or admit candidate/model
    work. Its receipt must be an engineering plan explicitly declaring the same
    purpose and model_revision='no-model-workload'. Use native-reference-replay
    only for the harness-owned live scorer callback, never a GPU workload.
    """
    if purpose not in {"engineering-preparation", "native-reference-replay"}:
        raise AdmissionError("PREPARATION_SCOPE_FORBIDDEN")
    if not isinstance(preparation_id, str) or not preparation_id:
        raise AdmissionError("RESERVATION_ID_REQUIRED")
    return _reserve(path, {"reservation_id": preparation_id, "purpose": purpose,
                          "seconds": _number(bound_seconds, "FINITE_PREPARATION_BOUND_REQUIRED", positive=True)}, now)


@_checked
def reserve_bundle(root, path, *, bundle_id, cell_ids, calibration_ref, expected_bindings, now=None):
    """Reserve a complete bundle using current real-host measured cost bounds.

    Calibration is ``suite-resource-calibration-v1`` with ``bindings``, ``host_ref``
    and per-cell ``costs``. Each cost declares upper_seconds and a retained native
    run receipt. The upper bound cannot be below that receipt's actual wall time.
    Source identity changes or incomplete per-cell measurements reject admission.
    """
    C, _, _, _ = _modules()
    root = C.root_path(root)
    if calibration_ref is None:
        raise AdmissionError("REAL_HOST_CALIBRATION_REQUIRED")
    calibration = C.load_file(C.verify_ref(root, calibration_ref))
    if calibration.get("version") != "suite-resource-calibration-v1":
        raise AdmissionError("CALIBRATION_VERSION_UNSUPPORTED")
    if not isinstance(expected_bindings, dict) or not expected_bindings:
        raise AdmissionError("CALIBRATION_IDENTITY_REQUIRED")
    for key in ("environment_digest", "model_revision", "code_digest", "benchmark_manifest_ref"):
        if key not in expected_bindings:
            raise AdmissionError("CALIBRATION_IDENTITY_REQUIRED", key)
    if calibration.get("bindings") != expected_bindings:
        raise AdmissionError("CALIBRATION_IDENTITY_MISMATCH")
    C.verify_ref(root, expected_bindings["benchmark_manifest_ref"])
    host = C.load_file(C.verify_ref(root, calibration["host_ref"]))
    if host.get("gpu_count") != 1 or "2080" not in str(host.get("gpu_model")) or "Ti" not in str(host.get("gpu_model")):
        raise AdmissionError("CALIBRATION_HOST_MISMATCH", "one RTX 2080 Ti is the authorized host")
    _timestamp(host["observed_at"])
    _refs(C, root, host.get("evidence_refs"), "HOST_OBSERVATION_REQUIRED")
    if (not isinstance(cell_ids, list) or not cell_ids or len(set(cell_ids)) != len(cell_ids)
            or any(not isinstance(x, str) or not x for x in cell_ids)):
        raise AdmissionError("COMPLETE_CELL_INVENTORY_REQUIRED")
    costs = calibration.get("costs", {})
    if not set(cell_ids).issubset(costs):
        raise AdmissionError("COMPLETE_BUNDLE_CALIBRATION_REQUIRED", ", ".join(sorted(set(cell_ids) - set(costs))))
    total = 0.0
    cell_seconds = {}
    for cell in cell_ids:
        measured = costs[cell]
        upper = _number(measured["upper_seconds"], "FINITE_COST_BOUND_REQUIRED", positive=True)
        receipt = C.verify_ref(root, measured["receipt_ref"], "experiment-run-receipt")
        if receipt["status"] != "completed" or receipt["purpose"] != "scientific" or not receipt["attempts"]:
            raise AdmissionError("COMPLETED_NATIVE_CALIBRATION_REQUIRED", cell)
        calibration_scope = receipt["provenance"].get("admission_scope")
        if calibration_scope != "baseline-calibration":
            if calibration_scope not in {"comparator-qualification", "candidate-calibration-child"}:
                raise AdmissionError("NATIVE_BASELINE_CALIBRATION_REQUIRED", cell)
            plan = C.verify_ref(root, measured["plan_ref"], "experiment-run-plan")
            _, R, _, M = _modules()
            R.validate_plan(root, plan)
            if (plan["plan_digest"] != receipt["plan_digest"] or plan["run_id"] != receipt["run_id"]
                    or receipt["provenance"] != plan["provenance"] or plan["provenance"].get("node_id") != cell):
                raise AdmissionError("CALIBRATION_EXACT_NODE_PLAN_REQUIRED", cell)
            ledger = _read_budget(path)
            if (C.safe_path(root, receipt["provenance"].get("qualification_budget_path", "")) != Path(path).resolve()
                    or not any(row["receipt_ref"] == measured["receipt_ref"] for row in ledger["charges"])):
                raise AdmissionError("CALIBRATION_ORIGINAL_BUDGET_CHARGE_REQUIRED", cell)
            source_protocol = C.load_file(C.verify_ref(root, plan["protocol_ref"]))
            if calibration_scope == "candidate-calibration-child":
                from .suite_precursor import validate_child_plan
                M.verify_run_design(root, plan, source_protocol)
                validate_child_plan(root, plan, source_protocol)
            else:
                from .suite_queue import validate_bound_comparator_plan
                _qualification(source_protocol, root)
                validate_bound_comparator_plan(root, plan, source_protocol)
        for key in ("environment_digest", "model_revision"):
            if receipt["provenance"].get(key) != expected_bindings[key]:
                raise AdmissionError("CALIBRATION_IDENTITY_MISMATCH", key)
        elapsed = _number(receipt["resources"]["seconds"], "MEASURED_COST_REQUIRED", positive=True)
        if upper < elapsed:
            raise AdmissionError("COST_BOUND_BELOW_OBSERVED", cell)
        for attempt in receipt["attempts"]:
            for key in ("stdout_ref", "stderr_ref"):
                C.verify_ref(root, attempt[key])
            _refs(C, root, attempt["code_refs"])
            if C.hashed(attempt["code_refs"]) != expected_bindings["code_digest"]:
                raise AdmissionError("CALIBRATION_CODE_MISMATCH", cell)
        total += upper
        cell_seconds[cell] = upper
    return _reserve(path, {"reservation_id": bundle_id, "purpose": "candidate",
                          "seconds": total, "cell_ids": list(cell_ids),
                          "cell_seconds": cell_seconds, "charged_cell_ids": [],
                          "calibration_ref": deepcopy(calibration_ref),
                          "bindings": deepcopy(expected_bindings)}, now)


@_checked
def charge_budget(root, path, *, reservation_id, receipt_ref, cell_id=None, final=True):
    """Charge a retained run, including failed attempts, exactly once.

    For a candidate bundle, supply its cell_id and final=False until the last
    child. Only that cell's reserved bound is released; unfinished cells retain
    their full reservations across resume. final=True requires every cell to be
    charged. Missing receipts leave the reservation in place; callers may not
    free capacity by dropping attempts. Completed IDs cannot be reused.
    """
    C, _, _, _ = _modules()
    root = C.root_path(root)
    receipt = C.verify_ref(root, receipt_ref, "experiment-run-receipt")
    elapsed = _number(receipt["resources"]["seconds"], "MEASURED_USAGE_REQUIRED")
    attempt_seconds = sum(_number(a["seconds"], "MEASURED_USAGE_REQUIRED") for a in receipt["attempts"])
    if elapsed < attempt_seconds:
        raise AdmissionError("RECEIPT_WALL_TIME_UNDERCOUNTS_ATTEMPTS")
    if _timestamp(receipt["completed_at"]) < _timestamp(receipt["started_at"]):
        raise AdmissionError("RECEIPT_TIME_ORDER_INVALID")
    with _budget_lock(path) as path:
        budget = _read_budget(path)
        prior = [x for x in budget["charges"] if x["receipt_ref"]["sha256"] == receipt_ref["sha256"]]
        if prior:
            if prior[0]["reservation_id"] != reservation_id or prior[0].get("cell_id") != cell_id:
                raise AdmissionError("RECEIPT_ALREADY_CHARGED")
            return budget
        if reservation_id not in budget["reservations"]:
            raise AdmissionError("BUDGET_RESERVATION_REQUIRED")
        reservation = budget["reservations"][reservation_id]
        expected_scope = reservation["purpose"]
        preparation = expected_scope in {"engineering-preparation", "native-reference-replay"}
        if (receipt["purpose"] != ("engineering" if preparation else "scientific")
                or receipt["provenance"].get("admission_scope") != expected_scope
                or (preparation and receipt["provenance"].get("model_revision") != "no-model-workload")):
            raise AdmissionError("BUDGET_RECEIPT_SCOPE_MISMATCH")
        if expected_scope == "comparator-qualification":
            if (receipt.get("provenance", {}).get("qualification_protocol_ref") != reservation["protocol_ref"]
                    or receipt.get("provenance", {}).get("qualification_arm_role") != reservation["arm_role"]):
                raise AdmissionError("QUALIFICATION_BUDGET_BINDING_MISMATCH")
        if expected_scope == "candidate-calibration-child":
            if (receipt["provenance"].get("qualification_protocol_ref") != reservation["protocol_ref"]
                    or receipt["provenance"].get("node_id") != cell_id):
                raise AdmissionError("CALIBRATION_CHILD_BUDGET_BINDING_MISMATCH")
        if expected_scope in {"candidate", "candidate-calibration-child"}:
            if cell_id not in reservation["cell_ids"]:
                raise AdmissionError("RESERVED_CELL_ID_REQUIRED")
            if cell_id in reservation["charged_cell_ids"]:
                raise AdmissionError("CELL_ALREADY_CHARGED")
            charged = [*reservation["charged_cell_ids"], cell_id]
            complete = set(charged) == set(reservation["cell_ids"])
            if final and not complete:
                raise AdmissionError("BUNDLE_STILL_HAS_UNCHARGED_CELLS")
            released = reservation["cell_seconds"][cell_id]
            reservation["charged_cell_ids"] = charged
            reservation["seconds"] = sum(bound for cell, bound in reservation["cell_seconds"].items()
                                          if cell not in charged)
            if complete:
                budget["reservations"].pop(reservation_id)
        else:
            if cell_id is not None or not final:
                raise AdmissionError("CALIBRATION_USES_SINGLE_RECEIPT")
            released = reservation["seconds"]
            budget["reservations"].pop(reservation_id)
        budget["charges"].append({"reservation_id": reservation_id, "receipt_ref": deepcopy(receipt_ref),
                                  "cell_id": cell_id, "seconds": elapsed, "reserved_seconds": released,
                                  "status": receipt["status"]})
        budget["cumulative_used_seconds"] = budget["initial_used_seconds"] + sum(x["seconds"] for x in budget["charges"])
        _write_budget(path, budget)
        return budget
