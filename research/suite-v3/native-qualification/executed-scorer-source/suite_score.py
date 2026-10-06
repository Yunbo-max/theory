"""Complete native scoring contracts for HumanEval+, MBPP+ and LiveCodeBench."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import uuid

if __package__ in (None, ""):
    # Explicit source root for the isolated -I child, never its working directory.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "recursive_ssd"

from .benchmarks import (
    COUNTS, DESIGN_DATA, EVAL_SHA256, FULL_FILES, LCB_COMMIT, LCB_REFERENCE_BLOB,
    LCB_REFERENCE_COMMIT, MBPP_SHA256, NAMES, PROJECT_ROOT, _git_blob,
    benchmark_tasks, lcb_source_path, task_id, validate_task_inventory, verify_lcb_source,
)
from .evaluation import evaluator_info, inventory, pass_at_k, run_native_process
from .io import atomic_json, digest, jsonl, object_hash, read_json, read_jsonl


def publish_native_result(source, target):
    """Publish official bytes verbatim, including native MBPP ±Infinity cases."""
    source, target = Path(source), Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    shutil.copyfile(source, temporary)
    os.replace(temporary, target)


def validate_samples(raw, tasks, expected_samples, benchmark):
    ids = [task_id(row, benchmark) for row in tasks]
    if type(expected_samples) is not int or expected_samples < 1:
        raise ValueError("expected_samples must be a positive integer")
    for row in raw:
        if (not isinstance(row, dict) or not isinstance(row.get("task_id"), str)
                or type(row.get("sample_id")) is not int
                or row["sample_id"] not in range(expected_samples)):
            raise ValueError("invalid zero-based sample identity")
        if not isinstance(row.get("text"), str):
            raise ValueError("sample text must be a string, including failed/empty generations")
    inventory(raw, [{"task_id": identifier} for identifier in ids], expected_samples)
    grouped = {identifier: [] for identifier in ids}
    for row in raw:
        grouped[row["task_id"]].append(row)
    return {identifier: sorted(rows, key=lambda row: row["sample_id"])
            for identifier, rows in grouped.items()}


def _summary(per_task, benchmark, expected_samples, ks):
    if not per_task:
        raise ValueError("empty official task inventory")
    ks = sorted(set(ks))
    if not ks or 1 not in ks or any(type(k) is not int or not 1 <= k <= expected_samples for k in ks):
        raise ValueError("invalid pass@k sample contract")
    for values in per_task.values():
        values.update({f"pass@{k}": pass_at_k(values["n"], values["correct"], k) for k in ks})
    return {"benchmark": benchmark, "task_count": len(per_task),
            "samples_per_task": expected_samples, "per_task": per_task,
            **{f"pass@{k}": sum(row[f"pass@{k}"] for row in per_task.values()) / len(per_task)
               for k in ks}, "primary_metric": "pass@1", "coverage_metric": f"pass@{max(ks)}",
            "metric_unit": "fraction", "aggregation": "equal-weight task mean",
            "scientific_result_verified": False}


def summarize_evalplus(native, task_ids, expected_samples, ks, benchmark):
    outcomes = native.get("eval", {})
    if len(set(task_ids)) != len(task_ids) or set(outcomes) != set(task_ids):
        raise ValueError("official EvalPlus output task inventory mismatch")
    per_task = {}
    for identifier in task_ids:
        rows = outcomes[identifier]
        if not isinstance(rows, list) or len(rows) != expected_samples:
            raise ValueError("missing official EvalPlus sample outcomes")
        if any(not isinstance(row, dict) or "base_status" not in row or "plus_status" not in row
               or row.get("task_id", identifier) != identifier for row in rows):
            raise ValueError("invalid official EvalPlus sample outcomes")
        base = sum(row["base_status"] == "pass" for row in rows)
        correct = sum(row["base_status"] == "pass" and row["plus_status"] == "pass" for row in rows)
        per_task[identifier] = {"n": expected_samples, "correct": correct,
                                "base_correct": base, "plus_correct": correct,
                                "base_pass1": base / expected_samples,
                                "plus_pass1": correct / expected_samples}
    return _summary(per_task, benchmark, expected_samples, ks)


def summarize_lcb(native, task_ids, expected_samples, ks):
    order = native.get("task_ids", [])
    if (len(set(order)) != len(order) or len(set(task_ids)) != len(task_ids)
            or set(order) != set(task_ids)):
        raise ValueError("official LiveCodeBench question_id inventory mismatch")
    metrics = native.get("metrics")
    if not isinstance(metrics, list) or len(metrics) != 3:
        raise ValueError("missing official LiveCodeBench outcomes/metadata")
    outcomes = metrics[1]
    if {str(index) for index in outcomes} != {str(index) for index in range(len(order))}:
        raise ValueError("official LiveCodeBench result index inventory mismatch")
    case_counts = native.get("case_counts")
    per_task = {}
    for index, identifier in enumerate(order):
        rows = outcomes.get(str(index), outcomes.get(index))
        if not isinstance(rows, list) or len(rows) != expected_samples:
            raise ValueError("missing official LiveCodeBench sample outcomes")
        grades = []
        for row in rows:
            if not isinstance(row, list) or not row or any(type(value) not in (int, float, bool) for value in row):
                raise ValueError("invalid/empty official LiveCodeBench testcase outcomes")
            passed = all(value > 0 for value in row)
            if passed and case_counts is not None and len(row) != case_counts[identifier]:
                raise ValueError("passing LiveCodeBench sample omitted native test outcomes")
            grades.append(passed)
        per_task[identifier] = {"n": expected_samples, "correct": sum(grades), "graded_list": grades}
    result = _summary(per_task, "livecodebench", expected_samples, ks)
    # Native codegen_metrics uses the identical estimator in fraction units.
    for k in ks:
        if f"pass@{k}" in metrics[0] and abs(metrics[0][f"pass@{k}"] - result[f"pass@{k}"]) > 1e-12:
            raise ValueError("official LiveCodeBench aggregate differs from retained outcomes")
    return result


def _task_source(tasks_path, tasks, benchmark, data_root, qualify):
    """Match every native row to its actual released source, including private tests."""
    root = Path(data_root).resolve() if data_root is not None else Path(tasks_path).resolve().parent
    if (root / "benchmarks-manifest.json").exists():
        full = benchmark_tasks(root, benchmark, "full")
        manifest = read_json(root / "benchmarks-manifest.json")
        receipt = {"manifest_sha256": digest(root / "benchmarks-manifest.json"),
                   "release": manifest["benchmarks"][benchmark]["source"],
                   "full_source_sha256": digest(root / FULL_FILES[benchmark])}
        if benchmark == "livecodebench":
            native_manifest = root / "livecodebench-native-sources.json"
            if digest(native_manifest) != manifest["files"][native_manifest.name]:
                raise ValueError("native test-source manifest hash changed")
            native_sources = read_json(native_manifest)
            for path, item in native_sources["files"].items():
                if manifest["files"].get(path) != item["sha256"] or digest(root / path) != item["sha256"]:
                    raise ValueError(f"native released test-source bytes changed: {path}")
            receipt["native_tests_manifest_sha256"] = digest(native_manifest)
    elif benchmark in ("humaneval", "mbpp"):
        source = PROJECT_ROOT / ("artifacts/data-check/HumanEvalPlus-full.jsonl"
                                if benchmark == "humaneval" else "artifacts/design-audit/MbppPlus-v0.2.0.jsonl")
        expected = EVAL_SHA256 if benchmark == "humaneval" else MBPP_SHA256
        if not source.is_file() or digest(source) != expected:
            raise ValueError("released benchmark source missing; prepare assets and pass data_root")
        full = read_jsonl(source)
        receipt = {"full_source_sha256": expected, "release": "v0.1.10" if benchmark == "humaneval" else "v0.2.0"}
    else:
        raise ValueError("released LiveCodeBench source manifest required; pass data_root")
    full_ids = validate_task_inventory(full, benchmark, COUNTS[benchmark])
    by_id = {task_id(row, benchmark): row for row in full}
    selected = [task_id(row, benchmark) for row in tasks]
    validate_task_inventory(tasks, benchmark, len(tasks))
    for row, identifier in zip(tasks, selected):
        # task_id is a generation convenience; native LCB source rows use question_id.
        native_row = {key: value for key, value in row.items()
                      if not (benchmark == "livecodebench" and key == "task_id")}
        if identifier not in by_id or native_row != by_id[identifier]:
            raise ValueError(f"task differs from pinned native release: {identifier}")
    if not qualify:
        inventories = [set(full_ids)]
        if benchmark == "humaneval":
            splits = read_json(DESIGN_DATA / "humaneval-manifest.json")
            inventories += [set(splits["dev_ids"]), set(splits["confirm_ids"])]
        if set(selected) not in inventories:
            raise ValueError("incomplete published benchmark/split denominator")
    receipt.update(task_ids=selected, full_count=len(full), selected_count=len(selected),
                   selection_scope="native-scorer qualification only" if qualify else "frozen full benchmark/split")
    return receipt, root


def lcb_evaluator_info(source, parallel=2, timeout=6):
    identity = verify_lcb_source(source)
    required = {"numpy": "2.2.6", "datasets": "3.5.0", "tqdm": "4.70.1"}
    for name, version in required.items():
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(f"native LiveCodeBench requires {name}=={version}") from exc
        if actual != version:
            raise RuntimeError(f"native LiveCodeBench requires {name}=={version}, found {actual}")
    return {"backend": "livecodebench-native", "source": identity,
            "interpreter": str(Path(sys.executable).resolve()), "python": platform.python_version(),
            "packages": {item.metadata["Name"]: item.version for item in importlib.metadata.distributions()},
            "required_packages": required, "parallel": parallel, "timeout_per_test_seconds": timeout,
            "extraction": "official extract_code, LMStyle.CodeQwenInstruct (last fenced block)",
            "wrapper_sha256": digest(Path(__file__).with_name("lcb_score.py")),
            "memory_limit": "official reliability_guard default; no OS memory sandbox asserted"}


def _reference_grades(raw, data_root):
    path = Path(data_root) / "sources/livecodebench-reference-outputs.json"
    if not path.is_file() or _git_blob(path) != LCB_REFERENCE_BLOB:
        raise ValueError("released LiveCodeBench reference outputs missing or changed")
    references = {row["question_id"]: row for row in read_json(path)}
    grades = {}
    for row in raw:
        index = row.get("source_sample_id")
        original = references.get(row["task_id"])
        if (type(index) is not int or original is None or not 0 <= index < len(original["output_list"])
                or row["text"] != original["output_list"][index]):
            raise ValueError("qualification sample is not a bound released model output")
        grades[(row["task_id"], row["sample_id"])] = bool(original["graded_list"][index])
    return grades, {"commit": LCB_REFERENCE_COMMIT, "git_blob": LCB_REFERENCE_BLOB,
                    "sha256": digest(path), "kind": "released model outputs, not canonical solutions"}


def score_benchmark(raw_path, tasks_path, output, benchmark, expected_samples, deadline, **options):
    """Run/cache one complete native scoring attempt, retaining all failed attempts.

    Options: qualify=False, data_root=None, source_root=None, parallel=2,
    timeout=6 (LCB native per-test seconds), max_seconds=1800, k_values=None.
    Qualification is scoped scorer replay; it does not qualify a model/baseline.
    """
    if benchmark not in NAMES:
        raise ValueError(f"unknown benchmark: {benchmark}")
    known = {"qualify", "data_root", "source_root", "parallel", "timeout", "max_seconds", "k_values"}
    if set(options) - known:
        raise TypeError(f"unsupported native scorer options: {sorted(set(options) - known)}")
    qualify = options.get("qualify", False)
    parallel, timeout = options.get("parallel", 2), options.get("timeout", 6)
    if type(parallel) is not int or parallel < 1 or type(timeout) is not int or timeout < 1:
        raise ValueError("invalid native scorer resource settings")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    result_path, receipt_path = output / "official_results.json", output / "native_receipt.json"
    if result_path.exists() != receipt_path.exists():
        raise ValueError("unbound cached native result: provenance receipt/result missing")
    tasks_path = Path(tasks_path).resolve()
    tasks = read_jsonl(tasks_path)
    canonical = qualify and benchmark != "livecodebench"
    if canonical and (expected_samples != 1 or raw_path is not None):
        raise ValueError("canonical qualification uses exactly one native reference per task and no raw file")
    raw = ([{"task_id": task_id(row, benchmark), "sample_id": 0, "text": ""} for row in tasks]
           if canonical else read_jsonl(raw_path))
    grouped = validate_samples(raw, tasks, expected_samples, benchmark)
    source_receipt, data_root = _task_source(tasks_path, tasks, benchmark, options.get("data_root"), qualify)
    coverage = min(5 if benchmark == "livecodebench" else 10, expected_samples)
    ks = sorted(set(options.get("k_values") or [1, coverage]))
    if 1 not in ks or any(type(k) is not int or not 1 <= k <= expected_samples for k in ks):
        raise ValueError("pass@k requires the full declared sample denominator")
    references, reference_receipt = (None, None)
    if qualify and benchmark == "livecodebench":
        references, reference_receipt = _reference_grades(raw, data_root)
    source = Path(options.get("source_root") or lcb_source_path(data_root)).resolve()
    if benchmark == "livecodebench":
        info = lcb_evaluator_info(source, parallel, timeout)
    else:
        info = evaluator_info()
        info.update(parallel=parallel, wrapper_sha256=digest(Path(__file__)))
    identity = {"benchmark": benchmark, "tasks_sha256": digest(tasks_path),
                "raw_sha256": digest(raw_path) if raw_path is not None else None,
                "expected_samples": expected_samples, "qualify": qualify, "k_values": ks,
                "source_receipt": source_receipt, "reference_receipt": reference_receipt,
                "evaluator": info, "adapter_sha256": digest(Path(__file__)),
                "benchmark_adapter_sha256": digest(Path(__file__).with_name("benchmarks.py"))}
    if result_path.exists():
        receipt = read_json(receipt_path)
        if receipt["identity"] != identity or receipt["result_sha256"] != digest(result_path):
            raise ValueError("cached native result provenance changed; use a new output identity")
        native = read_json(result_path)
    else:
        attempt = output / "native" / ("attempt-" + uuid.uuid4().hex[:12])
        inputs = attempt / "inputs"
        inputs.mkdir(parents=True)
        shutil.copyfile(tasks_path, inputs / "tasks.jsonl")
        wrapper = Path(__file__).with_name("lcb_score.py") if benchmark == "livecodebench" else Path(__file__)
        command = [sys.executable, "-I", str(wrapper), "--tasks", str(inputs / "tasks.jsonl"),
                   "--output", str(attempt), "--expected-samples", str(expected_samples),
                   "--parallel", str(parallel), "--k-values", ",".join(map(str, ks))]
        if benchmark == "livecodebench":
            shutil.copytree(source, inputs / "official-source", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            command += ["--source", str(inputs / "official-source"), "--timeout", str(timeout),
                        "--data-root", str(data_root), "--task-cache", str(output / "task-cache")]
        else:
            command += ["--benchmark", benchmark]
        if canonical:
            command.append("--qualify")
        else:
            shutil.copyfile(raw_path, inputs / "raw.jsonl")
            command += ["--raw", str(inputs / "raw.jsonl")]
        atomic_json(attempt / "input-receipt.json", identity)
        deadline.check(45)
        run_native_process(command, attempt, max(1, min(float(options.get("max_seconds", 1800)), deadline.remaining() - 20)))
        native = read_json(attempt / "official_results.json")
        # Validate before publishing an accepted cache; raw official attempt always remains.
        if benchmark == "livecodebench":
            summarize_lcb(native, list(grouped), expected_samples, ks)
        else:
            summarize_evalplus(native, list(grouped), expected_samples, ks, benchmark)
        publish_native_result(attempt / "official_results.json", result_path)
        retained = {str(path.relative_to(output)): digest(path) for path in attempt.rglob("*")
                    if path.is_file() and "__pycache__" not in path.parts}
        if benchmark == "livecodebench":
            retained.update({str(path.relative_to(output)): digest(path)
                             for path in (output / "task-cache").glob("*.json")})
        atomic_json(receipt_path, {"identity": identity, "result_sha256": digest(result_path),
                    "native_output": str((attempt / "official_results.json").relative_to(output)),
                    "execution": str((attempt / "execution.json").relative_to(output)),
                    "retained_files": retained})
    result = (summarize_lcb(native, list(grouped), expected_samples, ks) if benchmark == "livecodebench"
              else summarize_evalplus(native, list(grouped), expected_samples, ks, benchmark))
    if canonical and any(values["correct"] != 1 for values in result["per_task"].values()):
        raise RuntimeError("official canonical reference qualification failed")
    if references is not None:
        for identifier, samples in grouped.items():
            for row, passed in zip(samples, result["per_task"][identifier]["graded_list"]):
                if passed != references[(identifier, row["sample_id"])]:
                    raise RuntimeError("native replay differs from released reference grade; retain and investigate")
    result.update(native_receipt=str(receipt_path), native_receipt_sha256=digest(receipt_path),
                  official_result=str(result_path), official_result_sha256=digest(result_path),
                  source_receipt=source_receipt, qualification_replay=bool(qualify))
    atomic_json(output / "metrics.json", result)
    return result


def main():
    """Isolated EvalPlus native child. Dataset override must precede any import."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True, choices=("humaneval", "mbpp"))
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--raw")
    parser.add_argument("--qualify", action="store_true")
    parser.add_argument("--expected-samples", required=True, type=int)
    parser.add_argument("--parallel", type=int, default=2)
    parser.add_argument("--k-values", default="1")
    args = parser.parse_args()
    override = "HUMANEVAL_OVERRIDE_PATH" if args.benchmark == "humaneval" else "MBPP_OVERRIDE_PATH"
    os.environ[override] = str(Path(args.tasks).resolve())
    from evalplus.data import get_human_eval_plus, get_mbpp_plus
    from evalplus.evaluate import evaluate
    from evalplus.sanitize import sanitize
    problems = get_human_eval_plus() if args.benchmark == "humaneval" else get_mbpp_plus()
    if args.qualify:
        samples = [{"task_id": identifier, "solution": row["prompt"] + row["canonical_solution"]}
                   for identifier, row in problems.items()]
        mapping = [{"task_id": identifier, "sample_id": 0} for identifier in problems]
    else:
        grouped = validate_samples(read_jsonl(args.raw), list(problems.values()), args.expected_samples, args.benchmark)
        samples, mapping = [], []
        for identifier, rows in grouped.items():
            for row in rows:
                samples.append({"task_id": identifier,
                                "solution": sanitize(row["text"], entrypoint=problems[identifier]["entry_point"])})
                mapping.append({"task_id": identifier, "sample_id": row["sample_id"]})
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    jsonl(output / "samples.jsonl", samples)
    atomic_json(output / "sample-map.json", mapping)
    evaluate(dataset=args.benchmark, samples=str(output / "samples.jsonl"), parallel=args.parallel,
             test_details=True, min_time_limit=1, gt_time_limit_factor=4)
    publish_native_result(output / "samples_eval_results.json", output / "official_results.json")


if __name__ == "__main__":
    main()
