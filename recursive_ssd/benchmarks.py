"""Pinned native benchmark assets and generation-only prompt mirrors.

Acquisition is a CPU workload: invoke preparation through the project harness.
This module never loads private pickle payloads and never imports a scorer.
"""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import urllib.request

from .data import EVAL_SHA256, EVAL_URL, MBPP, MBPP_REV
from .io import atomic_json, digest, jsonl, object_hash, read_json, read_jsonl, stable_seed

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DESIGN_DATA = PROJECT_ROOT / "research/design-v2/data"
MBPP_URL = "https://github.com/evalplus/mbppplus_release/releases/download/v0.2.0/MbppPlus.jsonl.gz"
MBPP_SHA256 = "b54e762755248ca411b523c917fa9f93c07b5ff2966bf60b3917b853926a3dad"
CLEAN_TRAIN_SHA256 = "a9f7642d10f57d03abc77b55e46e8158f5cb2e1b9fd5335baf4d4ecc41e6092f"
LCB_DATASET = "livecodebench/code_generation_lite"
LCB_REVISION = "0fe84c3912ea0c4d4a78037083943e8f0c4dd505"
LCB_COMMIT = "28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24"
LCB_FILES = ("test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl")
LCB_FIELDS = ("question_title", "question_content", "platform", "question_id", "contest_id",
              "contest_date", "starter_code", "difficulty", "public_test_cases",
              "private_test_cases", "metadata")
NAMES = ("humaneval", "mbpp", "livecodebench")
FULL_FILES = {"humaneval": "HumanEvalPlus-full.jsonl", "mbpp": "MbppPlus-full.jsonl",
              "livecodebench": "LiveCodeBench-full.jsonl"}
COUNTS = {"humaneval": 164, "mbpp": 378, "livecodebench": 880}

# Native API import closure and its inspected parser/prompt/custom evaluator.
# Git blob IDs were read from the official repository at LCB_COMMIT.
LCB_SOURCE_BLOBS = {
    "LICENSE": "d3c6d337d2a28958523092bfced9258904135083",
    "pyproject.toml": "e2b5b910298ce7b0fb3f05da173f432e633ae371",
    "lcb_runner/benchmarks/__init__.py": "a7f5db8a74867a8fc596c3dd8044562075cf2d62",
    "lcb_runner/benchmarks/code_generation.py": "e2e48a73b68c4981dae44356dd4d52df87d9f070",
    "lcb_runner/benchmarks/code_execution.py": "f37a03b368fbb1a80b2fecda7f8c0961c70f9ad7",
    "lcb_runner/benchmarks/test_output_prediction.py": "12154ada92c2f539326d33106d711a4cc15d2c86",
    "lcb_runner/evaluation/__init__.py": "cad106ea1c384fe24fafdd28594209538a43df31",
    "lcb_runner/evaluation/compute_code_generation_metrics.py": "b8de33ee0631ef0747c3a0d32aa77130ba8659d8",
    "lcb_runner/evaluation/compute_code_execution_metrics.py": "2737aa392bf0cf96c9c8b952295f9271510dea92",
    "lcb_runner/evaluation/compute_test_output_prediction_metrics.py": "f9f624cc4b25793d28b941bee275c79fdb576d91",
    "lcb_runner/evaluation/pass_k_utils.py": "a49382f0857448ef494d77a4c6107db952c9823a",
    "lcb_runner/evaluation/testing_util.py": "ecc89474b813cb552ae060a53b3a7da534476968",
    "lcb_runner/evaluation/utils_execute.py": "85d9a0ef601fd313e30321a6a5ac17c19603eb5b",
    "lcb_runner/lm_styles.py": "d10b21176473e22caf55ca042c1eba47e6ceb580",
    "lcb_runner/utils/extraction_utils.py": "3491583ee6c4162896ec6e180cd8119069602892",
    "lcb_runner/prompts/code_generation.py": "22ba3f82979598dd1fe828cfb6647df53d4e82d2",
    "lcb_runner/runner/custom_evaluator.py": "58990fc364273698db51609df32440996aab10c2",
    "lcb_runner/runner/parser.py": "a047fc0d27e8a28fe3bd346ca0bbec05c8bb799c",
}
LCB_REFERENCE_COMMIT = "6ca212e9c2039373f6e5069d37ffa9db66e23736"
LCB_REFERENCE_FILE = "Qwen2.5-Coder-Ins-7B/Scenario.codegeneration_10_0.2_eval_all.json"
LCB_REFERENCE_BLOB = "0cba6247fc4aae0d1875d49d15812a1429c8a6dc"
MAX_NATIVE_LINE_BYTES = 512 * 1024 * 1024


def task_id(row, benchmark):
    if benchmark not in NAMES:
        raise ValueError(f"unknown benchmark: {benchmark}")
    key = "question_id" if benchmark == "livecodebench" else "task_id"
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"invalid {benchmark} task inventory: missing {key}")
    if benchmark == "livecodebench" and "task_id" in row and row["task_id"] != value:
        raise ValueError("LiveCodeBench task_id/question_id mismatch")
    return value


def validate_task_inventory(rows, benchmark, expected_count, expected_ids=None):
    ids = [task_id(row, benchmark) for row in rows]
    if len(ids) != expected_count or len(set(ids)) != len(ids) or not ids:
        raise ValueError(f"unexpected {benchmark} task inventory: {len(ids)} rows")
    if expected_ids is not None and ids != list(expected_ids):
        raise ValueError(f"changed {benchmark} task inventory or ordering")
    return ids


def benchmark_prompt(row, benchmark):
    """Exact native EvalPlus prompt or official LCB generic user-message template."""
    if benchmark in ("humaneval", "mbpp"):
        prompt = row["prompt"]
    elif benchmark == "livecodebench":
        prompt = f"### Question:\n{row['question_content']}\n\n"
        if row["starter_code"]:
            prompt += ("### Format: You will use the following starter code to write the solution "
                       "to the problem and enclose your code within delimiters.\n")
            prompt += f"```python\n{row['starter_code']}\n```\n\n"
        else:
            prompt += ("### Format: Read the inputs from stdin solve the problem and write the answer "
                       "to stdout (do not directly test on the sample inputs). Enclose your code within "
                       "delimiters as follows. Ensure that when the python program runs, it reads the "
                       "inputs, runs the algorithm and writes output to STDOUT.\n")
            prompt += "```python\n# YOUR CODE HERE\n```\n\n"
        prompt += "### Answer: (use the provided format with backticks)\n\n"
    else:
        raise ValueError(f"unknown benchmark: {benchmark}")
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("empty or invalid benchmark prompt")
    return prompt


def _git_blob(path):
    path = Path(path)
    h = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _download(url, target, *, download, sha256=None, git_blob=None, compressed=False):
    """Keep rejected/partial acquisitions; never overwrite an existing changed asset."""
    target = Path(target)
    if not target.exists():
        if not download:
            raise FileNotFoundError(f"pinned asset missing: {target}; prepare with download=True")
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + ".download")
        request = urllib.request.Request(url, headers={"User-Agent": "recursive-ssd-research"})
        with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as output:
            shutil.copyfileobj(response, output, length=1 << 20)
        if compressed:
            unpacked = target.with_name(target.name + ".unpacked")
            with gzip.open(partial, "rb") as source, unpacked.open("wb") as output:
                shutil.copyfileobj(source, output, length=1 << 20)
            candidate = unpacked
        else:
            candidate = partial
        if sha256 is not None and digest(candidate) != sha256:
            raise ValueError(f"released asset hash mismatch: {url}")
        if git_blob is not None and _git_blob(candidate) != git_blob:
            raise ValueError(f"official source git blob mismatch: {url}")
        os.replace(candidate, target)
    actual_sha256 = digest(target)
    if sha256 is not None and actual_sha256 != sha256:
        raise ValueError(f"released asset hash changed: {target}")
    if git_blob is not None and _git_blob(target) != git_blob:
        raise ValueError(f"official source git blob changed: {target}")
    return {"url": url, "sha256": actual_sha256, "bytes": target.stat().st_size,
            **({"git_blob": git_blob} if git_blob is not None else {})}


def _copy_pinned(source, target, expected):
    source, target = Path(source), Path(target)
    if digest(source) != expected:
        raise ValueError(f"retained source hash changed: {source}")
    if target.exists():
        if digest(target) != expected:
            raise ValueError(f"prepared file changed: {target}; choose a new root")
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)


def lcb_source_path(root):
    return Path(root) / "sources" / "livecodebench" / LCB_COMMIT


def prepare_lcb_source(root, *, download=True):
    source = lcb_source_path(root)
    files = {}
    for name, blob in LCB_SOURCE_BLOBS.items():
        url = f"https://raw.githubusercontent.com/LiveCodeBench/LiveCodeBench/{LCB_COMMIT}/{name}"
        files[name] = _download(url, source / name, download=download, git_blob=blob)
    receipt = {"repository": "LiveCodeBench/LiveCodeBench", "commit": LCB_COMMIT,
               "files": files, "import_scope": "official native code-generation API"}
    atomic_json(source / "source-receipt.json", receipt)
    return receipt


def verify_lcb_source(source):
    source = Path(source)
    files = {}
    for name, expected in LCB_SOURCE_BLOBS.items():
        path = source / name
        if not path.is_file() or path.is_symlink() or _git_blob(path) != expected:
            raise ValueError(f"pinned LiveCodeBench source changed or missing: {name}")
        files[name] = digest(path)
    # Reject import-path injection alongside the frozen native import closure.
    allowed = {name for name in LCB_SOURCE_BLOBS if name.endswith(".py")}
    if {str(path.relative_to(source)) for path in source.rglob("*.py")} != allowed:
        raise ValueError("unexpected Python file in pinned LiveCodeBench source")
    return {"commit": LCB_COMMIT, "files": files, "sha256": object_hash(files)}


def acquire_lcb_references(root, *, download=True):
    """Released model outputs, for scorer replay only; not canonical solutions."""
    root = Path(root)
    path = root / "sources" / "livecodebench-reference-outputs.json"
    receipt_path = path.with_suffix(".receipt.json")
    manifest_path = root / "benchmarks-manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else None
    if manifest is not None:
        for asset in (path, receipt_path):
            key = asset.relative_to(root).as_posix()
            expected = manifest["files"].get(key)
            if expected is not None and (not asset.is_file() or digest(asset) != expected):
                raise ValueError(f"bound released reference asset changed: {key}")
    url = (f"https://raw.githubusercontent.com/LiveCodeBench/submissions/{LCB_REFERENCE_COMMIT}/"
           + LCB_REFERENCE_FILE)
    receipt = _download(url, path, download=download, git_blob=LCB_REFERENCE_BLOB)
    receipt.update(repository="LiveCodeBench/submissions", commit=LCB_REFERENCE_COMMIT,
                   source_path=LCB_REFERENCE_FILE, local_path=str(path.resolve()))
    if receipt_path.exists():
        retained = read_json(receipt_path)
        # Keep the original acquisition path when a verified asset was moved.
        if (set(retained) != set(receipt) or not isinstance(retained["local_path"], str)
                or any(retained[key] != value for key, value in receipt.items() if key != "local_path")):
            raise ValueError("released reference acquisition receipt changed")
        receipt = retained
    else:
        atomic_json(receipt_path, receipt)
    if manifest is not None:
        files = {**manifest["files"], **{asset.relative_to(root).as_posix(): digest(asset)
                                       for asset in (path, receipt_path)}}
        if files != manifest["files"]:
            atomic_json(manifest_path, {**manifest, "files": files})
    return receipt


def index_lcb_sources(root, paths, *, expected_count=880):
    """Index exact byte ranges; peak input memory is one native JSONL record."""
    root = Path(root).resolve()
    handles, statistics = [], {}
    for path in paths:
        path = Path(path).resolve()
        if not path.is_relative_to(root):
            raise ValueError("native source must reside inside the prepared data root")
        file_hash = hashlib.sha256()
        count, largest = 0, 0
        with path.open("rb") as stream:
            while True:
                offset = stream.tell()
                line = stream.readline(MAX_NATIVE_LINE_BYTES + 1)
                if not line:
                    break
                if len(line) > MAX_NATIVE_LINE_BYTES:
                    raise ValueError("native task line exceeds the declared memory limit; retain source and qualify a larger envelope")
                file_hash.update(line)
                if not line.strip():
                    continue
                row = json.loads(line)
                if set(row) != set(LCB_FIELDS) or any(not isinstance(value, str) for value in row.values()):
                    raise ValueError(f"unexpected native LiveCodeBench row schema in {path.name}")
                handle = {key: value for key, value in row.items()
                          if key not in ("private_test_cases", "public_test_cases")}
                handle["_native_ref"] = {"path": str(path.relative_to(root)), "offset_bytes": offset,
                    "length_bytes": len(line), "sha256": hashlib.sha256(line).hexdigest()}
                handles.append(handle)
                count += 1
                largest = max(largest, len(line))
                del row, line
        statistics[str(path.relative_to(root))] = {"sha256": file_hash.hexdigest(),
            "bytes": path.stat().st_size, "count": count, "max_line_bytes": largest}
    validate_task_inventory(handles, "livecodebench", expected_count)
    handles.sort(key=lambda row: row["question_id"])
    return handles, {"files": statistics, "total_rows": len(handles),
        "max_line_bytes": max(value["max_line_bytes"] for value in statistics.values()),
        "max_supported_line_bytes": MAX_NATIVE_LINE_BYTES}


def load_native_task(handle, root):
    """Resolve one frozen LCB record without materializing its multi-GB source."""
    ref = handle.get("_native_ref")
    if not isinstance(ref, dict):
        raise ValueError("LiveCodeBench task must use its pinned native byte-range reference")
    root = Path(root).resolve()
    path = root / ref["path"]
    if path.is_symlink() or not path.resolve().is_relative_to(root):
        raise ValueError("native source path escapes the prepared data root")
    offset, length = ref["offset_bytes"], ref["length_bytes"]
    if type(offset) is not int or type(length) is not int or offset < 0 or not 0 < length <= MAX_NATIVE_LINE_BYTES:
        raise ValueError("invalid bounded native byte-range reference")
    with path.open("rb") as stream:
        stream.seek(offset)
        payload = stream.read(length)
    if len(payload) != length or hashlib.sha256(payload).hexdigest() != ref["sha256"]:
        raise ValueError("native task source line hash changed")
    row = json.loads(payload)
    if any(row.get(key) != value for key, value in handle.items() if key not in ("_native_ref", "task_id")):
        raise ValueError("native task locator metadata changed")
    return row


def _clone_file(source, target):
    """Copy-on-write clone when supported; never use writable hard links."""
    import fcntl
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    temporary = target.with_name(target.name + ".clone")
    try:
        with Path(source).open("rb") as src, temporary.open("wb") as dst:
            fcntl.ioctl(dst.fileno(), 0x40049409, src.fileno())  # Linux FICLONE
    except OSError:
        needed = Path(source).stat().st_size + 256 * 1024 * 1024
        if shutil.disk_usage(target.parent).free < needed:
            raise RuntimeError("copy-on-write clone unavailable and disk space is insufficient for a safe native-source copy")
        shutil.copyfile(source, temporary)
    os.replace(temporary, target)


def _lcb_file_metadata(root, name, download):
    """Pin huge-file bytes using the official immutable revision's HEAD metadata."""
    path = Path(root) / "sources/livecodebench-data" / f"{name}.source.json"
    url = f"https://huggingface.co/datasets/{LCB_DATASET}/resolve/{LCB_REVISION}/{name}"
    if path.exists():
        metadata = read_json(path)
    elif download:
        from huggingface_hub import get_hf_file_metadata
        response = get_hf_file_metadata(url, token=False, timeout=60)
        metadata = {"url": url, "commit_hash": response.commit_hash,
                    "sha256": response.etag, "bytes": response.size,
                    "source": "official Hugging Face HEAD x-linked-etag SHA-256"}
    else:
        raise FileNotFoundError("pinned LCB source metadata missing; allow metadata acquisition once")
    sha = metadata.get("sha256", "")
    if (metadata.get("url") != url or metadata.get("commit_hash") != LCB_REVISION
            or not isinstance(sha, str) or len(sha) != 64
            or any(character not in "0123456789abcdef" for character in sha)
            or type(metadata.get("bytes")) is not int or metadata["bytes"] <= 0):
        raise ValueError("official LCB immutable revision/LFS metadata mismatch")
    if not path.exists():
        atomic_json(path, metadata)
    return metadata


def prepare_benchmarks(root, *, download=True, include_lcb=True, reuse_lcb_root=None):
    """Prepare exact releases; ``include_lcb=False`` is asset staging, not a full suite."""
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    previous = root / "benchmarks-manifest.json"
    if previous.exists():
        for name, expected in read_json(previous)["files"].items():
            if not (root / name).is_file() or digest(root / name) != expected:
                raise ValueError(f"prepared benchmark hash changed: {name}")
    sources = {}
    for name, url, sha, retained in (
        ("humaneval", EVAL_URL, EVAL_SHA256, PROJECT_ROOT / "artifacts/data-check/HumanEvalPlus-full.jsonl"),
        ("mbpp", MBPP_URL, MBPP_SHA256, PROJECT_ROOT / "artifacts/design-audit/MbppPlus-v0.2.0.jsonl"),
    ):
        target = root / FULL_FILES[name]
        if not target.exists() and retained.exists():
            _copy_pinned(retained, target, sha)
        sources[name] = _download(url, target, download=download, sha256=sha, compressed=True)
    he = read_jsonl(root / FULL_FILES["humaneval"])
    mbpp = read_jsonl(root / FULL_FILES["mbpp"])
    validate_task_inventory(he, "humaneval", 164)
    mbpp_ids = validate_task_inventory(mbpp, "mbpp", 378,
        read_json(DESIGN_DATA / "mbpp-plus-audit.json")["all_task_ids"])
    he.sort(key=lambda row: stable_seed("eval-split-v1", row["task_id"]))
    retained_split = read_json(DESIGN_DATA / "humaneval-manifest.json")
    for split, rows, count in (("dev", he[:32], 32), ("confirm", he[32:], 132)):
        validate_task_inventory(rows, "humaneval", count, retained_split[split + "_ids"])
        jsonl(root / f"HumanEvalPlus-{split}.jsonl", rows)
    train_path = root / "train_prompts.jsonl"
    _copy_pinned(DESIGN_DATA / "train_prompts-clean-v2.jsonl", train_path, CLEAN_TRAIN_SHA256)
    train = read_jsonl(train_path)
    train_audit = read_json(DESIGN_DATA / "clean-training-manifest.json")
    if (len(train) != 32 or [row.get("task_id") for row in train] != train_audit["selected_ids"]
            or any(set(row) != {"task_id", "text"} for row in train)
            or any(not isinstance(row["text"], str) or not row["text"] for row in train)
            or {row["task_id"] for row in train} & {item.split("/")[-1] for item in mbpp_ids}):
        raise ValueError("clean-v2 training prompt inventory/field/overlap violation")
    all_rows = {"humaneval": {"full": read_jsonl(root / FULL_FILES["humaneval"]),
                              "dev": he[:32], "confirm": he[32:]}, "mbpp": {"full": mbpp}}
    if include_lcb:
        files = {}
        native_paths = []
        for name in LCB_FILES:
            url = f"https://huggingface.co/datasets/{LCB_DATASET}/resolve/{LCB_REVISION}/{name}"
            target = root / "sources" / "livecodebench-data" / name
            metadata = _lcb_file_metadata(root, name, download)
            if reuse_lcb_root is not None and not target.exists():
                _clone_file(Path(reuse_lcb_root) / name, target)
            files[name] = _download(url, target, download=download, sha256=metadata["sha256"])
            if target.stat().st_size != metadata["bytes"]:
                raise ValueError("official LCB native source byte count mismatch")
            files[name]["metadata"] = metadata
            native_paths.append(target)
        lcb, source_statistics = index_lcb_sources(root, native_paths)
        for path, stat in source_statistics["files"].items():
            if files[Path(path).name]["sha256"] != stat["sha256"]:
                raise ValueError("native source changed while preparing its byte-range index")
            files[Path(path).name].update(count=stat["count"], max_line_bytes=stat["max_line_bytes"])
        jsonl(root / FULL_FILES["livecodebench"], lcb)
        all_rows["livecodebench"] = {"full": lcb}
        sources["livecodebench"] = {"dataset": LCB_DATASET, "revision": LCB_REVISION,
            "release": "release_v5", "files": files, "storage": "indexed-native-jsonl",
            "max_line_bytes": source_statistics["max_line_bytes"]}
        atomic_json(root / "livecodebench-native-sources.json", {
            "dataset": LCB_DATASET, "revision": LCB_REVISION, "release": "release_v5",
            "storage": "indexed-native-jsonl", "index": {"path": FULL_FILES["livecodebench"],
                "sha256": digest(root / FULL_FILES["livecodebench"])},
            "files": source_statistics["files"], "total_count": len(lcb),
            "max_line_bytes": source_statistics["max_line_bytes"],
            "max_supported_line_bytes": MAX_NATIVE_LINE_BYTES})
        sources["livecodebench_evaluator"] = prepare_lcb_source(root, download=download)
    benchmarks = {}
    files = {"train_prompts.jsonl": digest(train_path)}
    for name, splits in all_rows.items():
        split_info = {}
        for split, rows in splits.items():
            task_file = FULL_FILES[name] if split == "full" else f"HumanEvalPlus-{split}.jsonl"
            mirror = f"{name}-prompts-{split}.jsonl"
            prompt_rows = [{"task_id": task_id(row, name), "prompt": benchmark_prompt(row, name),
                            **({"entry_point": row["entry_point"]} if name != "livecodebench" else {})}
                           for row in rows]
            jsonl(root / mirror, prompt_rows)
            files[task_file] = digest(root / task_file)
            files[mirror] = digest(root / mirror)
            split_info[split] = {"path": task_file, "prompts": mirror, "count": len(rows),
                                 "task_ids": [task_id(row, name) for row in rows]}
        benchmarks[name] = {"splits": split_info, "source": sources[name],
                           "storage": "indexed-native-jsonl" if name == "livecodebench" else "native-jsonl"}
    # Freeze all native assets in addition to normalized full files and mirrors.
    if include_lcb:
        files["livecodebench-native-sources.json"] = digest(root / "livecodebench-native-sources.json")
        files.update({path: stat["sha256"] for path, stat in source_statistics["files"].items()})
        for path in (root / "sources").rglob("*"):
            if path.is_file() and not path.name.endswith((".download", ".unpacked")):
                relative = str(path.relative_to(root))
                if relative not in files:
                    files[relative] = digest(path)
    manifest = {"version": "clean-v2-native-benchmarks-1", "benchmarks": benchmarks,
                "files": files, "sources": sources, "training": {
                    "dataset": MBPP, "revision": MBPP_REV, "split": "full/train",
                    "count": 32, "fields": ["task_id", "text"], "lineage": "clean-v2",
                    "must_restart_from_base": True, "exact_mbpp_plus_overlap": []},
                "complete_suite": include_lcb, "private_tests_deserialized": False,
                "scientific_qualification": False}
    atomic_json(previous, manifest)
    return manifest


def _split_info(root, benchmark, split):
    if benchmark not in NAMES or split not in (("dev", "confirm", "full")
                                                if benchmark == "humaneval" else ("full",)):
        raise ValueError(f"unsupported benchmark/split: {benchmark}/{split}")
    manifest = read_json(Path(root) / "benchmarks-manifest.json")
    try:
        info = manifest["benchmarks"][benchmark]["splits"][split]
    except KeyError as exc:
        raise ValueError(f"benchmark split has not been prepared: {benchmark}/{split}") from exc
    return manifest, info


def benchmark_tasks_path(root, benchmark, split):
    _, info = _split_info(root, benchmark, split)
    return Path(root) / info["path"]


def benchmark_prompts_path(root, benchmark, split):
    manifest, info = _split_info(root, benchmark, split)
    path = Path(root) / info["prompts"]
    if digest(path) != manifest["files"][info["prompts"]]:
        raise ValueError("benchmark prompt mirror hash changed")
    return path


def benchmark_tasks(root, benchmark, split):
    manifest, info = _split_info(root, benchmark, split)
    path = Path(root) / info["path"]
    if digest(path) != manifest["files"][info["path"]]:
        raise ValueError("benchmark task file hash changed")
    rows = read_jsonl(path)
    validate_task_inventory(rows, benchmark, info["count"], info["task_ids"])
    return rows


def select_lcb_reference_cases(tasks, references):
    """Select one functional and one stdin replay case; retain all ten draws.

    This is scorer branch qualification using released grades, never a sample
    selection rule for research outcomes. Prefer cases with both true/false
    grades so successful and unsuccessful native pathways are exercised.
    """
    by_id = {row["question_id"]: row for row in references}
    if len(by_id) != len(references):
        raise ValueError("duplicate released reference question_id")
    candidates = {"functional": [], "stdin": []}
    for task in sorted(tasks, key=lambda row: row["question_id"]):
        ref = by_id.get(task["question_id"])
        if ref is None:
            continue
        if (len(ref.get("output_list", [])) != 10 or len(ref.get("graded_list", [])) != 10
                or any(type(value) is not bool for value in ref["graded_list"])
                or any(not isinstance(value, str) for value in ref["output_list"])):
            raise ValueError("released reference must retain all ten original samples and boolean grades")
        if any(ref.get(key) != task[key] for key in ("question_content", "starter_code")):
            continue
        if not any(ref["graded_list"]):
            continue
        category = "functional" if json.loads(task["metadata"]).get("func_name") is not None else "stdin"
        candidates[category].append((all(ref["graded_list"]), task["question_id"], task, ref))
    chosen, raw, expected = [], [], {}
    for category in ("functional", "stdin"):
        if not candidates[category]:
            raise ValueError(f"no qualified released reference covers native {category} format")
        _, identifier, task, ref = sorted(candidates[category], key=lambda item: item[:2])[0]
        chosen.append(task)
        expected[identifier] = ref["graded_list"]
        raw.extend({"task_id": identifier, "sample_id": index, "source_sample_id": index, "text": text}
                   for index, text in enumerate(ref["output_list"]))
    return chosen, raw, expected


def prepare_reference_qualification(root, output):
    """Prepare source-bound CPU replay inputs; do not execute the native graders."""
    root, output = Path(root).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    jobs = []
    for name in ("humaneval", "mbpp"):
        rows = benchmark_tasks(root, name, "full")[:2]
        path = output / f"{name}-tasks.jsonl"
        # Native MBPP JSON permits ±Infinity; preserve that published transport.
        with path.open("w") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=True) + "\n")
        jobs.append({"benchmark": name, "tasks_path": str(path), "raw_path": None,
            "expected_samples": 1, "expected_grades": {row["task_id"]: [True] for row in rows},
            "tasks_sha256": digest(path), "data_root": str(root), "qualify": True})
    reference = root / "sources/livecodebench-reference-outputs.json"
    if not reference.is_file() or _git_blob(reference) != LCB_REFERENCE_BLOB:
        raise ValueError("released reference outputs missing; acquire_lcb_references first")
    chosen, raw, expected = select_lcb_reference_cases(benchmark_tasks(root, "livecodebench", "full"), read_json(reference))
    tasks_path, raw_path = output / "livecodebench-tasks.jsonl", output / "livecodebench-raw.jsonl"
    jsonl(tasks_path, chosen)
    jsonl(raw_path, raw)
    jobs.append({"benchmark": "livecodebench", "tasks_path": str(tasks_path), "raw_path": str(raw_path),
        "expected_samples": 10, "expected_grades": expected, "tasks_sha256": digest(tasks_path),
        "raw_sha256": digest(raw_path), "data_root": str(root), "qualify": True,
        "reference": {"repository": "LiveCodeBench/submissions", "commit": LCB_REFERENCE_COMMIT,
            "path": LCB_REFERENCE_FILE, "git_blob": LCB_REFERENCE_BLOB, "sha256": digest(reference)}})
    packet = {"kind": "native-reference-qualification-inputs", "jobs": jobs,
        "source_manifest_sha256": digest(root / "benchmarks-manifest.json"),
        "scope": "sourced scorer replay only; no model experiment or scientific gate advanced"}
    atomic_json(output / "qualification-inputs.json", packet)
    return packet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--without-lcb", action="store_true")
    parser.add_argument("--lcb-references", action="store_true")
    parser.add_argument("--reuse-lcb-root", help="Reuse the five original native files from a previous retained acquisition")
    parser.add_argument("--reference-output", help="Write source-bound native qualification inputs without scoring")
    args = parser.parse_args()
    manifest = prepare_benchmarks(args.root, download=not args.offline, include_lcb=not args.without_lcb,
                                  reuse_lcb_root=args.reuse_lcb_root)
    if args.lcb_references:
        acquire_lcb_references(args.root, download=not args.offline)
    if args.reference_output:
        prepare_reference_qualification(args.root, args.reference_output)
    print(json.dumps({"complete_suite": manifest["complete_suite"], "counts": {
        name: {split: info["count"] for split, info in item["splits"].items()}
        for name, item in manifest["benchmarks"].items()}}, indent=2))


if __name__ == "__main__":
    main()
