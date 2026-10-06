"""Adapter engineering contracts; synthetic metadata is never scorer evidence."""
import importlib
import importlib.util
import json
from pathlib import Path
import time

import pytest

from recursive_ssd.io import Deadline, atomic_json, digest, jsonl, read_json


def adapter(name):
    spec = importlib.util.find_spec("recursive_ssd." + name)
    assert spec is not None, f"native benchmark adapter {name} is not implemented"
    return importlib.import_module("recursive_ssd." + name)


@pytest.mark.parametrize("benchmark", ["humaneval", "mbpp"])
def test_evalplus_prompt_does_not_export_reference_or_hidden_tests(benchmark):
    module = adapter("benchmarks")
    row = {"task_id": "metadata", "prompt": "published prompt\n", "entry_point": "f",
           "canonical_solution": "REFERENCE_SENTINEL", "plus_input": "PRIVATE_SENTINEL",
           "assertion": "ASSERTION_SENTINEL", "contract": "CONTRACT_SENTINEL"}
    assert module.benchmark_prompt(row, benchmark) == "published prompt\n"


def test_lcb_prompt_uses_official_format_and_never_decodes_private_tests():
    module = adapter("benchmarks")
    row = {"question_id": "format-only", "question_content": "Published statement.",
           "starter_code": "class Solution:\n    pass", "private_test_cases": "not pickle",
           "public_test_cases": "[]", "metadata": "{\"secret\": \"PRIVATE_SENTINEL\"}"}
    expected = ("### Question:\nPublished statement.\n\n"
                "### Format: You will use the following starter code to write the solution "
                "to the problem and enclose your code within delimiters.\n"
                "```python\nclass Solution:\n    pass\n```\n\n"
                "### Answer: (use the provided format with backticks)\n\n")
    assert module.benchmark_prompt(row, "livecodebench") == expected


def test_external_benchmarks_reject_development_splits(tmp_path):
    module = adapter("benchmarks")
    for benchmark in ("mbpp", "livecodebench"):
        with pytest.raises(ValueError, match="split"):
            module.benchmark_tasks(tmp_path, benchmark, "dev")


def test_release_inventory_rejects_duplicate_and_incomplete_ids():
    module = adapter("benchmarks")
    with pytest.raises(ValueError, match="inventory"):
        module.validate_task_inventory([{"task_id": "A"}, {"task_id": "A"}], "humaneval", 2)
    with pytest.raises(ValueError, match="inventory"):
        module.validate_task_inventory([{"question_id": "A"}], "livecodebench", 880)


def test_task_file_tampering_is_rejected_before_reading_tasks(tmp_path):
    module = adapter("benchmarks")
    path = tmp_path / "HumanEvalPlus-dev.jsonl"
    jsonl(path, [{"task_id": "A", "prompt": "old"}])
    atomic_json(tmp_path / "benchmarks-manifest.json", {
        "files": {path.name: digest(path)},
        "benchmarks": {"humaneval": {"splits": {"dev": {"path": path.name,
            "count": 1, "task_ids": ["A"]}}}},
    })
    jsonl(path, [{"task_id": "A", "prompt": "changed"}])
    with pytest.raises(ValueError, match="hash|changed"):
        module.benchmark_tasks(tmp_path, "humaneval", "dev")


def test_native_sample_inventory_requires_exact_zero_based_sample_ids():
    module = adapter("suite_score")
    tasks = [{"task_id": "A"}]
    for samples in ([{"task_id": "A", "sample_id": 8, "text": "metadata"}],
                    [{"task_id": "A", "sample_id": True, "text": "metadata"}],
                    [{"task_id": "A", "sample_id": 0, "text": None}]):
        with pytest.raises(ValueError, match="sample|text"):
            module.validate_samples(samples, tasks, 1, "humaneval")


def test_lcb_results_follow_question_id_mapping_and_count_negative_status_as_failure():
    module = adapter("suite_score")
    native = {"task_ids": ["q2", "q10"],
              "metrics": [{}, {"1": [[True], [-2], [True]],
                                "0": [[True, True], [-4], [False]]}, [[], []]]}
    result = module.summarize_lcb(native, ["q10", "q2"], 3, [1, 3])
    assert result["per_task"]["q2"]["correct"] == 1
    assert result["per_task"]["q10"]["correct"] == 2
    assert result["per_task"]["q2"]["pass@1"] == pytest.approx(1 / 3)
    assert result["per_task"]["q10"]["pass@3"] == 1.0
    assert result["task_count"] == 2


@pytest.mark.parametrize("outcomes", [[], [[True]], [[True], [True], []]])
def test_lcb_incomplete_outcomes_cannot_reduce_the_denominator(outcomes):
    module = adapter("suite_score")
    native = {"task_ids": ["q"], "metrics": [{}, {"0": outcomes}, [[]]]}
    with pytest.raises(ValueError, match="outcome|inventory|sample"):
        module.summarize_lcb(native, ["q"], 3, [1, 3])


def test_lcb_runner_infrastructure_error_cannot_be_accepted_as_a_wrong_solution():
    module = adapter("suite_score")
    # Official error transport fixture only; no code or test input is executed.
    native = {"task_ids": ["q"], "metrics": [{"pass@1": 0.0}, {"0": [[-2]]},
        [[json.dumps({"error": "EOFError()", "error_code": -5, "error_message": "TestRunnerError"})]]]}
    with pytest.raises(RuntimeError, match="infrastructure|TestRunnerError"):
        module.summarize_lcb(native, ["q"], 1, [1])


def test_evalplus_summary_requires_both_base_and_plus_and_full_inventory():
    module = adapter("suite_score")
    native = {"eval": {"A": [{"base_status": "pass", "plus_status": "pass"},
                              {"base_status": "fail", "plus_status": "pass"}]}}
    result = module.summarize_evalplus(native, ["A"], 2, [1, 2], "mbpp")
    assert result["per_task"]["A"]["correct"] == 1
    assert result["per_task"]["A"]["pass@1"] == 0.5
    assert result["per_task"]["A"]["pass@2"] == 1.0
    with pytest.raises(ValueError, match="inventory"):
        module.summarize_evalplus(native, ["A", "B"], 2, [1], "mbpp")


def test_unbound_native_cache_is_rejected_before_any_evaluator(tmp_path):
    module = adapter("suite_score")
    tasks = tmp_path / "tasks.jsonl"
    raw = tmp_path / "raw.jsonl"
    jsonl(tasks, [{"task_id": "A"}])
    jsonl(raw, [{"task_id": "A", "sample_id": 0, "text": "not executed"}])
    output = tmp_path / "scoring"
    atomic_json(output / "official_results.json", {"eval": {}})
    with pytest.raises(ValueError, match="unbound|provenance"):
        module.score_benchmark(raw, tasks, output, "mbpp", 1, Deadline(time.time() + 60))


def test_restricted_private_case_decoder_rejects_object_execution():
    module = adapter("lcb_score")
    # Engineering security fixture only, never sent to a benchmark evaluator.
    import base64
    import zlib
    payload = base64.b64encode(zlib.compress(b"cos\nsystem\n(S'false'\ntR.")).decode()
    with pytest.raises(ValueError, match="pickle|private|object"):
        module.decode_private_cases(payload)


def test_private_case_decoder_preserves_released_json_string_representation():
    module = adapter("lcb_score")
    import base64
    import pickle
    import zlib
    payload = base64.b64encode(zlib.compress(pickle.dumps("[]", protocol=4))).decode()
    assert module.decode_private_cases(payload) == []


def test_reference_selection_retains_all_ten_released_outputs_in_order():
    module = adapter("benchmarks")
    choose = getattr(module, "select_lcb_reference_cases", None)
    assert callable(choose), "released reference replay selection is not implemented"
    # Format/provenance fixtures only; no executable code or evaluation inputs.
    tasks = [{"question_id": "stdio", "question_content": "public B", "starter_code": "",
              "metadata": "{}", "private_test_cases": "not decoded"},
             {"question_id": "function", "question_content": "public A", "starter_code": "signature",
              "metadata": '{"func_name": "metadata"}', "private_test_cases": "not decoded"}]
    references = [{"question_id": task["question_id"], "question_content": task["question_content"],
                   "starter_code": task["starter_code"],
                   "output_list": [f"metadata {index}" for index in range(10)],
                   "graded_list": [False, True] * 5} for task in tasks]
    chosen, raw, expected = choose(tasks, references)
    assert [row["question_id"] for row in chosen] == ["function", "stdio"]
    assert len(raw) == 20
    assert [row["sample_id"] for row in raw] == list(range(10)) * 2
    assert [row["source_sample_id"] for row in raw] == list(range(10)) * 2
    assert [row["text"] for row in raw[:10]] == [f"metadata {index}" for index in range(10)]
    assert expected == {"function": [False, True] * 5, "stdio": [False, True] * 5}


def test_reference_selection_cannot_treat_nine_samples_as_complete():
    module = adapter("benchmarks")
    choose = getattr(module, "select_lcb_reference_cases", None)
    assert callable(choose), "released reference replay selection is not implemented"
    tasks = [{"question_id": "format", "question_content": "public", "starter_code": "", "metadata": "{}"}]
    references = [{"question_id": "format", "question_content": "public", "starter_code": "",
                   "output_list": ["metadata"] * 9, "graded_list": [True] * 9}]
    with pytest.raises(ValueError, match="reference|ten|format|sample"):
        choose(tasks, references)


def test_native_result_publication_preserves_official_nonfinite_case_values(tmp_path):
    module = adapter("suite_score")
    publish = getattr(module, "publish_native_result", None)
    assert callable(publish), "lossless official native result publication is not implemented"
    # Official MBPP+ contains Infinity inputs. This is an I/O fixture, never scored.
    source, target = tmp_path / "source.json", tmp_path / "retained.json"
    source.write_text('{"failed_case": [-Infinity, Infinity]}\n')
    publish(source, target)
    assert target.read_bytes() == source.read_bytes()


def test_indexed_native_task_reads_only_its_line_and_rejects_changed_bytes(tmp_path, monkeypatch):
    module = adapter("benchmarks")
    load = getattr(module, "load_native_task", None)
    assert callable(load), "bounded native task loading is not implemented"
    import hashlib
    prefix = b'{"ignored":"first row"}\n'
    payload = b'{"question_id":"second","private_test_cases":"transport-only"}\n'
    path = tmp_path / "native.jsonl"
    path.write_bytes(prefix + payload)
    handle = {"question_id": "second", "_native_ref": {"path": "native.jsonl",
              "offset_bytes": len(prefix), "length_bytes": len(payload),
              "sha256": hashlib.sha256(payload).hexdigest()}}
    def no_whole_file(*args, **kwargs):
        raise AssertionError("must seek one native line instead of reading the full release")
    monkeypatch.setattr(Path, "read_text", no_whole_file)
    monkeypatch.setattr(Path, "read_bytes", no_whole_file)
    assert load(handle, tmp_path) == {"question_id": "second", "private_test_cases": "transport-only"}
    path.write_bytes(prefix + payload.replace(b"second", b"tamper"))
    with pytest.raises(ValueError, match="hash|changed"):
        load(handle, tmp_path)


def test_lcb_index_keeps_private_payload_out_of_task_handles(tmp_path):
    module = adapter("benchmarks")
    build = getattr(module, "index_lcb_sources", None)
    assert callable(build), "streaming native source indexing is not implemented"
    rows = [{"question_title": "format fixture", "question_content": "published", "platform": "atcoder",
             "question_id": identifier, "contest_id": "metadata", "contest_date": "2024-01-01T00:00:00",
             "starter_code": "", "difficulty": "easy", "public_test_cases": "[]",
             "private_test_cases": "PRIVATE_TRANSPORT" * 4096, "metadata": "{}"}
            for identifier in ("b", "a")]
    source = tmp_path / "native.jsonl"
    jsonl(source, rows)
    handles, stats = build(tmp_path, [source], expected_count=2)
    assert [row["question_id"] for row in handles] == ["a", "b"]
    assert all("private_test_cases" not in row and "public_test_cases" not in row for row in handles)
    assert all("PRIVATE_TRANSPORT" not in json.dumps(row) for row in handles)
    assert stats["max_line_bytes"] > 50000
    assert stats["total_rows"] == 2
    assert module.load_native_task(handles[1], tmp_path) == rows[0]


def test_reference_acquisition_extends_prepared_manifest_with_replay_inputs(tmp_path, monkeypatch):
    module = adapter("benchmarks")
    # Acquisition/manifest fixture only; no invented solution is scored.
    def acquired(url, target, *, download, git_blob):
        assert git_blob == module.LCB_REFERENCE_BLOB
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_text("released-output transport fixture\n")
        return {"url": url, "sha256": digest(target), "bytes": target.stat().st_size,
                "git_blob": git_blob}
    monkeypatch.setattr(module, "_download", acquired)
    path = tmp_path / "benchmarks-manifest.json"
    previous = {"files": {"native-index.jsonl": "a" * 64}, "complete_suite": True,
                "benchmarks": {"livecodebench": {"source": "unchanged release binding"}}}
    atomic_json(path, previous)
    module.acquire_lcb_references(tmp_path, download=False)
    prepared = read_json(path)
    reference = "sources/livecodebench-reference-outputs.json"
    receipt = "sources/livecodebench-reference-outputs.receipt.json"
    assert prepared == {**previous, "files": {**previous["files"],
        reference: digest(tmp_path / reference), receipt: digest(tmp_path / receipt)}}
    accepted = path.read_bytes()
    module.acquire_lcb_references(tmp_path, download=False)
    assert path.read_bytes() == accepted
