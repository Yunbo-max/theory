"""CPU child invoking pinned, unmodified LiveCodeBench native grading APIs.

Only the released private-case transport is normalized: a restricted unpickler
accepts its JSON string payload without allowing Python object construction.
CodeGenerationProblem still builds the native samples, and codegen_metrics
performs all execution, comparisons, error handling and pass@k calculation.
"""
import argparse
import base64
import gc
import io
import json
from pathlib import Path
import pickle
import sys
import zlib

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "recursive_ssd"

from .benchmarks import LCB_FIELDS, load_native_task, task_id, verify_lcb_source
from .io import atomic_json, digest, object_hash, read_json, read_jsonl

MAX_PRIVATE_BYTES = 512 * 1024 * 1024


class _DataUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise ValueError("private testcase pickle contains an executable object")

    def persistent_load(self, pid):
        raise ValueError("private testcase pickle contains a persistent object")


def _validate_cases(cases):
    if not isinstance(cases, list):
        raise ValueError("native testcase transport must contain a list")
    for row in cases:
        if (not isinstance(row, dict) or set(row) != {"input", "output", "testtype"}
                or not isinstance(row["input"], str) or not isinstance(row["output"], str)
                or row["testtype"] not in ("stdin", "functional")):
            raise ValueError("invalid released native testcase schema")
    return cases


def decode_private_cases(value):
    """Decode native JSON or compressed-pickled-JSON-string; reject object code."""
    if not isinstance(value, str):
        raise ValueError("private testcase transport is not a string")
    try:
        cases = json.loads(value)
    except json.JSONDecodeError:
        try:
            compressed = base64.b64decode(value.encode("ascii"), validate=True)
            inflater = zlib.decompressobj()
            payload = inflater.decompress(compressed, MAX_PRIVATE_BYTES + 1)
            if len(payload) > MAX_PRIVATE_BYTES or not inflater.eof or inflater.unused_data:
                raise ValueError("private testcase pickle exceeds bounded transport or has trailing data")
            stream = io.BytesIO(payload)
            decoded = _DataUnpickler(stream).load()
            if not isinstance(decoded, str) or stream.read():
                raise ValueError("private testcase pickle must contain only one JSON string")
            cases = json.loads(decoded)
        except (ValueError, TypeError, UnicodeError, zlib.error, pickle.UnpicklingError, EOFError) as exc:
            raise ValueError(f"invalid private testcase pickle/JSON transport: {exc}") from exc
    return _validate_cases(cases)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--raw", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--task-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-samples", required=True, type=int)
    parser.add_argument("--parallel", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=6)
    parser.add_argument("--k-values", default="1,5")
    args = parser.parse_args()
    source = Path(args.source).resolve()
    source_identity = verify_lcb_source(source)
    # Import ordinary official packages from the exact copied source closure.
    sys.path.insert(0, str(source))
    from lcb_runner.benchmarks.code_generation import CodeGenerationProblem, Test
    from lcb_runner.evaluation.compute_code_generation_metrics import codegen_metrics
    from lcb_runner.evaluation.pass_k_utils import compute_metrics_from_results
    from lcb_runner.lm_styles import LMStyle
    from lcb_runner.utils.extraction_utils import extract_code
    from .suite_score import summarize_lcb, validate_samples

    tasks = read_jsonl(args.tasks)
    grouped = validate_samples(read_jsonl(args.raw), tasks, args.expected_samples, "livecodebench")
    tasks.sort(key=lambda row: task_id(row, "livecodebench"))
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache = Path(args.task_cache).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    ks = sorted({int(value) for value in args.k_values.split(",")})
    environment_sha = object_hash(read_json(output / "input-receipt.json")["evaluator"])
    sample_map, case_counts, results, metadatas, task_receipts = [], {}, {}, [], {}
    with (output / "samples.jsonl").open("w") as sample_stream:
        for index, handle in enumerate(tasks):
            identifier = task_id(handle, "livecodebench")
            outputs = [generation["text"] for generation in grouped[identifier]]
            codes = [extract_code(text, LMStyle.CodeQwenInstruct) for text in outputs]
            sample_stream.write(json.dumps({"question_id": identifier, "output_list": outputs,
                                             "code_list": codes}, ensure_ascii=False) + "\n")
            sample_stream.flush()
            identity = {"task": handle, "raw_sha256": object_hash(grouped[identifier]),
                "source_sha256": source_identity["sha256"], "wrapper_sha256": digest(Path(__file__)),
                "environment_sha256": environment_sha, "expected_samples": args.expected_samples,
                "k_values": ks, "parallel": args.parallel, "timeout": args.timeout}
            cache_path = cache / (object_hash(identifier) + ".json")
            if cache_path.exists():
                packet = read_json(cache_path)
                if packet.get("identity") != identity or packet.get("result_sha256") != object_hash(packet["native"]):
                    raise ValueError(f"native task cache provenance changed: {identifier}")
                native = packet["native"]
                atomic_json(output / "native-task-results" / cache_path.name, native)
                print(json.dumps({"task": identifier, "status": "reused_native_task", "index": index}), flush=True)
            else:
                print(json.dumps({"task": identifier, "status": "scoring_native_task", "index": index,
                                  "total": len(tasks)}), flush=True)
                row = load_native_task(handle, args.data_root)
                private = decode_private_cases(row.pop("private_test_cases"))
                # Let the official loader construct public tests and metadata.
                # Populate private tests with its exact Test class after safe
                # transport decoding, avoiding a second huge JSON serialization.
                row["private_test_cases"] = "[]"
                problem = CodeGenerationProblem(**{key: row[key] for key in LCB_FIELDS})
                problem.private_test_cases = [Test(**case) for case in private]
                count = len(problem.public_test_cases) + len(problem.private_test_cases)
                if not count:
                    raise ValueError(f"released task has no native tests: {identifier}")
                counts = {"public": len(problem.public_test_cases), "private": len(problem.private_test_cases)}
                sample = problem.get_evaluation_sample()
                del private, row, problem
                gc.collect()
                native_metrics = codegen_metrics([sample], [codes], k_list=ks,
                    num_process_evaluate=args.parallel, timeout=args.timeout, debug=False)
                native = {"task_ids": [identifier], "case_counts": {identifier: count},
                          "counts": counts, "metrics": native_metrics}
                # Retain the complete official return before validation. An
                # infrastructure exception must leave its original metadata
                # available, without publishing it as an accepted task cache.
                atomic_json(output / "native-task-results" / cache_path.name, native)
                summarize_lcb(native, [identifier], args.expected_samples, ks)
                atomic_json(cache_path, {"identity": identity, "native": native,
                                        "result_sha256": object_hash(native)})
                del sample, native_metrics
                gc.collect()
            summarize_lcb(native, [identifier], args.expected_samples, ks)
            case_counts[identifier] = native["case_counts"][identifier]
            results[index] = native["metrics"][1].get("0", native["metrics"][1].get(0))
            metadatas.append(native["metrics"][2][0])
            sample_map.append({"index": index, "question_id": identifier,
                "sample_ids": [generation["sample_id"] for generation in grouped[identifier]],
                "native_ref": handle["_native_ref"], "public_case_count": native["counts"]["public"],
                "private_case_count": native["counts"]["private"]})
            task_receipts[identifier] = {"path": str(cache_path), "sha256": digest(cache_path)}
    atomic_json(output / "sample-map.json", sample_map)
    # Native aggregate from the official estimator; every task remains present.
    metrics = [compute_metrics_from_results(results, k_list=ks), results, metadatas]
    # Keep the official result verbatim, plus its otherwise positional task map.
    atomic_json(output / "native-metrics.json", metrics)
    atomic_json(output / "official_results.json", {
        "task_ids": [handle["question_id"] for handle in tasks], "case_counts": case_counts,
        "metrics": metrics, "source": source_identity,
        "task_receipts": task_receipts, "data_root": str(Path(args.data_root).resolve()),
        "execution_scope": "one native task at a time; all samples/tests retained",
        "private_transport": "restricted pickle JSON-string decoding; native test values unchanged",
    })


if __name__ == "__main__":
    main()
