# Native benchmark adapters

`recursive_ssd.benchmarks.prepare_benchmarks(root, download=True,
include_lcb=True, reuse_lcb_root=None)` prepares HumanEval+ v0.1.10 (164), MBPP+ v0.2.0 (378),
and LiveCodeBench code_generation_lite release_v5 (880). Acquisition and
all scorer processes must be invoked inside the project research harness.
`include_lcb=False` stages the two EvalPlus assets and explicitly records an
incomplete suite; it never makes a full experiment bundle ready.

The clean-v2 32-record prompt file is copied with its retained SHA-256 and
field allowlist. HumanEval uses the existing dev32/confirm132 ID order.
External benchmarks have only `full`; they have no tuning split. LCB reads
exactly `test.jsonl` through `test5.jsonl` at dataset revision
`0fe84c3912ea0c4d4a78037083943e8f0c4dd505`. Preparation verifies its complete
880-unique-ID inventory and never unpickles private tests.

Generation must read `benchmark_prompts_path(root, benchmark, split)`, which
contains only `task_id`, the exact prompt, and (for EvalPlus) `entry_point`.
`benchmark_tasks` returns native rows for EvalPlus and native byte-range
handles for LCB. `benchmark_prompt` uses
the released EvalPlus prompt or the official LCB generic user-message
template. LCB statement examples and starter code remain public; private
tests and metadata never enter this prompt. Apply the selected model's
recorded chat template once, outside the adapter.

The manifest `benchmarks-manifest.json` stores every split's task path,
prompt path, count, ordered IDs and file digests, native source receipts,
and clean training provenance. The full native test rows remain unchanged.
The manifest does not claim scientific qualification.

The five actual LCB files total several gigabytes. Preparation streams one
JSONL line at a time and writes a small `LiveCodeBench-full.jsonl` index with
public task fields plus `_native_ref` (`path`, `offset_bytes`, `length_bytes`,
and SHA-256 of the exact source line). Paths are relative to the prepared
data root. `load_native_task(handle, root)` seeks and verifies one record;
generation never resolves its private payload. The
`livecodebench-native-sources.json` sidecar binds the index and all original
source files, counts and observed maximum line sizes. A 512 MiB maximum
record/decoded-transport bound rejects oversized tasks explicitly; it never
drops them from the benchmark denominator. A larger required envelope must
be qualified before changing that bound.

`reuse_lcb_root` names a directory containing previously downloaded
`test.jsonl` through `test5.jsonl`. Reuse clones/copies immutable inputs into
the new root and preserves earlier attempts. It attempts filesystem
copy-on-write cloning and never makes writable hard links. Stage these
files as explicit native-harness inputs. Hugging Face HEAD metadata pins
the exact dataset commit, LFS SHA-256 and size; reused bytes must match.
This avoids downloading the same corpus after an interrupted preparation.

## Scoring interface

```python
score_benchmark(raw_path, tasks_path, output, benchmark,
                expected_samples, deadline, **options)
```

Benchmark names are `humaneval`, `mbpp`, `livecodebench`. Raw JSONL rows need
`task_id`, integer `sample_id` in `0..n-1`, and `text`; empty text remains a
failed candidate, never an excluded task. The adapter rejects missing or
duplicate samples, task changes relative to the pinned native release,
and incomplete normal benchmark denominators. Qualification allows a
native-source subset and is separately labelled.

| Option | Default | Meaning |
|---|---|---|
| `data_root` | task file's parent | Prepared release manifest and full native rows |
| `qualify` | `False` | Native scorer reference replay only |
| `source_root` | prepared LCB source directory | Exact official LCB import closure |
| `parallel` | `2` | CPU grader workers |
| `timeout` | `6` | Official LCB timeout per test, seconds |
| `max_seconds` | `1800` | Bounded native subprocess wall time, additionally limited by deadline |
| `k_values` | `[1, min(n, 10)]` for EvalPlus; `[1, min(n, 5)]` for LCB | Native unbiased pass@k estimates |

Results contain `task_count`, `samples_per_task`, `per_task` counts
(`n`, `correct`, `pass@1`, dynamic `pass@10`/`pass@5`), and the equal-weight
task means. Scores are fractions. EvalPlus correctness requires both base
and plus suites to pass. LCB correctness uses the official condition that
every returned test status is positive; all failures/timeouts remain in
the denominator. A successful program must return outcomes for every native
test. The wrapper also checks its aggregates against the official output.

Each attempt retains input files, sample mapping, sanitized/extracted code,
exact native test-source byte references (LCB), complete official results, evaluator logs,
and process execution receipt. `official_results.json` and
`native_receipt.json` bind raw/task bytes, native sources, interpreter,
installed environment, exact wrapper code, sample count and options.
A changed/unbound accepted cache is rejected. Earlier failed attempts stay
under `native/attempt-*`.

The LCB scorer resolves and grades one task at a time through the official
API, retaining all ten draws and all native tests. Task caches bind the
native source line, raw outputs, code, environment and evaluation options.
An interruption preserves completed native task outcomes for the next
authorized attempt. The final official aggregate uses every declared task;
there is no partial-denominator score. The original source files remain
the retained test-case artifact, avoiding a second multi-gigabyte decoded
test export. File hashing uses bounded buffers.

MBPP+ officially contains nonfinite input constants such as `Infinity`.
Use the supplied native task file directly or copy its bytes; do not
reserialize the full task corpus with strict finite JSON. Official native
result bytes are likewise copied without rewriting such failure details.
The reported metrics remain ordinary finite JSON.

## Official implementation bindings

EvalPlus is version 0.3.1 with tree-sitter 0.23.2/tree-sitter-python 0.23.6.
Dataset override variables are set before official imports. MBPP calls the
official `get_mbpp_plus`, preserving tuple/complex conversion and its native
special oracles. Generation is processed by official `sanitize`.

LCB source is commit `28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24`, with
an explicit import closure checked against official Git blob IDs. The
wrapper calls the unmodified `CodeGenerationProblem.get_evaluation_sample`,
`extract_code` (`CodeQwenInstruct`), and `codegen_metrics` APIs. The native
extractor uses the last fenced block and returns empty code if delimiters
are absent. The original custom-evaluator/parser files are retained as
source evidence. Native code-generation requires numpy 2.2.6, datasets
3.5.0 and tqdm 4.70.1; no inference provider SDK, vLLM or pyext is needed by
this adapter. The inspected parser specifies top_p 0.95, max_tokens 2000,
temperature 0.2 and 10 samples. Host length/cost/VRAM qualification is still
a separate requirement.

LCB's compressed private transport contains a pickled JSON string. A
restricted decoder forbids global object construction and persistent IDs,
then constructs the official `Test` objects with unchanged case input/output
strings and ordering. The official loader handles public tests, metadata
and evaluation-sample construction. This protects inspection from executing pickle
payloads; it does not change scoring. Official grader guards plus the
process deadline are native execution controls, not an OS security sandbox.
In particular, this LCB commit does not actually apply a 4 GiB memory cap
when calling its default reliability guard.

## Reference replay

EvalPlus `qualify=True` requires `raw_path=None`, `expected_samples=1`, and
uses the released prompt plus canonical solution.

LCB lite contains no canonical solution. `acquire_lcb_references` fetches
actual released Qwen2.5-Coder-7B-Instruct outputs from the official submissions
repository at `6ca212e9c2039373f6e5069d37ffa9db66e23736`, Git blob
`0cba6247fc4aae0d1875d49d15812a1429c8a6dc`. Qualification raw rows must include
the original `source_sample_id` and an exact original `output_list` string.
Replay compares native grades against those retained released grades,
including any selected known failures. Such replay qualifies only that
native scorer pathway; it is not a new model experiment or full benchmark
result. Never replace missing references with newly authored solutions.

`prepare_reference_qualification(root, output)` or `--reference-output`
writes a qualification input packet without scoring: two native canonical
HumanEval tasks, two native canonical MBPP tasks, and one functional plus
one stdin LCB task with every original ten-draw output and grade. The LCB
rule prefers tasks with both recorded successes and failures for scorer
path coverage, separately from research task selection. `--lcb-references`
acquires the released output source first.

Reference acquisition adds the released output and its acquisition receipt to
an existing asset manifest before qualification staging. Existing file bindings
must remain unchanged; repeated acquisition preserves the same manifest bytes.

The official LCB evaluator requires local socket IPC for
`multiprocessing.Manager`. A host that denies this capability cannot qualify
that scorer. Native `TestRunnerError` metadata is an infrastructure failure,
not a wrong solution: the adapter preserves the full return under
`native-task-results/` and rejects both a new accepted cache and any reused
result with that failure. Candidate compilation, runtime, and test failures
retain the official grading rules.

Engineering unit tests use metadata/security fixtures and never send
invented cases to a native grader. Full asset acquisition, reference replay,
and target GPU qualification have separate execution receipts.
