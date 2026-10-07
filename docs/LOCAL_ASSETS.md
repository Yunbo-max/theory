# Local input acquisition and integrity cards

Status: **generated_unexecuted**. This document was source-reviewed against the
delivered suite-v3 interfaces; its commands have not been executed by Web. Start
with [LOCAL_AGENT_RUNBOOK.md](../LOCAL_AGENT_RUNBOOK.md), then the current round's
`WEB_HANDOFF.md`. Local Codex runs on the user's computer and controls the Linux
GPU host over SSH. No Codex installation is required on the GPU host.

## Download datasets and models

All command cards below run on the GPU host, in the exact staged project checkout,
using its **active native Conda Python 3.11/3.12**. The SSH alias, remote checkout,
Conda installation/environment and available disk are facts that Local must
restore; this document does not invent them. `python` below means that verified
Conda interpreter, not a different system interpreter. See the runbook for source
staging and the host/path record.

**These are conditional execution cards, not permission to start a fresh window.**
Every preparation call supplies `--queue runs/multibench-v3`, which must resolve
to the existing original authorization. Its known intended start was
`2026-10-06T16:00:00+01:00`, with a maximum 28,800 seconds and a hard end of
`2026-10-07T00:00:00+01:00`. That boundary has passed. No actual start/consumption
has been observed by Web. Restore actual Local evidence first; do not reset the
start, omit `--queue`, or create another budget to make these cards run.

`scripts/research.py` is the only project controller. Its setup/acquisition/
qualification operations compile a bounded CPU task and execute it through the
**existing pinned** `vendor/research_autopilot/scripts/run_harness.py`. No card
requires Docker, Podman, Singularity or Apptainer. Do not call an inner module or
official benchmark executable directly to bypass the harness or budget.

### Environment and installed scoring packages

```bash
python scripts/research.py setup --queue runs/multibench-v3 --seconds 1800
python scripts/research.py pipcheck --queue runs/multibench-v3 --seconds 120
```

The first command submits two real installation tasks: PyTorch `2.6.0` from
`https://download.pytorch.org/whl/cu118`, then `pip install '.[test,evaluation]'`.
It then runs `pip check`. Each invocation retains its own attempt and cost; the
1800-second option is not a promised total time for the entire setup sequence.
The project dependency source is [pyproject.toml](../pyproject.toml):

| Component | Required version / use |
|---|---|
| Python | >=3.11,<3.13 in active Conda; `_conda()` verifies interpreter ownership |
| PyTorch | 2.6.0, CUDA 11.8 wheel source; target driver compatibility still needs Local evidence |
| transformers / peft / accelerate | 4.51.3 / 0.15.2 / 1.6.0 |
| datasets / huggingface-hub | 3.5.0 / 0.30.2 |
| numpy / safetensors | 2.2.6 / 0.5.3 |
| EvalPlus | 0.3.1, audited upstream commit `e5d0ed0bab96280b60b637ec7f15b5e4841b0cb2` |
| tree-sitter / tree-sitter-python | 0.23.2 / 0.23.6 |
| psutil / tqdm / pytest | 7.2.2 / 4.70.1 / 8.3.5 |
| jinja2 | >=3.1,<4; actual resolved version must be captured |

This is **not** a complete hash-locked transitive environment. Setup binds the
actual environment receipt; `evaluator_info()` also fingerprints installed
EvalPlus Python source. The PyPI version and upstream Git commit are distinct
identities; no unavailable wheel checksum is invented. Any environment change
after qualification invalidates its bound identity. Do not silently upgrade to
the current HF CLI or reinstall a different evaluator to overcome a failure.

The project's model acquisition uses the pinned HF **Python SDK**. It does not
depend on whether `hf download` exists in hub 0.30.2. Do not substitute CLI flags
from newer documentation. No external teacher API, judge, reward model, embedding
model, vLLM, FlashAttention or paid inference service is an input to this suite.

### Complete model and code inventory

| Input | Immutable identity | Consumers and acquired paths |
|---|---|---|
| Qwen/Qwen2.5-Coder-1.5B-Instruct | `2e1fd397ee46e1388853d2af2c993145b0f1098a` | All primary/development/tuning/confirmation/R5 arms; `artifacts/multibench-v3/model-manifest.json` and `models/<revision>/` |
| Qwen/Qwen2.5-Coder-0.5B-Instruct | `ea3f2471cf1b1f0db85067f1ef93848e38e88c25` | Separate smaller-model boundary only; `artifacts/multibench-v3-05b/model-manifest.json` and `models/<revision>/` |
| Original SSD released source subset | [apple-aiml-research/ml-ssd](https://github.com/apple-aiml-research/ml-ssd/tree/2637d2021f1bc523385b48a1f88ea9aa4812b0a9), commit `2637d2021f1bc523385b48a1f88ea9aa4812b0a9` | Acquired with this project checkout under `vendor/apple-ssd/`; per-file hashes in [SOURCE.json](../vendor/apple-ssd/SOURCE.json). Copied function template/license in `recursive_ssd/templates/` |
| Project trainer and all comparison arms | Exact delivered project commit and code references | `recursive_ssd/suite_train.py`, `train.py`, `methods.py`, `suite_design.py`; no separate baseline checkpoint download |
| Pinned execution runtime | 2026-10-06 snapshot with every file SHA-256 | [VENDOR.json](../vendor/research_autopilot/VENDOR.json); already included in the project checkout, do not upgrade it to the currently installed skill |

```bash
python scripts/research.py model --queue runs/multibench-v3 \
  --data artifacts/multibench-v3 \
  --model Qwen/Qwen2.5-Coder-1.5B-Instruct --seconds 1800
```

For the separately admitted smaller-model boundary, stage its complete input root
before building that boundary queue; do not overwrite the primary model manifest:

```bash
python scripts/research.py assets --queue runs/multibench-v3 \
  --data artifacts/multibench-v3-05b --seconds 1800
python scripts/research.py model --queue runs/multibench-v3 \
  --data artifacts/multibench-v3-05b \
  --model Qwen/Qwen2.5-Coder-0.5B-Instruct --seconds 1800
```

`suite.prepare_model()` fixes the revision through `suite_design.MODELS`, calls
`snapshot_download(..., local_dir=..., max_workers=2)`, and includes all matching
`*.json`, `*.safetensors`, `*.txt`, `*.model`, `*.tiktoken` files. This includes
config, generation config, tokenizer files and any matching shard-index/shards;
it does not intentionally select one weight shard. Real files are promoted into
the artifact root, rather than symlinks to an unbound global cache. The manifest
lists actual relative paths and SHA-256s. `verify_model_manifest()` rejects
missing, changed, extra or symlinked files and mismatched model/revision.

The loader is `Policy.load()` in [model.py](../recursive_ssd/model.py):
`AutoTokenizer` and `AutoModelForCausalLM` use the verified local snapshot,
`local_files_only=True`, `trust_remote_code=False`, FP16 and SDPA. They do not
download missing files during a training/evaluation task. The full base is
loaded once with student/teacher/lag LoRA adapters. Later-round adapters come
from actual predecessor output/receipt hashes, never a guessed HF checkpoint.

The current downloader checks weight presence and the acquired file inventory;
it does **not** separately prove every safetensors index reference is complete
before loading. Local model-load/preflight qualification must reject incomplete
weights/tokenizers. A local SHA-256 records acquired identity; it is not an
invented upstream checksum. Model storage has not been measured in this project:
rough planning estimates are about 3 GB for 1.5B FP16 weights and 1 GB for 0.5B,
plus metadata, retained attempts, promoted copies and checkpoint outputs. Use
actual downloaded bytes and filesystem capacity before admission, not these
estimates as certified RAM/VRAM/disk requirements.

SSD's public code supplies generation/prompt/evaluation semantics but does not
contain the paper's training implementation. This project's trainer is an
adaptation, not an exact author-code training reproduction. The six
`gkd_{fkl,rkl,jsd}_{cached,update}` controls are self-policy adaptations implemented
here. They share the same base model/data; they do not acquire a stronger GKD
teacher. [The retained GKD note](../research/literature/GKD_2306.13649v3.md) names
paper v3 and the inspected TRL implementation commit; its historical
“not yet implemented” paragraph predates suite-v3. Current implementation and
comparison obligations are in [G01_COVERAGE.md](../research/suite-v3/G01_COVERAGE.md).
Do not install TRL or SSD's original multi-GPU evaluation stack for this suite.

### Complete benchmark and training inputs

One source-inspected command downloads, builds and verifies **all** required
benchmark data, native source files and released replay outputs:

```bash
python scripts/research.py assets --queue runs/multibench-v3 \
  --data artifacts/multibench-v3 --seconds 1800
```

| Input | Released identity and source | Complete local inventory / consumer |
|---|---|---|
| Training prompts | `google-research-datasets/mbpp`, revision `4bb6404fdc6cacfda99d4ac4205087b89d32030c`, `full/train` | Retained clean-v2 32-record export copied from `research/design-v2/data/train_prompts-clean-v2.jsonl` into `train_prompts.jsonl`; only `task_id,text` are permitted |
| HumanEval+ | [v0.1.10 release](https://github.com/evalplus/humanevalplus_release/releases/download/v0.1.10/HumanEvalPlus.jsonl.gz) | `HumanEvalPlus-full.jsonl` 164; deterministic retained dev32/confirm132 split files and `humaneval-prompts-{full,dev,confirm}.jsonl` |
| MBPP+ | [v0.2.0 release](https://github.com/evalplus/mbppplus_release/releases/download/v0.2.0/MbppPlus.jsonl.gz) | `MbppPlus-full.jsonl` 378; `mbpp-prompts-full.jsonl`; native EvalPlus loader retains special oracles and nonfinite input representations |
| LiveCodeBench | `livecodebench/code_generation_lite`, revision `0fe84c3912ea0c4d4a78037083943e8f0c4dd505`, release_v5 | Complete `test.jsonl`…`test5.jsonl` under `sources/livecodebench-data/`; 880 original tasks, native test payloads unchanged |
| LCB evaluator | [LiveCodeBench/LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench/tree/28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24), commit `28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24` | `sources/livecodebench/<commit>/`; exact native API import closure and retained parser/prompt/extractor source |
| LCB scorer replay outputs | [LiveCodeBench/submissions](https://github.com/LiveCodeBench/submissions/tree/6ca212e9c2039373f6e5069d37ffa9db66e23736), commit `6ca212e9c2039373f6e5069d37ffa9db66e23736` | `sources/livecodebench-reference-outputs.json`, source path `Qwen2.5-Coder-Ins-7B/Scenario.codegeneration_10_0.2_eval_all.json`; requires **outputs only**, no 7B model weights |

The 32 training prompt bytes are already acquired in Git; fetching the exact
project commit is their acquisition step. The current assets command verifies
and copies them instead of downloading/reselecting the raw MBPP dataset. The
selection, all32 IDs, exclusion of108 exact MBPP+ overlapping IDs, and SHA-256
are in [clean-training-manifest.json](../research/design-v2/data/clean-training-manifest.json).
This excludes exact-ID overlap, not every possible semantic or pretraining
overlap. Never replace this with the legacy `data.prepare()` subset or reuse an
adapter trained on that older lineage.

LCB URLs have the literal pattern
`https://huggingface.co/datasets/livecodebench/code_generation_lite/resolve/0fe84c3912ea0c4d4a78037083943e8f0c4dd505/test.jsonl`,
with the other four exact names in the table below. The preparation worker
obtains immutable-revision HF HEAD metadata, verifies commit, linked SHA-256
and byte count, then downloads raw bytes. It creates
`LiveCodeBench-full.jsonl`, a byte-range index, and
`livecodebench-native-sources.json`, plus `livecodebench-prompts-full.jsonl`.
This is the **actual native test preparation step**, not simply downloading a
dataset repository. `benchmark_prompts_path()` supplies generation-only views;
`load_native_task()` resolves/verifies each original native record for scoring.
The scorer never substitutes the prompt mirror for the full tests.

Source-backed fixed identities:

| File | Bytes (retained acquisition) | SHA-256 |
|---|---:|---|
| clean-v2 prompt export | See actual retained file; 32 records | `a9f7642d10f57d03abc77b55e46e8158f5cb2e1b9fd5335baf4d4ecc41e6092f` |
| HumanEvalPlus-full.jsonl | 7,714,666 | `42526ec0e7d5f3ee0b06d6ced98f8c8bae3d76519151bfb3d36f79010645bd7f` |
| MbppPlus-full.jsonl | 2,592,369 | `b54e762755248ca411b523c917fa9f93c07b5ff2966bf60b3917b853926a3dad` |
| test.jsonl | 1,252,609,773 | `2bd02b38beb48e8c46b5b9987095d999ff38cd8efc255ea5d58974317c48f63f` |
| test2.jsonl | 713,377,060 | `095df7c5daf15f882c51a9deb84085cff1e073495a5dbcf95015a564d485f3a3` |
| test3.jsonl | 623,360,766 | `28ed26cc83363ce3f1fe2d5fad9f8393077beb1907b167a31bd3b32f80801b79` |
| test4.jsonl | 1,204,644,685 | `d711138ddaebfcf5f8ec6a4283ee677298c0f5c5d374a235af92aaf0584510da` |
| test5.jsonl | 557,699,297 | `7f77571c2a6df0c2a72a3277650309f67e01e0008e18117e624633df53f81214` |
| LCB released replay outputs | 10,858,879 | `586707158f8d1648964f54f0290dc42c94806b12684ea03b537202008170c992` |

The EvalPlus hashes apply to **decompressed** JSONL, not the gzip download.
LCB original files total 4,351,691,581 bytes; maximum observed record size is
92,838,755 bytes. These historical identities are retained in
[asset-identities](../research/suite-v3/native-qualification/asset-identities/).
The current preparation still verifies actual Local bytes. Account for the
isolated harness workspace, promoted copies, compressed/partial files, models
and retained failed attempts; 4.35 GB is not a total free-disk requirement.
Memory and full queue disk peaks remain unmeasured on the target host.

The official LCB import closure uses per-file **Git blob IDs** from
`benchmarks.LCB_SOURCE_BLOBS`; it downloads exact raw GitHub URLs and writes
`source-receipt.json`. `verify_lcb_source()` rejects changed files and unexpected
Python modules in that closure. Replay outputs additionally require Git blob
`0cba6247fc4aae0d1875d49d15812a1429c8a6dc`. Git blob IDs are not SHA-256s.

### Integrity, qualification and paths to retain

`assets` verifies the pinned release hashes, ordered task inventories, clean
prompt fields and non-overlap, all five LCB source-file identities and indexed
record hashes. `model` writes and verifies its local manifest. Both operations
publish only after a real completed native-harness receipt and output hashes
are checked. Required success records are:

* `artifacts/multibench-v3/benchmarks-manifest.json`, with `complete_suite=true`,
  all three benchmark inventories and every retained input digest;
* `artifacts/multibench-v3/livecodebench-native-sources.json` and the exact source
  files it names, plus replay-output receipt;
* `artifacts/multibench-v3/model-manifest.json` and all physical snapshot files;
* `runs/controller/<actual-assets-or-model-id>/preparation.json`,
  `environment.json`, `native-plan.json`, `harness-plan.json`,
  `harness-result.json`, `promoted-outputs.json` when promotion succeeds;
* native `receipt_ref`/`attempt_path` from those actual results, including all
  `stdout.log`, `stderr.log`, `attempt.json` and failed acquisitions. UUIDs are
  resolved from receipts, never filled with hypothetical paths.

Then submit source-specific semantic tests and the three separate native
reference replays through the same controller and budget:

```bash
python scripts/research.py check --queue runs/multibench-v3 --seconds 300 -- \
  tests/test_benchmarks.py tests/test_suite_runtime.py tests/test_native_runtime.py
python scripts/research.py qualify --queue runs/multibench-v3 \
  --data artifacts/multibench-v3 --benchmark humaneval --seconds 600
python scripts/research.py qualify --queue runs/multibench-v3 \
  --data artifacts/multibench-v3 --benchmark mbpp --seconds 600
python scripts/research.py qualify --queue runs/multibench-v3 \
  --data artifacts/multibench-v3 --benchmark livecodebench --seconds 600
```

These targeted tests supplement, not replace, the runbook's complete software
acceptance. Replay uses official HE/0,/1 and Mbpp/2,/3 canonical solutions and
the selected published LCB functional/stdin cases with all10 original draws.
No synthesized answer is accepted as a reference. New target-host receipts are
required even though historical authoring CPU HE/MB replay passed. The
historical LCB `multiprocessing.Manager` AF_UNIX `EPERM`/`EOFError` remains an
infrastructure failure, not a zero model score. Read raw native metadata and
the [retained failure](../research/suite-v3/native-qualification/README.md).
Fix the actual IPC/host restriction and requalify unchanged official code;
do not alter the scorer, ignore `TestRunnerError`, or use a container workaround.

Native reference replay does not qualify model performance, all benchmark
tasks, GPU memory, throughput, Parent/Gate0/IPCG or a frozen full design. Those
remain separate runbook steps.

### Reuse, failures and exact capability limits

The CPU preparation resource defaults are two cores,4096 MiB RAM,zero GPUs,
with exclusive key `recursive-ssd-single-host`. Do not concurrently mutate the
same asset root/cache. Network and elapsed preparation time count against the
same total allowance; every new call is a real new bounded attempt.

The inner `_download()` reuses an already present **verified final file** in
its workspace. Its HTTP path has a120-second socket timeout and writes
`.download`/`.unpacked` intermediates. It does not implement HTTP Range resume;
retrying the same path truncates the partial download. Preserve the original
failed attempt before any retry, and never delete a cache in use. HF SDK
partial-cache behavior is version-specific; retain its `.cache` and receipt
rather than promising that a new isolated task will reuse it.

The public controller currently has **no** `assets --offline`,
`assets --reuse-lcb-root`, model cache-import or generic arbitrary-command
flag. In particular, its assets/model preparation does not stage an existing
destination root as a task input. A second invocation may reacquire data in a
fresh workspace even if a complete promoted root exists. Do not run another
download merely to claim an integrity check; downstream binding/qualification
checks the existing manifests and inputs.

The inner `benchmarks.py` supports `--offline` and `--reuse-lcb-root`, but those
are **not public controller flags**. A future bounded recovery using that path
requires a reviewed harness preparation plan with every reused file as an
explicit immutable input, or a reviewed controller extension and Local
qualification. It must preserve output promotion and budget accounting. This
capability gap is not permission to execute the module raw, skip LCB, switch to
release_v6, create writable hard links or repair a digest in metadata.

On hash mismatch, HTML response, truncated JSONL, Git LFS pointer, incomplete
weight/index, expired time or insufficient disk: retain the attempt and exact
source/version/path, block dependent work, and repair only that evidenced cause.
Use a new artifact root for changed source bytes; it still shares the original
authorization. Do not substitute another model, test split or scorer version.
These inputs were public at retained acquisition; current access/network status
must be observed on Local. If permission is denied, use existing authorized
credentials only on the user's controller/host, never in tracked files, exported
commands or returned logs. Report the exact denied resource; no paid/gated
replacement is authorized by this handoff.
