# Native Conda/Python execution

The current user explicitly requires native evaluation. No runtime/setup/cleanup path invokes a container runtime.
The previous container implementation is superseded. The numerical default training configuration remains a small development run; the broader G01 design is separately conditional.

## Environment and commands

Use a project Conda environment with Python 3.11 or 3.12. An existing compatible environment can be reused.
scripts/python_env.sh chooses explicit PYTHON, then active CONDA_PREFIX, then an existing project venv; it never installs a daemon.

```bash
conda activate recursive-ssd
bash scripts/setup.sh
python -m recursive_ssd.cli qualify --artifacts artifacts --output artifacts/native-qualification
bash scripts/run_8h.sh
python -m recursive_ssd.cli report
python -m recursive_ssd.cli collect
```

Setup pins PyTorch 2.6.0 cu118, project dependencies, EvalPlus 0.3.1, tree-sitter 0.23.2 and tree-sitter-python 0.23.6.
It performs software checks, real data/model preparation, native official-reference qualification, then the real GPU preflight.
The authoring environment has no target GPU. The user-host installation, full GPU memory profile and elapsed throughput remain unmeasured.
No API service, external teacher, vLLM, FlashAttention or paid compute is required.

## Scoring contract

recursive_ssd/native_score.py sets HUMANEVAL_OVERRIDE_PATH before importing the official loader, uses the official sanitizer,
and invokes evalplus.evaluate.evaluate with parallel=2, test_details=True, min_time_limit=1 and gt_time_limit_factor=4.
Per-program memory is 4 GiB through the official EVALPLUS_MAX_MEMORY_BYTES option.
The exact official tests and base-plus success rule are retained. Empty/malformed model outputs remain scored failures under the native contract; missing inventory is an error.

The trainer launches a separate CPU Python process with an allowlisted environment and an overall timeout.
The child receives no GPU selection or API credentials from the inherited environment.
Native multiprocessing descendants are stopped on a local timeout; they inherit the worker process group so the supervisor's hard-stop applies to them.
EvalPlus resource/reliability controls and process cleanup are not an OS security sandbox; Conda is dependency isolation.
The current wrapper does not claim the old container's network or filesystem isolation.

Installed Python, all resolved package versions, the official EvalPlus source digest and wrapper bytes are bound into evaluator identity.
Inputs, native output, logs and execution argv/timestamps are retained in per-attempt directories.
A cache without a provenance receipt, with changed inputs/environment/source, or with changed result bytes is rejected.
Qualification also rechecks task count and canonical pass outcomes on cache reads.

## Migration and time budget

A native run has runtime_version=native-v1 and defaults to runs/2080ti-native-8h.
Old run folders/receipts are retained; their evaluator identity is incompatible and cannot be silently resumed.
Do not update an active GPU checkout in place. Use a new checkout/directory when adopting this revision.
A new directory does not grant more compute: if the original window began at 2026-10-06 16:00 Europe/London, its hard end remains 2026-10-07 00:00.

```bash
bash scripts/run_8h.sh --start-at '2026-10-06T16:00:00+01:00'
```

Use that original boundary only if it matches the actual authorized start. The web chat has not observed a GPU launch.
The process stores the deadline once; resuming that run preserves it.

## What qualification proves

The authoring CPU run uses official HumanEval/57 and HumanEval/72 and their released reference solutions.
A second raw-input/sanitizer replay passed both references, verified cache reuse and rejected changed input bytes. Exact logs, receipts and released reference inputs are retained under research/native-v1/verification/.
These are sourced evaluator checks, not generated model scores, GPU performance, baseline competitiveness, Gate A or design_verified.
All target-host and full-comparison evidence still needs the actual user-run experiment packet.
