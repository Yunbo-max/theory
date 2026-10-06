# Complete native multibench implementation plan

> Execution: apply superpowers:executing-plans, with independent domains delegated under superpowers:dispatching-parallel-agents. Existing user authorization requests implementation and GitHub delivery.

**Goal:** implement the entire retained design-v2 backlog as usable native components and a finite resumable queue, preserving the skill's scientific admission boundaries.

**Architecture:** published benchmark adapters feed one task/result contract. Explicit arm specifications bind all method-specific contrasts to the existing recursive trainer. One skill execution harness owns every workload; the project controller prepares, admits, resumes and reports complete comparisons without inventing qualification evidence.

**Tech stack:** Python 3.11/3.12, PyTorch/Transformers/PEFT, EvalPlus 0.3.1, pinned official LiveCodeBench evaluator, the installed research-autopilot standard-library harness.

**Spec:** research/design-v2/G01_EXPERIMENT_DESIGN.md, METHOD_MATRIX.md and implementation-gaps.json. These remain conditional scientific protocols until real host/baseline qualification is complete.

## Global constraints

- Native Conda/Python only; one GPU; original cumulative eight-hour authorization is not reset.
- Existing 20 mathematical candidates and ranked 15 are reused. No new methods are invented in this repair.
- Train only on the 32 prompt-only clean-v2 MBPP records; preserve old artifacts and start the new lineage at the original base model.
- HumanEval+ development 32 / confirmation 132; MBPP+ 378; LiveCodeBench release_v5 full official inventory, no score-based subset.
- Development seed 17; confirmation 23,47,71,101,131; round 3 fixed endpoint, trajectories retained, round 5 separately configured.
- Published/native scoring only. Engineering fixtures never become scientific evaluation evidence.
- No GPU launch here; missing scientific gates remain explicit and checked at dispatch.

## Interfaces and ownership

- `benchmarks.py`: prepare/verify released datasets, prompt-only generation inputs and exact test inventories; `prepare_benchmarks(root)`, `benchmark_tasks(root, benchmark, split)`, `benchmark_prompt(row, benchmark)`.
- `suite_score.py`: `score_benchmark(raw_path, tasks_path, output, benchmark, expected_samples, deadline, **options)` returns per-task native success counts plus source receipts. Official LCB implementation and parser are pinned.
- `methods.py` / `train.py`: implemented deterministic/factorial/noise/SGD/GKD controls. `train_round(..., rollout_callback=None)` supports current-student per-update generation and bounded training-clock comparisons with resumable optimizer state.
- `analysis.py`: strict inventory validation, paired crossed task/seed bootstrap, predeclared multiplicity and outcome rules; no formal verdict from incomplete or unqualified results.
- `suite_design.py`: complete arm registry, method-to-comparison mapping, development/tuning/confirmation/boundary inventories and exact dependencies.
- `suite.py`: native workload entry points for prepare, qualify, preflight, one training/evaluation unit and collection; stages preserve checkpoint/input/scorer identities.
- `harness.py` and `scripts/research.py`: project adapter for the actual skill `run_harness.py`, portable pinned runtime files, preparation/engineering/scientific purpose separation and original budget persistence.

## Review focus

1. Missing, duplicate or shuffled benchmark samples must not reduce a denominator or change task matching.
2. Resume after partial generation/training must preserve model lineage, RNG, prior failures and deadline.
3. Shared baselines must match data, model, decoder, seed, optimizer and actual training budget; display aliases alone never justify reuse.
4. Confirmation cannot select hyperparameters/checkpoints or reuse development seeds; multiple comparisons use the frozen full family.
5. Changed source/data/environment or unresolved scientific prerequisites must reject dispatch rather than inherit a stale PASS.

## Task 1: benchmark assets and native scoring

- [x] Inspect actual official source/loaders and native release rows.
- [x] Write failing contract tests for inventories, leakage, sanitizer/scorer dispatch and cache identity.
- [x] Implement HumanEval+, MBPP+ and LCB adapters with pinned releases and exact native metrics.
- [x] Execute source-backed scorer checks through the harness and retain inputs, raw outputs and logs.

## Task 2: matched training controls

- [x] Recheck the selected method code boundary and retain current report.
- [x] Write mathematical/gradient/resume tests for the missing controls.
- [x] Implement soft LR/mixing, head factorial, smoothing, tail iid, lag/mean-gate/mean-weight/noise-budget controls, SGD norm matching, clock matching, GKD divergence and per-update rollout support.
- [x] Keep coefficient calculations independent of vocabulary chunk boundaries; record actual optimizer and generation costs.
- [x] Verify all controls and recursive lineage under the harness.

## Task 3: complete experiment queue and execution ownership

- [x] Implement the complete versioned arm registry and all 15 comparison bundles.
- [x] Implement actual development/tuning/finalist selection/confirmation/boundary compilation with stable IDs and native split/sample contracts.
- [x] Implement clean data/model preparation, real host calibration and finite complete-bundle capacity admission.
- [x] Route setup/checks/qualification/train/eval/replay/report/collection through the skill harness. Bind exact code, inputs, environment and required scientific design batch.
- [x] Preserve all failed/incomplete attempts and original cumulative time across resumes; no automatic new window authorization.

## Task 4: analysis, integration and delivery

- [x] Implement full comparison inventory and crossed task/seed uncertainty with frozen endpoint/multiplicity rules.
- [x] Verify with engineering fixtures and actual official released scorer cases; report GPU validation separately.
- [x] Review source-to-design coverage, update every implementation obligation with concrete file/test evidence and retain remaining host/scientific evidence gaps.
- [x] Push source and bound evidence to Yunbo-max/theory and read back exact remote files.

## Observed integration verification

Actual pinned harness run `suite-v3-final-verification-73aefe2e5c`: 332 tests passed; dependency, compilation, both current/legacy CLI help and shell syntax checks passed (7 commands). Software fixtures are engineering checks only. Complete native asset inventories were verified; official HE/MB reference replay passed; LCB reference replay is blocked by the authoring host's required IPC denial. No GPU experiment or scientific gate is passed. Current coverage and remaining genuine prerequisites are in `research/suite-v3/G01_COVERAGE.md`.
