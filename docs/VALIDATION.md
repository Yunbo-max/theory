# Engineering validation and outstanding checks

Historical validation is retained below. Current suite-v3 historical CPU evidence
is in `research/suite-v3/verification/`; new Web handoff repairs are
`generated_unexecuted` and require Local checks. Start with
[LOCAL_AGENT_RUNBOOK.md](../LOCAL_AGENT_RUNBOOK.md) and the
[current handoff](../rounds/2026-10-07-local-handoff/WEB_HANDOFF.md).
`research/native-v1/verification/` is an earlier snapshot. All execution is native
Conda/Python; no container-based qualification is part of the current protocol.

The checked test suite exercises:

- Temperature/top-k/top-p order, normalized detached targets for all 15 selected candidates and controls.
- Floor-projection KKT ratios, brute finite-simplex objective comparison, bounded log-ratios, conditional diversity and fixed-prefix sampling identities.
- Stratified-tail expectation, noise-budget feasibility, prompt allocation, the power-map composition identity and the softmax cross-entropy gradient.
- Tiny randomly initialized Qwen2 causal masks, reference/base freezing, every candidate's backward pass, both optimizer interventions, two-round adapter lineage and exact CPU optimizer resume.
- Deterministic cached generation including historical-policy mixing, native pass@k combinatorial identity, missing/duplicate evaluation inventory rejection, atomic deadline state and required baseline configuration.

Real public-data preparation was also exercised: the pinned MBPP split is reduced to `task_id,text`; official HumanEval+ has 164 distinct tasks and disjoint 32/132 task IDs. No target model weights, GPU training result or benchmark score was fabricated.

Logs in `research/verification/` record the actual final test command/output, environment, public-data manifest and source fingerprint. Read their timestamp and file identity; these checks are scoped engineering evidence, not scientific outcomes. Random tiny-model fixtures and probability vectors are not a substitute benchmark.

**Not observed by Web:** target CUDA/RTX 2080 Ti execution, actual FP16 GPU
training and full-workload throughput. Current setup performs dependency/software/
asset preparation through the native harness; reference qualification and reviewed
GPU preflight are separate commands in the runbook. Target-host and complete
model-comparison readiness remain pending until their real receipts exist.

Full-scale/native confirmation, independent review, multiple seeds, all candidate-specific matched controls and a novelty review remain future evidence requirements. No code here should be described as empirically improving the model before those observations exist.

## Native-v1 更新

原生 runtime 回归与完整 CPU 检查保存在 `research/native-v1/verification/`，旧 `research/verification/` 是历史快照。官方 HumanEval/57 和 HumanEval/72 的参考答案在 native Python EvalPlus 路径通过；这只验证已运行的评分路径。目标 2080 Ti 的驱动、显存、耗时和实际模型结果仍未知。
