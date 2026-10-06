# Engineering validation and outstanding checks

The checked test suite exercises:

- Temperature/top-k/top-p order, normalized detached targets for all 15 selected candidates and controls.
- Floor-projection KKT ratios, brute finite-simplex objective comparison, bounded log-ratios, conditional diversity and fixed-prefix sampling identities.
- Stratified-tail expectation, noise-budget feasibility, prompt allocation, the power-map composition identity and the softmax cross-entropy gradient.
- Tiny randomly initialized Qwen2 causal masks, reference/base freezing, every candidate's backward pass, both optimizer interventions, two-round adapter lineage and exact CPU optimizer resume.
- Deterministic cached generation including historical-policy mixing, native pass@k combinatorial identity, missing/duplicate evaluation inventory rejection, atomic deadline state and required baseline configuration.

Real public-data preparation was also exercised: the pinned MBPP split is reduced to `task_id,text`; official HumanEval+ has 164 distinct tasks and disjoint 32/132 task IDs. No target model weights, GPU training result or benchmark score was fabricated.

Logs in `research/verification/` record the actual final test command/output, environment, public-data manifest and source fingerprint. Read their timestamp and file identity; these checks are scoped engineering evidence, not scientific outcomes. Random tiny-model fixtures and probability vectors are not a substitute benchmark.

**Not available in the authoring environment:** CUDA/RTX 2080 Ti, a Docker daemon, native sandbox execution, actual FP16 GPU training, and an eight-hour throughput measurement. `scripts/setup.sh` performs the on-machine GPU canary before the timed run. The worker qualifies the official scorer on actual canonical tasks in Docker before evaluating generated code. Until those run successfully, hardware and native-evaluation readiness remain pending.

Full-scale/native confirmation, independent review, multiple seeds, all candidate-specific matched controls and a novelty review remain future evidence requirements. No code here should be described as empirically improving the model before those observations exist.
