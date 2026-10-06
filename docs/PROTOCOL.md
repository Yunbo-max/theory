# Frozen pilot protocol

Configuration: `configs/2080ti_8h.json`. The launcher stores its exact bytes' semantics, a code/config/template content manifest, model/data identities, evaluator image ID and start/end timestamps in `run.json`. Changing any scientific setting requires a new run identity. The default queue is fixed in advance and does not inspect correctness to determine updates, weights, sampling, stopping or method order.

## Question, comparison and scope

Primary question: does recursive floor-constrained self-distillation improve the quality/retention trade-off over recursive hard SSD, full-soft decoder distillation and a simple arithmetic initial-policy mixture? Secondary controls: repeatedly train first-round data and halve hard-SSD learning rate. Recursion itself and generic KL anchoring are not claimed novel.

Each round continues from the preceding round's LoRA adapter. References are frozen copies of the same model: initial base, previous round and (only for relevant candidates) lagged round. Teachers and students see the same prefix. The target is recomputed from these frozen references during the round; it is not a function of the evolving student. Dropout is zero. Optimizer state resets at round boundaries for every arm and is preserved within a round on resume.

Hard/full-soft/floor/arithmetic/fixed-data/lower-LR arms use 32 fixed MBPP train prompts, two own-policy samples per prompt and one pass over each round's samples, microbatch one and gradient accumulation four. There is no correctness, length-quantile or empty-text filtering. EOS tokens and length-capped sequences are retained. Loss is a mean over completion tokens in each sequence and then an equal-sequence mean. Prompt tokens are masked. M14, if added, corrects unequal prompt counts with explicit per-record weights.

These arms match prompt/sample counts and nominal optimization steps, not realized tokens, FLOPs or wall time. Output lengths, additional reference forwards and different optimizers can change costs. Conclusions must include the recorded costs; the pilot cannot establish compute efficiency by nominal equality alone.

## Native evaluator and separation

HumanEval+ v0.1.10, EvalPlus 0.3.1, source tag commit `e5d0ed0bab96280b60b637ec7f15b5e4841b0cb2`. A deterministic SHA-256 task ordering assigns 32 tasks to the development pilot and 132 to held-out confirmation before any result is seen. Native task contents and tests are unchanged. The override file is a subset of official rows, not a newly invented benchmark.

Two samples per task, temperature .8, top-p .95 and no top-k truncation at evaluation; identical settings and sample seeds across arms. Raw generations are kept. Official sanitizer produces full Python solutions; the official base/plus checks execute only in a network-disabled Docker sandbox. Candidate code is never executed by the training process. The official scorer is first qualified on canonical solutions of two actual tasks inside this same sandbox. No canonical solution enters training.

Complete evaluation inventory is required: 32 tasks × 2 samples, no dropped failures or empty outputs. Native pass@1 and pass@2 use the unbiased per-task estimator, then average over tasks. pass@2 with two samples means whether at least one sample passes; it is not independent confirmation of a pass@1 gain. Both base and plus must pass for plus success.

The GPU evaluator reads a prompt-only mirror, not the benchmark's canonical solutions or tests. The prepared benchmark is accessible to the isolated scorer. Training is a fixed prompt/raw-output computation, not a correctness-feedback loop.

## Mathematical distinguishing observations

- Compare rounds at equal round index; later partial arms cannot be compared against earlier completed arms as if matched.
- Mandatory full-soft baseline integrates the local next-token label. Its performance determines whether a sampling-noise explanation is sufficient.
- Floor projection must additionally compete against arithmetic anchoring; merely preserving entropy is insufficient evidence of usefulness.
- On fixed base-generated prefixes, log KL in both directions, entropy, Gini diversity, initial-decoder excluded-support mass and Σ max(c p₀−p_student,0). A floor in q does not imply a floor in the fitted student.
- Do not claim component-level variance identities reduce the covariance of a whole autoregressive parameter gradient. Do not infer correctness from agreement or confidence.
- For optional M12 add full-soft SGD; for M09/M11 add matching scalar/fresh-label controls; M15 requires its additional generation cost to be charged. Candidate cards list remaining comparisons. They are not automatically included in this first window.

## Outcome rules and evidence gates

The first eight-hour run is a one-seed development pilot. Every method's scientific outcome remains **INCONCLUSIVE**. Report task-cluster paired bootstrap intervals descriptively (4,000 resamples with a fixed seed); these do not cover training-seed uncertainty, correlated task difficulty or multiple candidate selection. Never promote a method using only an interval excluding zero here.

Native scorer failure, missing inventory, data/source mismatch, nonfinite loss, zero successful updates and failed baseline execution are invalid execution outcomes, not negative scientific findings. AMP overflow/skipped updates are logged and mark optimization validity false even when a checkpoint exists. An interrupted generation/training/evaluation is incomplete, never a failed hypothesis.

Gate 0/IPCG: theory-driven investigation with the support-loss/temperature-drift counterexamples in `research/THEORY.md`; no new naturally observed GPU failure is claimed. CPU checks only establish engineering properties. Native baseline qualification, Gate A/G01 live replay, complete method-specific controls, held-out confirmation and E04 empirical review remain pending. The method-selection checker does not certify these stages.

No grant of extra compute is implied by an unfinished queue. At the persisted deadline the runner checkpoints/stops. A future confirmation or new eight-hour run requires a deliberate new user-run command; there is no background cloud spending or automated deployment.

## Exact version and limitations

Model and data revisions are recorded in `recursive_ssd/data.py`; downloads are checked against the run's stored file hashes. The complete official HumanEval+ source file SHA-256 is recorded during preparation. Docker image content ID is recorded and enforced on resume. Python package versions are pinned for the trainer; transitive environment versions are included in the validation log/environment manifest where available.

Code/model/data versions cannot remove pretraining contamination. The small instruct model, MBPP prompts, LoRA, short sequences and HumanEval+ evaluator differ from SSD's paper. This pilot supports at most a bounded exploratory result on this configuration.
