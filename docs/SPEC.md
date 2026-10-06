> 历史设计记录：原容器执行约束已由用户的原生运行要求取代。当前运行说明见 [NATIVE_RUNTIME.md](NATIVE_RUNTIME.md)，完整条件实验设计见 [G01](../research/design-v2/G01_EXPERIMENT_DESIGN.md)。下文保留原始方案语境，不是当前容器启动指令。

# Recursive SSD: bounded, label-free pilot

User-authorized delivery: implement recursive SSD and mathematically motivated alternatives in `Yunbo-max/theory`; prepare a single RTX 2080 Ti (11 GiB) run limited to eight hours. The user launches the GPU run. No remote GPU has been connected or scheduled.

## Scientific question

Can a frozen copy of the current policy generate unverified code and then train its successor repeatedly, without amplifying sampling noise or erasing useful support? Training consumes prompts and model outputs only. Correct answers, executable tests, test results, external teachers and reward models are excluded from generation and optimization. Benchmark correctness is used only in a separate evaluator after a checkpoint is frozen.

The parent problem is reliable multi-round self-training, not a claim that recursion itself is new. SSD, CRISP, iterative fine-tuning and recursive model-collapse work are related. No novelty or monotonic-improvement claim is established. Elementary lemmas below are tools for candidate construction, not claimed new theorems.

## Decisions

- Continue from the preceding round's adapter; refresh data and frozen teacher each round. Reusing one data set for several epochs is a separate control.
- Pin Apple SSD at `2637d2021f1bc523385b48a1f88ea9aa4812b0a9`. Its public repository provides generation/evaluation, not an SFT trainer. The trainer here is new, and this is a resource-constrained adaptation, not an exact reproduction of its paper.
- Preserve temperature → top-k → top-p decoding. Change BF16/vLLM/full-scale defaults to FP16, LoRA and sequential Hugging Face inference suitable for Turing GPUs. No quantization or flash-attention dependency.
- Use Qwen2.5-Coder-1.5B-Instruct, MBPP **train prompts only**, and official HumanEval+ tests via pinned EvalPlus. This is a smaller-scale transfer experiment; it cannot establish the original paper's LiveCodeBench result.
- One shared frozen base, separate student/teacher/lag adapters, completion-only losses, checkpointed vocabulary chunks. No two full model copies on GPU.
- Benchmark-generated Python executes only inside a constrained Docker container. The trainer never reads test outcomes.
- Default priority: base evaluation; three rounds each of hard SSD, full-soft decoder distillation and initial-distribution floor projection; fixed-data and arithmetic-anchor controls. Additional selected candidates are opt-in. An incomplete queue is reported as incomplete, never as a negative result.
- Downloads/installations happen before the timed window. Run start is local or an explicit timezone-aware `--start-at`. A persisted deadline prevents a resume from silently buying another eight hours.
- GitHub is the requested delivery destination. No Hugging Face upload or additional GPU spending is configured.

## Acceptance and limits

Engineering acceptance: numerical probability/constraint/gradient tests, a tiny random model round-trip test, recursive lineage and deadline/resume tests, package and CLI checks. These are not scientific benchmark evidence. Actual 2080 Ti memory/throughput, Docker live replay and native baseline qualification remain pending until the user's run. No Gate A/G01/E04 PASS is pre-issued.

The eight-hour pilot uses a fixed public development subset and one seed; all claims remain exploratory. Confirmation requires the held-out tasks, additional seeds, complete controls and matching information/token/wall-clock accounting. Failed runs and unfinished methods are retained.
