# Source audit (2026-10-06)

- SSD: https://arxiv.org/abs/2604.01193v2 and https://github.com/apple-aiml-research/ml-ssd . Inspected generation, configuration, README, license and evaluation. Pinned source commit: `2637d2021f1bc523385b48a1f88ea9aa4812b0a9`. Public code does not contain the paper's training implementation. Temperature/top-k/top-p are reused as semantics; the training system here is an adaptation.
- CRISP: https://arxiv.org/abs/2603.05433v7 . Iterative self-policy distillation already exists; a concise-instruction teacher and reverse KL differ from raw-output SSD. It prevents claiming that iterative self-policy distillation in general is new.
- Iterative Finetuning is Mostly Idempotent: https://arxiv.org/abs/2605.01130 . Directly relevant to repeated fine-tuning on predecessors' data, although objectives, tasks and reset/continuation conditions differ.
- The First-Extinction Law: https://arxiv.org/abs/2509.20101 . Relevant recursive resampling/mode-loss literature. Neutral finite-population collapse is not a new discovery here.
- Why Self-Training Helps or Hurts: https://arxiv.org/abs/2602.14029 . Theoretical denoising/forgetting context; its assumptions do not establish LLM guarantees.
- SoFT: https://arxiv.org/abs/2609.32493v1 . Soft-target supervised learning is related, but our training labels are unverified own-policy samples, not externally supplied demonstrations.
- Qwen model: https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct , revision `2e1fd397ee46e1388853d2af2c993145b0f1098a`.
- MBPP: https://huggingface.co/datasets/google-research-datasets/mbpp , revision `4bb6404fdc6cacfda99d4ac4205087b89d32030c`; only `full/train` text and task IDs are exported for training.
- EvalPlus: https://github.com/evalplus/evalplus/tree/v0.3.1 , commit `e5d0ed0bab96280b60b637ec7f15b5e4841b0cb2`; HumanEval+ v0.1.10. Official tests and scoring, including base and plus results, are retained. A task subset is a pilot, not the complete benchmark.

Search scope is bounded, not an exhaustive novelty proof. All 20 cards are research candidates; some explicitly instantiate familiar KL projections, mixtures or allocation rules. Their value and novelty in recursive SSD require comparisons. No card is labeled a publishable new algorithm on algebra alone.
