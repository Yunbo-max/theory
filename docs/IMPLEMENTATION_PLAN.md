# Implementation plan

Goal: a reproducible, bounded, recursive self-policy experiment that the user can run on one 2080 Ti.

Architecture: pinned prompt preparation → shared-base LoRA model → refreshed per-round generation → frozen-target training → isolated official evaluation → atomic state/report. Python 3.11, PyTorch, Transformers, PEFT and EvalPlus. Specification: `docs/SPEC.md`.

1. Audit upstream and primary related work; record adaptation, provenance and mathematical limits.
2. Write and semantically review 20 mathematical candidate cards; compare the complete pool and select 15. Run the research-autopilot selection evidence checker before candidate implementation.
3. Implement normalized decoder distributions and explicit target constructors with full-soft and hard controls. Test support, signs, normalization, expectation and constraints.
4. Implement shared-base adapters, causal completion masks, chunked FP32 losses with FP16 model activations, checkpoint lineage and resumable optimizer state. Verify on a tiny random causal model; this fixture is software testing only.
5. Implement pinned native data, isolated official scoring, deadline-aware generation/training, reports and return bundles. Test inventory completeness and resume behavior.
6. Freeze the eight-hour protocol and run CPU checks. Report GPU/native-evaluation gates as pending; do not invent measurements.
7. Publish source, derivations, manifests and run instructions to the authorized repository; verify the remote commit and contents.

Review focus: q is detached and refreshed per round; predecessor rather than base initialization; no hidden correctness filtering; hard/soft controls share prompt/sample accounting; no future-position leakage into stochastic target allocation; no mislabeling scalar target-space bounds as neural-network guarantees; no silent deadline reset or missing-task score inflation.
