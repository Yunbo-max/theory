# Recursive SSD — Local execution handoff

Round: `2026-10-07-local-handoff`. Role: `web_supervisor`.
Status: **generated_unexecuted**. The user's 2026-10-07 clarification is that
Web finishes code and complete design, and Local runs tests/experiments.
No GPU run, new software pass or scientific qualification is claimed here.

Read [AGENTS](../../AGENTS.md), the [Local runbook](../../LOCAL_AGENT_RUNBOOK.md)
and its [downloads section](../../LOCAL_AGENT_RUNBOOK.md#download-datasets-and-models)
at the delivered `main` commit. The delivery receipt supplies this file's exact
commit after publication; this file cannot name its own future commit.

## Pinned starting state and preserved scope

Starting main: `f8a4ca035424668dc8bb805017d1dbe93a9a3a65`.
Historical tested source: `f12e6b3a976d00d99fedff9c23b63c39e6f352d2`.
The existing 20 mathematical cards and ranked top15 remain selected; this handoff
does not reopen method discovery or invent another candidate list.

- [Math cards](../../research/cards), [selection](../../research/selection.json),
  [selection check](../../research/selection-check.json).
- [Implementation batch](../../research/suite-v3/implementation-batch.json),
  [mapping cards](../../research/suite-v3/implementations),
  [historical CPU checks](../../research/suite-v3/verification/engineering-checks.json).
- [Full G01 coverage](../../research/suite-v3/G01_COVERAGE.md),
  [per-method comparisons](../../research/design-v2/METHOD_MATRIX.md),
  [analysis rules](../../docs/ANALYSIS.md),
  [native benchmark adapters](../../docs/BENCHMARK_ADAPTERS.md).
- [Historical native reference replay](../../research/suite-v3/native-qualification/README.md):
  HE/MBPP checks passed only in their recorded scope; LCB IPC failed; target-host
  qualification remains pending for every scorer and actual model baseline.

All 15 candidates and complete controls remain in the catalog. The development
inventory is 64 trajectories /190 evaluation cells /203 comparisons; tuning is
12 trials plus initial evaluation; independent confirmation uses at most two
actual development finalists, five new seeds and all three native benchmarks.
R5 and smaller-model boundaries remain separate. Nothing was reduced to fit an
unmeasured eight-hour window.

## This delivery

| Item | Source and acceptance |
|---|---|
| Explicit Local entry and SSH topology | [runbook](../../LOCAL_AGENT_RUNBOOK.md), root AGENTS and README; source/navigation review only |
| Complete asset and source cards | [LOCAL_ASSETS](../../docs/LOCAL_ASSETS.md), both Qwen revisions, all three benchmarks/scorers, clean prompt export, Apple SSD and GKD scope |
| Missing input lifecycle | [LOCAL_INPUT_CONTRACTS](../../docs/LOCAL_INPUT_CONTRACTS.md), nine JSONC documents in [templates](../../research/templates); unknown evidence stays null/non-dispatchable |
| Finalization accounting repair | [METADATA_BUDGET_HANDOFF](../../docs/METADATA_BUDGET_HANDOFF.md), `suite_queue.metadata_operation`; new checks need Local execution |
| Comparator qualification and dependencies | [COMPARATOR_QUALIFICATION](../../docs/COMPARATOR_QUALIFICATION.md); exact catalog jobs, genuine ancestor records and per-comparator native evidence |
| Candidate-derived calibration | [CALIBRATION_CHILD_PROTOCOL](../../docs/CALIBRATION_CHILD_PROTOCOL.md), `suite_precursor.py`; finite independently frozen child with its own method identity, exact native endpoint and original shared budget |
| Test staging | `scripts/research.py` queued check source/input capture; Local must test the actual isolated staged path, not only an unstaged direct pytest run |

The pinned vendor runtime is unchanged. New source/config/test changes invalidate
matching historical implementation bindings until Local creates current tested
evidence. The old report is retained, not silently relabeled as current.

## Local acceptance tasks

Use the runbook's ordered commands on the actual SSH GPU host, under the original
valid cumulative authorization. All executable tasks go through the one harness.

1. Restore actual host/path/interpreter/Conda/GPU identities and remaining
   authority; stage this exact delivered revision in an isolated checkout.
2. Native bootstrap/setup, complete input acquisition and integrity checks,
   then `research.py check --queue ...`. Inspect actual code hashes and semantics.
3. For repairs, run `tests/test_suite_queue.py`, `tests/test_suite_admission.py`,
   `tests/test_comparator_qualification.py`, `tests/test_suite_precursor.py` and
   `tests/test_research_controller.py`.
   Use the full test suite after affected
   checks, all inside the harness. These are planned tests; Web has run none.
4. Replay official native reference cases, retaining raw receipts and any IPC
   failure. Qualify real model baselines and complete comparator dependencies
   under their actual reviewed native protocols; no model score is invented.
5. Construct and review the complete finite calibration-child design where the
   comparator guide requires it. Its newly generated controller path preserves
   candidate code/design boundaries, real dependencies and the original budget;
   it supplies calibration evidence without accepting the whole main comparison.
6. Recompute current code/design evidence, native qualification and canonical
   Parent/Gate0/IPCG/freeze. Measure full-node costs/VRAM and admit only complete
   eligible comparisons. Initial short preflight does not cover long LCB inputs.
7. Execute the approved queue, apply native E04 and collect/push actual small
   evidence with exact execution identities. Incomplete units remain incomplete.

The historical original planned start was 2026-10-06 16:00 Europe/London for
eight hours. Actual launch/usage is unobserved, and the planned window has
elapsed. This handoff does not reset or extend that budget. It also does not
require future GPU results before Web may finish authoring the delivery.

## Return locator

Use `rounds/2026-10-07-local-handoff/runs/<actual-run-id>/` for a unique review
packet. Include original logs/receipts, tested/executed versions, all failed and
missing outcomes, budget and current running state. Follow the runbook's transfer
and exact-commit readback instructions. The selected GitHub destination is
`Yunbo-max/theory`, branch `main`; no new HF output destination is created.

Return: repository URL, actual pushed commit, packet path, completed scope,
pending items and observed next legal action. Upload success is not experiment
success; source inspection is not testing.
