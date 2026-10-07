# Recursive SSD agent entry

Read this file, [LOCAL_AGENT_RUNBOOK.md](LOCAL_AGENT_RUNBOOK.md), and the current
[Web handoff](rounds/2026-10-07-local-handoff/WEB_HANDOFF.md) at the **exact delivered
commit** before setup, testing, execution or repair. Read the runbook's
[download section](LOCAL_AGENT_RUNBOOK.md#download-datasets-and-models) before
asset-dependent work; it links the complete [asset inventory](docs/LOCAL_ASSETS.md).

## Role and delivery

- Web writes/reviews the complete code and experiment design, marks new work
  `generated_unexecuted`, and delivers to `Yunbo-max/theory` **main**. Web and its
  subagents do not execute generated code, tests, training, inference or evaluation.
- Local Codex runs on the user's computer and controls the separate Linux GPU
  host over the user's existing SSH connection. The GPU host runs ordinary
  Python/Conda processes; it does not need Codex or a remote GPT.
- The user clarified on 2026-10-07: finish code/design here; Local will run it.
  Missing future GPU results do not prevent delivering the complete draft.
  They do prevent claims of target-host acceptance or scientific success.
- Publish through one integration writer; preserve other changes, active runs
  and pinned revisions. Use normal fast-forward/expected-head updates and read
  back exact commits. Never force-push or silently substitute a branch/PR.

## Execution constraints

Use `scripts/research.py` and the unmodified runtime pinned by
`vendor/research_autopilot/VENDOR.json`. All executable project tasks on the GPU
host use this single harness, including setup, software checks, downloads,
reference replay, model runs, analysis and collection. Source reading, host
inventory, SSH/file transfer and Git delivery are controller operations.

Use native Conda/Python; **no Docker, Podman, Singularity or Apptainer**. Do not
launch a second raw training/evaluation driver. Restore actual SSH/path/Conda/GPU
facts and the original cumulative authorization; no copied example timestamp
or new queue creates a new eight-hour budget. Do not launch if it has expired.

## Scientific scope and evidence

Preserve all selected methods and necessary controls, the complete three-native-
benchmark design, dev/tuning/confirmation separation and failed/missing units.
The authoritative coverage draft is `research/suite-v3/G01_COVERAGE.md`;
`research/design-v2/METHOD_MATRIX.md` records the per-method comparisons.
Do not reduce coverage to make a smoke check look complete.

Templates in `research/templates` are deliberately non-dispatchable. Derive real
input hashes, qualification, calibration, canonical gates and freezes from actual
evidence; do not fill them with fabricated PASS flags or fake references.
Historical 332 CPU tests refer to the earlier source revision, not newly generated
repairs. Local must run the handoff's affected tests and retain their real logs.
Use the [debug routing](LOCAL_AGENT_RUNBOOK.md#debug-routing) and retain every
attempt. A new mechanism, information source or outcome-driven tuning rule needs
the existing reviewed child-protocol process. Do not claim novelty or improvement
from a code review, a target-distribution bound or recursive training alone.
