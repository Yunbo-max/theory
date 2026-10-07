# Native candidate calibration children

Status: **generated_unexecuted**. This source review adds no Local qualification,
scientific PASS, new GPU authorization or validated runtime results. Local uses
the existing pinned `scripts/research.py` harness after reading
[LOCAL_AGENT_RUNBOOK.md](../LOCAL_AGENT_RUNBOOK.md). Do not execute these commands
in Web, run a raw model driver or replace the vendor runtime.

## Why this is a distinct G01 child

M06's measured-head control needs an actual M06 rollout; M13's learning-rate and
equal-clock controls need actual M13 updates/timing; M15's equal-clock control
needs actual M15 timing. Their main comparison cannot require the very same
control qualification before producing its source measurement. Renaming these
candidate workloads `baseline-calibration` does not resolve that dependency.

The finite child runs the specified candidate source and derivation under its
**own reviewed, frozen candidate protocol**. It retains `method_discovery`, a
complete independent actual code/design-verification batch, current canonical
candidate/freeze checks, initial/hard/full-soft native qualification and live
official scorer replay. It provides measured calibration/comparator evidence,
not acceptance of the whole main comparison. Every full comparison in
[G01_COVERAGE.md](../research/suite-v3/G01_COVERAGE.md) remains required.

The vendor Gate A freeze requires protocol `evidence_mode=prospective_confirmatory`.
The preregistered child retains that exact frozen protocol mode; its bounded
measurement jobs explicitly use native plan `evidence_mode=developmental` and
`admission_scope=candidate-calibration-child`. This supported vendor distinction
does not qualify them as the main method's prospective confirmation results.

M01 used as a comparator in M02 keeps `method_discovery.candidate_id=M01` and
`bundle_id=M02`. `floor_zero` keeps the M03 method identity;
`head_retention_zero` keeps M06; `allocation_without_weight_correction` keeps M14.
An alias does not remove the method-design prerequisites.

## Exact scope fields

The ordinary frozen native protocol contains a `suite_calibration_child` object
with **exactly** these fields. This is a field contract, not a ready-to-dispatch
JSON packet; values and file digests must come from Local's retained evidence.

| Field | Actual source |
|---|---|
| `purpose` | Literal `native-comparator-calibration` |
| `suite_ref` | Digest reference to the complete existing development/tuning catalog |
| `bundle_id` | Exact consuming bundle in that catalog |
| `node_ids` | Exact candidate train/evaluate and/or derivation node IDs, including every ancestor |
| `node_roles` | Map each node ID to `{group, arm_role}` in the child's native contracts |
| `node_seconds` | Map every declared node ID to its positive finite prospectively reviewed wall bound |
| `config` | Frozen project configuration; matches later comparator source usage |
| `calibration` | Frozen catalog calibration; no invented measured values |
| `tuning_budget` | `null` for development; exact `{trial_wall_seconds, source_refs}` for tuning, same allowance used across the later complete tuning families |
| `execution_provenance` | Actual `model_revision`, `environment_digest`, `data_revision` and any other frozen provenance bindings |

Node IDs are taken from `suite_queue.compile_nodes(suite)` via the catalog and
controller metadata interfaces. Do not invent strings or replace trajectory
settings in a request. A source M06/M13/M15 derivation uses its **source** arm's
native role/name even though the catalog node's `arm_id` identifies the eventual
ordinary comparator consuming the result. Every candidate/alias train or
evaluation node's underlying registry method must equal the child's own
`method_discovery.candidate_id`.

The entire catalog is reconstructed with `make_suite` and compared byte-for-value;
recomputing a digest after changing an arm cannot bypass the method identity.
For tuning children each trajectory's declared train-node bounds sum to at most
the frozen trial allowance. Native sample IDs, original label/test refs,
HumanEval project split and every decoder parameter must match the exact catalog
endpoint and prepared asset manifest. Benchmark-name substring matching is not
used.

The finite declared bounds sum to at most the original 28,800-second cap. They
are not a new grant, a measured cost profile or a promise that all work fits.
Before the first attempt, the controller reserves the complete child node
inventory plus every planned native replay. Each charge releases only its own
node/replay reservation. Already charged child nodes cannot execute again under
the same protocol identity. An initially unknown VRAM peak is permitted only on
the exclusive single GPU within these finite bounds; a partial supplied profile
is rejected, and OOM remains a failure. Main full-suite admission still requires
an actual complete memory profile.
Report/collection jobs reserve and charge their own finite allowance from the
same ledger when requested; child admission does not reserve extra finalization
capacity. Preserve headroom in the reviewed node/replay bounds. An expired
original authorization remains expired. Failed attempts retain their charges;
there are no automatic retries in a child.

## Materialization and execution inputs

Local supplies a request JSON containing only `{"node_id": "<actual catalog ID>"}`
to the project qualification route, with actual bindings for:

- `protocol_ref` and `verification_ref`: the frozen child and its actual complete
  code/design-verification batch;
- `group`, `arm_role`, `config`, `calibration`: exact frozen child values;
- `tuning_budget`: the exact frozen child allowance, or absent/null for development;
- `data`: project-relative prepared asset directory, containing the verified
  `benchmarks-manifest.json` and `model-manifest.json`;
- `budget_path`: the same project-relative original authorization ledger;
- `dependency_state_ref`: a digest reference containing **exactly the complete
  transitive ancestor entries** of this particular requested node; omit it for
  a dependency-free first node;
- host/GPU/resource and live replay settings required by the existing controller.

Use the existing controller's `qualify --job ... --bindings ... --queue ...`
CLI route as documented in the runbook. `prepare_child_job` only constructs the exact job; it never executes
it. The controller creates immutable job/bindings files and includes their refs
in the native plan. The permitted command is the current Python executable plus
`-m recursive_ssd.suite execute --job <absolute verified job path> --data
<project-relative data> --output results --seconds <frozen node bound>`.
The sole declared native output is `results/receipt.json`. The single pinned
harness owns process launch, isolation, timing, failures and output retention.

After each real attempt the controller retains `dependency-record.json`. To
assemble a later ancestor state, merge the actual returned `nodes` entries;
reject conflicting duplicate IDs. Include failed entries in the retained full
history, but only complete, native-verified entries can satisfy a dependency.
Do not manufacture `attempts`, `receipt_ref`, `workload_receipt_ref`,
`native_plan_ref` or `output_refs` from a narrative log.

## Historical receipt checks and downstream use

For a completed source, `validate_child_dependency` checks the retained native
plan, its exact materialized job/argv, immutable method-design evidence, native
attempt files, stdout/stderr and output digests. It requires the same suite,
model/data/environment/configuration, and an actual charge for the native receipt
in the same original ledger. The source candidate may cease to be canonical
current after it has run; historical verification does not rewrite it or demand
that the new current candidate be changed back. New dispatch still checks the
current candidate and replays all required native qualification evidence.

The ordinary comparator route can consume these measured derivation receipts;
it cannot execute candidate source nodes under a baseline label. The main full
bundle needs its own complete formal protocol, live comparator qualification,
measured resource admission and all required comparisons. No child result is a
complete development selection or independent confirmation result.

## Local acceptance and repairs

Run the affected software checks, including `tests/test_suite_precursor.py`, via
the runbook's pinned software-check harness. These generated tests cover finite
scope, source-role identity, complete ancestors, protected aliases,
candidate-as-control identity, arbitrary-job rejection and missing-receipt
rejection. They have **not** been executed by Web.

| Rejection | Local repair |
|---|---|
| `CALIBRATION_CHILD_OWN_METHOD_DESIGN_REQUIRED` | Select the actual underlying candidate's complete verified child design; never relabel it as hard/full-soft |
| `CALIBRATION_CHILD_COMPLETE_DEPENDENCY_SCOPE_REQUIRED` | Include all actual catalog ancestors and their bounded costs in the prospectively reviewed child |
| `CALIBRATION_CHILD_EXACT_DEPENDENCY_CLOSURE_REQUIRED` | Assemble the exact ancestor entries from genuine returned dependency records |
| `CALIBRATION_CHILD_CHARGED_DEPENDENCY_REQUIRED` | Reconcile the real harness receipt against the original ledger before proceeding; never add a fictional charge |
| `CALIBRATION_CHILD_NATIVE_ARM_MISMATCH` | Bind a derivation to its source candidate, and an alias to its exact registry arm name |
| `CALIBRATION_CHILD_DERIVED_JOB_MISMATCH` | Regenerate through the controller from exact source refs; do not edit a frozen job |
| Original budget expired/exhausted | Preserve progress; no start-time reset or silent new authorization |

Keep source changes, Local software validation, native target-host qualification
and scientific conclusions as separate evidence states.
