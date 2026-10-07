# Comparator qualification and measured child dependencies

Status: **generated_unexecuted**. All source and test changes in this handoff
await Local acceptance. There are no new model results, native qualifications,
frozen designs or gate approvals. The pinned vendor is unchanged.

The earlier four-name bootstrap could not qualify the full comparison catalogue.
This source repair adds a separate exact-catalogue comparator scope, a strict
scientific child route for selected methods and their measured dependencies,
and per-comparator source reuse. The complete main comparison is retained.
See [Local input contracts](LOCAL_INPUT_CONTRACTS.md),
[child protocol contract](CALIBRATION_CHILD_PROTOCOL.md), and the
[Local runbook](../LOCAL_AGENT_RUNBOOK.md).

## Ordinary comparator scope

`prepare_native_protocol` accepts the following `bindings.qualification` fields:

| Field | Source and validation |
|---|---|
| `purpose` | `comparator-qualification` |
| `allowed_arm_roles` | Exact native contract roles |
| `suite_ref` | Immutable complete development or tuning catalogue; reconstructed against `make_suite` |
| `bundle_id` | Existing selected method's complete comparison bundle |
| `arm_bindings` | Every native role maps to one distinct catalogue arm ID; `base` maps to native name `initial` |
| `method_verification_ref` | Actual current method batch; code boundary and exact implementation refs are reverified |
| `config`, `calibration` | Exact frozen configuration; external arbitrary calibration is rejected |
| `execution_provenance` | Exact `model_revision`, `data_revision` and `environment_digest` |
| `tuning_budget` | Null/absent for development; exact source-bound `trial_wall_seconds` allowance for tuning |

Every arm must be a required comparator of the specified bundle. The bundle's
main treatment is excluded. An arm whose registered method is a selected Mxx
method is rejected, even if its label looks like a baseline or an ablation.
The old `baseline-calibration` / `native-evaluator-qualification` scope and its
four-name allowlist remain unchanged.

In tuning, only the actual three-value arithmetic-control family belonging to
M03 or M01 and the initial model may use the ordinary scope. Candidate grid arms
retain their selected-method child route. Both pin the same reviewed trial
allowance; the ordinary controller includes paid ancestor attempts when checking
the remaining bound. Grid labels cannot be substituted for a different family.

A job request for this route is **only** `{"node_id": "exact catalogue node"}`.
Use the existing CLI:

```bash
python scripts/research.py qualify --job research/local/comparator-node.json --bindings research/local/comparator-bindings.json --queue runs/local-window --data data/suite --seconds 600
```

The bound `seconds`, paths and queue above must be replaced with the actual
reviewed Local files and retained authorization, not copied as an approval.
Bindings include `protocol_ref`, `group`, `arm_role`, `config`, `calibration`,
`gpu_uuid`, actual resources and, after round one, `dependency_state_ref`.
The controller supplies the real data and original budget paths. Each successful
controller result points to a `dependency-record.json` with native plan,
attempt, output and workload receipt refs. Combine the exact ancestor records
in a new immutable `{"nodes": {...}}` file for the next request. Missing or
extra ancestors are rejected; there are no future checkpoint placeholders.

The controller derives the workload from the catalogue. Direct native-plan
callers face the same check: exact interpreter/argv, job bytes, seed, role,
settings, current environment/source inventory and complete data/model inputs.
Parent native plans are revalidated, their outputs remain bound to actual
attempts, and their charges must exist in the same original budget. A changed
source, checkpoint, role or configuration invalidates reuse.

## Selected methods and calibration dependencies

`qualification_dependency_report(suite)` is a source-derived authoring diagnostic.
It lists every required comparator, preserves selected-candidate aliases, and
reports the following dependency edges. Eligibility is not actual qualification.

| Consumer | Required source |
|---|---|
| M02's M01 comparator | M01's own applicable method and independently frozen child design |
| `floor_zero` | M03 despite its control label |
| `head_retention_zero` | M06 despite its control label |
| `allocation_without_weight_correction` | M14 despite its control label |
| `temperature_matched_head` | M06 training and `head_temperature` derivation |
| `full_soft_lr_matched` | M13 training and `m13_step_scale` derivation |
| `full_soft_equal_clock` | M13 training and `clock_m13` derivation |
| `hard_equal_clock` | M15 training and `clock_m15` derivation |
| `arithmetic_anchor_tuned` | Actual complete development tuning selection |

`suite_precursor.py` implements the finite child route for selected methods and
source-calibration nodes. It retains `method_discovery`, the actual verified
child code/design batch, canonical current-candidate freeze and live native
baseline qualification. It cannot use the ordinary comparator exception.
The child protocol is prospectively frozen under the existing vendor schema;
its measurement jobs are explicitly developmental. Its complete node inventory
and node bounds are frozen, reserved together, and charged from the original
shared budget. No main efficacy claim or main comparison PASS follows.
Unknown peak VRAM may be measured by this finite child on the single exclusive
GPU; OOM remains a retained failure. A supplied measured profile must be complete
and valid. The complete main queue still requires its actual measured profile.

Once real child outputs exist, an ordinary calibrated comparator can consume
that exact dependency closure. The controller rechecks historical child method
and native plans, actual attempts/checkpoints, source suite/settings/provenance,
and the original budget charges. It derives calibration from those bytes.
Arbitrary supplied temperatures, clocks, step scales or approval booleans cannot
replace the child evidence. A new child review/freeze is Local scientific work;
this handoff provides its interfaces and draft, not a forged successful record.

## Per-comparator native evidence

The legacy `suite_qualification_evidence[group] = {protocol_ref, manifest_ref}`
remains supported. The new form is:

```json
{"comparators": {
  "baseline": {"protocol_ref": {"path": "...", "sha256": "..."}, "manifest_ref": {"path": "...", "sha256": "..."}, "source_arm_role": "treatment"},
  "full_soft": {"protocol_ref": {"path": "...", "sha256": "..."}, "manifest_ref": {"path": "...", "sha256": "..."}, "source_arm_role": "full_soft"}
}}
```

This structural example is not a runnable evidence packet. Supply **all** main
non-treatment roles, with actual immutable refs. Each role must have exactly the
same arm implementation and native sample/test/scorer/sampling identity as its
source. A selected source method's own full design is reverified. A source used
by several roles is replayed once per group, and every current qualification
rule is applied to its actual recomputed metric. Missing, failed or mismatched
controls remain blocking. Initial/strong/simple controls cannot be removed to
make a full main protocol freeze succeed.

For source-bound comparator/child cost observations, each `costs[cell_id]` also
pins `plan_ref`. The exact node ID, native plan/receipt identity, source scope,
model/environment/code and retained logs must match. This cannot assign a
candidate measurement to another workload. Final full-suite admission still
requires complete measured bounds and all main scientific prerequisites.

## Local acceptance

Run the existing admitted `scripts/research.py check --queue ...` path at the
exact delivered commit. New planned checks are in
`tests/test_comparator_qualification.py` and `tests/test_suite_precursor.py`.
They cover catalogue scope, protected aliases, code boundary, dependency closure,
per-role source reuse, missing controls, changed identities and finite child
scope. They have not been executed here. Passing software tests would establish
those software semantics only; actual child evidence, official scorer replay,
capacity, full main freeze and scientific conclusions remain Local tasks.
