# Native protocol and finite queue admission

`recursive_ssd.suite_admission` is the project bridge to the **unmodified**
research-autopilot runtime pinned in `vendor/research_autopilot/VENDOR.json`.
It compiles real supplied source/evidence bindings into runnable native plans.
It does not generate reviewers, qualification observations, canonical approvals,
or a `design_verified` field. The outer `run_harness.py` remains the execution
owner; these APIs launch no process.

The implementation is available for real GPU-host evidence. The current CPU
engineering tests establish contract behavior only. They do not qualify the
GPU, establish benchmark performance, or make any candidate scientifically ready.

## 1. Released assets to exact native definitions

All references are `{path, sha256}`, relative to the same project root. Paths
outside that root, symlinks, missing bytes, and stale hashes are rejected by the
vendor runtime. Source acquisition/authentication remains the host's obligation:
a hash of a fabricated source or a stored `qualified: true` is not evidence.

```python
assets = prepare_native_assets(
    root, asset_manifest_ref,
    groups={
        "primary-he-dev": {"benchmark": "humaneval", "split": "dev"},
        "primary-mbpp": {"benchmark": "mbpp", "split": "full"},
        "primary-lcb": {"benchmark": "livecodebench", "split": "full"},
    },
    definitions=retained_source_backed_definitions,
    output_dir="research/native-protocol-assets-v2",
    human_eval_split_ref=retained_humaneval_split_ref,
    subset_support_ref=official_evalplus_subset_entrypoint_ref,
)
```

`asset_manifest_ref` points to the actual `benchmarks-manifest.json` written by
`benchmarks.prepare_benchmarks`. The compiler checks the complete original
inventory (HumanEval 164, MBPP 378, LiveCodeBench 880) and every prepared file's
declared hash. It creates small immutable native sample and definition sidecars.
It does not download or change the original data.

Each `definitions[group]` must contain the captured upstream metadata:

| Field | Required meaning |
|---|---|
| `benchmark_id`, `benchmark_revision` | Exact released benchmark identity |
| `published_at`, `source_url`, `publication_refs` | Source-backed date, URL and retained original artifacts |
| `upstream_split` | Actual upstream split name |
| `primary_metric`, `metrics` | `pass@1` and native `pass@10` for HE/MBPP or `pass@5` for LCB; every metric has its exact native JSON `output_path` |
| `prediction_format` | Native JSON/JSONL format, `records_path`, `id_path` |
| `sampling` | Explicit policy and parameters; this suite requires `parameters.n=10` |
| `budget` | All finite native resource limits, with matching telemetry names |
| `scorer` | Official identity/revision, source/code refs, exact native argv/cwd, JSON output location and denominator path |

These fields must describe the inspected official source. The compiler does not
infer publication dates, decide qualification thresholds, or label a project
scorer as official. A faithful project wrapper instead supplies a `scorer` with
`kind="faithful_harness"` and a pinned parity contract when compiling the native
contract. The vendor checks it against the official scorer; live replay of both
scorers remains necessary.

**HumanEval project subsets.** The original fixed split uses ascending
`int(sha256("eval-split-v1|" + task_id)[:15], 16)`: 32 development tasks and the
remaining 132 confirmation tasks. This differs from the vendor's optional
integer-seeded `sha256_rank` selector. The bridge retains an explicit versioned
`humaneval-project-split-v1` adapter and verifies the original rule, both retained
ID lists, and every selected row against the complete original release. The
native sidecar names the scope `test/project-dev` or `test/project-confirm` when
the actual upstream split is `test`. These are **project-selected published
tasks**, not newly claimed upstream splits. The official subset-support source,
unchanged original test bytes, and adapter ref stay bound. No new tasks, labels,
tests, or metric are introduced. Low-level caller definitions for a
`/project-` split require the same adapter check.

**LiveCodeBench native shards.** The complete five-file release is large. Its
prepared 880-row index contains original-line locations, not private tests.
The bridge pins `livecodebench-native-sources.json`, the index, all five original
native files, and a labels manifest naming those refs. The definition's
`required_input_refs` contains that entire input closure. Plan validation rejects
missing shard inputs. File hashing streams; the compiler never deserializes the
private-test payloads. The native scorer adapter resolves the exact original
source lines from these pinned inputs.

The result supplies:

```python
{
    "benchmark_manifest": {group: sample_manifest_ref},
    "native_definition_refs": {group: native_definition_ref},
    "split_adapter_refs": {project_subset_group: adapter_ref},
}
```

## 2. Compile, freeze, and verify current evidence

`build_native_contract(root, sample_manifest_ref, arms, scorer_spec)` builds a
single vendor `native-eval-contract`. `arms` maps native roles to
`{name, revision, implementation_refs}`. Roles are `treatment`, `baseline`, and
one or more named controls; each control's role equals its arm name. The vendor
requires this full comparison structure even for qualification protocols.

`scorer_spec` contains:

```python
{
    "native_definition_ref": actual_definition_ref,
    "baseline_qualification": {
        "metric": native_metric, "operator": comparison_operator,
        "threshold": predeclared_threshold, "reference_ref": real_source_ref,
    },
    "control_qualifications": [
        {"role": control_role, "metric": native_metric, "operator": comparison_operator,
         "threshold": predeclared_threshold, "reference_ref": real_source_ref},
    ],
    # optional: scorer (verified faithful harness), selection (vendor-supported)
}
```

`prepare_native_protocol(root, benchmark_manifest, arms, seeds, groups,
scorer_spec, bindings, output)` expects both manifest/spec mappings keyed by
exactly `groups`. `bindings["protocol"]` provides the complete reviewed G01
gate-a/full-validation fields, including criterion/uncertainty/inventory policy,
evidence snapshot, method/source-tree version, and analysis plan where required.
It never fills missing scientific facts. A changed existing output requires a
new child path.

For a candidate, `bindings["method_discovery"]` is
`{candidate_id, batch_ref}` for the actual immutable current code-discovery batch.
The compiler calls `verify_methods.before_action(..., "experiment-design", id)`
and replays the canonical project prerequisites. Missing or stale evidence rejects
compilation; a prose design or a historical test count does not substitute.

Preparation computes the protocol digest but does **not** freeze it.
`freeze_native_protocol(root, relative_protocol_path, gate="gate-a",
replay_context=actual_live_callback)` replays existing native qualifications and calls the
vendor's real locked `freeze_gate` transaction. It validates the actual canonical
ledger, parent problem, Natural Gate 0, IPCG, route, evidence snapshot, native
contract and prospective analysis. An absent ledger stays absent; this bridge
does not create a simulated successful research state.

`build_scientific_plan(...)` returns the actual vendor `make_plan` result with
`purpose="scientific"`. Required keywords are:

```python
run_id, command, code_refs, input_refs, output_paths,
protocol_ref, verification_ref, seed, group, arm_role, provenance, seconds
```

Optional keywords are `replay_context`, `evidence_mode`, and `trial_id` (default
`job`). The **complete current design batch** goes in `verification_ref`, not a
saved checker report. The vendor reruns method verification, checks selection and
the exact frozen protocol reference, and verifies the implementation refs in
every job. The project bridge additionally checks the current canonical frozen
protocol, current candidate and parent/Gate 0/IPCG bindings.

Candidate protocols supply `suite_qualification_evidence` keyed by every required
group, with `{protocol_ref, manifest_ref}` for real qualification runs. These
must have the same native sample/test/scorer/sampling identity and matching
baseline/control implementations. `suite_qualification_provenance[group]` freezes
the matching `model_revision`, `environment_digest`, and `data_revision`; the
qualification manifest and current job must match them. Additional declared
provenance keys must also match the qualification manifest. The bridge invokes the vendor's live native
replay and derives qualification from actual recomputed metrics. No static
receipt flag or callback that merely reads saved metrics is acceptable. The
host's `replay_context` must execute the exact nonce-bound scorer request under
the outer harness's resource/process ownership. Keep replay execution refs and
charge their wall time to the same original budget.

`validate_scientific_plan(root, plan, replay_context=...)` repeats these checks
immediately before dispatch. Plan preparation does not grant later execution
authority or permit a stale plan to bypass changed evidence. Commands use native
Python/Conda argv; container and shell-wrapper launches are rejected.

`validate_candidate_binding(root, protocol_ref, verification_ref)` performs only
the read-only current canonical/method check before reserving capacity or paying
for scorer replay. Its returned digests do not replace live qualification or
grant dispatch authority.

## 3. Baseline/evaluator qualification without circular candidate approval

A narrowly bounded initial qualification protocol may omit `method_discovery`
only when it instead declares:

```python
bindings["qualification"] = {
    "purpose": "baseline-calibration",  # or native-evaluator-qualification
    "allowed_arm_roles": exact_native_contract_roles,
}
```

Every arm in every such contract must be an existing `initial`, `hard`,
`full_soft`, or `native_reference` arm. Candidate names are rejected even when
the caller labels their roles “baseline.” Such plans still use native scientific
contracts and exact pinned inputs, but their execution mode is developmental and
their provenance scope is qualification. They cannot make a candidate verdict.
The complete protocol still contains treatment/baseline/control roles as required
by the vendor native contract. Call `build_scientific_plan` with
`verification_ref=None` for these qualification plans.

This distinction is not a generic engineering flag to launch GPU experiments.
Code refs, arm IDs, exact commands and the live host's existing user authorization
remain binding. Qualifying controls or changing their scientific behavior uses
the applicable current method evidence; renaming a candidate is not allowed.

## 4. Original finite resource authorization

```python
initialize_budget(path, original_start=actual_aware_start,
                  cap_seconds=28800, already_used_seconds=actual_prior_use)
```

The original start is mandatory and timezone-aware. The maximum cumulative cap
is eight hours. Resume preserves `original_start`, `absolute_end`, prior charges
and all outstanding reservations. A changed start/cap, malformed number, duplicate
charge or fabricated extension is rejected. `remaining_budget(path)` returns
the minimum of absolute time left and cumulative time left, less outstanding
reservations. It is unreserved capacity, not an extra budget for already-reserved
jobs. The queue must also respect the absolute end when executing reserved jobs.

Initial calibration can be bounded before candidate costs are measured:

```python
reserve_calibration(path, calibration_id=stable_id, bound_seconds=explicit_bound,
                    arm_ids=["initial", "hard", "full_soft"])
```

This accepts only the fixed qualification arm allowlist. Candidate bundles
instead call:

```python
reserve_bundle(root, path, bundle_id=stable_id, cell_ids=complete_job_inventory,
               calibration_ref=real_host_calibration_ref,
               expected_bindings=current_environment_model_code_data)
```

The calibration record is `version="suite-resource-calibration-v1"` with:

| Field | Contract |
|---|---|
| `bindings` | Exact `environment_digest`, `model_revision`, `code_digest`, `benchmark_manifest_ref`; other caller bindings also match exactly |
| `host_ref` | Actual one-RTX-2080-Ti inventory: `gpu_count=1`, `gpu_model`, aware `observed_at`, raw `evidence_refs` |
| `costs[cell_id]` | `upper_seconds` and real retained native `receipt_ref` for every job in the complete bundle |

The cost receipt must be completed native scientific calibration with matching
environment/model, retained logs and current implementation refs.
`code_digest` is the vendor canonical hash of the actual attempt's `code_refs`.
Bounds below observed wall time are rejected. The host must establish a
conservative cost model for every different workload, including extra model
forwards, scoring/replays and saving; assigning a large number to an unrelated
receipt is not qualification. Measurements are empirical bounds, not a guarantee
that every future attempt will finish. The harness enforces the finite timeout.

The entire comparison is reserved before candidate admission; no partial set of
cheap controls makes an incomplete comparison ready. The returned reservation
includes `cell_seconds` and `charged_cell_ids`. For each retained child receipt:

```python
charge_budget(root, path, reservation_id=bundle_id, cell_id=job_id,
              receipt_ref=actual_child_receipt_ref, final=is_last_cell)
```

Use `final=False` until the last job. Each call charges actual elapsed time,
including failures, and releases only that cell's bound. Pending jobs stay
reserved. Premature finalization fails atomically. Duplicate receipt submission
is idempotent; a second receipt for the same completed cell is rejected.
Missing receipts never release reserved work. Resume of the same bundle returns
its remaining reservation without extending the original authorization.

CPU preparation and live native replay also consume the same authorization:

```python
reserve_preparation(path, preparation_id=stable_id, bound_seconds=explicit_bound,
                    purpose="engineering-preparation")  # or native-reference-replay
```

Their receipts must be actual engineering plans with matching
`provenance.admission_scope` and `model_revision="no-model-workload"`. This is
accounting for a finite CPU envelope, not candidate or GPU authorization. The
queue must include the full declared replay overhead in complete-bundle capacity
checks. Baseline calibration and preparation reservations use a single final
receipt and omit `cell_id`.

## 5. Validation and remaining host obligations

All executable verification uses the project harness:

```bash
python scripts/research.py check --label admission-check --seconds 180 -- tests/test_suite_admission.py
```

Tests are explicit temporary engineering metadata fixtures. They execute no
benchmark, generated solution, or GPU workload and provide no research proof.
They cover immutable native bindings, denominator/selection integrity, stale
sources, narrow qualification scope, missing method evidence, original deadline
preservation, complete cost admission and incremental failure accounting.

Real host inventory, baseline/scorer qualification, current complete method
reviews, the actual canonical research ledger, and appropriate native execution
isolation remain runtime inputs. Missing inputs produce named `AdmissionError`
codes and preserve pending work. They are never filled by a successful CPU test.
