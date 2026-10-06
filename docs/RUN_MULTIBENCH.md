# Native multi-benchmark controller

`python scripts/research.py --help` is the project entry point. Executable work is
submitted to the pinned research-autopilot `run_harness.py`; the controller only
compiles immutable inputs, reconciles receipts, and advances dependency nodes.
Use an activated Conda environment on the authorized single RTX 2080 Ti host.
The authoring environment has no GPU and has not produced scientific results.

The complete catalog retains all 15 method bundles, their required baselines,
matched controls, six GKD conditions, three recursive rounds, and every planned
evaluation. Confirmation fixes five independent seeds and all three benchmarks;
the five-round and smaller-model boundaries have separate suite identities.
An incomplete bundle remains incomplete. Running out of time, losing a process,
or an OOM never removes a cell and never becomes a scientific KILL.

## Prepare and inspect

```bash
conda activate recursive-ssd
python scripts/research.py budget --queue runs/multibench-v3 \
  --original-start '2026-10-06T16:00:00+01:00'
python scripts/research.py setup --queue runs/multibench-v3 --seconds 1800
python scripts/research.py pipcheck --queue runs/multibench-v3
python scripts/research.py check --queue runs/multibench-v3 --seconds 300
python scripts/research.py assets --queue runs/multibench-v3 --data artifacts/multibench-v3 --seconds 1800
python scripts/research.py model --queue runs/multibench-v3 --data artifacts/multibench-v3 --seconds 1800
python scripts/research.py design --stage development --output runs/development-suite.json
```

Record the real original start before preparation. The example timestamp is
only the previously discussed intended start; use it only if it matches the
actual authorization. `bash scripts/setup.sh --queue runs/multibench-v3` runs
setup, checks, assets, and model preparation with that same budget.

Assets and model acquisition are bounded engineering preparations. They are not
baseline qualification. `qualify` checks released native reference outputs;
`preflight` measures real GPU behavior using an explicit baseline-only native
protocol and a bounded scientific plan. These commands require a reviewed
qualification binding file for scientific GPU qualification; released-reference
`qualify --benchmark ... --queue ...` needs only the prepared native assets.
They do not synthesize a native qualification
threshold, a parent problem, a Gate 0, an IPCG decision, or a design review.
No command invokes Docker or another container runtime.

## Original time authorization

`budget` records the **actual original authorized start**, not the time a new
checkout or queue directory was created. The hard limit is the earlier of the
original start plus eight hours and the remaining cumulative authorization.
Supply earlier consumed compute with `--already-used-seconds` when needed.
`build` binds the existing authorization; it preserves preparation charges.
Resuming never creates a fresh eight-hour window. Pass `--budget-from` naming
the earlier queue when building tuning, development, or confirmation in a new
directory. The same normalized original start and cap also share one ledger.

```bash
python scripts/research.py build --suite runs/development-suite.json \
  --data artifacts/multibench-v3 --queue runs/multibench-v3 \
  --original-start '2026-10-06T16:00:00+01:00'
python scripts/research.py status --queue runs/multibench-v3
```

The timestamp above is only the previously discussed intended start. Use it
only when it matches the real authorization. Build creates a finite catalog;
it does not authorize candidate dispatch.

## Scientific preparation, freeze, admission, and execution

`native-assets` converts the released manifest and source-backed native
definitions to the actual skill's sample/definition contracts.
`protocol` compiles supplied reviewed G01 fields and a current method batch.
`freeze` invokes the skill's canonical freeze transaction. See
`SCIENTIFIC_ADMISSION.md` for the native evidence contracts.

```bash
python scripts/research.py native-assets --spec research/native-assets-input.json
python scripts/research.py protocol --spec research/protocol-input.json
python scripts/research.py freeze --protocol research/frozen-m03.json \
  --queue runs/multibench-v3 --replay-calls "$NATIVE_REPLAY_CALLS"
python scripts/research.py admit --queue runs/multibench-v3 --bundle M03 \
  --bindings research/m03-dispatch-bindings.json
python scripts/research.py run --queue runs/multibench-v3 --bundle M03
python scripts/research.py resume --queue runs/multibench-v3 --bundle M03
```

Set `NATIVE_REPLAY_CALLS` to the exact reviewed qualification replay inventory:
sum the native arm roles over required groups, counting each faithful-harness
role twice. This is a bounded allowance, not a claim that qualification passed.

Admission requires the current canonical parent/Gate 0/IPCG/protocol state,
current complete method-verification batch, native scorer and comparator
qualification with live replay, and real host calibration for **every** node
in the complete comparison. A measured upper bound is required per training,
evaluation, and matching-calibration node, plus explicit live-replay overhead
and finalization reserve. Unknown costs leave the bundle blocked.

The controller runs one native child plan at a time through the shared host
pool. A child checkpoint reference is created only after the parent's real
attempt receipt and output hashes are verified. No future checkpoint hashes
are invented. Recursive training loads the preceding checkpoint; fixed-data
and lagged controls also bind the appropriate retained generation/checkpoint.
Per-update files remain available to a retry under the original deadline;
failed attempts and their cost remain in the record. Confirmation has no
automatic retries.

An explicit `resume --retry-failed` permits at most one development retry per
node. Its additional complete cost and live-replay allowance are reserved while
all unfinished descendants keep their original reservations. The original
deadline and all failed-attempt costs remain unchanged. Confirmation retries
are rejected. A normal resume rejoins only the exact retained harness batch.

The dispatch binding JSON uses `protocol_ref`, `verification_ref`,
`calibration_ref`, one `gpu_uuid`, `resources`, `groups` keyed by
`benchmark/split`, and `arm_roles` keyed by every catalog arm. Each reference is
`{"path": "project-relative-file", "sha256": "actual-file-digest"}`.
`resources` includes actual `gpu_peak_mib` and `memory_profile_ref`, plus finite
`cpu_cores`/`ram_mib`; sharing is always disabled. `replay_timeout_seconds` and
`replay_calls_per_node` provide a finite allowance for live native verification;
every replay slot is reserved in advance and charged from its actual receipt.
`finalization_reserve_seconds` defaults to 900. Matching `calibration` records
retain actual source references; unresolved controls cannot dispatch.

The frozen protocol must contain `suite_binding` equal to the suite digest,
benchmark and model manifest references, `code_digest` of the ordered code
reference list, environment digest, and `config_digest` of the exact binding
`config` (empty configuration is hashed explicitly), plus `calibration_digest`
of the effective suite/binding calibration table. A binding cannot replace a
suite-frozen calibration value; shared baselines cannot change configuration
or calibration between bundle admissions. During tuning this also
contains the exact `tuning_budget` object. A matching arm name alone cannot
authorize different hyperparameters, data, models, or code.

Tuning has two independently admitted candidate bundles, `M03` and `M01`.
All four candidate/control grids require the same frozen
`tuning_budget={"trial_wall_seconds": ..., "source_refs": [...]}`. No numeric
allowance is invented by the controller. Each training child is capped by the
remaining trial allowance; prior failed attempts count. Selection retains
actual spent seconds, the common allowance, and the budget digest, and rejects
overtime/incomplete trials. Apply the selected parameters with
`design --tuning-selection <selection.json>` for development. Confirmation and
both boundary stages automatically inherit the selected development suite's
tuning parameters, measured calibration, and global training configuration;
an explicit differing tuning override or admission configuration is rejected.
Head-temperature parameters are derived on development and frozen before
confirmation; confirmation and boundary jobs cannot retune them. Their origin
model/revision, seed, lineage, parent training output, calibration output, and
actual native derivation attempt are verified. R5 prospectively carries the
development R3 beta into R4 and R5. The smaller-model boundary retains that beta
as an explicitly labeled transfer setting, so it cannot claim a same-model
matched-head control or a formal confirmation verdict. A separately reviewed
smaller-model development protocol is needed for that different claim.

```bash
python scripts/research.py design --stage confirmation \
  --selection runs/multibench-v3/finalists.json --output runs/confirmation-suite.json
python scripts/research.py build --suite runs/confirmation-suite.json \
  --data artifacts/multibench-v3 --queue runs/confirmation \
  --budget-from runs/multibench-v3 --original-start '2026-10-06T16:00:00+01:00'
```

Preparation and metadata workloads read staged snapshots and publish files
only after the controller verifies their real harness receipts and hashes.
Native live replay is the explicit exception: the vendored nonce-bound API
requires the actual original-root argv/cwd. Its pinned scorer/input checks run
inside one bounded harness-owned CPU job, retain the truthful executed argv,
and preserve every replay output reference. No argv or scorer success is forged.

Changed source, training data, benchmark tests, model snapshot, interpreter,
packages, scorer, decoder, or frozen protocol reject resume. Use a reviewed
child protocol and a new queue identity for a changed experiment. An expired
original authorization still cannot be extended by changing directories.

## Report, select, and collect

```bash
python scripts/research.py report --queue runs/multibench-v3 --seconds 600
python scripts/research.py select --queue runs/multibench-v3 \
  --output runs/multibench-v3/finalists.json
python scripts/research.py collect --queue runs/multibench-v3 \
  --output returns/multibench-v3.tar.gz
```

The report includes the entire planned denominator, every missing/failed cell,
actual native success counts, paired task/seed estimates, the frozen full
comparison family, costs, and checkpoint lineage. Scientific decisions remain
INCONCLUSIVE until all required native and reviewed evidence exists.

Selection requires a complete native development report. It ranks candidates
by pass@10 gain over their strongest required simple baseline, requires no
more than 1 percentage point pass@1 degradation, and breaks ties by measured
training time then the retained method order. It selects at most two methods;
it produces no scientific PASS/KILL. Tuning selects matching hyperparameters
only from its complete separate development inventory. Confirmation design
requires the retained selection artifact and never selects the best round or
seed from confirmation outcomes.

Collection preserves catalog/budget/attempt records, native plans and logs,
raw generations, exact native outputs, source/model/data identities, and
checkpoint hash lineage. Weights can be added explicitly with
`--include-checkpoints`; they are never silently committed to GitHub.
