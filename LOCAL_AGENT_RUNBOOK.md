# Local Codex: operate Recursive SSD over SSH

This is the project-specific entry for the Local agent. **Web delivery status:
generated_unexecuted.** The user requested code and complete experiment design;
Local performs testing and experimental execution. No remote GPU job is claimed.

Read [AGENTS.md](AGENTS.md) and the [round handoff](rounds/2026-10-07-local-handoff/WEB_HANDOFF.md)
at the delivered commit. Commands below are source-inspected instructions for
Local, not executed receipts. Restore the real host facts and authorization before
using them. The original proposed window has elapsed; these instructions do not
authorize replacing its start with today's time.

## Read first

| Item | Exact source |
|---|---|
| Source version | Resolve the delivered `main` commit from the Web receipt; record actual `git rev-parse HEAD` on controller and GPU host |
| Math, ranking and implementation | [selection](research/selection.json), [math cards](research/cards), [selected implementation batch](research/suite-v3/implementation-batch.json), [mapping cards](research/suite-v3/implementations) |
| Full design | [G01 coverage](research/suite-v3/G01_COVERAGE.md), [15-method matrix](research/design-v2/METHOD_MATRIX.md), [development catalog](research/suite-v3/catalogs/development.json), [tuning catalog](research/suite-v3/catalogs/tuning.json) |
| Canonical admission | [SCIENTIFIC_ADMISSION](docs/SCIENTIFIC_ADMISSION.md), [input contracts/templates](docs/LOCAL_INPUT_CONTRACTS.md) |
| Commands and implementation | [research.py](scripts/research.py), [suite_queue](recursive_ssd/suite_queue.py), [admission](recursive_ssd/suite_admission.py), [runtime pins](vendor/research_autopilot/VENDOR.json) |
| Status and history | [STATUS](research/STATUS.json), [current checkpoint](research/suite-v3/workflow-checkpoint.json), [handoff checkpoint](rounds/2026-10-07-local-handoff/workflow-checkpoint.json), [historical suite checkpoint](research/suite-v3/history/workflow-checkpoint-2026-10-06.json) |

The retained 332 CPU tests at source `f12e6b3a976d00d99fedff9c23b63c39e6f352d2`
are historical engineering evidence. They do not test the new handoff repairs,
qualify the GPU host or close the scientific design/result gates. Reuse unchanged
evidence only at its actual source scope; recalculate affected code bindings after
Local acceptance. Do not overwrite old cards with invented successful tests.

## Hosts and paths

| Location / fact | Source and Local action |
|---|---|
| Controller computer | Existing Local Codex session and authenticated delivery checkout; record absolute `SSD_LOCAL_CHECKOUT` |
| SSH target | Restore the user's existing authorized alias into `SSD_SSH_TARGET`; currently unknown to Web. Inspect the named target's config and verify access; do not guess a host |
| Remote project | Choose/restore an isolated absolute `SSD_REMOTE_PROJECT`; preserve running checkouts. Record it with remote hostname and boot ID |
| Runtime | Inside that checkout: `vendor/research_autopilot/scripts`; verify pinned bytes through `harness.runtime()` in Local acceptance |
| Bootstrap Python / Conda | Inventory existing Python 3.11/3.12 and absolute Conda executable; record `SSD_CONDA_BIN` and an external-to-checkout `SSD_CONDA_PREFIX`. Do not use a notebook/laptop Python by accident |
| Model/data paths | Project-relative `artifacts/multibench-v3`; separate `artifacts/multibench-v3-05b` for 0.5B. No symlink inputs: the queue rejects them |
| Hardware | One user-run RTX 2080 Ti, approximately 11 GiB was requested; UUID, actual free VRAM, host RAM/CPU/disk, availability and contention must be observed |
| Budget | Original one-card cumulative 28,800 seconds, planned `2026-10-06T16:00:00+01:00`; actual start/consumption unobserved. Read the real ledger; no automatic extension |

The controller handles source delivery and SSH. The remote host owns Conda,
harness processes, CPU/GPU workloads and receipts. The remote host does **not**
need Codex or GitHub credentials. Native Python/Conda only; no containers.

### Stage and inspect exact source

On the controller, fetch the authorized repository without changing a dirty/live
checkout. Set `SSD_DELIVERED_COMMIT` from the actual delivery receipt. If starting
fresh, cloning `https://github.com/Yunbo-max/theory.git` is the real source URL.

```bash
git -C "$SSD_LOCAL_CHECKOUT" fetch origin main
git -C "$SSD_LOCAL_CHECKOUT" show "$SSD_DELIVERED_COMMIT:AGENTS.md"
git -C "$SSD_LOCAL_CHECKOUT" show "$SSD_DELIVERED_COMMIT:LOCAL_AGENT_RUNBOOK.md"
git -C "$SSD_LOCAL_CHECKOUT" show "$SSD_DELIVERED_COMMIT:rounds/2026-10-07-local-handoff/WEB_HANDOFF.md"
ssh "$SSD_SSH_TARGET"
```

All following command cards execute in that authorized remote shell, with the
remote path variables restored there. Do not assume controller shell variables
were forwarded over SSH. If the GPU host can read public GitHub:

```bash
set -eu
: "${SSD_REMOTE_PROJECT:?Restore the actual isolated remote checkout path}"
: "${SSD_DELIVERED_COMMIT:?Use the exact delivered commit}"
test ! -e "$SSD_REMOTE_PROJECT"
git clone --no-checkout https://github.com/Yunbo-max/theory.git "$SSD_REMOTE_PROJECT"
git -C "$SSD_REMOTE_PROJECT" checkout --detach "$SSD_DELIVERED_COMMIT"
cd "$SSD_REMOTE_PROJECT"
git rev-parse HEAD
git status --short
hostname
cat /proc/sys/kernel/random/boot_id
command -v python3
command -v conda
df -h .
python3 vendor/research_autopilot/scripts/run_harness.py --inspect-host
```

Only run the clone block for a confirmed nonexistent destination; a failed
`test` is a stop, not permission to continue overwriting. If GitHub is unreachable
from that host, transfer a Git bundle from the controller: `git bundle create
source.bundle origin/main`, `scp` it through the existing SSH target, then use
`git clone --no-checkout /actual/transferred/source.bundle "$SSD_REMOTE_PROJECT"`
and the same detached checkout. These are controller transfer operations.
Verify the exact commit exists in the bundle first; do not transmit credentials.

Record observed host/path/inventory facts in Local's unique return packet, not in
the immutable Web draft. Runtime `--inspect-host` is inventory only. It neither
allocates the device nor measures training throughput.

## Download datasets and models

Read the complete [asset cards and acquisition commands](docs/LOCAL_ASSETS.md).
They cover both pinned Qwen models, clean-v2 training prompts, HumanEval+, MBPP+,
all release_v5 LiveCodeBench inputs and published reference outputs, the official
scorers, Apple SSD source/templates and the six GKD-inspired comparator settings.

Run the asset/model commands through `scripts/research.py` on the remote host
after Conda setup, always with the real `--queue`. Check promotion receipts and
model shard/tokenizer loading under the harness. The current acquisition workers
use isolated workspaces; no controller `--offline` or general cached-resume flag
is promised. Do not omit files, silently upgrade revisions or overwrite a
different existing manifest. Use a separately reviewed destination for changed
assets and retain partial attempts.

## Ordered command cards

### 1. Restore the original authority and budget

Before any executable preparation, read the saved budget/authorization and user
allocation. If it is expired, hold executable work and report the actual boundary;
do not omit `--queue` to get an uncharged alternative. If existing authority is
valid, set `SSD_ORIGINAL_START` to its actual original timestamp. On **first
ledger creation only**, set `SSD_INITIAL_USED_SECONDS` to observed carry-in usage
from before that ledger. When reusing an existing ledger, restore its original
`initial_used_seconds` or omit the flag: do not pass the now-larger cumulative
usage again, which would duplicate charges or trigger `INITIAL_USAGE_CHANGED`.
The first-creation card on the remote host is:

```bash
cd "$SSD_REMOTE_PROJECT"
: "${SSD_ORIGINAL_START:?Restore the real original authorization}"
: "${SSD_INITIAL_USED_SECONDS:?Restore initial carry-in usage before this ledger}"
python3 scripts/research.py budget --queue runs/multibench-v3 \
  --original-start "$SSD_ORIGINAL_START" \
  --already-used-seconds "$SSD_INITIAL_USED_SECONDS"
```

Use `--budget-from` for later queues. Never create a fresh start to bypass expiry.
`budget` is a metadata controller; later preparation tasks reserve and charge
the same ledger before executing. A new authoring delivery grants no compute.

### 2. Native Conda bootstrap and dependency setup

Reuse a compatible existing native environment after inspecting its identity.
`setup` does not create Conda; `_conda()` requires its Python interpreter and
`CONDA_PREFIX` to agree. The following is a **Local-only controller card** for an
absent project environment, using the inspected internal engineering adapter so
the Conda command itself remains a budgeted, CPU-only harness job. Conda must
already be installed; a missing Conda installation is a named host prerequisite.

```bash
export SSD_CONDA_BIN SSD_CONDA_PREFIX
python3 - <<'PY'
import json, os, uuid
from pathlib import Path
from recursive_ssd import suite_queue as Q
from recursive_ssd.harness import ROOT
conda = Path(os.environ['SSD_CONDA_BIN']).resolve(strict=True)
prefix = Path(os.environ['SSD_CONDA_PREFIX'])
if not prefix.is_absolute() or prefix.exists() or prefix.resolve().is_relative_to(ROOT):
    raise SystemExit('Use an absent absolute Conda prefix outside the project; inspect existing environments instead')
folder = ROOT / 'runs/controller' / ('conda-bootstrap-' + uuid.uuid4().hex[:12])
result = Q._engineering(ROOT, folder,
    [str(conda), 'create', '--prefix', str(prefix), 'python=3.11', '-y'],
    seconds=600, scope='engineering-preparation',
    budget_path=Q.authorization_path(ROOT, 'runs/multibench-v3'))
print(json.dumps(result, indent=2))
raise SystemExit(0 if result['status'] == 'completed' else 1)
PY
```

Retain the actual bootstrap receipt and check the created interpreter exists.
The environment is an external mutable output: the receipt proves only the
command result, and the following environment/dependency checks still matter.
With the observed Conda shell initialization, activate the actual prefix:

```bash
conda activate "$SSD_CONDA_PREFIX"
python scripts/research.py setup --queue runs/multibench-v3 --seconds 1800
python scripts/research.py pipcheck --queue runs/multibench-v3
```

Setup installs `torch==2.6.0` from the CUDA 11.8 wheel index, then `.[test,evaluation]`
from [pyproject.toml](pyproject.toml), then checks dependencies. Each install is
a separate harness task: `--seconds 1800` is a per-command maximum, not a total
setup allowance. Actual charged costs and remaining wall deadline control entry.
Driver/CUDA incompatibility is a host problem to diagnose, not permission for a
blind package upgrade. The 2080 Ti workload uses FP16, not BF16/FlashAttention.

### 3. Acquire inputs and run Local software acceptance

```bash
python scripts/research.py assets --queue runs/multibench-v3 \
  --data artifacts/multibench-v3 --seconds 1800
python scripts/research.py model --queue runs/multibench-v3 \
  --data artifacts/multibench-v3 --seconds 1800
python scripts/research.py check --queue runs/multibench-v3 --seconds 600
```

The full check covers numerical targets, causal shifting/masking, teacher detach,
recursive checkpoint lineage, controls, queue/frozen identities, native adapters,
statistics and budget behavior. Local must inspect semantic failures, not only
the exit code. Focused rechecks use the real remaining-arguments interface:

```bash
python scripts/research.py check --queue runs/multibench-v3 --seconds 300 -- \
  tests/test_suite_admission.py tests/test_suite_queue.py
```

Additional repair checks are listed in the [round handoff](rounds/2026-10-07-local-handoff/WEB_HANDOFF.md).
Queued software checks retain the research fixtures through `project_refs`;
normal scientific source capture must also include the clean-v2 source and method
specifications used inside isolated attempts. Local's staged-input tests check
this closure, so a successful unstaged import alone is insufficient.
No expected pass count is supplied for the new revision. Record actual tested
code/environment hashes and logs, then review the derivation-to-code mapping.

### 4. Native scorer reference qualification

```bash
python scripts/research.py qualify --queue runs/multibench-v3 \
  --data artifacts/multibench-v3 --benchmark humaneval --seconds 600
python scripts/research.py qualify --queue runs/multibench-v3 \
  --data artifacts/multibench-v3 --benchmark mbpp --seconds 600
python scripts/research.py qualify --queue runs/multibench-v3 \
  --data artifacts/multibench-v3 --benchmark livecodebench --seconds 600
```

Actual inputs are the retained official HumanEval/0 and /1, Mbpp/2 and /3, and
published LCB reference outputs for 2728 (functional) and 1873_A (stdin), with
their native tests. These checks qualify interfaces, not model performance or
the complete benchmark. LCB previously failed before solution execution because
the authoring host forbade `multiprocessing.Manager` AF_UNIX IPC. Verify the
real remote outcome; `TestRunnerError` / `error_code=-5` remains infrastructure
failure and must not be cached as a model score.

### 5. Compile inventories, construct real native inputs and measure costs

```bash
python scripts/research.py design --stage tuning --output runs/tuning-suite.json
python scripts/research.py design --stage development --output runs/development-preview.json
python scripts/research.py build --suite runs/tuning-suite.json \
  --data artifacts/multibench-v3 --queue runs/tuning \
  --budget-from runs/multibench-v3 --original-start "$SSD_ORIGINAL_START"
python scripts/research.py status --queue runs/tuning
```

Compilation/building is not dispatch. Follow [LOCAL_INPUT_CONTRACTS](docs/LOCAL_INPUT_CONTRACTS.md)
to derive real `native-assets-input.json`, `protocol-input.json`, protocol output
and bundle bindings. The `research/templates/*.jsonc` files contain unknowns and
comments by design; do not pass them directly to `--spec`/`--bindings`.

`development-preview.json` is an inspection draft, not a queued tuning result.
Do not overwrite any suite already bound by a queue. Complete and select the
separately admitted tuning queue before constructing the tuned development suite.
The M03 development template requires a corresponding tuning-specific protocol
and equal `tuning_budget` for the M03/M01 tuning bundles; it cannot be relabeled
as a ready tuning freeze.

The real model baseline qualification interfaces are `qualify --job JOB_JSON
--bindings BINDINGS_JSON --queue runs/multibench-v3 --data artifacts/multibench-v3`
and `preflight --bindings BINDINGS_JSON --queue runs/multibench-v3 --data
artifacts/multibench-v3 [--config CONFIG_JSON]`. Each needs a reviewed native
qualification protocol and real inputs. There is no `calibrate` CLI. Matching
calibrations are explicit `derive-calibration` dependency nodes in the queue.

Read [comparator qualification](docs/COMPARATOR_QUALIFICATION.md) and
[candidate calibration children](docs/CALIBRATION_CHILD_PROTOCOL.md) before
constructing these inputs. Ordinary controls use exact catalog node requests;
selected-method aliases and candidate-derived calibration retain their own
independently reviewed, frozen child design. The child declares every ancestor,
its finite wall bound and native role. The same controller reserves its complete
inventory and native replays against the original budget, and returns real
`dependency-record.json` files for later nodes. Use the nine annotated input
templates and field sources in the input guide; no hand-written checkpoint or
`completed` flag can substitute for returned evidence.

For example, after constructing the real JSON files and recording the actual
prospective node bound, both scoped routes use this existing command:

```bash
python scripts/research.py qualify --job research/qualification-node-request.json \
  --bindings research/qualification-bindings.json --queue runs/multibench-v3 \
  --data artifacts/multibench-v3 --seconds "$QUALIFICATION_SECONDS"
```

The request contains only `node_id`; bindings identify either the ordinary
comparator protocol or the independently verified candidate child. For a child,
`QUALIFICATION_SECONDS` must equal that node's frozen bound. Unknown peak VRAM
may be measured by this bounded, exclusive one-GPU child; it does not waive the
measured profile required to admit a complete main bundle. Retain OOM and
timeout outcomes and their charges.

Default preflight measures hard/full-soft at a 512+384-token shape. It does not
measure the full 4096+2000-token LCB workload or every candidate/control. Obtain
actual per-node upper bounds, exclusive GPU memory profiles, matching-control
calibrations, live replay allowance and finalization reserve for the complete
bundle. Do not extrapolate a short probe into all-benchmark acceptance.

### 6. Review, freeze and admit the complete design

Restore the real canonical ledger, Parent, Natural Gate 0, IPCG, current method
implementation review, native model/comparator qualification and actual resource
evidence. [SCIENTIFIC_ADMISSION](docs/SCIENTIFIC_ADMISSION.md) and the input guide
define the exact bindings. The complete design draft exists; these future
evidence packets must be produced by Local, not invented by Web.

```bash
python scripts/research.py native-assets --spec research/native-assets-input.json
python scripts/research.py protocol --spec research/protocol-input.json
python vendor/research_autopilot/scripts/verify_methods.py --root . \
  --batch research/suite-v3/implementation-batch.json --candidate M03 --before experiment-design
```

The last command checks the current batch; a nonzero result requires actual
current Local code evidence and a versioned batch. Do not edit historical passed
logs to make it green. Add complete design records in a new batch linked to the
actual frozen protocol and feed that batch to admission. The freeze/dispatch
sequence and finite replay allowance are specified in [RUN_MULTIBENCH](docs/RUN_MULTIBENCH.md).
No static template or nominal `approved` field closes these gates.

### 7. Dispatch, inspect and resume

Only after current scientific admission succeeds, on the remote host:

First complete the actual tuning queue with its tuning-specific binding files,
including both M03 and M01 bundles and their matched simple-control trials. Run
`report --queue runs/tuning` and `select --queue runs/tuning --output
runs/tuning/selection.json` within its remaining authority. Then generate the
real development suite, leaving the preview untouched:

```bash
python scripts/research.py design --stage development \
  --tuning-selection runs/tuning/selection.json --output runs/development-tuned-suite.json
python scripts/research.py build --suite runs/development-tuned-suite.json \
  --data artifacts/multibench-v3 --queue runs/multibench-v3 \
  --original-start "$SSD_ORIGINAL_START"
```

Here `runs/multibench-v3` previously held the original budget pointer only, not
another immutable queue. If it already contains a different queue, use a new
development directory and `--budget-from runs/multibench-v3`; update every
subsequent command's queue path consistently. Freeze/admit the exact tuned
development identity before the following example:

```bash
python scripts/research.py admit --queue runs/multibench-v3 --bundle M03 \
  --bindings research/m03-dispatch-bindings.json
python scripts/research.py run --queue runs/multibench-v3 --bundle M03
python scripts/research.py status --queue runs/multibench-v3
python scripts/research.py resume --queue runs/multibench-v3 --bundle M03
```

Use an already available authorized `tmux` session if persistence across SSH
disconnects is needed; keep this controller and all native children foreground
inside their execution owner. Do not also start raw train/eval commands.

`status` reports retained state, not a fresh liveness guarantee. Reconnect to
the same host and inspect `runs/harness/<actual-batch>/state.json`, remote boot
ID and `/proc` process identities before resuming. A lost acknowledgment is not
proof that a job stopped. Never delete locks, invent a replacement run ID or
reset the ledger. `resume --retry-failed` permits one explicitly budgeted
development retry per node; confirmation retries are rejected.

### 8. Complete stages and collect before the hard deadline

For actual full tuning results, `select` produces the tuning selection consumed
by `design --stage development --tuning-selection PATH`. For actual complete
development results, `select` produces at most two finalists. Then use `design
--stage confirmation --selection PATH`; boundary/model_boundary use the same
retained development selection. Build each queue with `--budget-from
runs/multibench-v3` and the same original start. The smaller-model stage uses
the separate asset/model directory described in the asset cards.

```bash
python scripts/research.py report --queue runs/multibench-v3 --seconds 600
python scripts/research.py select --queue runs/multibench-v3 \
  --output runs/multibench-v3/finalists.json --seconds 600
python scripts/research.py collect --queue runs/multibench-v3 \
  --output returns/multibench-v3.tar.gz --seconds 600
```

Report/select/collect are CPU harness workloads and must fit the original
remaining budget. The finalization reserve is headroom, not permission to
compute after expiry. Selection rejects incomplete native results. Collection
does not silently include model weights; `--include-checkpoints` is explicit
and may require substantially more disk/time. If the deadline is already past,
retrieve retained files over SSH for review without launching fresh analysis or
relabeling a partial queue as complete.

## Method and benchmark coverage

The [15-row method matrix](research/design-v2/METHOD_MATRIX.md) enumerates every
selected method, its strongest simple alternatives and decisive ablations. Each
row maps to the same ID in [implementation cards](research/suite-v3/implementations),
[suite registry](recursive_ssd/suite_design.py), [training controls](recursive_ssd/suite_train.py)
and [target construction](recursive_ssd/methods.py). Do not infer comparison
coverage from a single `--bundle M03` example; execute all eligible required
bundles within actual capacity and retain every unfinished one.

| Stage / benchmark | Complete obligation and native output |
|---|---|
| Development | 15 candidate bundles; 64 trajectories, 190 evaluation cells, 203 comparisons; seed 17, R1/R2/R3, HE development32 |
| Tuning | 12 scalar trials plus initial evaluation; candidate/simple-control equal bounded time; three values per family |
| Confirmation | At most two real selected finalists and all required arms; seeds 23/47/71/101/131; fixed R3 endpoint |
| HumanEval+ | v0.1.10, 164 IDs split32/132; n10, T0.8, top-p0.95, maxnew384; native base+plus pass@1/pass@10 |
| MBPP+ | v0.2.0, all378; same generation settings; native base+plus pass@1/pass@10 |
| LCB | release_v5 all880; n10, T0.2, top-p0.95, maxprompt4096/maxnew2000; official pass@1/pass@5 |
| Boundaries | Separate R5 and Qwen0.5B queues and identities; descriptive boundaries, not substitutes for R3 confirmation |

[benchmarks.py](recursive_ssd/benchmarks.py) owns original task identities;
[suite_score.py](recursive_ssd/suite_score.py) and [lcb_score.py](recursive_ssd/lcb_score.py)
call native scorers; [analysis.py](recursive_ssd/analysis.py) retains paired
task×seed intervals and the full comparison family. Native generations are
`raw.jsonl`; scoring outputs include `metrics.json`, native receipts and official
per-task results. Locate them through actual attempt references. A failed cell
remains in the planned denominator. No arbitrary proxy benchmark is authorized.

## Debug routing

| Observation | Inspect actual evidence / cause | Scoped repair and recheck |
|---|---|---|
| ACTIVE_NATIVE_CONDA_REQUIRED / import / CUDA | Interpreter and CONDA_PREFIX; attempt stderr; pyproject pins; setup receipt | Restore the observed native environment. Re-run pipcheck and affected tests; no blind global upgrade or container workaround |
| Missing asset/shard or digest mismatch | Asset promotion receipt, manifests and [asset cards](docs/LOCAL_ASSETS.md); distinguish access, partial transfer, wrong revision and wrong loader path | Recover the same pinned input through admitted preparation; retain failed files. Changed model/data requires child bindings |
| OOM / foreign GPU / unknown profile | Actual host UUID/telemetry, execution-context, effective sequence lengths and `_resources`; include LCB's full shape | Resolve contention or profile the exact reviewed workload. Do not quietly halve length, precision, batch or model |
| NaN / loss-mask / teacher gradient / lineage | `suite_train.py`, `methods.py`, `train.py`; failing test input and checkpoint refs | Repair demonstrated cause; run affected semantic tests and full acceptance. Preserve changed-source evidence boundaries |
| LCB EPERM / TestRunnerError / -5 | Native attempt stderr, Manager IPC setup, `lcb_score.py` error handling | Qualify on the actual native host. Infrastructure failure stays unscored; do not count it as incorrect model output |
| Wrong sample IDs / score denominator | `benchmarks.py`, raw generations, scorer input map, official results and protocol | Reconcile released IDs and n10; replay unchanged native tests through harness; keep missing/failed units |
| Frozen protocol / qualification / code mismatch | `suite_admission.py`, actual referenced bytes, current canonical ledger and method batch | Rebuild only from real evidence and reviewed current code; [input guide](docs/LOCAL_INPUT_CONTRACTS.md) explains missing fields; no fake qualifiers |
| Budget expired / reservation failure | Original shared ledger, every prior attempt and finalization reserve | Stop new workloads; retrieve retained evidence. Do not change timestamps, omit queue or grant a new allowance |
| SSH disconnect / stale state | Same host boot ID, process start-time/PID tuple, controller/harness state and receipts | Reconcile first, then same identity resume; no duplicate queue or deleted locks |
| Mixed performance / no gain | Complete paired results, strongest simple baseline, actual parameters, matched costs and E04 | Use frozen dev-only tuning rules and preserve adverse results; confirmation is not a tuning set |
| Git push conflict / lost acknowledgment | Remote main parent, actual commit and content identities | Reconcile once with one writer, normal fast-forward/expected-head update; do not rerun experiments to fix a transfer |

For each actual repair retain observed version/input/error, competing causes,
evidence supporting the selected cause, exact diff, all attempts/costs and the
affected rechecks. The existing handoff has no target-host pass receipts.

## Report and return

Actual controller folders are `runs/controller/<label>-<generated-id>/`; resolve
their `native-plan.json`, `harness-plan.json`, `environment.json` and real receipt
refs. Harness state/logs are `runs/harness/<actual-batch>/`. Queue `queue.json`,
`state.json`, original budget pointer/ledger, `admissions`, reports and referenced
native attempts jointly describe completion. Do not fabricate an attempt ID.

On the controller retrieve the real collected archive with `scp`/`rsync` over
the same authorized SSH target, compare SHA256 with the remote archive, and keep
remote and local paths distinct. Read-only `sha256sum` or macOS `shasum -a 256`
can verify a transferred file. Preserve its original project-relative contents;
do not rewrite execution paths/hashes to make the copy look locally executed.

Create a unique `rounds/2026-10-07-local-handoff/runs/<actual-run-id>/` review
packet containing:

- actual executed commit/dirty patch, protocol/source/model/data/scorer/env
  hashes, remote hostname/boot/GPU identity, commands and times;
- every attempt/native receipt, stdout/stderr, raw scores, E04 scope and all
  missing, failed, adverse or invalid comparisons;
- cumulative budget and running/pending identities; measured costs/VRAM and
  exact next action; no binary scientific verdict from a partial inventory;
- an index to large immutable outputs plus observed hashes. Do not commit
  model checkpoints, raw private credentials or multi-GB caches to GitHub.

GitHub result destination remains `Yunbo-max/theory`, `main`, through one
authenticated Local integration writer. No new Hugging Face output repository
has been selected by this guide. Use existing authorized storage for large files;
input HF model IDs do not authorize output uploads. Commit scoped small evidence,
push normally, read back the exact commit and packet paths, then return that
locator. If transfer fails, retain outputs and report the specific failure;
never rerun a completed experiment because its upload failed.
