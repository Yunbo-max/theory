# Observed authoring-host native checks

These are retained **CPU scorer checks**, not training results or target-GPU qualification.

| Check | Observed scope | Outcome |
|---|---|---|
| Published assets | HE+ 164 (32/132), MBPP+ 378, LCB release_v5 880, clean training32 | Complete source inventories and fixed revisions checked |
| HumanEval+ | Official canonical solutions for HumanEval/0 and /1, all base+plus tests | Both passed |
| MBPP+ | Official canonical solutions for Mbpp/2 and /3, all base+plus tests | Both passed |
| LiveCodeBench | Published Qwen2.5-Coder-7B outputs,10 draws each for functional2728 and stdin1873_A | **Not qualified: host infrastructure failure** |

The LCB official runner's multiprocessing Manager could not create its local
socket (`PermissionError: EPERM`). Its20 returned `TestRunnerError`/`EOFError`
metadata entries occurred before solution grading. The all-failure values in
the archived official output are invalid infrastructure outcomes and must never
be used as model accuracy. The outer three-scorer job is honestly **failed** even
though HE/MB subchecks completed. Its receipt and logs are preserved.

The adapter has since been repaired to reject this official infrastructure status
before accepting either task or aggregate caches, while retaining the exact
native task return. No official evaluator code, grade, test, reference score or
host permission was changed. Current software verification covers that repair;
LCB still requires the same reference replay on a native host permitting its IPC.

The published LCB reference grades are7/10 for2728 and6/10 for1873_A. Those are
the upstream model's released grades, **not scores produced by this project**.
All10 reference outputs per task were retained, including upstream failures.

`retention-manifest.json` maps the original file path and SHA-256 to each copied
repository path. Copies preserve original bytes, including absolute runtime
paths in native receipts; no path rewriting or retrospective success flag was
applied. The `executed-scorer-source` directory is the exact source snapshot used
for these observed checks, before the infrastructure-rejection repair. Its files
are evidence, not the current importable package. Large released datasets remain
at their pinned upstream sources; manifests retain their revisions, sizes and
hashes. `collection-receipt.json` is the actual bounded harness collection result.

Asset acquisition also retained adverse attempts: the first full-memory staging
was interrupted at high memory/disk use; a concurrent source edit invalidated a
later staging hash; another staged large-file copy was shorter than its verified
original. The successful asset check used the original explicitly pinned source
files under the same bounded harness and wrote only metadata/index outputs.
Original raw hashes were not weakened or replaced. These authoring-host incidents
do not certify the target host's copying capacity or throughput.
