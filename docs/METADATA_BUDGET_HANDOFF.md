# Reporting and export under the original budget

Status: **generated_unexecuted**. This change was source-reviewed on the Web;
the new tests and the target-host behavior have not been executed. Historical
332-test evidence does not validate this change.

`scripts/research.py report`, `select`, and `collect` execute CPU work through
the pinned harness. They now reserve their `--seconds` bound on the queue's
original shared ledger before capturing input hashes, then give that exact
reservation to the existing harness adapter. Actual receipts, including failed
attempts, are charged by the adapter. The reservation precedes snapshotting
because the ledger itself is a retained input; changing it afterward would
invalidate the worker's input hashes.

Use the commands in `RUN_MULTIBENCH.md` while the existing authorization is
still active. `--seconds` is the upper bound for each metadata operation, so
budget report, selection, and collection separately. `budget_ref` in the
controller result points to the current ledger after that attempt. Reports and
archives describe their frozen input snapshot: by construction that snapshot
predates settlement of the export's own receipt. Return the current ledger and
the metadata controller/native/harness records together with the exported file
to retain that final charge.

`finalization_reserve_seconds` is an admission headroom requirement and a
training stop margin. Existing code does **not** make it a separate ledger
reservation, and it cannot authorize work after the absolute deadline. An
expired window therefore blocks new report, selection, and archive jobs. Do
not omit `--queue`, alter the start, create a fresh budget, call the internal
workers directly, or use an uncharged generic engineering plan to evade that
block. Local may still use ordinary `rsync`/`scp` file transfer to retrieve
already retained `runs/`, raw logs, queue state, and outputs; copying existing
files is not a new analysis or model run. Further computational analysis needs
an explicitly authorized next scope, preserving the original ledger.

Local semantic checks to run through the normal controller while authorized:

```bash
python scripts/research.py check --queue "$RESEARCH_QUEUE" --seconds 300 -- \
  tests/test_suite_queue.py -k metadata
```

The new fixtures check all three operations: shared ledger identity,
reservation-before-hashing order, unchanged original deadline, expiration
rejection, and rejection when unfinished work already reserves the remaining
capacity. They do not replace native target-host tests, official scorer replay,
or actual resource and scientific evidence.
