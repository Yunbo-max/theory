# Native comparison analysis

`recursive_ssd.analysis` is an analysis adapter for the retained G01 design. Its
software tests use small scalar/count engineering fixtures, never fabricated
benchmark tasks or evidence of model quality. A numerical `PASS` is a scoped
assessment conditional on the supplied reviewed evidence. It does not advance a
research gate, authenticate a reviewer, qualify a scorer, or establish a result.

## Cell contract

Every planned cell has these exact identity fields:

```python
{
    "cell_id": "stable-queue-cell-id",
    "arm_id": "M03", "model": "Qwen2.5-Coder-1.5B-Instruct",
    "benchmark": "humaneval", "split": "confirmation",
    "seed": 23, "round": 3,
    "task_ids": ["official/task/id", "another/official/id"],
    "expected_samples": 10,
    "identity_refs": {"scorer": "frozen-identity", "decoder": "frozen-identity"},
}
```

`expected_samples` can alternatively map every task ID to its fixed draw count.
Observations repeat all seven identity fields and `identity_refs`, and provide:

```python
{
    "status": "completed", "optimization_valid": True,
    "cost_matching_valid": True, "budget_valid": True,
    "training_seconds": 123.4, "optimization_seconds": 80.0,
    "per_task": {"official/task/id": {"n": 10, "correct": 3}, ...},
}
```

The EvalPlus adapter's `plus_correct` is accepted instead of `correct`. If both
exist they must agree. Scores are recomputed from integer success counts with
the existing official-formula `pass_at_k`; a stored average cannot replace its
denominator. Zero successes, timeouts classified as failed official samples,
failed cells, and missing cells remain visible. `training_seconds` is the
cumulative measured cost of producing the endpoint model. The suite's
`training_cost_summary` sums generation, proxy computation, optimization, saving,
and model loading across every preceding round, including charged failed work
and required extra forwards. In receipt fields this is
`sum(costs.total_round_seconds + costs.model_load_seconds)`; each
`total_round_seconds` is the disjoint sum of generation, proxy, optimization,
and saving. `optimization_seconds` separately reports cumulative optimization
time and does not replace the full cost denominator. Evaluation costs remain
separately reported in the reviewed cost record.

`validate_inventory(expected_cells, observations)` checks exact cell identity,
task sets, frozen draw counts, duplicate IDs and semantic keys, successful
optimization, finite timings and success counts, and expected identity references.
It returns explicit counts, eligible cells, every ineligible observation, missing
cells, and structured issues. Duplicates are never silently reduced to one row.
An unexpected row is retained as an ineligible observation. Explicit
`cost_matching_valid=False` or `budget_valid=False` makes a row ineligible and
also rejects direct paired/factorial analysis. The suite derives cost matching
from every preceding round that requested a matched clock; ordinary arms have
no such clock obligation. These checks do not infer scientific validity from
an `optimization_valid` flag.

## Estimates and multiplicity

`crossed_interval(candidate_cells, control_cells, metric, repeats=20000,
seed=20261006, alpha=.05)` requires exactly paired training seeds, tasks, draw
counts, model, benchmark and split. It accepts one endpoint round per arm.
Shared file identities are compared by verified SHA-256, so identical protocol
bytes can live at different per-cell paths. A stale cited file is rejected.

For a task metric, calculate official pass@k for each task/seed, average tasks
within each training seed, then average seeds. Each bootstrap draw independently
resamples the task IDs and training seeds with replacement. The same two index
vectors apply to both arms; every matched comparison/round with the same sorted
tasks/seeds and analysis seed uses the same deterministic index schedule. Decode
draws and rounds are never counted as independent training repetitions.

The return includes the paired difference, crossed interval, task-only and
seed-only intervals, per-seed means, raw draw/count summaries, and Monte Carlo
tail-resolution diagnostics. Pass metrics use absolute probabilities, so `.02`
is 2 percentage points. Percentage-point equivalents are also reported. The
`training_seconds_relative` endpoint is the ratio of seed-mean cumulative costs
defined above minus one; `-.10` means a 10% reduction. Its bootstrap resamples
training seeds only, with the same seed indices as quality metrics.

`factorial_interval(cells00, cells01, cells10, cells11, metric, ...)` uses the
same bootstrap indices for all four arms and estimates `11 - 10 - 01 + 00`.
It validates all four task/seed/scorer identities and the common endpoint round.
Its raw count report names all four arms. Interactions are in native pass@k
units; relative timing ratios do not use this additive factorial contract.

Intervals use the linear empirical percentile quantile. Fewer than two training
seeds or fewer than ten expected bootstrap draws in either requested tail is
explicitly flagged as insufficient for a verdict. The requested interval is
still returned for inspection. Before freezing the analysis, use
`recommended_repeats(family_size, min_tail_draws=20, alpha=.05)`. It returns
`max(20000, ceil(2 * family_size * min_tail_draws / alpha))`. Freeze that budget
before examining results and reserve its CPU cost. This resolves numerical tail
sampling, not limited task/seed precision. Temporary resampling arrays are
bounded in 256-draw chunks; three scalar distributions are retained for quantiles.
Changing resamples later is an analysis-protocol change to record, not permission
to change the family.

`comparison_report(expected_cells, observations, comparisons, *, family_size,
criteria, evidence_context, repeats=20000, seed=20261006, alpha=.05)` analyzes a
finite predeclared comparison inventory. `compare_report` is an equivalent name.

```python
{
    "comparison_id": "M03-vs-full-soft-HE",
    "candidate_arm": "M03", "control_arm": "full_soft",
    "model": "Qwen2.5-Coder-1.5B-Instruct",
    "benchmark": "humaneval", "split": "confirmation",
    "candidate_round": 3, "control_round": 3,
    "control_role": "baseline",
    "metrics": ["pass@1", "pass@10"],
}
```

The exact benchmark IDs are `humaneval`, `mbpp`, and `livecodebench`.
Initial-model comparisons use `control_role="initial_model"` and
`control_round=0`. HumanEval+/MBPP+ use pass@1/pass@10; LiveCodeBench uses
pass@1/pass@5. Benchmarks and models are analyzed separately. `family_size` is an
externally frozen positive integer covering the entire planned family, including
failed comparisons and any other predeclared models/benchmarks. It must cover at
least all endpoint slots requested in this call. Every interval uses
`alpha / family_size`; missing rows never reduce that divisor. Missing or invalid
comparisons retain named invalid estimate entries and full count accounting.
Development reporting works with `evidence_context={}` and remains inconclusive.

For a factorial comparison, supply `factorial_arms=[a00, a01, a10, a11]` and
`round=3`, replacing the two arm names and their round fields; keep the same
comparison ID, model/benchmark/split and metrics. Each interaction endpoint
occupies its own frozen-family slot. Reports include `factorial_cells` for all
four arms. Pairwise G01 thresholds do not automatically become interaction
thresholds; the distinguishing prediction must declare its direction and margin.

## Frozen rules and evidence

`g01_criteria(comparisons, method_id=..., self_improvement_claim=True)` translates
the retained G01 thresholds into explicit rules. Retention methods require
pass@1 improvement of 2pp over the initial model when claiming improvement,
coverage noninferiority of -1pp to the initial model, coverage improvement of 2pp
and pass@1 noninferiority of -1pp against every required simple comparator.
For M07, full-soft quality must be noninferior at -1pp on both endpoints and
training wall time must decrease by at least 10%. Distinctive mechanism controls
are bound through separately reviewed mechanism evidence, not by inventing a
universal task-effect threshold for all ablations.

`evaluate_criteria(estimates, criteria, evidence_context)` accepts rules of this
form, with thresholds fixed before confirmation:

```python
{"criterion_id": "M03-coverage", "estimate_id": "M03-vs-full-soft-HE:pass@10",
 "relation": "ge", "threshold": .02, "unit": "absolute_probability",
 "kind": "task"}
```

`relation="le"` is used for relative training time. For `ge`, the lower bound
must meet the threshold; an upper bound below it excludes the required effect.
The reverse applies to `le`. Point estimates never substitute for bounds.

Confirmation uses exactly seeds 23, 47, 71, 101 and 131, and candidate R3.
Development seed 17 cannot support confirmation. R5 boundary comparisons retain
their descriptive estimates and intervals, but remain `INCONCLUSIVE` under the
primary G01 R3 criteria; they cannot replace or extend an R3 verdict.
`evidence_context` supplies
`phase="confirmation"`, `expected_seeds`, `endpoint_round=3`, and cited reviewed
assessments in `prerequisites`. Required assessment keys are `protocol_freeze`,
`native_scoring`, `baseline_qualification`, `implementation_verification`,
`optimization_diagnostics`, `cost_accounting`, `independent_confirmation`, and
`e04_review`. Each assessment contains `assessment="supported"` and nonempty
`evidence_refs=[{"path": ..., "sha256": ...}]`. Paths must exist and match their
SHA-256. Naked booleans and missing/stale citations are not assessments. These
checks bind the cited files; they do not verify the scientific truth of a review.

`required_mechanisms` names the predeclared mechanism claims. Each entry under
`mechanisms` uses the same citation contract, with `assessment` equal to
`supported`, `contradicted` or `unresolved`. Missing mechanism data cannot support
a positive claim. Full valid evidence excluding a required task/resource effect
returns scoped `KILL`; supported task effects with a contradicted mechanism return
`REVISE`; all required effects and mechanisms supported return scoped `PASS`.
Otherwise the decision is `INCONCLUSIVE`. Incomplete inventories, invalid
intervals and missing scientific prerequisites take precedence over every
scientific decision. Every result retains `gate_advanced=False` and
`scientific_result_verified=False`.
