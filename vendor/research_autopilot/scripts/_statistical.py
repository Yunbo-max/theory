"""Analysis of native benchmark results, never a replacement task scorer.

Independent-unit paired means are the built-in design. Other estimands and
dependence structures require the frozen project's live analysis adapter.
"""
import math
import statistics
import warnings
import _autoresearch as C


def adjusted_confidence(confidence, family_size, policy):
    if not isinstance(family_size, int) or isinstance(family_size, bool) or family_size < 1:
        C.fail("INVALID_COMPARISON_FAMILY")
    if policy == "single_claim":
        if family_size != 1:
            C.fail("MULTIPLE_COMPARISON_POLICY_REQUIRED")
        return confidence
    if policy == "bonferroni":
        return 1.0 - (1.0 - confidence) / family_size
    if policy == "external":
        return confidence
    C.fail("UNSUPPORTED_MULTIPLE_COMPARISON_POLICY")


def analyze(values, rule, *, unit_ids, unit_policy, family_size=1,
            multiple_policy="single_claim", analysis_context=None, frozen_scope=None):
    C.finite(values)
    if not values or len(values) != len(unit_ids):
        C.fail("ANALYSIS_UNIT_COUNT_MISMATCH")
    if any(isinstance(v, bool) or not isinstance(v, (float, int)) for v in values):
        C.fail("ANALYSIS_NUMERIC_RESULTS_REQUIRED")
    if len({C.canonical(u) for u in unit_ids}) != len(unit_ids):
        C.fail("DUPLICATE_INDEPENDENT_UNIT")
    if not isinstance(unit_policy, dict) or not unit_policy.get("unit") or not unit_policy.get("independence_justification"):
        C.fail("INDEPENDENT_UNIT_POLICY_REQUIRED")
    method = rule["uncertainty"]
    if unit_policy.get("clustered") and method != "external":
        C.fail("CLUSTERED_DESIGN_REQUIRES_PROJECT_ANALYSIS")
    confidence = adjusted_confidence(rule["confidence"], family_size, multiple_policy)
    if multiple_policy == "external" and method != "external":
        C.fail("EXTERNAL_MULTIPLICITY_REQUIRES_PROJECT_ANALYSIS")
    mean = math.fsum(values) / len(values)
    result = {"n": len(values), "effect": mean, "lower": mean, "upper": mean,
              "method": method, "adjusted_confidence": confidence,
              "independent_unit": unit_policy["unit"], "family_size": family_size,
              "multiple_comparison_policy": multiple_policy}
    if method == "external":
        if not callable(analysis_context):
            C.fail("PROJECT_ANALYSIS_REPLAY_REQUIRED")
        if not frozen_scope or not frozen_scope.get("analysis_plan_digest"):
            C.fail("FROZEN_ANALYSIS_SCOPE_REQUIRED")
        request = {"values": list(values), "unit_ids": unit_ids, "unit_policy": unit_policy,
                   "rule": rule, "family_size": family_size, "multiple_policy": multiple_policy,
                   "scope": frozen_scope, "adjusted_confidence": confidence}
        request["input_digest"] = C.hashed({k: v for k, v in request.items() if k != "scope"})
        request["scope_digest"] = C.hashed(frozen_scope)
        try:
            answer = analysis_context(request)
        except C.Failure:
            raise
        except Exception:
            C.fail("PROJECT_ANALYSIS_REPLAY_UNAVAILABLE")
        if not isinstance(answer, dict) or answer.get("input_digest") != request["input_digest"] or answer.get("scope_digest") != request["scope_digest"]:
            C.fail("PROJECT_ANALYSIS_INPUT_MISMATCH")
        if answer.get("n") != len(values) or not answer.get("method") or not answer.get("software_version"):
            C.fail("PROJECT_ANALYSIS_PROVENANCE_REQUIRED")
        for key in ("effect", "lower", "upper"):
            if isinstance(answer.get(key), bool) or not isinstance(answer.get(key), (float, int)):
                C.fail("PROJECT_ANALYSIS_NUMERIC_RESULT_REQUIRED")
        if not answer["lower"] <= answer["effect"] <= answer["upper"]:
            C.fail("PROJECT_ANALYSIS_INTERVAL_INVALID")
        result.update({k: answer[k] for k in ("n", "effect", "lower", "upper", "method", "software_version")})
        result["analysis_input_digest"] = request["input_digest"]
        result["analysis_scope_digest"] = request["scope_digest"]
    elif method != "none":
        if len(values) < 2:
            C.fail("UNCERTAINTY_INSUFFICIENT_RUNS")
        if method == "normal":
            critical = statistics.NormalDist().inv_cdf((1 + confidence) / 2)
            margin = critical * statistics.stdev(values) / math.sqrt(len(values))
            result.update(lower=mean - margin, upper=mean + margin)
        elif method in {"paired_t", "bootstrap"}:
            try:
                import numpy as np
                import scipy
                from scipy import stats
            except ImportError:
                C.fail("SCIPY_ANALYSIS_UNAVAILABLE")
            array = np.asarray(values, dtype=float)
            if method == "paired_t":
                margin = float(stats.t.ppf((1 + confidence) / 2, len(values) - 1)) * statistics.stdev(values) / math.sqrt(len(values))
                result.update(lower=mean - margin, upper=mean + margin)
            else:
                plan = unit_policy.get("bootstrap", {})
                if set(plan) != {"method", "resamples", "seed"} or plan["method"] not in {"BCa", "basic", "percentile"}:
                    C.fail("FROZEN_BOOTSTRAP_POLICY_REQUIRED")
                if not isinstance(plan["resamples"], int) or isinstance(plan["resamples"], bool) or plan["resamples"] < 2 or not isinstance(plan["seed"], int):
                    C.fail("INVALID_BOOTSTRAP_POLICY")
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", stats.DegenerateDataWarning)
                        interval = stats.bootstrap((array,), np.mean, confidence_level=confidence,
                            n_resamples=plan["resamples"], batch=min(plan["resamples"], 256),
                            method=plan["method"], rng=np.random.default_rng(plan["seed"])).confidence_interval
                    result.update(lower=float(interval.low), upper=float(interval.high))
                except (ValueError, RuntimeWarning, stats.DegenerateDataWarning):
                    C.fail("BOOTSTRAP_DEGENERATE_OR_INVALID")
            result["software_version"] = "scipy-" + scipy.__version__
        else:
            C.fail("UNSUPPORTED_UNCERTAINTY_METHOD")
    C.finite(result)
    result["analysis_digest"] = C.hashed(result)
    return result
