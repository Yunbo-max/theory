"""Importance-preserving workflow checks, not a scientific importance oracle."""
import copy
import math
import _autoresearch as C
from _literature import canonical_work_id

OUTCOMES={"freeze_parent_problem","record_natural_gate_0","importance_decision","restart_problem_exploration"}
CONTRIBUTION_ROUTES={"natural_failure":"failure_census","method":"failure_census",
    "replication_or_transfer":"failure_census","efficiency":"performance_frontier",
    "systems":"performance_frontier","phenomenon":"replicated_observations",
    "data":"data_coverage","theory":"formal_obligations"}
VALUE_THRESHOLDS={
    "formal_obligations":{"min_obligations":"count","min_strict_improvements":"count",
                          "min_consequential_obligations":"count"},
    "performance_frontier":{"min_observations":"count","min_mean_resource_gap":"positive",
        "max_quality_shortfall":"nonnegative","min_mean_consequence":"positive","min_unresolved_fraction":"fraction"},
    "replicated_observations":{"min_observations":"count","min_independent_replications":"replications",
        "min_mean_effect":"positive","min_mean_consequence":"positive","min_unresolved_fraction":"fraction"},
    "data_coverage":{"min_observations":"count","min_coverage_gap":"fraction",
        "min_mean_consequence":"positive","min_unresolved_fraction":"fraction"}}


def _thresholds(card,route):
    thresholds=card["value_gate"]["thresholds"]
    specification=VALUE_THRESHOLDS.get(route,{})
    if not specification or set(thresholds)!=set(specification):C.fail("CONTRIBUTION_VALUE_THRESHOLDS_REQUIRED")
    for key,kind in specification.items():
        value=thresholds[key]
        if type(value) not in {int,float} or not math.isfinite(value):C.fail("INVALID_CONTRIBUTION_VALUE_THRESHOLD")
        if kind in {"count","replications"}:
            if type(value) is not int or value<(2 if kind=="replications" else 1):C.fail("INVALID_CONTRIBUTION_VALUE_THRESHOLD")
        elif (value<0 if kind=="nonnegative" else value<=0):C.fail("INVALID_CONTRIBUTION_VALUE_THRESHOLD")
        if kind=="fraction" and value>1:C.fail("INVALID_CONTRIBUTION_VALUE_THRESHOLD")
    return thresholds


def _problem_contract(root,card):
    contribution=card.get("contribution_type","natural_failure")
    if contribution not in CONTRIBUTION_ROUTES:C.fail("UNSUPPORTED_CONTRIBUTION_TYPE")
    route=CONTRIBUTION_ROUTES[contribution]
    if "value_gate" in card:
        value=card["value_gate"]
        if value["route"]!=route:C.fail("CONTRIBUTION_VALUE_ROUTE_MISMATCH")
        for reference in value["evidence_refs"]:C.verify_ref(root,reference)
        if route!="failure_census":_thresholds(card,route)
    if route=="formal_obligations":
        contract=card["formal_contract"]
        C.verify_ref(root,contract["definitions_ref"])
        C.verify_ref(root,contract["theorem_statement_ref"])
        statements=contract.get("obligation_statement_refs")
        if len(contract["obligation_ids"])>1 and not statements:C.fail("FROZEN_FORMAL_OBLIGATION_STATEMENTS_REQUIRED")
        if statements is not None:
            if set(statements)!=set(contract["obligation_ids"]):C.fail("FROZEN_FORMAL_OBLIGATION_STATEMENTS_REQUIRED")
            for reference in statements.values():C.verify_ref(root,reference)
        refs=contract["closest_theorem_work_refs"]
        if contract["strongest_known_alternative_work_ref"] not in refs:
            C.fail("STRONGEST_KNOWN_THEOREM_MUST_BE_COMPARED")
        for reference in refs:
            work=C.verify_ref(root,reference,"work-record")
            if not work["primary_source"] or work["read_depth"] not in {"D2","D3"}:
                C.fail("CLOSEST_THEOREM_PRIMARY_FULL_TEXT_REQUIRED")
            for evidence in work["evidence_refs"]:C.verify_ref(root,evidence)
    else:
        for reference in card["benchmark_contract"]["source_evidence_refs"]:C.verify_ref(root,reference)
        if route=="data_coverage":
            reference=card["value_gate"].get("coverage_definition_ref")
            if not reference:C.fail("PUBLISHED_NATIVE_COVERAGE_DEFINITION_REQUIRED")
            coverage=C.load_file(C.verify_ref(root,reference))
            benchmark=card["benchmark_contract"]
            if (not isinstance(coverage,dict) or coverage.get("capture_format")!="native-coverage-definition-v1" or
                coverage.get("benchmark_id")!=benchmark["benchmark_id"] or coverage.get("benchmark_version")!=benchmark["version"] or
                not coverage.get("coverage_meaning") or not coverage.get("source_url") or
                not coverage.get("source_evidence_refs") or not isinstance(coverage.get("qualification"),dict)):
                C.fail("PUBLISHED_NATIVE_COVERAGE_DEFINITION_REQUIRED")
            for evidence in coverage["source_evidence_refs"]:C.verify_ref(root,evidence)
    return route


def _result(report,outcome,metrics,reasons,route):
    if report["outcome"]!=outcome:C.fail("NATURAL_GATE_OUTCOME_MISMATCH")
    return {"outcome":outcome,"metrics":metrics,"reason_codes":reasons or ["CONTRIBUTION_VALUE_SCREEN_PASSES"],
            "value_gate_route":route}


def _attestation(root,pref,card,raw,report,kind,validator,historical,required_refs):
    key="formal_review_attestation_ref" if kind=="formal" else "native_eval_attestation_ref"
    reference=raw.get(key)
    if not reference:return None,"FORMAL_REVIEW_ATTESTATION_REQUIRED" if kind=="formal" else "NATIVE_EVAL_ATTESTATION_REQUIRED"
    capture=C.load_file(C.verify_ref(root,reference))
    expected_format="formal-review-attestation-v1" if kind=="formal" else "value-gate-attestation-v1"
    digest=C.hashed({k:v for k,v in raw.items() if k!=key})
    if (not isinstance(capture,dict) or capture.get("capture_format")!=expected_format or
        capture.get("parent_problem_sha256")!=pref["sha256"] or capture.get("observations_digest")!=digest):
        C.fail("VALUE_GATE_ATTESTATION_BINDING_MISMATCH")
    if capture.get("status")!="verified" or not capture.get("proof_refs"):
        return None,"VALUE_GATE_ATTESTATION_UNVERIFIED"
    keys={(r["path"],r["sha256"]) for r in capture["proof_refs"]}
    if not {(r["path"],r["sha256"]) for r in required_refs}.issubset(keys):
        C.fail("VALUE_GATE_ATTESTATION_PROOF_CLOSURE_REQUIRED")
    for proof in capture["proof_refs"]:C.verify_ref(root,proof)
    C.scan(capture);C.finite(capture)
    if not historical:
        if not callable(validator):
            return None,"FORMAL_REVIEW_VALIDATOR_REQUIRED" if kind=="formal" else "NATIVE_EVAL_RECEIPT_VALIDATOR_REQUIRED"
        checked=validator(root,card,raw,report) if kind=="formal" else validator(root,card,raw)
        if not isinstance(checked,dict) or checked.get("status")!="verified" or checked.get("attestation_ref")!=reference:
            return None,"FORMAL_REVIEW_UNVERIFIED" if kind=="formal" else "NATIVE_EVAL_RECEIPTS_UNVERIFIED"
        if checked.get("proof_refs")!=capture["proof_refs"]:C.fail("LIVE_VALUE_GATE_ATTESTATION_DISAGREEMENT")
        field="obligation_ids" if kind=="formal" else "observations"
        if checked.get(field)!=capture.get(field):C.fail("LIVE_VALUE_GATE_ATTESTATION_DISAGREEMENT")
    return capture,None


def _formal_gate(root,pref,card,report,raw,validator,historical):
    route="formal_obligations";contract=card["formal_contract"]
    metrics={"obligations":0,"verified_obligations":0,"strict_improvements":0,"consequential_obligations":0}
    rows=raw.get("obligations") if isinstance(raw,dict) else None
    if not isinstance(rows,list) or not rows or raw.get("sampling_kind")!=route or raw.get("formal_system")!=contract["formal_system"]:
        return _result(report,"INCONCLUSIVE",metrics,["FORMAL_OBLIGATIONS_REQUIRED"],route)
    proof_refs=[];seen=set()
    for row in rows:
        if (not isinstance(row,dict) or not row.get("obligation_id") or row["obligation_id"] in seen or
            any(not row.get(k) for k in ("statement_ref","proof_ref","review_ref","comparison_ref","consequence_ref"))):
            return _result(report,"INCONCLUSIVE",metrics,["FORMAL_PROOF_REVIEW_EVIDENCE_REQUIRED"],route)
        seen.add(row["obligation_id"])
        statements=contract.get("obligation_statement_refs",{identifier:contract["theorem_statement_ref"] for identifier in contract["obligation_ids"]})
        if row["statement_ref"]!=statements.get(row["obligation_id"]):
            return _result(report,"INCONCLUSIVE",metrics,["FORMAL_STATEMENT_NOT_FROZEN"],route)
        for key in ("statement_ref","proof_ref","review_ref","comparison_ref","consequence_ref"):
            C.verify_ref(root,row[key]);proof_refs.append(row[key])
        review=C.load_file(C.verify_ref(root,row["review_ref"]))
        binding={"formal_system":contract["formal_system"],"obligation_id":row["obligation_id"],
            "statement_sha256":row["statement_ref"]["sha256"],"proof_sha256":row["proof_ref"]["sha256"]}
        if not isinstance(review,dict) or any(review.get(k)!=v for k,v in binding.items()):
            return _result(report,"INCONCLUSIVE",metrics,["FORMAL_PROOF_BINDING_REQUIRED"],route)
        if not report.get("evaluated_at") or not review.get("checked_at"):
            return _result(report,"INCONCLUSIVE",metrics,["FRESH_FORMAL_PROOF_REVIEW_REQUIRED"],route)
        age=(C.stamp(report["evaluated_at"])-C.stamp(review["checked_at"])).total_seconds()
        if age<0 or age>contract["review_max_age_seconds"]:
            return _result(report,"INCONCLUSIVE",metrics,["FRESH_FORMAL_PROOF_REVIEW_REQUIRED"],route)
        if review.get("review_kind")=="proof_assistant":
            if not review.get("checker_version") or not review.get("command") or not review.get("checker_output_ref"):
                return _result(report,"INCONCLUSIVE",metrics,["FORMAL_CHECKER_OUTPUT_REQUIRED"],route)
            output=C.load_file(C.verify_ref(root,review["checker_output_ref"]));proof_refs.append(review["checker_output_ref"])
            if (not isinstance(output,dict) or any(output.get(k)!=v for k,v in binding.items()) or
                output.get("exit_code")!=0 or output.get("unresolved_goals")!=[] or output.get("admitted_axioms")!=[]):
                return _result(report,"INCONCLUSIVE",metrics,["FORMAL_PROOF_NOT_CLOSED"],route)
        elif review.get("review_kind")=="independent_formal_review":
            if (review.get("independent") is not True or review.get("conclusion")!="proved" or
                review.get("unresolved_obligations")!=[] or review.get("admitted_axioms")!=[] or
                not review.get("reviewer_role") or not review.get("primary_evidence_refs")):
                return _result(report,"INCONCLUSIVE",metrics,["FORMAL_PROOF_NOT_CLOSED"],route)
            for evidence in review["primary_evidence_refs"]:C.verify_ref(root,evidence);proof_refs.append(evidence)
        else:return _result(report,"INCONCLUSIVE",metrics,["FORMAL_PROOF_REVIEW_EVIDENCE_REQUIRED"],route)
        comparison=C.load_file(C.verify_ref(root,row["comparison_ref"]))
        if (not isinstance(comparison,dict) or comparison.get("obligation_id")!=row["obligation_id"] or
            comparison.get("strongest_known_alternative")!=card["strong_simple_alternative"] or
            comparison.get("closest_theorem_work_ref")!=contract["strongest_known_alternative_work_ref"] or
            comparison.get("comparison_proof_ref")!=row["proof_ref"] or comparison.get("review_ref")!=row["review_ref"]):
            return _result(report,"INCONCLUSIVE",metrics,["CLOSEST_THEOREM_FORMAL_COMPARISON_REQUIRED"],route)
        relation=comparison.get("relation")
        if relation not in {"strict_strengthening","assumption_removal","tight_bound","separation","equivalent","weaker"}:
            return _result(report,"INCONCLUSIVE",metrics,["CLOSEST_THEOREM_FORMAL_COMPARISON_REQUIRED"],route)
        consequence=C.load_file(C.verify_ref(root,row["consequence_ref"]))
        if (not isinstance(consequence,dict) or consequence.get("endpoint")!=card["consequence"]["endpoint"] or
            not consequence.get("scientific_implication") or not consequence.get("justification") or not consequence.get("primary_evidence_refs")):
            return _result(report,"INCONCLUSIVE",metrics,["FORMAL_CONSEQUENCE_EVIDENCE_REQUIRED"],route)
        for evidence in consequence["primary_evidence_refs"]:C.verify_ref(root,evidence);proof_refs.append(evidence)
        metrics["strict_improvements"]+=relation not in {"equivalent","weaker"}
        metrics["verified_obligations"]+=1;metrics["consequential_obligations"]+=1
    metrics["obligations"]=len(rows)
    if seen!=set(contract["obligation_ids"]):
        return _result(report,"INCONCLUSIVE",metrics,["FORMAL_OBLIGATION_COVERAGE_REQUIRED"],route)
    attestation,reason=_attestation(root,pref,card,raw,report,"formal",validator,historical,proof_refs)
    if reason:return _result(report,"INCONCLUSIVE",metrics,[reason],route)
    if attestation.get("obligation_ids")!=sorted(seen):C.fail("FORMAL_REVIEW_OBLIGATION_BINDING_MISMATCH")
    t=_thresholds(card,route)
    if metrics["obligations"]<t["min_obligations"]:
        return _result(report,"INCONCLUSIVE",metrics,["FORMAL_OBLIGATION_SAMPLE_INSUFFICIENT"],route)
    reasons=[]
    if metrics["strict_improvements"]<t["min_strict_improvements"]:reasons.append("KNOWN_THEOREM_ALREADY_SUFFICIENT")
    if metrics["consequential_obligations"]<t["min_consequential_obligations"]:reasons.append("FORMAL_CONSEQUENCE_TOO_SMALL")
    return _result(report,"KILL" if reasons else "PASS",metrics,reasons,route)


def native_value_gate_validator(root,card,raw,replay_context=None):
    """Bind a value screen to existing native receipts; this never scores samples.

    The native evaluator supplies all measured values and qualification flags.
    Resource/qualification bindings select retained native receipt fields or
    predeclared native qualification rules, rather than defining another metric.
    """
    from _native_eval import verify_protocol,verify_run
    runtime=C.get_evaluation_runtime() if hasattr(C,"get_evaluation_runtime") else {}
    replay_context=replay_context or runtime.get("native_replay_context")
    protocol=C.load_file(C.verify_ref(root,raw["native_protocol_ref"]))
    contract=verify_protocol(root,protocol)
    benchmark=card["benchmark_contract"]
    expected={"benchmark_id":benchmark["benchmark_id"],"benchmark_revision":benchmark["version"],
              "split":benchmark["evaluation_split"],"primary_metric":benchmark["primary_metric"]}
    if any(contract.get(k)!=v for k,v in expected.items()):C.fail("VALUE_GATE_NATIVE_CONTRACT_MISMATCH")
    if (contract["scorer"]["identity"]!=raw["scorer_identity"] or
        contract["scorer"]["revision"]!=raw["scorer_revision"] or
        contract["contrasts"]["baseline"]!=card["strong_simple_alternative"]):
        C.fail("VALUE_GATE_NATIVE_BASELINE_OR_SCORER_MISMATCH")
    runs={};proofs=[raw["native_protocol_ref"]];receipts=[];outputs=[]
    for reference in raw["native_run_manifest_refs"]:
        manifest=C.verify_ref(root,reference,"run-manifest")
        if manifest["run_id"] in runs:C.fail("DUPLICATE_VALUE_GATE_NATIVE_RUN")
        observed=verify_run(root,protocol,manifest,replay_context)
        if not observed["positive_control_passed"]:C.fail("VALUE_GATE_NATIVE_CONTROLS_UNQUALIFIED")
        runs[manifest["run_id"]]=(manifest,observed)
        proofs+=[reference]+observed["proof_refs"]
        for receipt in manifest["native_eval_receipts"].values():
            receipts.append(receipt)
            outputs.append(C.verify_ref(root,receipt,"native-eval-receipt")["native_output_ref"])
    ref_keys=lambda refs:{(r["path"],r["sha256"]) for r in refs}
    if ref_keys(receipts)!=ref_keys(raw["native_eval_receipt_refs"]) or ref_keys(outputs)!=ref_keys(raw["native_output_refs"]):
        C.fail("VALUE_GATE_NATIVE_RECEIPT_INVENTORY_MISMATCH")
    rows=copy.deepcopy(raw["observations"]);replications=set()
    required={"downstream_loss","alternative_solves"}
    route=card["value_gate"]["route"]
    coverage=None
    if route=="data_coverage":
        coverage=C.load_file(C.verify_ref(root,card["value_gate"]["coverage_definition_ref"]))
        role=coverage["qualification"].get("role")
        rule=contract["baseline_qualification"] if role=="baseline" else next(
            (r for r in contract["control_qualifications"] if r["role"]==role),None)
        if rule is None or coverage["qualification"].get("rule")!=rule:
            C.fail("VALUE_GATE_NATIVE_COVERAGE_RULE_MISMATCH")
        proofs.append(card["value_gate"]["coverage_definition_ref"])
        proofs+=coverage["source_evidence_refs"]
    required|={"resource","quality"} if route=="performance_frontier" else (
               {"observed_metric","reference_metric"} if route=="replicated_observations" else {"covered"})
    for row in rows:
        bindings=row.get("measurement_bindings",{})
        if set(bindings)!=required:C.fail("NATIVE_VALUE_MEASUREMENT_BINDINGS_REQUIRED")
        own=set()
        for field,binding in bindings.items():
            run_id=binding.get("run_id")
            if run_id not in runs:C.fail("VALUE_GATE_NATIVE_RUN_BINDING_REQUIRED")
            own.add(run_id);manifest,observed=runs[run_id]
            kind=binding.get("kind","metric")
            metric=binding.get("metric");role=binding.get("arm_role")
            if kind=="metric":
                if role not in observed["arm_metrics"] or metric not in observed["arm_metrics"][role]:
                    C.fail("VALUE_GATE_NATIVE_METRIC_REQUIRED")
                expected_metric=card["consequence"]["endpoint"] if field=="downstream_loss" else benchmark["primary_metric"]
                if field in {"resource","alternative_solves","covered"} or metric!=expected_metric:
                    C.fail("VALUE_GATE_NATIVE_ENDPOINT_MAPPING_REQUIRED")
                value=observed["arm_metrics"][role][metric]
            elif kind=="resource":
                if field!="resource" or role not in observed["arm_metrics"] or metric!=card["value_gate"]["resource_metric"]:
                    C.fail("VALUE_GATE_NATIVE_RESOURCE_MAPPING_REQUIRED")
                value=manifest["resources"].get(metric)
            elif kind=="qualification":
                qualification=binding.get("role")
                if field not in {"alternative_solves","covered"}:C.fail("VALUE_GATE_NATIVE_QUALIFICATION_MAPPING_REQUIRED")
                if field=="alternative_solves" and qualification!="baseline":
                    C.fail("VALUE_GATE_STRONG_ALTERNATIVE_BINDING_REQUIRED")
                if field=="covered" and (coverage is None or qualification!=coverage["qualification"]["role"]):
                    C.fail("VALUE_GATE_NATIVE_COVERAGE_RULE_MISMATCH")
                if qualification=="baseline":value=observed["baseline_qualified"]
                elif qualification in observed["control_results"]:value=observed["control_results"][qualification]
                else:C.fail("VALUE_GATE_NATIVE_QUALIFICATION_MAPPING_REQUIRED")
            else:C.fail("VALUE_GATE_NATIVE_MEASUREMENT_KIND_UNSUPPORTED")
            if type(value)!=type(row[field]) and not (type(value) in {float,int} and type(row[field]) in {float,int}):
                C.fail("NATIVE_VALUE_MEASUREMENT_DISAGREEMENT")
            if row[field]!=value:C.fail("NATIVE_VALUE_MEASUREMENT_DISAGREEMENT")
        if own!={row["observation_id"]}:C.fail("VALUE_GATE_NATIVE_OBSERVATION_BINDING_REQUIRED")
        if route=="replicated_observations":
            manifest,_=runs[row["observation_id"]]
            replicate=(manifest["seed"],manifest["group"])
            if row.get("replication_id")!=manifest["run_id"] or replicate in replications:
                C.fail("VALUE_GATE_INDEPENDENT_REPLICATION_BINDING_REQUIRED")
            replications.add(replicate)
        proofs.append(row["source_ref"])
    attestation=C.load_file(C.verify_ref(root,raw["native_eval_attestation_ref"]))
    if rows!=attestation.get("observations"):C.fail("NATIVE_VALUE_MEASUREMENT_DISAGREEMENT")
    if not ref_keys(proofs).issubset(ref_keys(attestation.get("proof_refs",[]))):
        C.fail("VALUE_GATE_ATTESTATION_PROOF_CLOSURE_REQUIRED")
    return {"status":"verified","observations":rows,"proof_refs":attestation["proof_refs"],
            "attestation_ref":raw["native_eval_attestation_ref"]}


def _empirical_gate(root,pref,card,report,raw,validator,historical):
    route=card["value_gate"]["route"];benchmark=card["benchmark_contract"]
    metrics={"observations":0,"mean_consequence":0.0,"unresolved_fraction":0.0}
    rows=raw.get("observations") if isinstance(raw,dict) else None
    if not isinstance(rows,list) or not rows or raw.get("sampling_kind")!=route:
        return _result(report,"INCONCLUSIVE",metrics,["CONTRIBUTION_OBSERVATIONS_REQUIRED"],route)
    identity={"benchmark_id":benchmark["benchmark_id"],"benchmark_version":benchmark["version"],
        "evaluation_split":benchmark["evaluation_split"],"primary_metric":benchmark["primary_metric"]}
    if any(raw.get(k)!=v for k,v in identity.items()):
        return _result(report,"INCONCLUSIVE",metrics,["CONTRIBUTION_NATIVE_BENCHMARK_BINDING_REQUIRED"],route)
    if (not raw.get("native_output_refs") or not raw.get("native_eval_receipt_refs") or
        not raw.get("native_protocol_ref") or not raw.get("native_run_manifest_refs")):
        return _result(report,"INCONCLUSIVE",metrics,["NATIVE_EVAL_RECEIPTS_REQUIRED"],route)
    if any(not benchmark.get(k) or raw.get(k)!=benchmark[k] for k in ("scorer_identity","scorer_revision")):
        return _result(report,"INCONCLUSIVE",metrics,["NATIVE_SCORER_IDENTITY_REQUIRED"],route)
    proofs=raw["native_output_refs"]+raw["native_eval_receipt_refs"]+[raw["native_protocol_ref"]]+raw["native_run_manifest_refs"]
    if route=="data_coverage":proofs.append(card["value_gate"]["coverage_definition_ref"])
    for reference in proofs:C.verify_ref(root,reference)
    seen=set();scope={"population_id":card["population_id"],"endpoint":card["consequence"]["endpoint"],
                      "strong_simple_alternative":card["strong_simple_alternative"]}
    number=lambda value:type(value) in {int,float} and math.isfinite(value)
    for row in rows:
        if (not isinstance(row,dict) or not isinstance(row.get("observation_id"),str) or not row["observation_id"] or
            row["observation_id"] in seen or any(row.get(k)!=v for k,v in scope.items()) or
            not number(row.get("downstream_loss")) or row["downstream_loss"]<0 or
            type(row.get("alternative_solves")) is not bool or not row.get("source_ref")):
            return _result(report,"INCONCLUSIVE",metrics,["CONTRIBUTION_OBSERVATION_SCOPE_OR_MEASUREMENTS_REQUIRED"],route)
        seen.add(row["observation_id"])
        source=C.load_file(C.verify_ref(root,row["source_ref"]));proofs.append(row["source_ref"])
        if source!={k:v for k,v in row.items() if k!="source_ref"}:C.fail("CONTRIBUTION_OBSERVATION_RAW_DISAGREEMENT")
        if route=="performance_frontier":
            resource_metric=card["value_gate"].get("resource_metric")
            limit=benchmark["resource_constraints"].get(resource_metric)
            if (not number(limit) or limit<=0 or row.get("resource_limit")!=limit or
                not number(row.get("resource")) or row["resource"]<0 or not number(row.get("quality")) or
                not number(card["value_gate"].get("required_quality"))):
                return _result(report,"INCONCLUSIVE",metrics,["PERFORMANCE_RESOURCE_FRONTIER_REQUIRED"],route)
        elif route=="replicated_observations":
            if (not number(row.get("observed_metric")) or not number(row.get("reference_metric")) or
                not isinstance(row.get("replication_id"),str) or not row["replication_id"]):
                return _result(report,"INCONCLUSIVE",metrics,["REPLICATED_PHENOMENON_MEASUREMENTS_REQUIRED"],route)
        elif type(row.get("covered")) is not bool:
            return _result(report,"INCONCLUSIVE",metrics,["NATIVE_DATA_COVERAGE_REQUIRED"],route)
    attestation,reason=_attestation(root,pref,card,raw,report,"empirical",validator,historical,proofs)
    if reason:return _result(report,"INCONCLUSIVE",metrics,[reason],route)
    if attestation.get("observations")!=rows:C.fail("NATIVE_VALUE_MEASUREMENT_DISAGREEMENT")
    count=len(rows);t=_thresholds(card,route)
    metrics.update(observations=count,mean_consequence=sum(r["downstream_loss"] for r in rows)/count,
                   unresolved_fraction=sum(not r["alternative_solves"] for r in rows)/count)
    if count<t["min_observations"]:
        return _result(report,"INCONCLUSIVE",metrics,["CONTRIBUTION_SAMPLE_INSUFFICIENT"],route)
    reasons=[]
    if metrics["mean_consequence"]<t["min_mean_consequence"]:reasons.append("DOWNSTREAM_CONSEQUENCE_TOO_SMALL")
    if metrics["unresolved_fraction"]<t["min_unresolved_fraction"]:reasons.append("SIMPLE_ALTERNATIVE_SUFFICIENT")
    direction=1 if card["value_gate"].get("effect_direction","maximize")=="maximize" else -1
    if route=="performance_frontier":
        metrics["mean_resource_gap"]=sum(max(0,r["resource"]-r["resource_limit"])/r["resource_limit"] for r in rows)/count
        metrics["max_quality_shortfall"]=max(max(0,direction*(card["value_gate"]["required_quality"]-r["quality"])) for r in rows)
        if metrics["mean_resource_gap"]<t["min_mean_resource_gap"]:reasons.append("KNOWN_RESOURCE_FRONTIER_SUFFICIENT")
        if metrics["max_quality_shortfall"]>t["max_quality_shortfall"]:reasons.append("PERFORMANCE_FRONTIER_TARGET_NOT_MET")
    elif route=="replicated_observations":
        metrics["independent_replications"]=len({r["replication_id"] for r in rows})
        metrics["mean_effect"]=sum(direction*(r["observed_metric"]-r["reference_metric"]) for r in rows)/count
        if metrics["independent_replications"]<t["min_independent_replications"]:
            return _result(report,"INCONCLUSIVE",metrics,["INDEPENDENT_PHENOMENON_REPLICATIONS_REQUIRED"],route)
        if metrics["mean_effect"]<t["min_mean_effect"]:reasons.append("PHENOMENON_EFFECT_TOO_SMALL")
    else:
        metrics["coverage_gap"]=sum(not r["covered"] for r in rows)/count
        if metrics["coverage_gap"]<t["min_coverage_gap"]:reasons.append("EXISTING_DATA_COVERAGE_SUFFICIENT")
    return _result(report,"KILL" if reasons else "PASS",metrics,reasons,route)

def parent(root,state):
    ref=state.get("parent_problem_ref")
    if not ref:C.fail("FROZEN_PARENT_PROBLEM_REQUIRED")
    card=C.verify_ref(root,ref,"parent-problem-card")
    if card["strong_simple_alternative"] not in card["simple_alternatives"]:
        C.fail("STRONG_SIMPLE_ALTERNATIVE_MUST_BE_INCLUDED")
    _problem_contract(root,card)
    return card


def natural_gate(root,pref,gref,receipt_validator=None,formal_review_validator=None,historical=False):
    card=C.verify_ref(root,pref,"parent-problem-card")
    report=C.verify_ref(root,gref,"natural-gate-0")
    if report["parent_problem_ref"]!=pref:C.fail("NATURAL_GATE_PARENT_MISMATCH")
    raw=C.load_file(C.verify_ref(root,report["observations_ref"]))
    route=_problem_contract(root,card)
    if report.get("contribution_type",card.get("contribution_type"))!=card.get("contribution_type"):
        C.fail("VALUE_GATE_CONTRIBUTION_TYPE_MISMATCH")
    if report.get("value_gate_route",route)!=route:C.fail("CONTRIBUTION_VALUE_ROUTE_MISMATCH")
    runtime=C.get_evaluation_runtime() if hasattr(C,"get_evaluation_runtime") else {}
    if route=="formal_obligations":
        return _formal_gate(root,pref,card,report,raw,formal_review_validator or runtime.get("formal_review_validator"),historical)
    if route!="failure_census":
        validator=receipt_validator or runtime.get("value_gate_validator")
        if validator is None and callable(runtime.get("native_replay_context")):
            validator=native_value_gate_validator
        return _empirical_gate(root,pref,card,report,raw,validator,historical)
    metrics={"cases":0,"affected":0,"prevalence":0.0,"mean_loss":0.0,"unresolved_fraction":0.0}
    reasons=[]
    rows=raw.get("cases") if isinstance(raw,dict) else None
    benchmark=card["benchmark_contract"]
    identity={"benchmark_id":benchmark["benchmark_id"],"benchmark_version":benchmark["version"],
        "evaluation_split":benchmark["evaluation_split"],"primary_metric":benchmark["primary_metric"]}
    if isinstance(raw,dict) and any(raw.get(k)!=v for k,v in identity.items()):
        reasons=["NATURAL_CENSUS_BENCHMARK_BINDING_REQUIRED"]
    elif not isinstance(raw,dict) or raw.get("sampling_kind")!="natural_failures" or not isinstance(rows,list) or not rows:
        reasons=["NATURAL_FAILURE_CENSUS_REQUIRED"]
    else:
        seen=set();eligible=[]
        scope={"population_id":card["population_id"],"natural_failure":card["natural_failure"],
            "endpoint":card["consequence"]["endpoint"],"strong_simple_alternative":card["strong_simple_alternative"]}
        for row in rows:
            if not isinstance(row,dict) or not isinstance(row.get("case_id"),str) or not row["case_id"] or row["case_id"] in seen:
                reasons.append("INVALID_OR_DUPLICATE_NATURAL_EPISODE");break
            seen.add(row["case_id"])
            if any(row.get(k)!=v for k,v in scope.items()) or row.get("natural") is not True or row.get("failed_episode") is not True:
                reasons.append("NATURAL_EPISODE_SCOPE_UNVERIFIED");break
            loss=row.get("downstream_loss")
            if type(row.get("affected")) is not bool or type(row.get("alternative_solves")) is not bool or type(loss) not in (float,int) or not math.isfinite(loss) or loss<0:
                reasons.append("NATURAL_EPISODE_MEASUREMENTS_REQUIRED");break
            if not row.get("source_ref"):
                reasons.append("NATURAL_EPISODE_SOURCE_REQUIRED");break
            source=C.load_file(C.verify_ref(root,row["source_ref"]))
            expected={k:v for k,v in row.items() if k!="source_ref"}
            if source!=expected:C.fail("NATURAL_EPISODE_RAW_DISAGREEMENT")
            eligible.append(row)
        if not reasons:
            affected=[r for r in eligible if r["affected"]]
            n,a=len(eligible),len(affected)
            metrics.update(cases=n,affected=a,prevalence=a/n,
                mean_loss=sum(r["downstream_loss"] for r in affected)/a if a else 0.0,
                unresolved_fraction=sum(not r["alternative_solves"] for r in affected)/a if a else 0.0)
            t=card["gate_0_thresholds"]
            if n<t["min_cases"]:reasons.append("NATURAL_SAMPLE_INSUFFICIENT")
    if reasons:outcome="INCONCLUSIVE"
    else:
        t=card["gate_0_thresholds"]
        checks=((metrics["affected"]>=t["min_affected"],"NATURAL_FAILURE_TOO_RARE"),
            (metrics["prevalence"]>=t["min_prevalence"],"NATURAL_PREVALENCE_TOO_LOW"),
            (metrics["mean_loss"]>=t["min_mean_loss"],"DOWNSTREAM_CONSEQUENCE_TOO_SMALL"),
            (metrics["unresolved_fraction"]>=t["min_unresolved_fraction"],"SIMPLE_ALTERNATIVE_SUFFICIENT"))
        reasons=[reason for passed,reason in checks if not passed]
        outcome="KILL" if reasons else "PASS"
    if report["outcome"]!=outcome:C.fail("NATURAL_GATE_OUTCOME_MISMATCH")
    return {"outcome":outcome,"metrics":metrics,"reason_codes":reasons or ["NATURAL_PROBLEM_SCREEN_PASSES"]}


def require_natural(root,state,candidate=None):
    card=parent(root,state)
    if not state.get("natural_gate_0_ref"):C.fail("NATURAL_GATE_0_REQUIRED")
    result=natural_gate(root,state["parent_problem_ref"],state["natural_gate_0_ref"],historical=True)
    if result["outcome"]!="PASS":C.fail("NATURAL_GATE_0_PASS_REQUIRED")
    if state.get("importance_outcome")=="KILL":C.fail("IMPORTANCE_KILL_REQUIRES_STOP_OR_REROUTE")
    if candidate is None and state.get("candidate_ref"):
        candidate=C.verify_ref(root,state["candidate_ref"],"idea-atom")
    if candidate is not None:
        if candidate.get("parent_problem_ref")!=state["parent_problem_ref"]:
            C.fail("CANDIDATE_PARENT_PROBLEM_MISMATCH")
        case=candidate.get("necessity_case")
        if not isinstance(case,dict):C.fail("CANDIDATE_NECESSITY_COMPARISON_REQUIRED")
        compared=set(case["simple_alternatives_compared"])
        if not set(card["simple_alternatives"]).issubset(compared):
            C.fail("FROZEN_SIMPLE_ALTERNATIVES_COMPARISON_REQUIRED")
        if case["memory_representation_change"] and "plain_text_memory" not in compared:
            C.fail("PLAIN_TEXT_MEMORY_COMPARISON_REQUIRED")
        for r in case["baseline_evidence_refs"]:C.verify_ref(root,r)
    return result


def assess(root,state,ref):
    card=parent(root,state)
    report=C.verify_ref(root,ref,"importance-decision")
    if report["parent_problem_ref"]!=state["parent_problem_ref"] or report["natural_gate_0_ref"]!=state.get("natural_gate_0_ref"):
        C.fail("IMPORTANCE_NOT_BOUND_TO_CURRENT_PROBLEM")
    if report.get("candidate_ref")!=state.get("candidate_ref"):C.fail("IMPORTANCE_NOT_BOUND_TO_CURRENT_CANDIDATE")
    for r in report["evidence_refs"]:C.verify_ref(root,r)
    outcome=report["outcome"]
    works=[C.verify_ref(root,r,"work-record") for r in report["functional_collision_work_refs"]]
    ids={canonical_work_id(root,identifier,works) if card.get("contribution_type") else identifier
         for identifier in state.get("major_collision_work_ids",[])}
    for r in report["functional_collision_work_refs"]:
        work=C.verify_ref(root,r,"work-record")
        if not work["primary_source"] or work["read_depth"] not in {"D2","D3"} or not work["evidence_refs"]:
            C.fail("MAJOR_FUNCTIONAL_COLLISION_PRIMARY_EVIDENCE_REQUIRED")
        for evidence in work["evidence_refs"]:C.verify_ref(root,evidence)
        ids.add(canonical_work_id(root,work,works) if card.get("contribution_type") else work["work_id"])
    if outcome=="CONCURRENT":
        require_natural(root,state)
        expected={"natural_failure":card["natural_failure"],"population_id":card["population_id"],
            "endpoint":card["consequence"]["endpoint"],"strong_simple_alternative":card["strong_simple_alternative"],
            "falsifiable_prediction":card["falsifiable_prediction"]}
        if report["retained_problem"]!=expected:C.fail("PARENT_SCOPE_CHANGED_RESTART_REQUIRED")
        if report["exclusion_basis"]:C.fail("NOVELTY_BY_EXCLUSION_REQUIRES_NEW_NATURAL_PROBLEM")
    # Fresh papers do not reset this per-parent budget. The second distinct
    # declared major functional collision forces I-style exploration.
    if len(ids)>=2 and outcome!="KILL":outcome="REROUTE"
    return report,outcome,sorted(ids)


def require_ready(root,state):
    require_natural(root,state)
    ref=state.get("importance_decision_ref")
    if not ref or state.get("importance_outcome")!="CONCURRENT":C.fail("CURRENT_IMPORTANCE_DECISION_REQUIRED")
    report,outcome,ids=assess(root,state,ref)
    if outcome!="CONCURRENT":C.fail("COLLISION_BUDGET_RESTART_REQUIRED")
    return report


def reset_for_exploration(state):
    new=copy.deepcopy(state)
    if state.get("parent_problem_ref"):new["previous_parent_problem_ref"]=state["parent_problem_ref"]
    for k in ("parent_problem_ref","natural_gate_0_ref","natural_gate_0_outcome","natural_gate_0_metrics",
              "candidate_id","candidate_ref","gate_a_protocol_ref","full_validation_protocol_ref"):
        new.pop(k,None)
    new.update(current_route="explore_problem",lifecycle_phase="landscape",suspension=None,
        gate_state="not_ready",validation_state="not_ready",writing_state="not_ready")
    return new


def apply(root,state,outcome,payload,historical=False):
    """Pure state decision reused during both commit and historical replay."""
    new=copy.deepcopy(state);refs=[]
    route,phase=state["current_route"],state["lifecycle_phase"]
    if outcome=="freeze_parent_problem":
        if route not in {"explore_problem","audit_method","audit_rejection"} or phase!="landscape":C.fail("PARENT_FREEZE_REQUIRES_EXPLORATION")
        pref=payload["parent_problem_ref"];card=C.verify_ref(root,pref,"parent-problem-card")
        if not historical and not card.get("contribution_type"):C.fail("CONTRIBUTION_TYPE_REQUIRED")
        parent(root,{"parent_problem_ref":pref})
        if state.get("parent_problem_ref") and pref!=state["parent_problem_ref"]:C.fail("PARENT_PROBLEM_IMMUTABLE_REROUTE_REQUIRED")
        previous=state.get("previous_parent_problem_ref")
        if previous:
            old=C.verify_ref(root,previous,"parent-problem-card")
            if pref==previous or card["problem_id"]==old["problem_id"] or card.get("previous_parent_problem_ref")!=previous:C.fail("NEW_PARENT_PROBLEM_LINEAGE_REQUIRED")
        new["parent_problem_ref"]=pref;refs+=C.payload_refs(card)
        if not state.get("parent_problem_ref"):
            new["major_collision_work_ids"]=[]
            new.pop("importance_decision_ref",None);new.pop("importance_outcome",None)
    elif outcome=="record_natural_gate_0":
        parent(root,state)
        if route not in {"explore_problem","audit_method","audit_rejection","develop_candidate"}:C.fail("NATURAL_GATE_REASSESSMENT_REQUIRES_EXPLORATION")
        gref=payload["natural_gate_0_ref"];checked=natural_gate(root,state["parent_problem_ref"],gref,historical=historical)
        report=C.verify_ref(root,gref,"natural-gate-0")
        raw=C.load_file(C.verify_ref(root,report["observations_ref"]))
        refs+=C.payload_refs(report)+C.payload_refs(raw)
        new.update(natural_gate_0_ref=gref,natural_gate_0_outcome=checked["outcome"],natural_gate_0_metrics=checked["metrics"])
        new.pop("importance_decision_ref",None)
        if state.get("importance_outcome")!="KILL":new.pop("importance_outcome",None)
    elif outcome=="importance_decision":
        iref=payload["importance_decision_ref"];report,choice,ids=assess(root,state,iref)
        refs+=C.payload_refs(report)
        for r in report["functional_collision_work_refs"]:refs+=C.payload_refs(C.verify_ref(root,r,"work-record"))
        if choice=="REROUTE":new=reset_for_exploration(state)
        new.update(importance_decision_ref=iref,importance_outcome=choice,major_collision_work_ids=ids)
        if choice=="KILL":new.update(gate_state="not_ready",validation_state="not_ready",writing_state="not_ready")
    elif outcome=="restart_problem_exploration":
        new=reset_for_exploration(state)
        new.pop("importance_decision_ref",None);new["importance_outcome"]="REROUTE"
    else:C.fail("UNKNOWN_IMPORTANCE_OUTCOME")
    return new,refs


def verify_event(root,prior,event):
    outcome=event["payload"].get("outcome")
    if event["event_type"]!="transition" or outcome not in OUTCOMES:return
    expected,_=apply(root,prior,outcome,event["payload"],historical=True)
    expected["sequence"]=event["sequence"]
    if expected!=event["next_state"]:C.fail("IMPORTANCE_STATE_REPLAY_MISMATCH")
