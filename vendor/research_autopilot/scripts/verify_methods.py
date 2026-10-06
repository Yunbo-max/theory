#!/usr/bin/env python3
"""Derive scoped method readiness from pinned artifacts and documented reviews.

This checks evidence contracts, not mathematical truth or reviewer identity.
Scientific outcomes reuse the existing evaluator and its trusted live contexts.
It neither launches jobs nor advances the research ledger's scientific gates.
"""
import argparse
from pathlib import Path
import re

import _autoresearch as C

MATH_CHECKS = ("formal_object", "operations_and_conditions", "derivation", "assumptions",
               "method_expression", "prediction_and_falsifier", "closest_alternative")
RANK_CHECKS = ("structural_distinctness", "all_candidates_compared", "ranking_basis", "top_selection")
CODE_CHECKS = ("math_to_code", "software_checks", "actual_code_identity")
DESIGN_CHECKS = ("native_benchmark", "strong_simple_baseline", "distinguishing_controls",
                 "fair_information_and_cost", "splits_seeds_uncertainty", "outcome_rules", "bounded_schedule")
RESULT_CHECKS = ("execution_and_parameters", "implementation_semantics", "native_measurement_and_baseline",
                 "optimization_validity", "confirmation_and_scope", "complete_inventory")
RANK_BASIS = ("problem_value", "mathematical_consequence", "closest_work_delta",
              "distinguishing_prediction", "feasibility_and_cost")
STAGES = ("math_verified", "code_verified", "design_verified", "results_verified", "verdict_verified")


def text(value, code="NONEMPTY_TEXT_REQUIRED"):
    if not isinstance(value, str) or not value.strip():
        C.fail(code)
    return value


def array(value, code, nonempty=True):
    if not isinstance(value, list) or (nonempty and not value):
        C.fail(code)
    return value


def obj(value, code="OBJECT_REQUIRED"):
    if not isinstance(value, dict):
        C.fail(code)
    return value


def equal(actual, expected, code):
    if actual != expected:
        C.fail(code)


def reference(root, value):
    obj(value, "REFERENCE_REQUIRED")
    if set(value) != {"path", "sha256"} or not re.fullmatch(r"[0-9a-f]{64}", str(value.get("sha256", ""))):
        C.fail("INVALID_REFERENCE")
    return C.verify_ref(root, value)


def read(root, ref, kind):
    value = C.load_file(reference(root, ref))
    obj(value)
    C.finite(value)
    C.scan(value)
    equal(value.get("kind"), kind, "PACKET_KIND_MISMATCH")
    equal(value.get("version"), "1.0.0", "PACKET_VERSION_UNSUPPORTED")
    return value


def required_ref(record, key):
    if key not in record:
        C.fail(key.upper() + "_REQUIRED")
    return record[key]


def refs(root, values):
    values = array(values, "EVIDENCE_REFS_REQUIRED")
    if len({C.canonical(r) for r in values}) != len(values):
        C.fail("DUPLICATE_REFERENCE")
    for value in values:
        reference(root, value)
    return values


def review(root, review_ref, subject, scope, artifact_ref, checks):
    value = read(root, review_ref, "method-review")
    equal(value.get("subject_id"), subject, "REVIEW_SUBJECT_MISMATCH")
    equal(value.get("scope"), scope, "REVIEW_SCOPE_MISMATCH")
    equal(value.get("artifact_ref"), artifact_ref, "REVIEW_ARTIFACT_MISMATCH")
    reference(root, value["artifact_ref"])
    text(value.get("reviewer"), "REVIEWER_REQUIRED")
    C.stamp(text(value.get("reviewed_at"), "REVIEW_TIME_REQUIRED"))
    equal(value.get("outcome"), "verified", "REVIEW_NOT_VERIFIED")
    records = obj(value.get("checks"), "REVIEW_CHECKS_REQUIRED")
    for key in checks:
        record = obj(records.get(key), "REVIEW_CHECK_REQUIRED")
        equal(record.get("status"), "verified", "REVIEW_CHECK_PENDING")
        text(record.get("rationale"), "SEMANTIC_REVIEW_RATIONALE_REQUIRED")
        refs(root, record.get("evidence_refs"))
    return value


def verify_math(root, entry):
    card_ref = required_ref(entry, "math_card_ref")
    card = read(root, card_ref, "math-card")
    equal(card.get("candidate_id"), entry["candidate_id"], "CANDIDATE_BINDING_MISMATCH")
    for key in ("formal_object", "method_expression", "distinguishing_prediction", "falsifier",
                "closest_alternative", "decisive_comparison"):
        text(card.get(key), "MATH_FIELD_REQUIRED")
    for assumption in array(card.get("assumptions"), "ASSUMPTIONS_REQUIRED"):
        text(assumption, "ASSUMPTION_REQUIRED")
    for op in array(card.get("operations"), "PERFORMED_MATH_OPERATIONS_REQUIRED"):
        obj(op)
        if not re.fullmatch(r"[A-H]0[1-6]", str(op.get("id", ""))):
            C.fail("UNKNOWN_MATH_OPERATION")
        for key in ("input", "condition", "output"):
            text(op.get(key), "OPERATION_DERIVATION_REQUIRED")
        if op.get("condition_status") not in ("established", "conditional"):
            C.fail("OPERATION_CONDITION_NOT_SUPPORTED")
    steps = array(card.get("derivation_steps"), "CONSEQUENTIAL_DERIVATION_REQUIRED")
    ids = []
    for step in steps:
        obj(step)
        ids.append(text(step.get("id"), "DERIVATION_STEP_ID_REQUIRED"))
        text(step.get("statement"), "DERIVATION_STATEMENT_REQUIRED")
        text(step.get("justification"), "DERIVATION_JUSTIFICATION_REQUIRED")
    if len(set(ids)) != len(ids):
        C.fail("DUPLICATE_DERIVATION_STEP")
    for source in array(card.get("source_refs", []), "INVALID_SOURCE_REFS", nonempty=False):
        reference(root, source)
    review(root, required_ref(entry, "math_review_ref"), entry["candidate_id"], "math", card_ref, MATH_CHECKS)
    return card


def verify_selection(root, batch, entries, pool_target, selection_target):
    selection_ref = required_ref(batch, "selection_ref")
    value = read(root, selection_ref, "method-selection")
    equal(value.get("batch_id"), batch["batch_id"], "BATCH_BINDING_MISMATCH")
    equal(value.get("pool_target"), pool_target, "POOL_TARGET_MISMATCH")
    equal(value.get("selection_target"), selection_target, "SELECTION_TARGET_MISMATCH")
    expected = [{"candidate_id": e["candidate_id"], "math_card_ref": e["math_card_ref"],
                 "math_review_ref": e["math_review_ref"]} for e in entries]
    equal(value.get("math_bindings"), expected, "MATH_BINDING_MISMATCH")
    ranking = array(value.get("ranking"), "RANKING_REQUIRED")
    ids = []
    for item in ranking:
        obj(item)
        ids.append(text(item.get("candidate_id")))
        reasons = obj(item.get("rationale"), "RANKING_RATIONALE_REQUIRED")
        for key in RANK_BASIS:
            text(reasons.get(key), "RANKING_BASIS_REQUIRED")
    if len(set(ids)) != len(ids) or set(ids) != {e["candidate_id"] for e in entries}:
        C.fail("RANKING_POOL_MISMATCH")
    equal(value.get("selected_ids"), ids[:selection_target], "TOP_SELECTION_MISMATCH")
    review(root, required_ref(batch, "selection_review_ref"), batch["batch_id"], "selection", selection_ref, RANK_CHECKS)
    return value["selected_ids"]


def verify_code(root, batch, entry, math):
    packet_ref = required_ref(entry, "implementation_ref")
    packet = read(root, packet_ref, "implementation-card")
    equal(packet.get("candidate_id"), entry["candidate_id"], "CANDIDATE_BINDING_MISMATCH")
    equal(packet.get("math_card_ref"), entry["math_card_ref"], "MATH_BINDING_MISMATCH")
    equal(packet.get("selection_ref"), batch["selection_ref"], "SELECTION_BINDING_MISMATCH")
    code_refs = refs(root, packet.get("code_refs"))
    equal(packet.get("code_version"), C.hashed(code_refs), "CODE_VERSION_MISMATCH")
    identity = obj(C.load_file(reference(root, required_ref(packet, "source_identity_ref"))),
                   "SOURCE_IDENTITY_REQUIRED")
    version = text(packet.get("method_version"), "METHOD_VERSION_REQUIRED")
    if not re.fullmatch(r"[0-9a-f]{64}", version):
        C.fail("INVALID_METHOD_VERSION")
    equal(identity.get("git_tree_digest"), version, "SOURCE_TREE_VERSION_MISMATCH")
    equal(identity.get("method_code_refs"), code_refs, "SOURCE_CODE_BINDING_MISMATCH")
    mapped = []
    for item in array(packet.get("mapping"), "MATH_CODE_MAPPING_REQUIRED"):
        obj(item)
        mapped.append(text(item.get("derivation_step")))
        if item.get("code_ref") not in code_refs:
            C.fail("MAPPING_CODE_IDENTITY_MISMATCH")
        text(item.get("location"), "CODE_LOCATION_REQUIRED")
        text(item.get("rationale"), "CODE_MAPPING_RATIONALE_REQUIRED")
    equal(set(mapped), {s["id"] for s in math["derivation_steps"]}, "DERIVATION_CODE_COVERAGE_INCOMPLETE")
    for check in array(packet.get("software_checks"), "SOFTWARE_CHECKS_REQUIRED"):
        obj(check)
        equal(check.get("status"), "passed", "SOFTWARE_CHECK_NOT_PASSED")
        for arg in array(check.get("command"), "CHECK_COMMAND_REQUIRED"):
            text(arg)
        equal(check.get("tested_code_refs"), code_refs, "TESTED_CODE_IDENTITY_MISMATCH")
        refs(root, check.get("log_refs"))
    review(root, required_ref(entry, "code_review_ref"), entry["candidate_id"], "code", packet_ref, CODE_CHECKS)
    return packet


def verify_design(root, batch, entry, code):
    packet_ref = required_ref(entry, "design_ref")
    packet = read(root, packet_ref, "experiment-design")
    equal(packet.get("candidate_id"), entry["candidate_id"], "CANDIDATE_BINDING_MISMATCH")
    for key in ("math_card_ref", "implementation_ref"):
        equal(packet.get(key), entry[key], key.upper() + "_BINDING_MISMATCH")
    equal(packet.get("selection_ref"), batch["selection_ref"], "SELECTION_BINDING_MISMATCH")
    protocol = obj(C.load_file(reference(root, required_ref(packet, "protocol_ref"))), "NATIVE_PROTOCOL_REQUIRED")
    if protocol.get("schema_id") not in ("gate-a-protocol", "full-validation-protocol"):
        C.fail("NATIVE_PROTOCOL_REQUIRED")
    C.validate(protocol)
    if protocol.get("protocol_digest") != C.protocol_hash(protocol) or not protocol.get("frozen_at"):
        C.fail("PROTOCOL_NOT_FROZEN")
    C.stamp(protocol["frozen_at"])
    equal(protocol.get("method_version"), code["method_version"], "METHOD_VERSION_MISMATCH")
    equal(obj(protocol.get("provenance_constraints"), "SOURCE_TREE_VERSION_REQUIRED").get("git_tree_digest"), code["method_version"],
          "METHOD_VERSION_NOT_BOUND_TO_IMPLEMENTATION")
    from _native_eval import verify_protocol
    verify_protocol(root, protocol)
    if "method_discovery" in protocol:
        discovery_batch, discovery_entry = verify_discovery(root, protocol)
        equal(discovery_batch["selection_ref"], batch["selection_ref"], "DISCOVERY_BINDING_MISMATCH")
        for key in ("math_card_ref", "math_review_ref", "implementation_ref", "code_review_ref"):
            equal(discovery_entry[key], entry[key], "DISCOVERY_BINDING_MISMATCH")
    contracts = protocol.get("native_eval_contracts") or {"main": protocol["native_eval_contract"]}
    for contract in contracts.values():
        treatment = contract["arm_requirements"]["treatment"]
        equal(treatment.get("implementation_refs"), code["code_refs"], "TREATMENT_CODE_BINDING_MISMATCH")
    review(root, required_ref(entry, "design_review_ref"), entry["candidate_id"], "design", packet_ref, DESIGN_CHECKS)
    return packet, protocol


def verify_result(root, entry, design, protocol):
    packet_ref = required_ref(entry, "result_ref")
    packet = read(root, packet_ref, "method-result")
    equal(packet.get("candidate_id"), entry["candidate_id"], "CANDIDATE_BINDING_MISMATCH")
    for key in ("design_ref", "implementation_ref"):
        equal(packet.get(key), entry[key], key.upper() + "_BINDING_MISMATCH")
    equal(packet.get("protocol_ref"), design["protocol_ref"], "RESULT_PROTOCOL_MISMATCH")
    gate = "gate-a" if protocol["schema_id"] == "gate-a-protocol" else "full-validation"
    state_key = "gate_a_protocol_ref" if gate == "gate-a" else "full_validation_protocol_ref"
    decision, _ = C.recompute_decision(root, {state_key: design["protocol_ref"]},
                                      {"decision_ref": required_ref(packet, "decision_ref")}, gate)
    review(root, required_ref(entry, "e04_review_ref"), entry["candidate_id"], "e04", packet_ref, RESULT_CHECKS)
    return decision, gate


def blocker(stage, error):
    return {"stage": stage, "code": error.code, "path": error.path}


def verify_batch(root, batch, through="results"):
    root = C.root_path(root)
    obj(batch)
    C.finite(batch)
    C.scan(batch)
    if through not in ("code", "design", "results"):
        C.fail("UNKNOWN_VERIFICATION_BOUNDARY")
    equal(batch.get("kind"), "method-verification-batch", "PACKET_KIND_MISMATCH")
    equal(batch.get("version"), "1.0.0", "PACKET_VERSION_UNSUPPORTED")
    text(batch.get("batch_id"), "BATCH_ID_REQUIRED")
    pool_target, selection_target = batch.get("pool_target", 20), batch.get("selection_target", 15)
    if type(pool_target) is not int or type(selection_target) is not int or not 1 <= selection_target <= pool_target <= 20:
        C.fail("INVALID_BATCH_TARGETS")
    if (pool_target, selection_target) != (20, 15):
        # Recorded scope, not an authorization token or proof of the user's identity.
        text(batch.get("target_override"), "EXPLICIT_TARGET_OVERRIDE_REQUIRED")
    entries = array(batch.get("candidates"), "CANDIDATES_REQUIRED", nonempty=False)
    if len(entries) > 20:
        C.fail("ACTIVE_CANDIDATE_CAP_EXCEEDED")
    ids = [text(obj(e).get("candidate_id"), "CANDIDATE_ID_REQUIRED") for e in entries]
    if len(set(ids)) != len(ids):
        C.fail("DUPLICATE_CANDIDATE_ID")
    rows, cards, fingerprints = [], {}, {}
    for entry in entries:
        cid = entry["candidate_id"]
        row = {"candidate_id": cid, "selected": False, **{key: False for key in STAGES},
               "outcome": "PENDING", "blockers": []}
        rows.append(row)
        try:
            card = verify_math(root, entry)
            cards[cid] = card
            fingerprint = C.hashed({key: card[key] for key in ("formal_object", "method_expression",
                                                               "distinguishing_prediction", "falsifier")})
            row["math_verified"] = True
            if fingerprint in fingerprints:
                other = fingerprints[fingerprint]
                other["math_verified"] = row["math_verified"] = False
                failure = blocker("math", C.Failure("DUPLICATE_CONSTRUCTION"))
                other["blockers"].append(failure)
                row["blockers"].append(failure)
            else:
                fingerprints[fingerprint] = row
        except C.Failure as error:
            row["blockers"].append(blocker("math", error))
    math_count = sum(r["math_verified"] for r in rows)
    pool_verified = len(entries) == pool_target and math_count == pool_target
    selection_verified, selected, selection_blockers = False, [], []
    try:
        if not pool_verified:
            C.fail("MATHEMATICAL_POOL_INCOMPLETE")
        selected = verify_selection(root, batch, entries, pool_target, selection_target)
        selection_verified = True
    except C.Failure as error:
        selection_blockers.append(blocker("selection", error))
    for entry, row in zip(entries, rows):
        if not selection_verified:
            if entry.get("implementation_ref"):
                row["blockers"].append(blocker("code", C.Failure("VERIFIED_SELECTION_REQUIRED")))
            continue
        if entry["candidate_id"] not in selected:
            continue
        row["selected"] = True
        stage = "code"
        try:
            code = verify_code(root, batch, entry, cards[entry["candidate_id"]])
            row["code_verified"] = True
            row["code_version"] = code["code_version"]
            row["method_version"] = code["method_version"]
            if through == "code":
                continue
            stage = "design"
            design, protocol = verify_design(root, batch, entry, code)
            row["design_verified"] = True
            if through == "design":
                continue
            stage = "results"
            decision, gate = verify_result(root, entry, design, protocol)
            if gate == "gate-a":
                row["gate_a_outcome"] = decision["outcome"]
                C.fail("FULL_COMPARISON_REQUIRED")
            row["outcome"] = decision["outcome"]
            row["reason_codes"] = decision["reason_codes"]
            if decision["outcome"] in {"PASS", "KILL", "REVISE"}:
                row["results_verified"] = True
                row["verdict_verified"] = decision["outcome"] in {"PASS", "KILL"}
            else:
                C.fail("SCIENTIFIC_OUTCOME_INCONCLUSIVE")
        except C.Failure as error:
            row["blockers"].append(blocker(stage, error))
    counts = {key: sum(r[key] for r in rows) for key in STAGES}
    counts.update(candidates=len(rows), selected=len(selected),
                  succeeded=sum(r["verdict_verified"] and r["outcome"] == "PASS" for r in rows),
                  failed=sum(r["verdict_verified"] and r["outcome"] == "KILL" for r in rows))
    return {"kind": "method-verification-report", "version": "1.0.0", "batch_id": batch["batch_id"], "through": through,
            "batch_digest": C.hashed(batch), "pool_target": pool_target, "selection_target": selection_target,
            "pool_deficit": max(0, pool_target - math_count),
            "selection_deficit": max(0, selection_target - math_count),
            "math_pool_verified": pool_verified, "selection_verified": selection_verified,
            "selection_blockers": selection_blockers, "counts": counts, "candidates": rows,
            "verification_scope": "Pinned evidence and documented semantic reviews; not an automated proof of mathematics. "
                                  "Empirical verdicts require existing native evaluation with trusted live replay and full comparison. "
                                  "This report does not replace Natural Gate 0, IPCG, Gate A, G01 or E04."}


def verify_discovery(root, protocol):
    """Native admission check for a newly tagged method, without circular refs.

    The immutable discovery batch ends at code; designs/results live in a later
    batch revision referencing the frozen protocol. Legacy protocols stay intact.
    """
    link = obj(protocol.get("method_discovery"), "METHOD_DISCOVERY_REQUIRED")
    cid = text(link.get("candidate_id"), "CANDIDATE_ID_REQUIRED")
    batch = read(root, required_ref(link, "batch_ref"), "method-verification-batch")
    report = verify_batch(root, batch, through="code")
    rows = [r for r in report["candidates"] if r["candidate_id"] == cid]
    if not report["selection_verified"] or not rows or not rows[0]["code_verified"]:
        C.fail("METHOD_DISCOVERY_NOT_VERIFIED")
    entry = next(e for e in batch["candidates"] if e["candidate_id"] == cid)
    code = read(root, entry["implementation_ref"], "implementation-card")
    equal(protocol.get("method_version"), code["method_version"], "METHOD_VERSION_MISMATCH")
    equal(obj(protocol.get("provenance_constraints"), "SOURCE_TREE_VERSION_REQUIRED").get("git_tree_digest"),
          code["method_version"], "METHOD_VERSION_NOT_BOUND_TO_IMPLEMENTATION")
    contracts = obj(protocol.get("native_eval_contracts") or {"main": protocol.get("native_eval_contract")},
                    "NATIVE_EVAL_CONTRACT_REQUIRED")
    for contract in contracts.values():
        obj(contract, "NATIVE_EVAL_CONTRACT_REQUIRED")
        arm = obj(obj(contract.get("arm_requirements")).get("treatment"))
        equal(arm.get("implementation_refs"), code["code_refs"], "TREATMENT_CODE_BINDING_MISMATCH")
    return batch, entry


def meets(report, stage, candidate_id=None):
    if stage in {"math_pool_verified", "selection_verified"}:
        if candidate_id is not None:
            C.fail("BATCH_REQUIREMENT_HAS_NO_CANDIDATE")
        return report[stage]
    if candidate_id is not None:
        rows = [r for r in report["candidates"] if r["candidate_id"] == candidate_id]
        if not rows:
            C.fail("UNKNOWN_CANDIDATE_ID")
    else:
        rows = [r for r in report["candidates"] if r["selected"]]
    return bool(rows) and all(r[stage] for r in rows)


def before_action(root, batch, action, candidate_id):
    """Check the required evidence before generating or dispatching a method.

    A ready boundary is neither launch authority nor a replacement for the
    canonical project ledger, Natural Gate 0/IPCG or native scoring checks.
    The action fixes the required stage; conversational metadata cannot lower it.
    """
    boundaries = {"code": ("code", "selection_verified"),
                  "experiment-design": ("code", "code_verified"),
                  "dispatch": ("design", "design_verified"),
                  "verdict": ("results", "verdict_verified")}
    if action not in boundaries:
        C.fail("UNKNOWN_WORKFLOW_BOUNDARY")
    text(candidate_id, "WORKFLOW_CANDIDATE_REQUIRED")
    through, prerequisite = boundaries[action]
    report = verify_batch(root, batch, through=through)
    rows = [r for r in report["candidates"] if r["candidate_id"] == candidate_id]
    if not rows:
        C.fail("UNKNOWN_CANDIDATE_ID")
    row = rows[0]
    ready = report["selection_verified"] and row["selected"]
    if prerequisite != "selection_verified":
        ready = ready and row[prerequisite]
    reasons = []
    if not report["selection_verified"]:
        reasons.extend(b["code"] for b in report["selection_blockers"])
    elif not row["selected"]:
        reasons.append("CANDIDATE_NOT_SELECTED")
    elif not ready:
        reasons.extend(b["code"] for b in row["blockers"])
    report["workflow_boundary"] = {"action": action, "candidate_id": candidate_id,
        "required_stage": prerequisite, "ready": bool(ready), "reason_codes": list(dict.fromkeys(reasons)),
        "batch_digest": report["batch_digest"], "authorization": "not_granted",
        "scope": "Current method evidence at this boundary; existing project/scientific/host gates still apply."}
    return report


def verify_run_design(root, plan, protocol):
    """A tagged new method needs its complete design, not only code receipts."""
    proof = plan["provenance"].get("method_verification_ref")
    if proof is None:
        C.fail("RUN_METHOD_DESIGN_EVIDENCE_REQUIRED")
    batch = read(root, proof, "method-verification-batch")
    candidate_id = protocol["method_discovery"]["candidate_id"]
    report = before_action(root, batch, "dispatch", candidate_id)
    if not report["workflow_boundary"]["ready"]:
        C.fail("RUN_METHOD_DESIGN_NOT_VERIFIED")
    discovery = read(root, protocol["method_discovery"]["batch_ref"], "method-verification-batch")
    equal(batch["batch_id"], discovery["batch_id"], "RUN_METHOD_DISCOVERY_BINDING_MISMATCH")
    equal(batch["selection_ref"], discovery["selection_ref"], "RUN_METHOD_DISCOVERY_BINDING_MISMATCH")
    entry = next(e for e in batch["candidates"] if e["candidate_id"] == candidate_id)
    design = read(root, entry["design_ref"], "experiment-design")
    equal(design["protocol_ref"], plan["protocol_ref"], "RUN_METHOD_DESIGN_PROTOCOL_MISMATCH")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--batch", required=True, help="Project-relative verification batch JSON/YAML")
    parser.add_argument("--require", choices=("math_pool_verified", "selection_verified", *STAGES))
    parser.add_argument("--through", choices=("code", "design", "results"), default="results")
    parser.add_argument("--candidate", help="Candidate-specific requirement; otherwise all selected candidates")
    parser.add_argument("--before", choices=("code", "experiment-design", "dispatch", "verdict"),
                        help="Check the fixed prerequisite before this candidate action")
    args = parser.parse_args(argv)
    try:
        root = C.root_path(args.root)
        batch = C.load_file(C.safe_path(root, args.batch))
        if args.before:
            report = before_action(root, batch, args.before, args.candidate)
            passed = report["workflow_boundary"]["ready"]
        else:
            report = verify_batch(root, batch, through=args.through)
            passed = not args.require or meets(report, args.require, args.candidate)
        print(C.canonical(report))
        return 0 if passed else 2
    except C.Failure as error:
        print(C.canonical({"status": "blocked", "code": error.code, "path": error.path}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
