"""Versioned replay, cross-artifact closure and transactional migration.

Captured JSON is an adapter format, not a substitute for primary source text.
Unsupported provider formats require an explicit versioned parser.
"""
import copy
import hashlib
import urllib.parse
from pathlib import Path
import _autoresearch as C

VERSIONS = ("capture-json-v1", "work-v1", "identifier-v1")
FAMILIES = {"problem","mechanism","objective","operator","differentiator","assumptions",
            "observable_behavior","component_combinations","historical_aliases","cross_domain"}

def replay_search(root, search):
    C.validate(search,"search-run")
    if (search["parser_version"],search["canonicalization_version"],search["deduplication_version"]) != VERSIONS:
        C.fail("UNSUPPORTED_SEARCH_NORMALIZER")
    if not search["query_strings"] or not search["query_families"] or not search["raw_capture_refs"]:
        C.fail("SEARCH_METADATA_INCOMPLETE")
    if search["pagination"].get("complete") is not True:
        C.fail("SEARCH_PAGINATION_INCOMPLETE")
    C.stamp(search["searched_at"])
    try:
        cutoff=C.dt.date.fromisoformat(search["cutoff"])
    except ValueError:
        C.fail("SEARCH_CUTOFF_INVALID")
    if search["requested_count"] < 1 or search["returned_count"] < 0:
        C.fail("SEARCH_COUNT_INVALID")
    works, units={},{}
    raw_count=0
    for capture_ref in search["raw_capture_refs"]:
        capture=C.load_file(C.verify_ref(root,capture_ref))
        C.schema_check(capture,{"type":"object","required":["works","evidence_units"],
            "properties":{"works":{"type":"array"},"evidence_units":{"type":"array"}}})
        C.scan(capture);C.finite(capture)
        raw_count+=len(capture["works"])
        for raw in capture["works"]:
            work=copy.deepcopy(C.validate(raw,"work-record"))
            if work.get("publication_date"):
                try:
                    date=C.dt.date.fromisoformat(work["publication_date"])
                except ValueError:
                    C.fail("WORK_PUBLICATION_DATE_INVALID")
                if date>cutoff:C.fail("WORK_AFTER_SEARCH_CUTOFF")
            key=work["work_id"]
            if key.startswith("doi:"):
                key=key.lower()
            work["work_id"]=key
            work["title"]=" ".join(work["title"].split())
            url=urllib.parse.urlsplit(work["canonical_url"])
            if url.scheme not in {"http","https"} or not url.hostname:C.fail("WORK_URL_INVALID")
            work["canonical_url"]=urllib.parse.urlunsplit((url.scheme.lower(),url.netloc.lower(),url.path,url.query,""))
            if key in works and works[key]!=work:C.fail("CONFLICTING_CANONICAL_WORK")
            works[key]=work
            for evidence in work["evidence_refs"]:C.verify_ref(root,evidence)
        for raw in capture["evidence_units"]:
            unit=copy.deepcopy(C.validate(raw,"evidence-unit"));key=unit["evidence_id"]
            if unit["work_id"].startswith("doi:"):unit["work_id"]=unit["work_id"].lower()
            if key in units and units[key]!=unit:C.fail("CONFLICTING_EVIDENCE_ID")
            units[key]=unit
            for locator in unit["locators"]:
                if not locator.get("artifact_ref") or not any(locator.get(k) for k in ("section","page","figure","table","lines","span")):
                    C.fail("EVIDENCE_LOCATOR_REQUIRED")
                C.verify_ref(root,locator["artifact_ref"])
    if raw_count!=search["returned_count"]:C.fail("SEARCH_CAPTURE_COUNT_MISMATCH")
    if any(u["work_id"] not in works for u in units.values()):C.fail("ORPHAN_EVIDENCE_WORK")
    normalized={"works":[works[k] for k in sorted(works)],"evidence_units":[units[k] for k in sorted(units)]}
    if C.hashed(normalized)!=search["normalized_output_digest"]:C.fail("SEARCH_NORMALIZED_DIGEST_MISMATCH")
    # Preserve work-v1 bytes/IDs and the captured digest for historical replay.
    # Version-aware identity is additional metadata, not a rewritten capture.
    from _literature import canonicalize_search_records
    normalized["canonical_identity"] = canonicalize_search_records(root, normalized["works"])
    return normalized


def verify_problem_eval_binding(root, protocol, parent_ref):
    parent = C.verify_ref(root, parent_ref, "parent-problem-card")
    if parent.get("contribution_type") == "theory":
        C.fail("FORMAL_CONFIRMATION_ADAPTER_REQUIRED")
    benchmark = parent.get("benchmark_contract")
    native = protocol.get("native_eval_contract")
    if not isinstance(benchmark, dict) or not isinstance(native, dict):
        C.fail("PARENT_BENCHMARK_CONTRACT_REQUIRED")
    if any(benchmark.get(k) != native.get(v) for k, v in (
        ("benchmark_id", "benchmark_id"), ("version", "benchmark_revision"), ("primary_metric", "primary_metric"))):
        C.fail("PARENT_NATIVE_EVALUATION_MISMATCH")
    splits = benchmark.get("allowed_eval_splits", [benchmark.get("evaluation_split")])
    if not isinstance(splits, list) or not splits or benchmark.get("evaluation_split") not in splits or native.get("split") not in splits:
        C.fail("PARENT_NATIVE_SPLIT_MISMATCH")
    if not any(r.get("indispensable") and r.get("metric") == native["primary_metric"] for r in protocol["criteria"]):
        C.fail("PARENT_PRIMARY_ENDPOINT_CRITERION_REQUIRED")
    return parent

def verify_collision_proofs(root, report):
    if not report.get("candidate_ref"):C.fail("FROZEN_COLLISION_CLAIMS_REQUIRED")
    candidate=C.verify_ref(root,report["candidate_ref"],"idea-atom")
    if candidate["candidate_id"]!=report["candidate_id"]:C.fail("STALE_COLLISION_CANDIDATE")
    claim_ids=set(candidate.get("claim_ids",[candidate["dominant_claim"]]))
    if set(report.get("essential_claim_coverage",[]))!=claim_ids:C.fail("COLLISION_CLAIM_COVERAGE_INCOMPLETE")
    coverage=report["coverage"]
    if not FAMILIES.issubset(coverage["required_families"]):C.fail("COLLISION_QUERY_PROTOCOL_INCOMPLETE")
    normalized=[replay_search(root,C.verify_ref(root,r,"search-run")) for r in report["search_run_refs"]]
    primary={w["work_id"]:w for batch in normalized for w in batch["works"]}
    for wref in report["work_refs"]:
        work=C.verify_ref(root,wref,"work-record")
        if primary.get(work["work_id"])!=work:C.fail("WORK_NOT_BOUND_TO_CAPTURE")
        if not work.get("publication_date"):C.fail("PRIORITY_DATE_REQUIRED")
    batches=report.get("coverage_batches",[])
    if len(batches)<2 or any(b.get("new_high_risk_families")!=[] or not b.get("search_run_refs") for b in batches[-2:]):
        C.fail("SEARCH_STOP_EVIDENCE_REQUIRED")
    known={(r["path"],r["sha256"]) for r in report["search_run_refs"]}
    if any((r["path"],r["sha256"]) not in known for b in batches for r in b.get("search_run_refs",[])):
        C.fail("COVERAGE_BATCH_NOT_CAPTURED")
    last_sets=[{(r["path"],r["sha256"]) for r in b["search_run_refs"]} for b in batches[-2:]]
    if last_sets[0]&last_sets[1]:C.fail("SEARCH_STOP_ROUNDS_NOT_DISTINCT")
    audits=report.get("claim_audits",[])
    if {a.get("claim_id") for a in audits}!=claim_ids:C.fail("COLLISION_CLAIM_PROOFS_REQUIRED")
    contexts=set()
    expected_roles={"retriever","prosecutor","defender","equivalence_verifier","provenance_auditor","adjudicator"}
    for audit in audits:
        if not audit.get("evidence_refs") or not audit.get("work_ids") or not set(audit["work_ids"]).issubset(primary):
            C.fail("COLLISION_PRIMARY_PROOF_REQUIRED")
        for evidence in audit["evidence_refs"]:
            unit=C.verify_ref(root,evidence,"evidence-unit")
            if unit["claim"]!=audit["claim_id"] or unit["work_id"] not in audit["work_ids"]:
                C.fail("COLLISION_PROOF_CLAIM_MISMATCH")
        roles=audit.get("role_refs",{})
        if set(roles)!=expected_roles:C.fail("INDEPENDENT_COLLISION_ROLES_REQUIRED")
        for role,r in roles.items():
            record=C.load_file(C.verify_ref(root,r));C.scan(record)
            if record.get("role")!=role or not record.get("context_id") or not record.get("evidence_refs"):
                C.fail("COLLISION_ROLE_PROOF_REQUIRED")
            context=record["context_id"]
            if context in contexts:C.fail("COLLISION_ROLES_NOT_INDEPENDENT")
            contexts.add(context)
            for evidence in record["evidence_refs"]:C.verify_ref(root,evidence)
            if role=="adjudicator" and any(k in record for k in ("generator_identity","vote_counts","previous_verdict")):
                C.fail("ADJUDICATION_NOT_BLINDED")
        if report["outcome"]=="KILL" and not audit.get("one_primary_covers_essential_elements"):
            C.fail("KILL_ESSENTIAL_ELEMENTS_NOT_COVERED")
    return report

def verify_claim_closure(root, snapshot, protocol, decision):
    indispensable=set(protocol.get("indispensable_claim_ids",[]))
    if not indispensable or not indispensable.issubset(snapshot["claim_ids"]):C.fail("CLAIM_EVIDENCE_CLOSURE_REQUIRED")
    records=snapshot.get("closure",{}).get("claims",[])
    mapping={r.get("claim_id"):r for r in records if isinstance(r,dict)}
    if len(mapping)!=len(records):C.fail("DUPLICATE_CLAIM_CLOSURE")
    closed={claim for claim,item in mapping.items() if item.get("status")=="closed"}
    frozen=set(protocol.get("claim_set",[]))
    if not (indispensable|closed).issubset(frozen & set(snapshot["claim_ids"])):
        C.fail("CLAIM_OUTSIDE_FROZEN_SCOPE")
    results={}
    for r in decision["criteria_results"]:results.setdefault(r["criterion_id"],[]).append(r)
    rules={r["id"]:r for r in protocol["criteria"]}
    retained={(r["path"],r["sha256"]) for r in snapshot["artifact_refs"]+snapshot["negative_result_refs"]}
    manifest_refs = decision.get("run_manifest_refs", [])
    manifests = {C.verify_ref(root, r, "run-manifest")["run_id"]: r for r in manifest_refs}
    for claim in sorted(indispensable|closed):
        item=mapping.get(claim)
        if not item or item.get("status")!="closed" or item.get("unresolved_confounds")!=[]:
            C.fail("UNCLOSED_INDISPENSABLE_CLAIM")
        own={key for key,r in rules.items() if r.get("claim_id")==claim}
        needed={key for key in own if rules[key]["indispensable"]} if claim in indispensable else own
        supplied=set(item.get("criterion_ids",[]))
        if not needed or not needed.issubset(supplied) or not supplied.issubset(own):
            C.fail("CLAIM_CRITERION_CLOSURE_REQUIRED")
        if any(key not in results or not all(r["passed"] for r in results[key]) for key in supplied):C.fail("CLAIM_CRITERION_NOT_PASSED")
        if not item.get("evidence_refs"):C.fail("CLAIM_RAW_EVIDENCE_REQUIRED")
        for evidence in item["evidence_refs"]:
            if (evidence["path"],evidence["sha256"]) not in retained:C.fail("CLAIM_PROOF_NOT_IN_SNAPSHOT")
            C.verify_ref(root,evidence)
        supplied_refs = {(r["path"], r["sha256"]) for r in item["evidence_refs"]}
        for key in supplied:
            for result in results[key]:
                run_ids = result.get("run_ids", [])
                if not run_ids or not set(run_ids).issubset(set(decision["eligible_run_ids"]) & set(manifests)):
                    C.fail("CLAIM_CRITERION_RUN_PROOF_REQUIRED")
                required_refs = {(manifests[r]["path"], manifests[r]["sha256"]) for r in run_ids}
                if not required_refs.issubset(supplied_refs):
                    C.fail("CLAIM_PROOF_NOT_BOUND_TO_ELIGIBLE_RUNS")
                proof_refs = {(r["path"], r["sha256"]) for r in result.get("proof_refs", [])}
                for rid in run_ids:
                    run = C.verify_ref(root, manifests[rid], "run-manifest")
                    if run["protocol_digest"] != protocol["protocol_digest"]:
                        C.fail("CLAIM_RUN_PROTOCOL_MISMATCH")
                    native_refs = list(run.get("native_eval_receipts", {}).values())
                    if not native_refs or not {(r["path"], r["sha256"]) for r in native_refs}.issubset(proof_refs):
                        C.fail("CLAIM_NATIVE_RESULT_PROOF_REQUIRED")
                for evidence in result.get("proof_refs", []):
                    C.verify_ref(root, evidence)
    if snapshot.get("closure",{}).get("result_inventory_run_ids")!=sorted(decision["eligible_run_ids"]):
        C.fail("WRITING_RESULT_INVENTORY_INCOMPLETE")
    return True

def verify_validation_binding(root, state, events=None):
    if events is None:
        _, events, _ = C.project(root)
    if state.get("validation_state") != "pass" or state.get("validated_snapshot_ref") != state.get("evidence_snapshot_ref"):
        C.fail("VALIDATION_BINDING_STALE_OR_MISSING")
    anchored = [e for e in events if e["event_type"] == "transition" and e["payload"].get("outcome") == "full_validation_PASS"]
    if not anchored:
        C.fail("ANCHORED_FULL_VALIDATION_REQUIRED")
    event = anchored[-1]
    decision_ref = state.get("validation_decision_ref")
    if not decision_ref or event["payload"].get("decision_ref") != decision_ref or event["payload"].get("evidence_snapshot_ref") != state["evidence_snapshot_ref"]:
        C.fail("VALIDATION_BINDING_NOT_ANCHORED")
    snapshot = C.verify_snapshot(root, state["evidence_snapshot_ref"])
    if snapshot.get("closure", {}).get("decision_ref") != decision_ref:
        C.fail("CLOSURE_NOT_BOUND_TO_VALIDATION_DECISION")
    protocol = C.verify_ref(root, state["full_validation_protocol_ref"], "full-validation-protocol")
    decision = C.verify_ref(root, decision_ref, "full-validation-decision")
    if event["next_state"].get("full_validation_protocol_ref") != state["full_validation_protocol_ref"] or decision["outcome"] != "PASS" or decision["protocol_digest"] != protocol["protocol_digest"]:
        C.fail("VALIDATION_PROTOCOL_BINDING_MISMATCH")
    if state.get("validated_method_version") != protocol.get("method_version") or state.get("validated_project_version") != protocol.get("project_version"):
        C.fail("VALIDATION_VERSION_BINDING_MISMATCH")
    from _writing import validate_claim_cards
    validate_claim_cards(protocol)
    verify_claim_closure(root, snapshot, protocol, decision)
    return snapshot, protocol, decision

def verify_handoff(root,manifest):
    C.validate(manifest,"handoff")
    state,events,_=C.project(root)
    if state["validation_state"]!="pass" or state["current_route"] not in {"prepare_writing","retarget"}:
        C.fail("HANDOFF_FULL_VALIDATION_REQUIRED")
    if manifest["evidence_snapshot_ref"]!=state["evidence_snapshot_ref"]:C.fail("HANDOFF_SNAPSHOT_NOT_CURRENT")
    snapshot, protocol, decision = verify_validation_binding(root, state, events)
    if decision["outcome"]!="PASS" or decision["protocol_digest"]!=protocol["protocol_digest"]:C.fail("HANDOFF_DECISION_MISMATCH")
    verify_claim_closure(root,snapshot,protocol,decision)
    closed={r["claim_id"] for r in snapshot["closure"]["claims"] if r["status"]=="closed"}
    if not manifest["allowed_claims"] or not set(manifest["allowed_claims"]).issubset(closed):C.fail("HANDOFF_UNSUPPORTED_CLAIM")
    if manifest["method_version"] != state["validated_method_version"] or manifest["project_version"] != state["validated_project_version"]:
        C.fail("HANDOFF_VERSION_NOT_VALIDATED")
    if manifest["stage"] != state["writing_state"]:
        C.fail("HANDOFF_STAGE_NOT_CURRENT")
    frozen_cards = {c["claim_id"]: c for c in protocol["claim_cards"]}
    cards = {c.get("claim_id"): c for c in manifest.get("claim_cards", []) if isinstance(c, dict)}
    if len(cards) != len(manifest.get("claim_cards", [])) or set(cards) != set(manifest["allowed_claims"]) or any(cards[c] != frozen_cards[c] for c in cards):
        C.fail("HANDOFF_CLAIM_SCOPE_NOT_FROZEN")
    if set(manifest["allowed_claims"])&set(manifest["forbidden_claims"]):C.fail("HANDOFF_CLAIM_CONTRADICTION")
    if manifest["provenance"].get("evidence_mode")!=protocol["evidence_mode"]:C.fail("HANDOFF_EVIDENCE_MODE_REQUIRED")
    negative={(r["path"],r["sha256"]) for r in manifest["negative_result_refs"]}
    if not {(r["path"],r["sha256"]) for r in snapshot["negative_result_refs"]}.issubset(negative):C.fail("HANDOFF_NEGATIVE_EVIDENCE_MISSING")
    for r in manifest["source_refs"]+manifest["negative_result_refs"]:C.verify_ref(root,r)
    return manifest

def migrate_project(root, target, transform=None):
    if target!=C.VERSION:C.fail("UNSUPPORTED_MIGRATION_READ_ONLY")
    # Validate the canonical ledger first. A stale projection can be reconstructed.
    events=C.verify_chain((root/"event-ledger.jsonl").read_text())
    anchor=C.validate(C.load_file(root/"ledger-anchor.json"),"ledger-anchor")
    if anchor["head_digest"]!=events[-1]["event_digest"] or anchor["sequence"]!=len(events):C.fail("LEDGER_TAIL_OR_ANCHOR_MISMATCH")
    files={};size=0
    for path in sorted(root.rglob("*")):
        if path.is_symlink():C.fail("SYMLINK_FORBIDDEN")
        if ".backups" in path.parts or path.name in {".state.lock",".pending-commit.json"}:continue
        if path.is_file():
            content=path.read_text();C.scan(content);size+=len(content.encode())
            if size>16*1024*1024:C.fail("BACKUP_REQUIRES_HOST_DURABLE_STORAGE")
            files[path.relative_to(root).as_posix()]=content
    backup=C.envelope("backup-manifest",files=files,source_head=anchor["head_digest"],source_sequence=anchor["sequence"])
    backup_digest=C.hashed(backup);backup_path=".backups/"+backup_digest+".json"
    C.validate(backup);C.atomic(C.safe_path(root,backup_path),C.canonical(backup)+"\n")
    try:
        candidate=transform(copy.deepcopy(files)) if transform else copy.deepcopy(files)
        if not isinstance(candidate,dict) or set(candidate)!=set(files):C.fail("MIGRATION_FILE_SET_CHANGED")
        replay=C.verify_chain(candidate["event-ledger.jsonl"])
        if replay[-1]["next_state"]!=events[-1]["next_state"]:C.fail("MIGRATION_STATE_SEMANTICS_CHANGED")
        for path,text in candidate.items():
            C.safe_path(root,path);C.scan(text)
            if path not in {"research-state.json"} and text!=files[path]:C.fail("SAME_VERSION_MIGRATION_CANNOT_CHANGE_EVIDENCE")
        candidate["research-state.json"]=C.canonical(events[-1]["next_state"])+"\n"
    except Exception:
        C.fail("MIGRATION_FAILED_ROLLED_BACK")
    C.commit_files(root,candidate,anchor["head_digest"])
    C.project(root)
    return {"status":"supported_version_preserved","version":target,"backup_digest":backup_digest,"backup_path":backup_path,
            "sequence":len(events)}
