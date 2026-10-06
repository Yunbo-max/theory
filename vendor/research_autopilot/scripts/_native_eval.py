"""Verify existing native benchmark contracts and replay pinned native scorers.

No metric is computed here. Metric values are extracted from the native scorer's
output at frozen paths. A trusted host must inject a live replay callable; stored
receipts, hashes and pass booleans do not authorize or prove scorer execution.
"""
import collections
import copy
import hashlib
import math
import operator
import uuid
import urllib.parse
from pathlib import Path
import _autoresearch as C

_OPERATORS = {"ge": operator.ge, "gt": operator.gt, "le": operator.le, "lt": operator.lt, "eq": operator.eq}


def _cwd(root, relative):
    return root if relative == "." else C.safe_path(root, relative)


def _refs(root, values, code="NATIVE_SOURCE_REFS_REQUIRED"):
    if not isinstance(values, list) or not values:
        C.fail(code)
    for item in values:
        C.schema_check(item, {"type":"object", "required":["path","sha256"], "properties":{
            "path":{"type":"string","minLength":1}, "sha256":{"type":"string","pattern":"^[0-9a-f]{64}$"}},
            "additionalProperties":False})
        C.verify_ref(root, item)
    return values


def _path(value, steps, code="NATIVE_OUTPUT_PATH_UNAVAILABLE"):
    if not isinstance(steps, list): C.fail("NATIVE_OUTPUT_PATH_INVALID")
    for step in steps:
        try:
            if isinstance(value, list) and isinstance(step, int) and not isinstance(step, bool) and step >= 0:
                value = value[step]
            elif isinstance(value, dict) and isinstance(step, str):
                value = value[step]
            else:
                C.fail(code)
        except (IndexError, KeyError):
            C.fail(code)
    return value


def _number(value, code="NATIVE_METRIC_NOT_NUMERIC"):
    if not isinstance(value, (int, float)) or isinstance(value, bool): C.fail(code)
    try: valid = math.isfinite(value)
    except (OverflowError,ValueError): valid = False
    if not valid: C.fail(code)
    return value


def sample_manifest_ref(contract):
    """The effective released sample manifest, including a frozen native subset."""
    return contract.get("selection", {}).get("selected_manifest_ref", contract["sample_manifest_ref"])


def _samples(root, contract):
    manifest = C.load_file(C.verify_ref(root, sample_manifest_ref(contract)))
    required = ("benchmark_id","benchmark_revision","split","sample_ids","denominator","predictions_per_sample",
                "labels_or_tests_ref","sampling","budget")
    if not isinstance(manifest, dict) or any(k not in manifest for k in required): C.fail("NATIVE_SAMPLE_MANIFEST_INCOMPLETE")
    for key in ("benchmark_id","benchmark_revision","split","labels_or_tests_ref","sampling","budget"):
        if manifest[key] != contract[key]: C.fail("NATIVE_SAMPLE_MANIFEST_IDENTITY_MISMATCH")
    ids = manifest["sample_ids"]
    if not isinstance(ids, list) or not ids or any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        C.fail("NATIVE_SAMPLE_IDENTITIES_INVALID")
    if _number(manifest["denominator"], "NATIVE_DENOMINATOR_INVALID") <= 0: C.fail("NATIVE_DENOMINATOR_INVALID")
    repetitions = manifest["predictions_per_sample"]
    if not isinstance(repetitions, int) or isinstance(repetitions, bool) or repetitions < 1:
        C.fail("NATIVE_SAMPLING_INVALID")
    C.verify_ref(root, manifest["labels_or_tests_ref"])
    return manifest


def _selection(root, contract, definition):
    selection = contract.get("selection")
    if selection is None: return
    capability = definition.get("subset_capability")
    if not isinstance(capability,dict) or not capability.get("source_ref"):
        C.fail("NATIVE_SUBSET_CAPABILITY_REQUIRED")
    allowed = capability.get("allowed_rules")
    if not isinstance(allowed,list) or not allowed or any(not isinstance(rule,str) for rule in allowed):
        C.fail("NATIVE_SUBSET_CAPABILITY_REQUIRED")
    if selection["capability_ref"] != capability["source_ref"] or selection["rule"] not in allowed:
        C.fail("NATIVE_SUBSET_CAPABILITY_REQUIRED")
    C.verify_ref(root,selection["capability_ref"])
    if selection["source_manifest_ref"] != contract["sample_manifest_ref"]: C.fail("NATIVE_SUBSET_SOURCE_MISMATCH")
    full = C.load_file(C.verify_ref(root,contract["sample_manifest_ref"]))
    if not isinstance(full,dict) or not isinstance(full.get("sample_ids"),list): C.fail("NATIVE_SAMPLE_MANIFEST_INCOMPLETE")
    ids = full["sample_ids"]
    if not ids or any(not isinstance(i,str) or not i for i in ids) or len(set(ids)) != len(ids):
        C.fail("NATIVE_SAMPLE_IDENTITIES_INVALID")
    count = selection["count"]
    if count > len(ids): C.fail("NATIVE_SUBSET_COUNT_INVALID")
    if selection["rule"] == "prefix": expected = ids[:count]
    elif selection["rule"] == "sha256_rank":
        expected = sorted(ids,key=lambda sample:hashlib.sha256(C.canonical([selection["seed"],sample]).encode()).hexdigest())[:count]
    else: C.fail("NATIVE_SUBSET_RULE_UNSUPPORTED")
    selected = C.load_file(C.verify_ref(root,selection["selected_manifest_ref"]))
    if not isinstance(selected,dict) or selected.get("sample_ids") != expected: C.fail("NATIVE_SUBSET_SELECTION_MISMATCH")
    if capability.get("denominator_rule") != "sample_count" or full.get("denominator") != len(ids):
        C.fail("NATIVE_SUBSET_DENOMINATOR_UNSUPPORTED")
    if selected.get("denominator") != count: C.fail("NATIVE_DENOMINATOR_MISMATCH")
    source_metadata = {k:v for k,v in full.items() if k not in {"sample_ids","denominator"}}
    selected_metadata = {k:v for k,v in selected.items() if k not in {"sample_ids","denominator"}}
    if source_metadata != selected_metadata: C.fail("NATIVE_SUBSET_METADATA_MISMATCH")


def _scorer(root, scorer):
    required = ("kind","identity","revision","source_refs","code_refs","command","cwd","output","denominator_path")
    if not isinstance(scorer, dict) or any(k not in scorer for k in required): C.fail("NATIVE_SCORER_IDENTITY_REQUIRED")
    if scorer["kind"] not in {"official","faithful_harness"}: C.fail("NATIVE_SCORER_KIND_UNSUPPORTED")
    if any(not isinstance(scorer[key], str) or not scorer[key] for key in ("identity","revision","cwd")):
        C.fail("NATIVE_SCORER_IDENTITY_REQUIRED")
    _refs(root, scorer["source_refs"]); _refs(root, scorer["code_refs"], "NATIVE_SCORER_CODE_REQUIRED")
    argv = scorer["command"]
    if not isinstance(argv, list) or not argv or any(not isinstance(a, str) or not a for a in argv):
        C.fail("NATIVE_SCORER_COMMAND_REQUIRED")
    _cwd(root, scorer["cwd"])
    output = scorer["output"]
    if not isinstance(output, dict) or output.get("format") != "json" or output.get("source") not in {"stdout","file"}:
        C.fail("NATIVE_OUTPUT_FORMAT_UNSUPPORTED")
    if output["source"] == "file":
        if not output.get("path"): C.fail("NATIVE_OUTPUT_FILE_REQUIRED")
        C.safe_path(root, output["path"])
        if "{output}" not in argv: C.fail("NATIVE_OUTPUT_PLACEHOLDER_REQUIRED")
    for arg in argv:
        rest = arg
        for token in ("{predictions}","{samples}","{labels}","{output}","{seed}"):
            rest = rest.replace(token, "")
        if "{" in rest or "}" in rest: C.fail("NATIVE_SCORER_PLACEHOLDER_UNSUPPORTED")
    if not isinstance(scorer["denominator_path"], list) or not scorer["denominator_path"]:
        C.fail("NATIVE_DENOMINATOR_PATH_REQUIRED")


def _qualification(root, rule, metric_names):
    if not isinstance(rule, dict) or any(k not in rule for k in ("metric","operator","threshold","reference_ref")):
        C.fail("NATIVE_QUALIFICATION_PROOF_REQUIRED")
    if rule["metric"] not in metric_names or rule["operator"] not in _OPERATORS:
        C.fail("NATIVE_QUALIFICATION_METRIC_INVALID")
    _number(rule["threshold"]); C.verify_ref(root, rule["reference_ref"])


def contract_for_group(protocol, group):
    """Select a declared group's existing contract, retaining scalar defaults."""
    mapping = protocol.get("native_eval_contracts")
    if mapping is None:
        contract = protocol.get("native_eval_contract")
        if not isinstance(contract,dict): C.fail("NATIVE_EVAL_CONTRACT_REQUIRED")
        return contract
    required = protocol.get("required_groups",[])
    if not isinstance(mapping,dict) or not required or set(mapping) != set(required):
        C.fail("NATIVE_GROUP_CONTRACTS_INCOMPLETE")
    if group not in mapping: C.fail("NATIVE_GROUP_CONTRACT_UNDECLARED")
    if not isinstance(mapping[group],dict): C.fail("NATIVE_EVAL_CONTRACT_REQUIRED")
    return mapping[group]


def verify_protocol(root, protocol):
    """Validate every group-scoped native contract and return the primary one."""
    root = C.root_path(root)
    if "method_discovery" in protocol:
        from verify_methods import verify_discovery
        verify_discovery(root, protocol)
    primary = protocol.get("native_eval_contract")
    if not isinstance(primary,dict): C.fail("NATIVE_EVAL_CONTRACT_REQUIRED")
    if "native_eval_contracts" not in protocol: return _verify_single_protocol(root,protocol)
    required = protocol.get("required_groups",[])
    # Selection validates exact inventory even when all contracts share a metric.
    for group in required:
        contract = contract_for_group(protocol,group)
        view = copy.deepcopy(protocol); view.pop("native_eval_contracts")
        view["native_eval_contract"] = contract
        view["criteria"] = [rule for rule in protocol.get("criteria",[]) if group in rule.get("groups",required)]
        view["required_groups"] = [group]
        _verify_single_protocol(root,view)
    if not required: C.fail("NATIVE_GROUP_CONTRACTS_INCOMPLETE")
    if not any(primary == contract for contract in protocol["native_eval_contracts"].values()):
        C.fail("NATIVE_PRIMARY_GROUP_CONTRACT_MISMATCH")
    return primary


def _verify_single_protocol(root, protocol):
    """Validate the inline native contract against retained existing definitions.

    This read-only function performs artifact closure, not publication/source
    authentication or scorer execution. The authorized host owns acquisition of
    authentic official artifacts and revision-bound executable dependencies.
    """
    root = C.root_path(root)
    contract = protocol.get("native_eval_contract")
    if not isinstance(contract, dict): C.fail("NATIVE_EVAL_CONTRACT_REQUIRED")
    C.validate(contract, "native-eval-contract")
    definition = C.load_file(C.verify_ref(root, contract["native_definition_ref"]))
    required = ("benchmark_id","benchmark_revision","published_at","source_url","primary_metric","metrics","splits",
                "prediction_format","sampling","budget","scorer","publication_refs")
    if not isinstance(definition, dict) or any(k not in definition for k in required):
        C.fail("NATIVE_PUBLISHED_DEFINITION_REQUIRED")
    if not isinstance(definition["source_url"],str): C.fail("NATIVE_PUBLICATION_URL_REQUIRED")
    try: url = urllib.parse.urlsplit(definition["source_url"])
    except ValueError: C.fail("NATIVE_PUBLICATION_URL_REQUIRED")
    if url.scheme not in {"https","http"} or not url.hostname: C.fail("NATIVE_PUBLICATION_URL_REQUIRED")
    C.stamp(definition["published_at"])
    if (definition["benchmark_id"], definition["benchmark_revision"]) != (contract["benchmark_id"], contract["benchmark_revision"]):
        C.fail("NATIVE_BENCHMARK_IDENTITY_MISMATCH")
    _refs(root, contract["published_source_refs"])
    _refs(root, definition["publication_refs"])
    if contract["published_source_refs"] != definition["publication_refs"]: C.fail("NATIVE_PUBLICATION_REFS_MISMATCH")
    splits = definition["splits"]
    if not isinstance(splits, list) or any(not isinstance(s, dict) for s in splits): C.fail("NATIVE_SPLITS_REQUIRED")
    selected = [s for s in splits if s.get("name") == contract["split"]]
    if len(selected) != 1: C.fail("NATIVE_SPLIT_UNAVAILABLE")
    if selected[0].get("sample_manifest_ref") != contract["sample_manifest_ref"]:
        C.fail("NATIVE_SPLIT_MANIFEST_MISMATCH")
    if selected[0].get("labels_or_tests_ref") != contract["labels_or_tests_ref"]:
        C.fail("NATIVE_LABELS_OR_TESTS_MISMATCH")
    C.verify_ref(root, contract["labels_or_tests_ref"])
    if contract["primary_metric"] != definition["primary_metric"]: C.fail("NATIVE_PRIMARY_METRIC_MISMATCH")
    if contract["metrics"] != definition["metrics"]: C.fail("NATIVE_METRIC_DEFINITION_MISMATCH")
    metrics = contract["metrics"]
    names = {m["name"] for m in metrics}
    if len(names) != len(metrics) or contract["primary_metric"] not in names: C.fail("NATIVE_PRIMARY_METRIC_UNAVAILABLE")
    if any(rule.get("metric") not in names for rule in protocol.get("criteria", [])): C.fail("NATIVE_CRITERION_METRIC_REQUIRED")
    for key in ("prediction_format","sampling","budget"):
        if contract[key] != definition[key]: C.fail("NATIVE_" + key.upper() + "_MISMATCH")
        if key in protocol and protocol[key] != contract[key]: C.fail("NATIVE_PROTOCOL_" + key.upper() + "_MISMATCH")
    for name, limit in contract["budget"].items():
        if not isinstance(name, str) or _number(limit, "NATIVE_BUDGET_INVALID") <= 0: C.fail("NATIVE_BUDGET_INVALID")
    _selection(root,contract,definition); _samples(root, contract)
    scorer = contract["scorer"]; official = definition["scorer"]
    _scorer(root, scorer); _scorer(root, official)
    if official["kind"] != "official": C.fail("NATIVE_OFFICIAL_SCORER_REQUIRED")
    if scorer["kind"] == "official":
        if scorer != official: C.fail("NATIVE_SCORER_IDENTITY_MISMATCH")
    else:
        if not scorer.get("verification_ref"): C.fail("NATIVE_FAITHFUL_HARNESS_VERIFICATION_REQUIRED")
        verification = C.load_file(C.verify_ref(root, scorer["verification_ref"]))
        harness = {k:v for k,v in scorer.items() if k != "verification_ref"}
        if not isinstance(verification, dict) or verification.get("official_scorer") != official or verification.get("harness_scorer") != harness:
            C.fail("NATIVE_FAITHFUL_HARNESS_IDENTITY_MISMATCH")
        if verification.get("sample_manifest_ref") != contract["sample_manifest_ref"]:
            C.fail("NATIVE_FAITHFUL_HARNESS_SAMPLE_MISMATCH")
        _refs(root, verification.get("source_refs"))
        tolerances = verification.get("metric_tolerances")
        if not isinstance(tolerances, dict) or set(tolerances) != names: C.fail("NATIVE_FAITHFUL_HARNESS_TOLERANCES_REQUIRED")
        for tolerance in tolerances.values():
            if _number(tolerance) < 0: C.fail("NATIVE_FAITHFUL_HARNESS_TOLERANCE_INVALID")
        native_tolerances = definition.get("faithful_harness_tolerances", {name:0 for name in names})
        if tolerances != native_tolerances: C.fail("NATIVE_FAITHFUL_HARNESS_TOLERANCE_UNSOURCED")
    if contract["contrasts"] != protocol.get("contrasts"): C.fail("NATIVE_CONTRASTS_MISMATCH")
    roles = {"treatment": contract["contrasts"]["treatment"], "baseline": contract["contrasts"]["baseline"],
             **{role:role for role in contract["contrasts"]["controls"]}}
    if len(roles) != 2 + len(contract["contrasts"]["controls"]): C.fail("NATIVE_ARM_ROLE_COLLISION")
    arms = contract["arm_requirements"]
    if set(arms) != set(roles): C.fail("NATIVE_ARM_REQUIREMENTS_INCOMPLETE")
    for role, arm in arms.items():
        if not isinstance(arm, dict) or arm.get("name") != roles[role] or not isinstance(arm.get("revision"), str) or not arm["revision"]:
            C.fail("NATIVE_ARM_IDENTITY_REQUIRED")
        _refs(root, arm.get("implementation_refs"), "NATIVE_ARM_IMPLEMENTATION_PROOF_REQUIRED")
    _qualification(root, contract["baseline_qualification"], names)
    controls = contract["control_qualifications"]
    if {r.get("role") for r in controls} != set(contract["contrasts"]["controls"]) or len(controls) != len(contract["contrasts"]["controls"]):
        C.fail("NATIVE_CONTROL_QUALIFICATIONS_INCOMPLETE")
    for rule in controls: _qualification(root, rule, names)
    return contract


def _prediction_ids(root, contract, receipt, samples):
    path = C.verify_ref(root, receipt["predictions_ref"])
    fmt = contract["prediction_format"]
    if fmt["format"] == "jsonl":
        try: records = [C.parse_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        except (OSError, UnicodeError): C.fail("NATIVE_PREDICTION_OUTPUT_UNAVAILABLE")
    else:
        records = _path(C.load_file(path), fmt["records_path"], "NATIVE_PREDICTION_RECORDS_UNAVAILABLE")
    if not isinstance(records, list): C.fail("NATIVE_PREDICTION_RECORDS_INVALID")
    observed = [_path(record, fmt["id_path"], "NATIVE_PREDICTION_ID_UNAVAILABLE") for record in records]
    if any(not isinstance(i, str) for i in observed): C.fail("NATIVE_PREDICTION_SAMPLE_MISMATCH")
    expected = collections.Counter({sample_id:samples["predictions_per_sample"] for sample_id in samples["sample_ids"]})
    if collections.Counter(observed) != expected: C.fail("NATIVE_PREDICTION_SAMPLE_MISMATCH")


def _command(root, scorer, receipt, *, output_path=None):
    replacements = {"{predictions}":str(C.verify_ref(root, receipt["predictions_ref"])),
        "{samples}":str(C.verify_ref(root, receipt["sample_manifest_ref"])),
        "{labels}":str(C.verify_ref(root, receipt["labels_or_tests_ref"])), "{seed}":str(receipt["seed"])}
    if output_path is not None: replacements["{output}"] = str(output_path)
    result = []
    for arg in scorer["command"]:
        for placeholder, value in replacements.items(): arg = arg.replace(placeholder, value)
        result.append(arg)
    return result


def _replay(root, contract, receipt, scorer, replay_context):
    request = {"nonce":uuid.uuid4().hex, "arm_role":receipt["arm_role"], "receipt":copy.deepcopy(receipt),
        "scorer":copy.deepcopy(scorer), "command":_command(root, scorer, receipt),
        "cwd":str(_cwd(root, scorer["cwd"])), "budget":copy.deepcopy(contract["budget"]),
        "input_refs":[receipt["predictions_ref"], receipt["sample_manifest_ref"], receipt["labels_or_tests_ref"]]}
    request["request_digest"] = C.hashed(request)
    try: result = replay_context(copy.deepcopy(request))
    except C.Failure: raise
    except Exception: C.fail("NATIVE_SCORER_REPLAY_UNAVAILABLE")
    if not isinstance(result, dict) or result.get("nonce") != request["nonce"] or result.get("request_digest") != request["request_digest"] or not result.get("execution_id"):
        C.fail("NATIVE_SCORER_REPLAY_NOT_LIVE")
    for key, expected in (("argv",request["command"]),("cwd",request["cwd"]),("input_refs",request["input_refs"]),
                          ("code_refs",scorer["code_refs"]),("scorer_identity",scorer["identity"]),("scorer_revision",scorer["revision"])):
        # A file-output host replaces only {output} with its isolated replay path.
        actual = result.get(key)
        if key == "argv" and scorer["output"]["source"] == "file":
            output_ref = result.get("native_output_ref")
            if not isinstance(output_ref, dict): C.fail("NATIVE_REPLAY_OUTPUT_REQUIRED")
            expected = _command(root, scorer, receipt, output_path=C.verify_ref(root, output_ref))
        if actual != expected: C.fail("NATIVE_SCORER_REPLAY_BINDING_MISMATCH")
    started = C.stamp(result.get("started_at")); ended = C.stamp(result.get("completed_at"))
    if ended < started: C.fail("NATIVE_SCORER_REPLAY_TIME_INVALID")
    if result.get("exit_code") != 0: C.fail("NATIVE_SCORER_REPLAY_FAILED")
    for key in ("stdout_ref","stderr_ref","native_output_ref"):
        if not result.get(key): C.fail("NATIVE_REPLAY_OUTPUT_REQUIRED")
        C.verify_ref(root, result[key])
    for item in request["input_refs"] + scorer["code_refs"]: C.verify_ref(root,item)
    if scorer["output"]["source"] == "stdout" and result["stdout_ref"] != result["native_output_ref"]:
        C.fail("NATIVE_SCORER_OUTPUT_BINDING_MISMATCH")
    output = C.load_file(C.verify_ref(root, result["native_output_ref"]))
    denominator = _number(_path(output, scorer["denominator_path"]), "NATIVE_DENOMINATOR_INVALID")
    if denominator != receipt["denominator"]: C.fail("NATIVE_DENOMINATOR_MISMATCH")
    metrics = {m["name"]:_number(_path(output, m["output_path"])) for m in contract["metrics"]}
    return metrics, result


def verify_run(root, protocol, manifest, replay_context=None):
    """Return native metrics/proofs only after trusted live scorer recomputation.

    proof_refs are stable frozen artifact refs; replay_audit contains separately
    retained live executions and must not participate in deterministic gate IDs.
    Qualification flags are derived from native metrics, never manifest booleans.
    """
    root = C.root_path(root); verify_protocol(root, protocol)
    contract = contract_for_group(protocol,manifest["group"])
    if not callable(replay_context): C.fail("NATIVE_SCORER_REPLAY_REQUIRED")
    roles = set(contract["arm_requirements"])
    receipt_refs = manifest.get("native_eval_receipts")
    if not isinstance(receipt_refs, dict) or set(receipt_refs) != roles: C.fail("NATIVE_ARM_RECEIPTS_INCOMPLETE")
    if manifest.get("status") != "completed": C.fail("NATIVE_RUN_INCOMPLETE")
    samples = _samples(root, contract); names = {m["name"] for m in contract["metrics"]}
    proofs = [contract["native_definition_ref"],contract["sample_manifest_ref"],contract["labels_or_tests_ref"]]
    if contract.get("selection"):
        proofs += [contract["selection"]["selected_manifest_ref"],contract["selection"]["capability_ref"]]
    proofs += contract["published_source_refs"] + contract["scorer"]["source_refs"] + contract["scorer"]["code_refs"]
    proofs.append(contract["baseline_qualification"]["reference_ref"])
    proofs += [r["reference_ref"] for r in contract["control_qualifications"]]
    if contract["scorer"].get("verification_ref"): proofs.append(contract["scorer"]["verification_ref"])
    arm_metrics, audits = {}, []
    for role in sorted(roles):
        receipt = C.verify_ref(root, receipt_refs[role], "native-eval-receipt")
        identity = {"run_id":manifest["run_id"],"protocol_id":protocol["protocol_id"],"protocol_digest":manifest["protocol_digest"],
            "contract_digest":C.hashed(contract),"arm_role":role,"arm_name":contract["arm_requirements"][role]["name"],
            "implementation_revision":contract["arm_requirements"][role]["revision"],
            "implementation_refs":contract["arm_requirements"][role]["implementation_refs"],
            "seed":manifest["seed"],"group":manifest["group"],
            **{k:contract[k] for k in ("benchmark_id","benchmark_revision","split","labels_or_tests_ref","sampling","budget")},
            "sample_manifest_ref":sample_manifest_ref(contract),
            "scorer_identity":contract["scorer"]["identity"],"scorer_revision":contract["scorer"]["revision"],
            "scorer_code_refs":contract["scorer"]["code_refs"],"scorer_command":contract["scorer"]["command"]}
        if manifest["protocol_digest"] != protocol.get("protocol_digest"): C.fail("NATIVE_PROTOCOL_DIGEST_MISMATCH")
        if any(receipt.get(k) != v for k,v in identity.items()): C.fail("NATIVE_RECEIPT_IDENTITY_MISMATCH")
        if receipt["denominator"] != samples["denominator"]: C.fail("NATIVE_DENOMINATOR_MISMATCH")
        if set(receipt["native_metrics"]) != names: C.fail("NATIVE_RECEIPT_METRICS_INCOMPLETE")
        if receipt["recorded_at"] != manifest["result_recorded_at"]: C.fail("NATIVE_RECEIPT_TIME_MISMATCH")
        C.stamp(receipt["recorded_at"])
        if "native_arm_resources" in manifest:
            arm_resources = manifest["native_arm_resources"]
            if not isinstance(arm_resources,dict) or set(arm_resources) != roles: C.fail("NATIVE_ARM_RESOURCES_INCOMPLETE")
            expected_resources = arm_resources[role]
        else: expected_resources = manifest["resources"]
        if receipt["resources"] != expected_resources: C.fail("NATIVE_RECEIPT_RESOURCE_MISMATCH")
        for name, limit in contract["budget"].items():
            observed = receipt["resources"].get(name)
            if observed is None or _number(observed, "NATIVE_RESOURCE_PROOF_REQUIRED") < 0: C.fail("NATIVE_RESOURCE_PROOF_REQUIRED")
            if observed > limit: C.fail("NATIVE_BUDGET_EXCEEDED")
        _refs(root, receipt["implementation_refs"]); _refs(root, receipt["raw_output_refs"])
        C.verify_ref(root, receipt["native_output_ref"]); _prediction_ids(root, contract, receipt, samples)
        observed, audit = _replay(root, contract, receipt, contract["scorer"], replay_context); audits.append(audit)
        if contract["scorer"]["kind"] == "faithful_harness":
            verification = C.load_file(C.verify_ref(root, contract["scorer"]["verification_ref"]))
            official, official_audit = _replay(root, contract, receipt, verification["official_scorer"], replay_context)
            audits.append(official_audit)
            if any(abs(observed[m]-official[m]) > verification["metric_tolerances"][m] for m in names):
                C.fail("NATIVE_FAITHFUL_HARNESS_PARITY_FAILED")
            observed = official
        if observed != receipt["native_metrics"]: C.fail("NATIVE_METRIC_REPLAY_DISAGREEMENT")
        arm_metrics[role] = observed
        proofs += [receipt_refs[role], receipt["predictions_ref"], receipt["native_output_ref"]]
        proofs += receipt["raw_output_refs"] + receipt["implementation_refs"]
    metrics = {name:{"treatment":arm_metrics["treatment"][name],"control":arm_metrics["baseline"][name]} for name in sorted(names)}
    rule = contract["baseline_qualification"]
    baseline = _OPERATORS[rule["operator"]](arm_metrics["baseline"][rule["metric"]],rule["threshold"])
    controls = {r["role"]:_OPERATORS[r["operator"]](arm_metrics[r["role"]][r["metric"]],r["threshold"])
                for r in contract["control_qualifications"]}
    stable = {(r["path"],r["sha256"]):r for r in proofs}
    _refs(root,list(stable.values()))
    return {"metrics":metrics,"arm_metrics":arm_metrics,"proof_refs":[stable[key] for key in sorted(stable)],
            "baseline_qualified":baseline,"positive_control_passed":all(controls.values()),
            "control_results":controls,"replay_audit":audits,
            "evaluation_scope":{"benchmark_id":contract["benchmark_id"],"benchmark_revision":contract["benchmark_revision"],
                "split":contract["split"],"group":manifest["group"],"source_manifest_ref":contract["sample_manifest_ref"],
                "selected_manifest_ref":sample_manifest_ref(contract),"selection":copy.deepcopy(contract.get("selection"))}}
