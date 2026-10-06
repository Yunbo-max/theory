"""Frozen claim scopes and dependency tracking for editable research assets."""
import copy
import _autoresearch as C


def _validate_eval_scope(scope, native, required_metrics):
    if not isinstance(scope, dict):
        C.fail("CLAIM_NATIVE_SCOPE_REQUIRED")
    benchmark = {"benchmark_id": native.get("benchmark_id"), "version": native.get("benchmark_revision"), "split": native.get("split")}
    if any(scope.get(k) != v for k, v in benchmark.items()):
        C.fail("CLAIM_BENCHMARK_SCOPE_MISMATCH")
    if not isinstance(native.get("metrics"), list) or any(not isinstance(m, dict) or not m.get("name") for m in native["metrics"]):
        C.fail("CLAIM_NATIVE_METRIC_SCOPE_MISMATCH")
    sample_ref = native.get("selection", {}).get("selected_manifest_ref", native.get("sample_manifest_ref"))
    if scope.get("sample_manifest_ref") != sample_ref or scope.get("budget") != native.get("budget"):
        C.fail("CLAIM_SAMPLE_OR_BUDGET_SCOPE_MISMATCH")
    if scope.get("selection") != native.get("selection"):
        C.fail("CLAIM_SELECTION_SCOPE_MISMATCH")
    metrics = scope.get("metrics")
    if not isinstance(metrics, list) or not metrics or not set(metrics).issubset({m["name"] for m in native["metrics"]}):
        C.fail("CLAIM_NATIVE_METRIC_SCOPE_MISMATCH")
    if not required_metrics.issubset(metrics):
        C.fail("CLAIM_CRITERION_SCOPE_MISMATCH")


def validate_claim_cards(protocol):
    cards = protocol.get("claim_cards", [])
    mapping = {c.get("claim_id"): c for c in cards if isinstance(c, dict)}
    if len(mapping) != len(cards) or set(mapping) != set(protocol.get("claim_set", [])):
        C.fail("FROZEN_CLAIM_CARDS_REQUIRED")
    native = protocol.get("native_eval_contract")
    if not isinstance(native, dict):
        C.fail("CLAIM_NATIVE_SCOPE_REQUIRED")
    from _native_eval import contract_for_group
    for claim, card in mapping.items():
        for key in ("supported_wording", "forbidden_wording", "limitations", "scope", "method_version", "project_version"):
            if not card.get(key):
                C.fail("CLAIM_CARD_FIELD_REQUIRED")
        if card["method_version"] != protocol.get("method_version") or card["project_version"] != protocol.get("project_version"):
            C.fail("CLAIM_VERSION_MISMATCH")
        if not isinstance(card["supported_wording"], str) or not card["supported_wording"].strip() or any(
            not isinstance(card[k], list) or any(not isinstance(v, str) or not v.strip() for v in card[k])
            for k in ("forbidden_wording", "limitations")):
            C.fail("CLAIM_CARD_WORDING_INVALID")
        scope = card["scope"]
        if not isinstance(scope, dict):
            C.fail("CLAIM_NATIVE_SCOPE_REQUIRED")
        rules = [r for r in protocol["criteria"] if r.get("claim_id") == claim]
        if not isinstance(scope.get("metrics"), list) or not scope["metrics"] or not rules or not {r["metric"] for r in rules}.issubset(scope["metrics"]):
            C.fail("CLAIM_CRITERION_SCOPE_MISMATCH")
        groups = {g for r in rules for g in r.get("groups", protocol["required_groups"])}
        if not scope.get("groups") or set(scope["groups"]) != groups:
            C.fail("CLAIM_GROUP_SCOPE_MISMATCH")
        evaluations = scope.get("evaluation_scopes")
        if evaluations is not None and (not isinstance(evaluations, dict) or set(evaluations) != groups):
            C.fail("CLAIM_GROUP_SCOPE_MISMATCH")
        for group in sorted(groups):
            group_metrics = {r["metric"] for r in rules if group in r.get("groups", protocol["required_groups"])}
            _validate_eval_scope(evaluations[group] if evaluations is not None else scope,
                contract_for_group(protocol, group), group_metrics)
    return mapping


def refresh_assets(root, manifest, cards, *, method_version, project_version):
    mapping = {c["claim_id"]: c for c in cards}
    if len(mapping) != len(cards):
        C.fail("DUPLICATE_CLAIM_CARD")
    result = copy.deepcopy(manifest)
    result["rewrite_actions"] = []
    seen = set()
    for asset in result.get("assets", []):
        if not asset.get("asset_id") or asset["asset_id"] in seen:
            C.fail("DUPLICATE_OR_MISSING_ASSET_ID")
        seen.add(asset["asset_id"])
        claims = asset.get("claim_ids", [])
        reasons = []
        unknown = sorted(set(claims) - set(mapping))
        if not claims or unknown:
            reasons.append("unknown_or_unbound_claim")
        if any(asset.get("rendered_from", {}).get(c) != C.hashed(mapping[c]) for c in claims if c in mapping):
            reasons.append("claim_scope_or_wording_changed")
        if asset.get("method_version") != method_version or asset.get("project_version") != project_version:
            reasons.append("method_or_project_changed")
        try:
            C.verify_ref(root, asset["asset_ref"])
        except (C.Failure, KeyError):
            reasons.append("asset_edited_or_unavailable")
        asset["status"] = "blocked" if unknown or not claims else "stale" if reasons else "current"
        asset["reason_codes"] = reasons
        if reasons:
            result["rewrite_actions"].append({"asset_id": asset["asset_id"], "kind": asset.get("kind", "unknown"),
                "claim_ids": claims, "action": "review_and_rewrite_from_current_evidence", "reasons": reasons})
    result["method_version"] = method_version
    result["project_version"] = project_version
    result["claim_card_digests"] = {c: C.hashed(card) for c, card in mapping.items()}
    result["all_current"] = all(a["status"] == "current" for a in result.get("assets", [])) and bool(result.get("assets"))
    return result
