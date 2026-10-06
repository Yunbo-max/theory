"""Private local coordination of route instances; no chat, compute or hosting adapter."""
import contextlib
import copy
import datetime as dt
import fcntl
import os
from pathlib import Path

import _autoresearch as C


VERSION = "1.0.0"
ACTIVE = {"active", "waiting", "blocked"}
OPERATIONS = {"register", "checkpoint", "reserve", "release", "record-usage", "sync-receipt", "set-destination"}


def _time(value=None):
    return C.stamp(value or C.now()).astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _text(value, code="NONEMPTY_STRING_REQUIRED"):
    if not isinstance(value, str) or not value.strip():
        C.fail(code)
    return value


def _amounts(value):
    if not isinstance(value, dict):
        C.fail("RESOURCE_AMOUNTS_REQUIRED")
    C.finite(value)
    C.scan(value)
    for key, number in value.items():
        _text(key)
        if isinstance(number, bool) or not isinstance(number, (int, float)) or number < 0:
            C.fail("NONNEGATIVE_RESOURCE_AMOUNT_REQUIRED")
    return copy.deepcopy(value)


def _root(value):
    return C.root_path(value)


@contextlib.contextmanager
def _lock(root, *, create=False, exclusive=False):
    root = _root(root)
    if create:
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not root.is_dir():
        C.fail("REGISTRY_UNAVAILABLE")
    path = C.safe_path(root, ".supervisor.lock")
    flags = os.O_RDWR | os.O_CREAT if create or exclusive else os.O_RDONLY
    try:
        fd = os.open(str(path), flags, 0o600)
    except OSError:
        C.fail("REGISTRY_UNAVAILABLE")
    try:
        try:
            fcntl.flock(fd, (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH) | fcntl.LOCK_NB)
        except BlockingIOError:
            C.fail("REGISTRY_BUSY")
        yield root
    finally:
        os.close(fd)


def _initial_state():
    return {"projects": {}, "routes": {}, "leases": {}, "usage": {},
            "sync_receipts": [], "fencing_counter": 0}


def _initial_head(registry):
    return C.hashed({key: registry[key] for key in ("registry_id", "created_at", "configuration")})


def _config(resources, budget, destination, stale_after_seconds):
    resources = copy.deepcopy(resources or {})
    if not isinstance(resources, dict):
        C.fail("RESOURCE_INVENTORY_REQUIRED")
    for resource_id, spec in resources.items():
        _text(resource_id)
        if not isinstance(spec, dict) or spec.get("kind") not in {"gpu", "workspace", "other"}:
            C.fail("RESOURCE_KIND_REQUIRED")
        capacity = _amounts({resource_id: spec.get("capacity")})[resource_id]
        if capacity <= 0 or (spec["kind"] == "workspace" and capacity != 1):
            C.fail("INVALID_RESOURCE_CAPACITY")
        if spec["kind"] == "gpu" and capacity != int(capacity):
            C.fail("INTEGER_GPU_CAPACITY_REQUIRED")
    if isinstance(stale_after_seconds, bool) or not isinstance(stale_after_seconds, (int, float)) or stale_after_seconds <= 0:
        C.fail("INVALID_STALE_INTERVAL")
    if destination is not None:
        if not isinstance(destination, dict) or destination.get("kind") not in {"repository", "page", "drive"}:
            C.fail("INVALID_DURABLE_DESTINATION")
        destination = copy.deepcopy(destination)
        _text(destination.get("uri"), "DURABLE_DESTINATION_URI_REQUIRED")
        destination.setdefault("private", True)
        if not isinstance(destination["private"], bool):
            C.fail("INVALID_DESTINATION_VISIBILITY")
    result = {"resources": resources, "budget": _amounts(budget or {}),
              "destination": destination, "stale_after_seconds": stale_after_seconds}
    C.scan(result)
    return result


def initialize(root, registry_id, *, resources=None, budget=None, destination=None,
               stale_after_seconds=3600, at=None):
    """Initialize or resume an unchanged private registry. Capacity is a declaration."""
    _text(registry_id)
    configuration = _config(resources, budget, destination, stale_after_seconds)
    with _lock(root, create=True, exclusive=True) as root:
        path = C.safe_path(root, "registry.json")
        if path.exists():
            saved = _load(root)
            if saved["registry_id"] != registry_id or saved["configuration"] != configuration:
                C.fail("REGISTRY_CONFIGURATION_CONFLICT")
            return {"status": "resumed", "revision": saved["revision"], "head_digest": saved["head_digest"]}
        if any(p.name != ".supervisor.lock" for p in root.iterdir()):
            C.fail("REGISTRY_ROOT_NOT_EMPTY")
        registry = C.envelope("supervisor-registry", registry_id=registry_id, created_at=_time(at),
                              configuration=configuration, events=[], revision=0, state=_initial_state())
        registry["head_digest"] = _initial_head(registry)
        C.validate(registry, "supervisor-registry")
        C.atomic(path, C.canonical(registry) + "\n")
        return {"status": "initialized", "revision": 0, "head_digest": registry["head_digest"]}


def _active(lease, at):
    return lease["status"] == "reserved" and C.stamp(lease["expires_at"]) > C.stamp(at)


def _accounting(state, route_id=None):
    spent, held = {}, {}
    for lease in state["leases"].values():
        if route_id is not None and lease["route_id"] != route_id:
            continue
        for key, amount in lease["spent"].items():
            spent[key] = spent.get(key, 0) + amount
        if lease["status"] != "denied" and not lease["settled"]:
            for key, amount in lease["budget"].items():
                held[key] = held.get(key, 0) + max(0, amount - lease["spent"].get(key, 0))
    return {"spent": spent, "held": held}


def _isolation(state, route, exclude=None, *, at=None):
    if not route["writer"] or route["status"] not in ACTIVE:
        return
    workspace = Path(route["workspace"])
    for other in state["routes"].values():
        leased = any(lease["route_id"] == other["route_id"] and _active(lease, at or _time()) for lease in state["leases"].values())
        if other["route_id"] == exclude or not other["writer"] or (other["status"] not in ACTIVE and not leased):
            continue
        if other["repository"] == route["repository"] and other["branch"] == route["branch"]:
            C.fail("WRITER_BRANCH_CONFLICT")
        other_workspace = Path(other["workspace"])
        if workspace.is_relative_to(other_workspace) or other_workspace.is_relative_to(workspace):
            C.fail("WRITER_WORKSPACE_CONFLICT")


def _assert_lease_state(state, lease_id, route_id, fencing_token, at, *, active=True):
    lease = state["leases"].get(lease_id)
    if lease is None:
        C.fail("LEASE_UNKNOWN")
    if lease["route_id"] != route_id:
        C.fail("LEASE_OWNER_MISMATCH")
    if isinstance(fencing_token, bool) or fencing_token != lease["fencing_token"]:
        C.fail("FENCING_TOKEN_MISMATCH")
    if lease["status"] == "denied":
        C.fail("LEASE_NOT_RESERVED")
    if active:
        if C.stamp(lease["expires_at"]) <= C.stamp(at):
            C.fail("LEASE_EXPIRED")
        if lease["status"] != "reserved":
            C.fail("LEASE_RELEASED")
        if state["routes"].get(route_id, {}).get("status") != "active":
            C.fail("LEASE_OWNER_NOT_ACTIVE")
    return lease


def _reduce(state, kind, payload, timestamp, configuration):
    """Deterministic replay; never fetch chat history, run a job or alter a project."""
    state = copy.deepcopy(state)
    if kind == "register":
        route = copy.deepcopy(payload["route"])
        if route["route_id"] in state["routes"]:
            C.fail("ROUTE_ALREADY_REGISTERED")
        _isolation(state, route, at=timestamp)
        project = state["projects"].get(route["project_id"])
        if project and project["canonical_root"] != route["project_root"]:
            C.fail("PROJECT_CANONICAL_ROOT_CONFLICT")
        if not project:
            state["projects"][route["project_id"]] = {"project_id": route["project_id"],
                "name": route["imported_state"]["state"]["project_name"], "canonical_root": route["project_root"],
                "upstream_validation": None}
        else:
            route["upstream_validation"] = copy.deepcopy(project["upstream_validation"])
        state["routes"][route["route_id"]] = route
        receipt = {"status": "registered", "route_id": route["route_id"], "route_revision": route["revision"]}
    elif kind == "checkpoint":
        route = copy.deepcopy(payload["route"])
        prior = state["routes"].get(route["route_id"])
        if not prior or route["revision"] != prior["revision"] + 1:
            C.fail("ROUTE_REVISION_CONFLICT")
        _isolation(state, route, exclude=route["route_id"], at=timestamp)
        for field in ("project_id", "project_root", "owner", "repository", "branch", "workspace", "goal", "scope", "budget", "writer"):
            if route[field] != prior[field]:
                C.fail("ROUTE_IDENTITY_IMMUTABLE")
        project = state["projects"][route["project_id"]]
        if route["checkpoint"]["changed_results"]:
            obligation = copy.deepcopy(route["upstream_validation"])
            previous = project["upstream_validation"] or {}
            obligation["project_sequence"] = max(obligation["project_sequence"], previous.get("project_sequence", 0))
            obligation["route_ids"] = sorted(set(previous.get("route_ids", [])) | {route["route_id"]})
            project["upstream_validation"] = obligation
        elif route["checkpoint"].get("validation_resolution"):
            project["upstream_validation"] = None
        for other in state["routes"].values():
            if other["project_id"] == route["project_id"]:
                other["upstream_validation"] = copy.deepcopy(project["upstream_validation"])
        route["upstream_validation"] = copy.deepcopy(project["upstream_validation"])
        state["routes"][route["route_id"]] = route
        receipt = {"status": "checkpointed", "route_id": route["route_id"],
                   "route_revision": route["revision"], "verification": route["checkpoint"]["verification"]}
    elif kind == "reserve":
        lease = copy.deepcopy(payload["lease"])
        C.validate(lease, "resource-lease")
        if lease["lease_id"] in state["leases"]:
            C.fail("LEASE_ID_REUSED")
        route = state["routes"].get(lease["route_id"])
        if not route:
            C.fail("ROUTE_UNKNOWN")
        if lease["status"] == "reserved":
            if lease["fencing_token"] != state["fencing_counter"] + 1:
                C.fail("FENCING_SEQUENCE_MISMATCH")
            state["fencing_counter"] = lease["fencing_token"]
            route["resource_blockers"] = []
        else:
            route["resource_blockers"] = list(lease["reason_codes"])
        state["leases"][lease["lease_id"]] = lease
        receipt = {"status": "reserved" if lease["status"] == "reserved" else "blocked",
                   "lease_id": lease["lease_id"], "fencing_token": lease["fencing_token"],
                   "reason_codes": lease["reason_codes"]}
    elif kind == "release":
        lease = _assert_lease_state(state, payload["lease_id"], payload["route_id"], payload["fencing_token"], timestamp, active=False)
        lease.update(status="released", released_at=timestamp)
        receipt = {"status": "released", "lease_id": lease["lease_id"], "budget": "held_until_settlement"}
    elif kind == "record-usage":
        lease = _assert_lease_state(state, payload["lease_id"], payload["route_id"], payload["fencing_token"], timestamp, active=False)
        if lease["settled"]:
            C.fail("LEASE_ALREADY_SETTLED")
        for key, amount in payload["usage"].items():
            total = lease["spent"].get(key, 0) + amount
            if key not in lease["budget"] or total > lease["budget"][key]:
                C.fail("USAGE_EXCEEDS_RESERVATION")
            lease["spent"][key] = total
        lease["settled"] = payload["final"]
        state["usage"][payload["usage_id"]] = copy.deepcopy(payload)
        receipt = {"status": "usage_recorded", "lease_id": lease["lease_id"], "settled": lease["settled"]}
    elif kind == "set-destination":
        destination = _config({}, {}, payload["destination"], 3600)["destination"]
        if destination is None:
            C.fail("DURABLE_DESTINATION_MISSING")
        state["destination"] = destination
        receipt = {"status": "destination_recorded_not_synced", "destination": copy.deepcopy(destination)}
    elif kind == "sync-receipt":
        state["sync_receipts"].append(copy.deepcopy(payload))
        receipt = {"status": "host_receipt_recorded", "independently_verified_by_helper": False,
                   "source_revision": payload["source_revision"]}
    else:
        C.fail("UNKNOWN_SUPERVISOR_OPERATION")
    return state, receipt


def _load(root):
    registry = C.validate(C.load_file(C.safe_path(root, "registry.json")), "supervisor-registry")
    _config(**registry["configuration"])
    state, head, request_ids = _initial_state(), _initial_head(registry), set()
    last_time = registry["created_at"]
    for index, event in enumerate(registry["events"], 1):
        C.validate(event, "supervisor-event")
        if C.hashed({k: v for k, v in event.items() if k != "event_digest"}) != event["event_digest"]:
            C.fail("REGISTRY_EVENT_DIGEST_MISMATCH")
        if event["revision"] != index or event["previous_event_digest"] != head:
            C.fail("REGISTRY_CONTINUITY_MISMATCH")
        if event["request_id"] in request_ids:
            C.fail("REQUEST_ID_REUSED")
        if C.stamp(event["timestamp"]) < C.stamp(last_time):
            C.fail("REGISTRY_TIMESTAMP_ROLLBACK")
        request_ids.add(event["request_id"])
        state, receipt = _reduce(state, event["event_type"], event["payload"], event["timestamp"], registry["configuration"])
        if C.hashed(state) != event["state_digest"] or receipt != event["receipt"]:
            C.fail("REGISTRY_REPLAY_MISMATCH")
        head, last_time = event["event_digest"], event["timestamp"]
    if registry["revision"] != len(registry["events"]) or registry["head_digest"] != head:
        C.fail("REGISTRY_TAIL_OR_HEAD_MISMATCH")
    if registry["state"] != state:
        C.fail("REGISTRY_PROJECTION_MISMATCH")
    return registry


def replay(root):
    with _lock(root) as root:
        return _load(root)


def _receipt(event):
    return {**copy.deepcopy(event["receipt"]), "revision": event["revision"], "head_digest": event["event_digest"]}


def _mutate(root, kind, request_id, expected_revision, request, prepare, at):
    _text(request_id, "REQUEST_ID_REQUIRED")
    if isinstance(expected_revision, bool) or not isinstance(expected_revision, int) or expected_revision < 0:
        C.fail("EXPECTED_REVISION_REQUIRED")
    C.finite(request)
    C.scan(request)
    digest = C.hashed({"operation": kind, "expected_revision": expected_revision, "request": request})
    with _lock(root, exclusive=True) as root:
        registry = _load(root)
        for event in registry["events"]:
            if event["request_id"] == request_id:
                if event["request_digest"] != digest:
                    C.fail("REQUEST_ID_REUSED")
                return _receipt(event)
        if registry["revision"] != expected_revision:
            C.fail("REGISTRY_REVISION_CONFLICT")
        timestamp = _time(at)
        last_time = registry["events"][-1]["timestamp"] if registry["events"] else registry["created_at"]
        if C.stamp(timestamp) < C.stamp(last_time):
            C.fail("REGISTRY_TIMESTAMP_ROLLBACK")
        payload = prepare(registry, timestamp)
        state, receipt = _reduce(registry["state"], kind, payload, timestamp, registry["configuration"])
        event = C.envelope("supervisor-event", revision=expected_revision + 1, request_id=request_id,
            request_digest=digest, event_type=kind, timestamp=timestamp, previous_event_digest=registry["head_digest"],
            payload=payload, state_digest=C.hashed(state), receipt=receipt)
        event["event_digest"] = C.hashed(event)
        C.validate(event, "supervisor-event")
        registry.update(revision=event["revision"], head_digest=event["event_digest"], state=state)
        registry["events"].append(event)
        C.validate(registry, "supervisor-registry")
        C.atomic(C.safe_path(root, "registry.json"), C.canonical(registry) + "\n")
        return _receipt(event)


def _read_project(value, at):
    """Read under the project's existing lock; never recover or repair its files."""
    root = _root(value)
    lock = C.safe_path(root, ".state.lock")
    try:
        fd = os.open(str(lock), os.O_RDONLY)
    except OSError:
        C.fail("CANONICAL_STATE_UNAVAILABLE")
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            C.fail("CANONICAL_PROJECT_BUSY")
        if (root / ".pending-commit.json").exists():
            C.fail("CANONICAL_COMMIT_PENDING")
        try:
            state, events, anchor = C.project(root)
            imported = {"verification": "verified_project_ledger", "project_id": state["project_id"],
                "sequence": anchor["sequence"], "head_digest": anchor["head_digest"], "verified_at": at,
                "state": copy.deepcopy(state), "state_digest": C.hashed(state),
                "ledger_ref": C.reference(root, root / "event-ledger.jsonl"),
                "anchor_ref": C.reference(root, root / "ledger-anchor.json"),
                "projection_ref": C.reference(root, root / "research-state.json")}
            return imported, events
        except (OSError, KeyError, ValueError, UnicodeError):
            C.fail("CANONICAL_STATE_UNAVAILABLE")
    finally:
        os.close(fd)


def _overlay(value, project_root, *, verify_evidence):
    C.schema_check(value, {"type": "object", "required": ["path", "current_node", "next_node", "last_evidence"]})
    registry = C.load_file(C.SKILL / "assets/research-nodes.json")
    nodes = {node["id"] for node in registry["nodes"]}
    edges = {(edge["from"], edge["to"]) for edge in registry["edges"]}
    path = value["path"]
    if not isinstance(path, list) or not path or any(node not in nodes for node in path):
        C.fail("UNKNOWN_MAP_NODE")
    current, next_node = value["current_node"], value["next_node"]
    if current not in nodes or (next_node is not None and next_node not in nodes):
        C.fail("UNKNOWN_MAP_NODE")
    if path[-1] != current:
        C.fail("MAP_CURRENT_NODE_PATH_MISMATCH")
    if any((a, b) not in edges for a, b in zip(path, path[1:])) or (next_node is not None and (current, next_node) not in edges):
        C.fail("MAP_EDGE_MISSING")
    evidence = value["last_evidence"]
    if evidence is not None:
        C.schema_check(evidence, {"type": "object", "required": ["path", "sha256"], "properties": {
            "path": {"type": "string", "minLength": 1}, "sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"}},
            "additionalProperties": False})
        if verify_evidence:
            C.verify_ref(_root(project_root), evidence)
    return {"path": copy.deepcopy(path), "current_node": current, "next_node": next_node,
            "last_evidence": copy.deepcopy(evidence)}


def register_route(root, request_id, expected_revision, request, *, at=None):
    def prepare(registry, timestamp):
        value = copy.deepcopy(request)
        required = {"route_id", "project_root", "owner", "repository", "branch", "workspace", "goal", "scope", "budget", "writer", "status", "overlay"}
        if not isinstance(value, dict) or set(value) != required:
            C.fail("ROUTE_REGISTRATION_FIELDS_REQUIRED")
        for field in ("route_id", "repository", "branch", "goal"):
            _text(value[field])
        if not isinstance(value["owner"], dict) or set(value["owner"]) != {"agent_id", "chat_id", "model"}:
            C.fail("OWNER_AGENT_CHAT_MODEL_REQUIRED")
        for identity in value["owner"].values():
            _text(identity)
        if not isinstance(value["scope"], list) or not value["scope"]:
            C.fail("ROUTE_SCOPE_REQUIRED")
        for scope in value["scope"]:
            _text(scope)
        if not isinstance(value["writer"], bool) or value["status"] not in ACTIVE | {"paused", "completed"}:
            C.fail("INVALID_ROUTE_STATUS")
        value["project_root"] = str(_root(value["project_root"]))
        if not Path(value["workspace"]).is_absolute():
            C.fail("ABSOLUTE_WORKSPACE_REQUIRED")
        value["workspace"] = str(_root(value["workspace"]))
        value["budget"] = _amounts(value["budget"])
        imported, _ = _read_project(value["project_root"], timestamp)
        value["overlay"] = _overlay(value["overlay"], value["project_root"], verify_evidence=True)
        value.update(project_id=imported["project_id"], revision=1, imported_state=imported,
            registered_at=timestamp, resource_blockers=[], upstream_validation=None,
            checkpoint={"schema_id": "route-checkpoint", "schema_version": VERSION, "route_id": value["route_id"],
                "kind": "verified", "verification": "verified_project_ledger", "observed_at": timestamp,
                "summary": "Route registered against canonical project ledger", "status": value["status"],
                "blockers": [], "changed_results": False, "overlay": copy.deepcopy(value["overlay"]), "lease_tokens": []})
        C.validate(value["checkpoint"], "route-checkpoint")
        return {"route": value}
    return _mutate(root, "register", request_id, expected_revision, request, prepare, at)


def _compare_import(prior, imported, events):
    if prior["project_id"] != imported["project_id"]:
        C.fail("CANONICAL_PROJECT_ID_CONFLICT")
    if imported["sequence"] < prior["sequence"]:
        C.fail("CANONICAL_STATE_ROLLBACK")
    if imported["sequence"] == prior["sequence"] and imported["head_digest"] != prior["head_digest"]:
        C.fail("CANONICAL_HEAD_CONFLICT")
    if imported["sequence"] > prior["sequence"] and events[prior["sequence"] - 1]["event_digest"] != prior["head_digest"]:
        C.fail("CANONICAL_HISTORY_CONFLICT")


def _resolve_validation(route, imported, events):
    obligation = route["upstream_validation"]
    if not obligation or imported["state"]["validation_state"] != "pass":
        C.fail("UPSTREAM_VALIDATION_NOT_ESTABLISHED")
    records = [e for e in events if e["payload"].get("outcome") == "full_validation_PASS"
               and e["sequence"] > obligation["project_sequence"]
               and C.stamp(e["timestamp"]) > C.stamp(obligation["changed_at"])]
    if not records:
        C.fail("UPSTREAM_VALIDATION_NOT_ESTABLISHED")
    root, payload = _root(route["project_root"]), records[-1]["payload"]
    from _evidence import verify_validation_binding
    snapshot, protocol, decision = verify_validation_binding(root, imported["state"], events)
    if decision["outcome"] != "PASS" or payload["evidence_snapshot_ref"] != imported["state"].get("evidence_snapshot_ref"):
        C.fail("UPSTREAM_VALIDATION_NOT_ESTABLISHED")
    return {"resolved_by": copy.deepcopy(payload["decision_ref"]), "snapshot_ref": copy.deepcopy(payload["evidence_snapshot_ref"])}


def checkpoint_route(root, request_id, expected_revision, request, *, at=None):
    def prepare(registry, timestamp):
        checkpoint = copy.deepcopy(request)
        checkpoint.setdefault("schema_id", "route-checkpoint")
        checkpoint.setdefault("schema_version", VERSION)
        C.validate(checkpoint, "route-checkpoint")
        if {"verification", "validation_resolution"} & set(request):
            C.fail("CHECKPOINT_VERIFICATION_IS_DERIVED")
        route = copy.deepcopy(registry["state"]["routes"].get(checkpoint["route_id"]))
        if not route:
            C.fail("ROUTE_UNKNOWN")
        route["upstream_validation"] = copy.deepcopy(registry["state"]["projects"][route["project_id"]]["upstream_validation"])
        if checkpoint.get("expected_route_revision") != route["revision"]:
            C.fail("ROUTE_REVISION_CONFLICT")
        observed = C.stamp(checkpoint["observed_at"])
        if observed < C.stamp(route["checkpoint"]["observed_at"]):
            C.fail("OLD_CHECKPOINT")
        if observed > C.stamp(timestamp):
            C.fail("FUTURE_CHECKPOINT")
        for token in checkpoint["lease_tokens"]:
            _assert_lease_state(registry["state"], token["lease_id"], route["route_id"], token["fencing_token"], timestamp)
        verified = checkpoint["kind"] == "verified"
        if verified:
            imported, events = _read_project(route["project_root"], timestamp)
            _compare_import(route["imported_state"], imported, events)
            route["imported_state"] = imported
            checkpoint["verification"] = "verified_project_ledger"
        else:
            _text(checkpoint.get("source"), "MANUAL_CHECKPOINT_SOURCE_REQUIRED")
            checkpoint["verification"] = "unverified_manual_report"
            events = []
        checkpoint["overlay"] = _overlay(checkpoint["overlay"], route["project_root"], verify_evidence=verified)
        if checkpoint.get("resolve_upstream_validation"):
            if not verified or checkpoint["changed_results"]:
                C.fail("UPSTREAM_VALIDATION_NOT_ESTABLISHED")
            checkpoint["validation_resolution"] = _resolve_validation(route, route["imported_state"], events)
            route["upstream_validation"] = None
        if checkpoint["changed_results"]:
            route["upstream_validation"] = {"required": True,
                "project_sequence": route["imported_state"]["sequence"], "changed_at": timestamp}
        route.update(revision=route["revision"] + 1, checkpoint=checkpoint,
                     status=checkpoint["status"], overlay=copy.deepcopy(checkpoint["overlay"]))
        return {"route": route}
    return _mutate(root, "checkpoint", request_id, expected_revision, request, prepare, at)


def reserve(root, request_id, expected_revision, request, *, at=None):
    def prepare(registry, timestamp):
        if not isinstance(request, dict) or set(request) != {"lease_id", "route_id", "resources", "budget", "expires_at"}:
            C.fail("LEASE_REQUEST_FIELDS_REQUIRED")
        _text(request["lease_id"])
        route = registry["state"]["routes"].get(request["route_id"])
        if not route:
            C.fail("ROUTE_UNKNOWN")
        if route["status"] not in ACTIVE:
            C.fail("ROUTE_NOT_ACTIVE")
        resources, budget = _amounts(request["resources"]), _amounts(request["budget"])
        if not resources or any(amount <= 0 for amount in resources.values()):
            C.fail("POSITIVE_RESOURCE_RESERVATION_REQUIRED")
        expires_at = _time(request["expires_at"])
        if C.stamp(expires_at) <= C.stamp(timestamp):
            C.fail("LEASE_EXPIRY_MUST_BE_FUTURE")
        reasons = []
        inventory = registry["configuration"]["resources"]
        for resource_id, amount in resources.items():
            if resource_id not in inventory:
                C.fail("RESOURCE_UNKNOWN")
            if inventory[resource_id]["kind"] == "gpu" and amount != int(amount):
                C.fail("INTEGER_GPU_RESOURCE_REQUIRED")
            if inventory[resource_id]["kind"] == "workspace" and amount != 1:
                C.fail("WORKSPACE_RESOURCE_EXCLUSIVE")
            used = sum(lease["resources"].get(resource_id, 0) for lease in registry["state"]["leases"].values() if _active(lease, timestamp))
            if used + amount > inventory[resource_id]["capacity"]:
                reasons.append("RESOURCE_CAPACITY_CONFLICT")
        for limits, accounting, exceeded in ((registry["configuration"]["budget"], _accounting(registry["state"]), "GLOBAL_BUDGET_EXCEEDED"),
            (route["budget"], _accounting(registry["state"], route["route_id"]), "ROUTE_BUDGET_EXCEEDED")):
            for key, amount in budget.items():
                if key not in limits:
                    reasons.append("BUDGET_LIMIT_UNKNOWN")
                elif accounting["spent"].get(key, 0) + accounting["held"].get(key, 0) + amount > limits[key]:
                    reasons.append(exceeded)
        reasons = sorted(set(reasons))
        lease = C.envelope("resource-lease", lease_id=request["lease_id"], route_id=route["route_id"],
            resources=resources, budget={} if reasons else budget, requested_budget=budget,
            created_at=timestamp, expires_at=expires_at, status="denied" if reasons else "reserved",
            fencing_token=0 if reasons else registry["state"]["fencing_counter"] + 1,
            spent={}, settled=bool(reasons), reason_codes=reasons)
        return {"lease": lease}
    return _mutate(root, "reserve", request_id, expected_revision, request, prepare, at)


def _lease_request(request, extra=()):
    if not isinstance(request, dict) or set(request) != {"lease_id", "route_id", "fencing_token", *extra}:
        C.fail("LEASE_REQUEST_FIELDS_REQUIRED")
    for field in ("lease_id", "route_id"):
        _text(request[field])
    if isinstance(request["fencing_token"], bool) or not isinstance(request["fencing_token"], int) or request["fencing_token"] < 1:
        C.fail("FENCING_TOKEN_REQUIRED")
    return copy.deepcopy(request)


def release(root, request_id, expected_revision, request, *, at=None):
    def prepare(registry, timestamp):
        return _lease_request(request)
    return _mutate(root, "release", request_id, expected_revision, request, prepare, at)


def record_usage(root, request_id, expected_revision, request, *, at=None):
    def prepare(registry, timestamp):
        payload = _lease_request(request, ("usage", "final", "receipt"))
        payload["usage"] = _amounts(payload["usage"])
        if not isinstance(payload["final"], bool):
            C.fail("USAGE_FINAL_FLAG_REQUIRED")
        _text(payload["receipt"], "USAGE_RECEIPT_REQUIRED")
        payload["usage_id"] = request_id
        return payload
    return _mutate(root, "record-usage", request_id, expected_revision, request, prepare, at)


def assert_lease(root, lease_id, route_id, fencing_token, *, at=None):
    registry = replay(root)
    _assert_lease_state(registry["state"], lease_id, route_id, fencing_token, _time(at))
    return True


def query_leases(root, *, at=None):
    registry, timestamp = replay(root), _time(at)
    result = {"revision": registry["revision"], "active": [], "expired": [], "released": [], "denied": []}
    for lease in sorted(registry["state"]["leases"].values(), key=lambda value: value["lease_id"]):
        category = "active" if _active(lease, timestamp) else "expired" if lease["status"] == "reserved" else lease["status"]
        result[category].append(copy.deepcopy(lease))
    return result


def _destination(registry):
    return registry["state"].get("destination", registry["configuration"]["destination"])


def set_destination(root, request_id, expected_revision, request, *, at=None):
    def prepare(registry, timestamp):
        if not isinstance(request, dict) or set(request) != {"destination"}:
            C.fail("DESTINATION_REQUEST_FIELDS_REQUIRED")
        destination = _config({}, {}, request["destination"], 3600)["destination"]
        if destination is None:
            C.fail("DURABLE_DESTINATION_MISSING")
        return {"destination": destination}
    return _mutate(root, "set-destination", request_id, expected_revision, request, prepare, at)


def record_sync_receipt(root, request_id, expected_revision, request, *, at=None):
    def prepare(registry, timestamp):
        required = {"destination", "source_revision", "source_head_digest", "source_sha256",
                    "remote_revision_before", "remote_revision_after", "operation_id", "readback_sha256"}
        if not isinstance(request, dict) or set(request) != required:
            C.fail("HOST_SYNC_RECEIPT_FIELDS_REQUIRED")
        if _destination(registry) is None:
            C.fail("DURABLE_DESTINATION_MISSING")
        if request["destination"] != _destination(registry):
            C.fail("SYNC_DESTINATION_MISMATCH")
        if request["source_revision"] != registry["revision"] or request["source_head_digest"] != registry["head_digest"]:
            C.fail("SYNC_SOURCE_MISMATCH")
        source = C.filehash(C.safe_path(_root(root), "registry.json"))
        if request["source_sha256"] != source or request["readback_sha256"] != source:
            C.fail("SYNC_READBACK_MISMATCH")
        for field in ("remote_revision_before", "remote_revision_after", "operation_id"):
            _text(request[field], "HOST_REVISION_RECEIPT_REQUIRED")
        return {**copy.deepcopy(request), "recorded_at": timestamp, "independently_verified_by_helper": False}
    return _mutate(root, "sync-receipt", request_id, expected_revision, request, prepare, at)


def overview(root, *, at=None):
    registry, timestamp = replay(root), _time(at)
    state, configuration = registry["state"], registry["configuration"]
    destination, receipts = _destination(registry), state["sync_receipts"]
    coordination_blockers = []
    if destination is None:
        durability_status = "local_only"
        coordination_blockers.append("DURABLE_DESTINATION_MISSING")
    elif not receipts:
        durability_status = "not_synced"
        coordination_blockers.append("HOST_WRITE_RECEIPT_MISSING")
    else:
        source_revision = receipts[-1]["source_revision"]
        unsynced = any(e["event_type"] != "sync-receipt" and e["revision"] > source_revision for e in registry["events"])
        durability_status = "unsynced_changes" if unsynced else "host_receipt_recorded"
        if unsynced:
            coordination_blockers.append("HOST_WRITE_RECEIPT_STALE")
    canonical = {}
    routes = []
    for route in sorted(state["routes"].values(), key=lambda value: value["route_id"]):
        reasons, classification = [], "fresh"
        if route["project_root"] not in canonical:
            try:
                imported, events = _read_project(route["project_root"], timestamp)
                canonical[route["project_root"]] = (imported, events, None)
            except C.Failure as failure:
                canonical[route["project_root"]] = (None, [], failure.code)
        current, events, error = canonical[route["project_root"]]
        if error:
            classification = "blocked"
            reasons.extend(["CANONICAL_STATE_UNAVAILABLE", error])
        else:
            try:
                _compare_import(route["imported_state"], current, events)
            except C.Failure as failure:
                classification = "conflicts"
                reasons.append(failure.code)
            if current["sequence"] > route["imported_state"]["sequence"]:
                reasons.append("CANONICAL_STATE_CHANGED")
        if route["checkpoint"]["verification"] == "unverified_manual_report":
            reasons.append("MANUAL_CHECKPOINT_UNVERIFIED")
        if (C.stamp(timestamp) - C.stamp(route["checkpoint"]["observed_at"])).total_seconds() > configuration["stale_after_seconds"]:
            reasons.append("CHECKPOINT_STALE")
        reasons.extend(route["resource_blockers"])
        if route["checkpoint"]["blockers"] or route["status"] in {"blocked", "waiting"}:
            reasons.append("WORKER_REPORTED_BLOCKER")
        if route["upstream_validation"]:
            reasons.append("UPSTREAM_VALIDATION_REQUIRED")
        if classification != "conflicts":
            if error or route["resource_blockers"] or route["checkpoint"]["blockers"] or route["status"] in {"blocked", "waiting"} or route["upstream_validation"]:
                classification = "blocked"
            elif reasons:
                classification = "stale"
        routes.append({"route_id": route["route_id"], "project_id": route["project_id"],
            "project_root": route["project_root"], "goal": route["goal"], "scope": route["scope"],
            "owner": copy.deepcopy(route["owner"]), "status": route["status"], "writer": route["writer"],
            "repository": route["repository"], "branch": route["branch"], "workspace": route["workspace"],
            "route_revision": route["revision"], "classification": classification, "reason_codes": sorted(set(reasons)),
            "checkpoint": copy.deepcopy(route["checkpoint"]), "overlay": copy.deepcopy(route["overlay"]),
            "imported_sequence": route["imported_state"]["sequence"], "imported_head_digest": route["imported_state"]["head_digest"],
            "canonical_sequence": current["sequence"] if current else None,
            "budget": copy.deepcopy(route["budget"]),
            "budget_accounting": _accounting(state, route["route_id"])})
    return {"registry_id": registry["registry_id"], "revision": registry["revision"],
            "head_digest": registry["head_digest"], "observed_at": timestamp,
            "coordination_blockers": coordination_blockers, "routes": routes,
            "projects": copy.deepcopy(state["projects"]), "budget_accounting": _accounting(state),
            "durability": {"status": durability_status, "destination": copy.deepcopy(destination),
                "last_receipt": copy.deepcopy(receipts[-1]) if receipts else None,
                "independently_verified_by_helper": False},
            "capabilities": {"arbitrary_chat_enumeration": False, "live_chat_stream": False,
                "background_chat_start": False, "compute_authority": False, "publication_authority": False}}


def export_overview(root, output, *, format="html", at=None):
    view = overview(root, at=at)
    output = _root(output)
    if output == _root(root) / "registry.json" or output == _root(root) / ".supervisor.lock":
        C.fail("EXPORT_OVERWRITE_REGISTRY_FORBIDDEN")
    if format == "html":
        from _route_visuals import render_html
        content = render_html(view)
    elif format == "mermaid":
        from _route_visuals import render_mermaid
        content = render_mermaid(view)
    elif format == "json":
        content = C.canonical(view) + "\n"
    else:
        C.fail("UNKNOWN_EXPORT_FORMAT")
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    C.atomic(output, content)
    return {"status": "exported_private_snapshot", "path": str(output), "format": format, "revision": view["revision"], "published": False}
