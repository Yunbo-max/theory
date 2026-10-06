"""Evidence-linked work identity, with version-specific scientific status.

No network lookup or directory scan occurs during canonicalization. Callers pass
the exact captured records, so later publications cannot rewrite older replay.
"""
import copy
import re
import urllib.parse
import _autoresearch as C


def normalized_identifier(identifier):
    if not isinstance(identifier, str) or not identifier.strip():
        C.fail("WORK_IDENTIFIER_REQUIRED")
    value = identifier.strip()
    url = urllib.parse.urlsplit(value)
    if url.scheme in {"http", "https"} and url.hostname in {"doi.org", "dx.doi.org"}:
        value = "doi:" + urllib.parse.unquote(url.path.lstrip("/"))
    elif url.scheme in {"http", "https"} and url.hostname in {"arxiv.org", "www.arxiv.org"}:
        match = re.fullmatch(r"/(?:abs|pdf)/(.+?)(?:\.pdf)?", url.path)
        if match:
            value = "arxiv:" + match.group(1)
    if value.lower().startswith("doi:"):
        return "doi:" + value[4:].lower()
    if value.lower().startswith("arxiv:"):
        return "arxiv:" + re.sub(r"v[1-9][0-9]*$", "", value[6:])
    return value


def _version_id(record):
    if record.get("version_id"):
        return record["version_id"]
    original = record["work_id"].strip()
    identifier = normalized_identifier(original)
    if identifier.startswith("arxiv:"):
        supplied = re.search(r"v[1-9][0-9]*$", original.removesuffix(".pdf"))
        version = supplied.group() if supplied else record["version"]
        if re.fullmatch(r"v[1-9][0-9]*", version):
            return identifier + version
    return identifier + "#" + record["version"]


def _refs(items):
    unique = {(r["path"], r["sha256"]): r for r in items}
    return [unique[k] for k in sorted(unique)]


def _identity_index(root, records):
    leaders, retained = {}, []

    def find(identifier):
        leaders.setdefault(identifier, identifier)
        if leaders[identifier] != identifier:
            leaders[identifier] = find(leaders[identifier])
        return leaders[identifier]

    def preference(identifier):
        # This is an identifier choice, never evidence of formal publication.
        return (0 if identifier.startswith("doi:") else
                1 if identifier.startswith("proceedings:") else 2, identifier)

    for record in records:
        own = normalized_identifier(record["work_id"])
        find(own)
        for link in record.get("identity_links", []):
            first = normalized_identifier(link["from_id"])
            second = normalized_identifier(link["to_id"])
            if own not in {first, second} or link.get("relation") != "same_work" or not link.get("evidence_refs"):
                C.fail("PRIMARY_WORK_IDENTITY_LINK_REQUIRED")
            for reference in link["evidence_refs"]:
                source = C.load_file(C.verify_ref(root, reference))
                if not isinstance(source, dict):
                    C.fail("PRIMARY_WORK_IDENTITY_LINK_REQUIRED")
                identifiers = source.get("identifiers", [])
                if not isinstance(identifiers, list):
                    C.fail("PRIMARY_WORK_IDENTITY_LINK_REQUIRED")
                source_url = urllib.parse.urlsplit(source.get("source_url", ""))
                if (source.get("capture_format") != "work-identity-link-v1" or
                    source.get("primary_source") is not True or source.get("relation") != "same_work" or
                    not {first, second}.issubset({normalized_identifier(i) for i in identifiers}) or
                    source_url.scheme not in {"http", "https"} or not source_url.hostname or
                    not source.get("locator") or not source.get("excerpt")):
                    C.fail("PRIMARY_WORK_IDENTITY_LINK_REQUIRED")
                C.scan(source)
                retained.append(reference)
            left, right = find(first), find(second)
            preferred = min((left, right), key=preference)
            leaders[left] = preferred
            leaders[right] = preferred
    return {key: find(key) for key in sorted(leaders)}, _refs(retained)


def canonical_work_id(root, work_record_or_id, records=None):
    """Return a canonical underlying-work identifier using only supplied evidence."""
    is_record = isinstance(work_record_or_id, dict)
    record = C.validate(work_record_or_id, "work-record") if is_record else None
    identity = normalized_identifier(record["work_id"] if is_record else work_record_or_id)
    known = list(records or [])
    if record is not None and record not in known:
        known.append(record)
    for item in known:
        C.validate(item, "work-record")
    mapping, _ = _identity_index(root, known)
    return mapping.get(identity, identity)


def canonicalize_search_records(root, records):
    """Return deterministic version records and separately deduplicated work IDs.

    Work records retain their normalized source identifier in ``work_id``.
    ``canonical_work_id`` binds equivalent sources without promoting one
    version's reading, publication or reproduction status to another version.
    """
    records = [copy.deepcopy(C.validate(record, "work-record")) for record in records]
    mapping, evidence = _identity_index(root, records)
    versions = {}
    for record in records:
        version = _version_id(record)
        identity = normalized_identifier(record["work_id"])
        record["work_id"] = identity
        record["title"] = " ".join(record["title"].split())
        url = urllib.parse.urlsplit(record["canonical_url"])
        if url.scheme not in {"http", "https"} or not url.hostname:
            C.fail("WORK_URL_INVALID")
        record["canonical_url"] = urllib.parse.urlunsplit((url.scheme.lower(), url.netloc.lower(), url.path, url.query, ""))
        record["canonical_work_id"] = mapping.get(identity, identity)
        record["version_id"] = version
        record.setdefault("publication_status", "unknown")
        record.setdefault("reproduction_status", "unknown")
        record["identity_status"] = "verified_link" if any(key != identity and value == record["canonical_work_id"]
                                                          for key, value in mapping.items()) else "single_identifier"
        record["unresolved_aliases"] = sorted({normalized_identifier(alias) for alias in record.get("identity_aliases", [])
            if mapping.get(normalized_identifier(alias), normalized_identifier(alias)) != record["canonical_work_id"]})
        for reference in record["evidence_refs"]:
            C.verify_ref(root, reference)
        key = (record["canonical_work_id"], version)
        if key in versions and versions[key] != record:
            C.fail("CONFLICTING_CANONICAL_WORK_VERSION")
        versions[key] = record
    works = [versions[key] for key in sorted(versions)]
    return {"works": works, "work_ids": sorted({w["canonical_work_id"] for w in works}),
            "version_ids": sorted({w["version_id"] for w in works}),
            "d2_work_ids": sorted({w["canonical_work_id"] for w in works if w["read_depth"] in {"D2", "D3"}}),
            "identity_evidence_refs": evidence, "identity_map": mapping}
