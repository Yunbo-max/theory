#!/usr/bin/env python3
"""Coordinate private route records; external chat/host/compute actions remain separate."""
import sys
from pathlib import Path

import _autoresearch as C
import _supervisor as S


def main(argv=None):
    parser = C.SafeArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init")
    init.add_argument("--root", required=True)
    init.add_argument("--registry-id", required=True)
    init.add_argument("--configuration", help="JSON resources/budget/destination/stale_after_seconds")
    for name in sorted(S.OPERATIONS):
        command = commands.add_parser(name)
        command.add_argument("--root", required=True)
        command.add_argument("--request-id", required=True)
        command.add_argument("--expected-revision", required=True, type=int)
        command.add_argument("--request", required=True, help="JSON request file")
    for name in ("overview", "replay", "leases"):
        command = commands.add_parser(name)
        command.add_argument("--root", required=True)
    export = commands.add_parser("export")
    export.add_argument("--root", required=True)
    export.add_argument("--output", required=True)
    export.add_argument("--format", choices=("html", "mermaid", "json"), default="html")
    fence = commands.add_parser("check-lease")
    fence.add_argument("--root", required=True)
    fence.add_argument("--lease-id", required=True)
    fence.add_argument("--route-id", required=True)
    fence.add_argument("--fencing-token", required=True, type=int)
    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            configuration = C.load_file(Path(args.configuration)) if args.configuration else {}
            if not isinstance(configuration, dict) or set(configuration) - {"resources", "budget", "destination", "stale_after_seconds"}:
                C.fail("INVALID_REGISTRY_CONFIGURATION")
            result = S.initialize(args.root, args.registry_id, **configuration)
        elif args.command in S.OPERATIONS:
            method = {"register": S.register_route, "checkpoint": S.checkpoint_route, "reserve": S.reserve,
                      "release": S.release, "record-usage": S.record_usage, "sync-receipt": S.record_sync_receipt,
                      "set-destination": S.set_destination}[args.command]
            result = method(args.root, args.request_id, args.expected_revision, C.load_file(Path(args.request)))
        elif args.command == "overview":
            result = S.overview(args.root)
        elif args.command == "replay":
            result = S.replay(args.root)
        elif args.command == "leases":
            result = S.query_leases(args.root)
        elif args.command == "check-lease":
            result = {"valid": S.assert_lease(args.root, args.lease_id, args.route_id, args.fencing_token)}
        else:
            result = S.export_overview(args.root, args.output, format=args.format)
        print(C.canonical(result))
        return 0
    except C.Failure as failure:
        print(C.canonical({"error": {"code": failure.code, "path": failure.path}}))
        return 2
    except (OSError, KeyError, TypeError, ValueError, UnicodeError):
        print(C.canonical({"error": {"code": "INVALID_OR_UNAVAILABLE_SUPERVISOR_INPUT", "path": "$"}}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
