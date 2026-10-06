"""Report stale manuscript, figure, website and promotion assets from claim cards."""
import argparse
from pathlib import Path
import _autoresearch as C
from _writing import refresh_assets


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--claims", required=True, help="Frozen protocol containing claim_cards and version identities")
    p.add_argument("--output", help="Optional project-relative private dependency report")
    args = p.parse_args()
    try:
        root = C.root_path(args.root)
        protocol = C.load_file(Path(args.claims))
        report = refresh_assets(root, C.load_file(Path(args.manifest)), protocol["claim_cards"],
            method_version=protocol["method_version"], project_version=protocol["project_version"])
        C.scan(report)
        if args.output:
            C.atomic(C.safe_path(root, args.output), C.canonical(report) + "\n")
        print(C.canonical(report))
    except C.Failure as e:
        print(C.canonical({"error": {"code": e.code, "path": e.path}}))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
