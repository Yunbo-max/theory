#!/usr/bin/env python3
"""Single project controller; all executable workloads use the skill harness."""
import argparse
import json
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def engineering(command, *, label, seconds, queue=None, code=None):
    from recursive_ssd.harness import execute_engineering
    if queue is None:
        if code is not None:
            raise ValueError("Explicit engineering source capture requires an original queue budget")
        return execute_engineering(command, label=label, seconds=seconds)
    from recursive_ssd import suite_queue as Q
    folder = ROOT / "runs/controller" / (label + "-" + uuid.uuid4().hex[:12])
    result = Q._engineering(ROOT, folder, command, seconds=seconds,
                            budget_path=Q.authorization_path(ROOT, queue), code=code)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "completed" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check", help="bounded CPU engineering tests")
    check.add_argument("--label", default="software-check")
    check.add_argument("--seconds", type=int, default=180)
    check.add_argument("--queue", help="charge this check to the original authorized window")
    check.add_argument("tests", nargs=argparse.REMAINDER)
    for name in ("setup", "pipcheck"):
        command = commands.add_parser(name)
        command.add_argument("--seconds", type=int, default=1800 if name == "setup" else 120)
        command.add_argument("--queue", help="charge preparation to the original authorized window")
    for name in ("assets", "model", "qualify"):
        command = commands.add_parser(name)
        command.add_argument("--data", default="artifacts/multibench-v3")
        command.add_argument("--seconds", type=int, default=1800 if name != "qualify" else 600)
        command.add_argument("--queue", help="charge preparation to this original authorized window")
        if name == "model":
            command.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
        if name == "qualify":
            command.add_argument("--benchmark", choices=("humaneval", "mbpp", "livecodebench"), default="humaneval")
            command.add_argument("--job", help="baseline job or source-bound comparator/calibration-child node request")
            command.add_argument("--bindings", help="reviewed native protocol/group/arm and exact dependency bindings")
    preflight = commands.add_parser("preflight")
    preflight.add_argument("--data", default="artifacts/multibench-v3")
    preflight.add_argument("--queue", required=True)
    preflight.add_argument("--bindings", required=True)
    preflight.add_argument("--seconds", type=int, default=1800)
    preflight.add_argument("--config")
    design = commands.add_parser("design")
    design.add_argument("--stage", choices=("development", "tuning", "confirmation", "boundary", "model_boundary"), required=True)
    design.add_argument("--selection", help="verified development selection artifact for confirmation/boundaries")
    design.add_argument("--tuning-selection", help="verified development hyperparameters for candidates and matched controls")
    design.add_argument("--output", required=True)
    build = commands.add_parser("build")
    build.add_argument("--suite", required=True)
    build.add_argument("--data", default="artifacts/multibench-v3")
    build.add_argument("--queue", default="runs/multibench-v3")
    build.add_argument("--original-start", required=True)
    build.add_argument("--cap-seconds", type=float, default=28800)
    build.add_argument("--already-used-seconds", type=float, default=0)
    build.add_argument("--budget-from", help="share the original cumulative authorization from an earlier queue")
    budget = commands.add_parser("budget", help="record the original authorization before preparation")
    budget.add_argument("--queue", default="runs/multibench-v3")
    budget.add_argument("--original-start", required=True)
    budget.add_argument("--cap-seconds", type=float, default=28800)
    budget.add_argument("--already-used-seconds", type=float, default=0)
    budget.add_argument("--budget-from")
    for name in ("native-assets", "protocol"):
        command = commands.add_parser(name)
        command.add_argument("--spec", required=True, help="JSON keyword arguments for the native admission compiler")
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--protocol", required=True)
    freeze.add_argument("--gate", choices=("gate-a", "full-validation"), default="gate-a")
    freeze.add_argument("--queue", required=True, help="original budget for actual native qualification replay")
    freeze.add_argument("--replay-timeout-seconds", type=float, default=60)
    freeze.add_argument("--replay-calls", type=int, required=True)
    admit = commands.add_parser("admit")
    admit.add_argument("--queue", default="runs/multibench-v3")
    admit.add_argument("--bundle", required=True)
    admit.add_argument("--bindings", required=True)
    for name in ("run", "resume", "status", "report", "select", "collect"):
        command = commands.add_parser(name)
        command.add_argument("--queue", default="runs/multibench-v3")
        if name in {"run", "resume"}:
            command.add_argument("--bundle", required=True)
            command.add_argument("--max-nodes", type=int)
            command.add_argument("--retry-failed", action="store_true", help="one explicit development retry; confirmation forbids retries")
        if name in {"report", "select", "collect"}:
            command.add_argument("--seconds", type=int, default=600)
        if name == "report":
            command.add_argument("--evidence-context")
        if name in {"select", "collect"}:
            command.add_argument("--output", required=True)
        if name == "collect":
            command.add_argument("--include-checkpoints", action="store_true")
    args = parser.parse_args(argv)
    from recursive_ssd.harness import execute_engineering
    if args.command == "check" and args.queue is None:
        selected = args.tests[1:] if args.tests[:1] == ["--"] else args.tests
        return execute_engineering([sys.executable, "-m", "pytest", "-q", *selected],
                                   label=args.label, seconds=args.seconds)
    from recursive_ssd import suite_queue as Q
    from recursive_ssd import suite_admission as A
    from recursive_ssd.io import atomic_json, read_json
    from recursive_ssd.suite_design import make_suite
    try:
        if args.command == "check":
            from recursive_ssd.harness import project_refs
            selected = args.tests[1:] if args.tests[:1] == ["--"] else args.tests
            # Software semantics depend on retained research fixtures as well as
            # package code. project_refs excludes runtime caches and attempts.
            return engineering([sys.executable, "-m", "pytest", "-q", *selected],
                               label=args.label, seconds=args.seconds, queue=args.queue,
                               code=project_refs(ROOT))
        if args.command in {"setup", "pipcheck"}:
            Q._conda()
            if args.command == "setup":
                code = engineering([sys.executable, "-m", "pip", "install", "torch==2.6.0",
                    "--index-url", "https://download.pytorch.org/whl/cu118"], label="native-conda-torch",
                    seconds=args.seconds, queue=args.queue)
                if code:
                    return code
                # A normal install retains an environment independent of a disposable checkout.
                code = engineering([sys.executable, "-m", "pip", "install", ".[test,evaluation]"],
                                   label="native-conda-dependencies", seconds=args.seconds, queue=args.queue)
                if code:
                    return code
            return engineering([sys.executable, "-m", "pip", "check"], label="native-conda-pipcheck",
                               seconds=min(args.seconds, 120), queue=args.queue)
        if args.command in {"assets", "model", "qualify"}:
            if args.command == "qualify" and args.job:
                if not args.bindings or not args.queue:
                    parser.error("baseline qualification --job requires --bindings and --queue")
                result = Q.qualification_job(ROOT, job=read_json(args.job), data=args.data,
                    bindings=read_json(args.bindings), seconds=args.seconds,
                    budget_path=Q.authorization_path(ROOT, args.queue))
            else:
                result = Q.prepare_operation(ROOT, kind=args.command, data=args.data, seconds=args.seconds,
                    model=getattr(args, "model", None), benchmark=getattr(args, "benchmark", "humaneval"), queue=args.queue)
        elif args.command == "preflight":
            result = Q.qualification_job(ROOT, job={"kind": "preflight", **({"config": read_json(args.config)} if args.config else {})},
                data=args.data, bindings=read_json(args.bindings), seconds=args.seconds,
                budget_path=Q.authorization_path(ROOT, args.queue))
        elif args.command == "design":
            tuning = Q.validate_selection(ROOT, args.tuning_selection, tuning=True)["tuning_parameters"] if args.tuning_selection else None
            if args.stage in {"confirmation", "boundary", "model_boundary"}:
                if not args.selection:
                    parser.error("confirmation and boundaries require --selection from complete native development")
                selected = Q.validate_selection(ROOT, args.selection)
                result = Q.compile_selected_suite(ROOT, args.stage, selected, tuning_override=tuning)
            else:
                result = make_suite(args.stage, **({"tuning_selection": tuning} if tuning else {}))
            atomic_json(Q._path(ROOT, args.output), result)
            result = {"suite": args.output, "suite_digest": result["suite_digest"],
                      "scientific_dispatch_ready": False, "scope": "complete catalog; canonical admission remains required"}
        elif args.command == "build":
            data = Q._path(ROOT, args.data)
            result = Q.build_queue(ROOT, args.queue, suite_ref=Q.ref(ROOT, args.suite),
                benchmark_manifest_ref=Q.ref(ROOT, data / "benchmarks-manifest.json"),
                model_manifest_ref=Q.ref(ROOT, data / "model-manifest.json"), original_start=args.original_start,
                cap_seconds=args.cap_seconds, already_used_seconds=args.already_used_seconds, budget_from=args.budget_from)
            result = {"queue": args.queue, "queue_digest": result["queue_digest"], "planned_nodes": len(result["nodes"]),
                      "scientific_dispatch_ready": False}
        elif args.command == "budget":
            result = Q.initialize_authorization(ROOT, args.queue, original_start=args.original_start,
                cap_seconds=args.cap_seconds, already_used_seconds=args.already_used_seconds, budget_from=args.budget_from)
        elif args.command == "native-assets":
            result = A.prepare_native_assets(ROOT, **read_json(args.spec))
        elif args.command == "protocol":
            result = A.prepare_native_protocol(ROOT, **read_json(args.spec))
        elif args.command == "freeze":
            Q._conda()
            directory = Q._path(ROOT, args.queue)
            Q._time_left(directory)
            result = A.freeze_native_protocol(ROOT, args.protocol, gate=args.gate,
                replay_context=Q._live_replay(ROOT, directory, {"replay_timeout_seconds": args.replay_timeout_seconds},
                                              {"remaining_replay_calls": args.replay_calls}))
        elif args.command == "admit":
            result = Q.admit_bundle(ROOT, args.queue, args.bundle, read_json(args.bindings))
        elif args.command in {"run", "resume"}:
            if args.max_nodes is not None and args.max_nodes < 1:
                parser.error("--max-nodes must be positive")
            result = Q.run_queue(ROOT, args.queue, bundle=args.bundle, max_nodes=args.max_nodes, retry_failed=args.retry_failed)
        elif args.command == "status":
            queue, state, _ = Q.verify_queue(ROOT, args.queue)
            result = {**Q.inventory_status(queue["nodes"], state["nodes"]),
                "original_budget": read_json(Q.authorization_path(ROOT, args.queue)),
                "remaining_unreserved_seconds": A.remaining_budget(Q.authorization_path(ROOT, args.queue)),
                "scientific_dispatch_ready": False, "scope": "retained status; dispatch rechecks exact live evidence"}
        else:
            result = Q.metadata_operation(ROOT, operation=args.command, directory=args.queue, seconds=args.seconds,
                output=getattr(args, "output", None), evidence_context=getattr(args, "evidence_context", None),
                include_checkpoints=getattr(args, "include_checkpoints", False))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 1 if isinstance(result, dict) and result.get("status") in {"failed", "budget_exhausted"} else 0
    except (ValueError, FileNotFoundError, KeyError) as error:
        print(json.dumps({"status": "blocked", "reason": str(error), "gate_advanced": False}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
