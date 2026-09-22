from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from .backend import RayServeBackend
from .benchmark import AIPerfRunner
from .config import load_config
from .runner import environment_check, run_benchmarks, run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="autoscale-harness")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("check", help="Validate the local and Ray environments")
    for name in ("validate", "status", "teardown", "benchmark"):
        command = subparsers.add_parser(name)
        command.add_argument("config", type=Path)
    deploy = subparsers.add_parser("deploy")
    deploy.add_argument("config", type=Path)
    deploy.add_argument("--replace", action="store_true")
    run = subparsers.add_parser("run")
    run.add_argument("config", type=Path)
    run.add_argument("--replace", action="store_true")
    run.add_argument("--keep-deployment", action="store_true", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "check":
            print(json.dumps(environment_check(connect=True), indent=2, default=str))
            return 0

        config = load_config(args.config)
        if args.command == "validate":
            print(yaml.safe_dump(config.resolved_dict(), sort_keys=False))
        elif args.command == "status":
            backend = RayServeBackend(config)
            backend.connect()
            print(json.dumps(backend.status(), indent=2, default=str))
        elif args.command == "deploy":
            backend = RayServeBackend(config, replace=args.replace)
            print(json.dumps(backend.connect(), indent=2, default=str))
            print(json.dumps(backend.deploy(), indent=2, default=str))
            print(json.dumps(backend.wait_healthy(), indent=2, default=str))
        elif args.command == "teardown":
            backend = RayServeBackend(config)
            backend.connect()
            backend.teardown()
        elif args.command == "benchmark":
            root = config.runtime.results_dir / f"manual-{config.name}"
            (root / "benchmark").mkdir(parents=True, exist_ok=True)
            print(json.dumps(run_benchmarks(config, root), indent=2, default=str))
        elif args.command == "run":
            root = run_experiment(
                config,
                args.config,
                replace=args.replace,
                keep_deployment=args.keep_deployment,
            )
            print(f"Run artifacts: {root}")
        return 0
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

