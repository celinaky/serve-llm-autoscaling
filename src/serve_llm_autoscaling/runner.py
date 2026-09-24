from __future__ import annotations

import platform
import subprocess
import sys
import time
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from typing import Any, Callable

from .artifacts import RunArtifacts, format_run_summary, write_json
from .backend import RayServeBackend
from .benchmark import AIPerfRunner, aiperf_command
from .config import CONTINUOUS_MODES, AgentXWorkloadConfig, ExperimentConfig, write_config
from .telemetry import TelemetrySession


def _aiperf_version() -> str:
    completed = subprocess.run(
        [*aiperf_command(), "--version"], capture_output=True, text=True, check=True
    )
    return completed.stdout.strip()


def environment_check(connect: bool = True) -> dict[str, Any]:
    # The driver must share the workers' environment.
    if sys.prefix != sys.base_prefix:
        raise RuntimeError(f"run on the Ray image's Python, not a venv ({sys.prefix})")
    try:
        import ray
    except ImportError as exc:
        raise RuntimeError("Ray is not importable") from exc
    result: dict[str, Any] = {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "aiperf_version": _aiperf_version(),
    }
    result.update({"ray_version": ray.__version__, "ray_path": ray.__file__})
    if connect:
        ray.init(address="auto", ignore_reinit_error=True)
        result["cluster_resources"] = ray.cluster_resources()
        if result["cluster_resources"].get("GPU", 0) < 1:
            raise RuntimeError("the connected Ray cluster does not report any GPUs")
    return result


def run_benchmarks(
    config: ExperimentConfig, root: Path, telemetry: TelemetrySession | None = None
) -> list[dict[str, Any]]:
    """Run the configured benchmarks, collecting ``telemetry`` for their duration."""
    runner = AIPerfRunner(config, root / "benchmark")
    summary: list[dict[str, Any]] = []
    workload = config.benchmark.workload
    jobs: list[tuple[Any, Callable[[], dict[str, Any]]]]
    # Continuous modes are one AIPerf process, summarized as a single row.
    if config.benchmark.mode == "request_rate_series":
        jobs = [("series", runner.run_series)]
    elif isinstance(workload, AgentXWorkloadConfig):
        jobs = [(workload.target_concurrency, runner.run_agentx_ramp)]
    else:
        jobs = [
            (
                int(level) if float(level).is_integer() else level,
                partial(runner.run_point, level),
            )
            for level in config.benchmark.levels
        ]
    # Collectors stop and flush on success, benchmark failure, or interrupt.
    with telemetry or nullcontext():
        for level, job in jobs:
            try:
                row = job()
            except Exception as exc:
                row = {"mode": config.benchmark.mode, "level": level}
                if isinstance(workload, AgentXWorkloadConfig):
                    row["target_concurrency"] = workload.target_concurrency
                    row["ramp_duration_s"] = workload.concurrency_ramp_duration_s
                row["error"] = f"{type(exc).__name__}: {exc}"
                summary.append(row)
                write_json(root / "benchmark" / "sweep_summary.json", summary)
                if config.benchmark.fail_fast:
                    raise
            else:
                summary.append(row)
                write_json(root / "benchmark" / "sweep_summary.json", summary)
    return summary


def run_analysis(root: Path) -> dict[str, Any]:
    """Derive a continuous run's analysis; failures are recorded, never raised."""
    from .analysis import analyze_run

    try:
        return analyze_run(root)
    except Exception as exc:
        return {"analysis_error": f"{type(exc).__name__}: {exc}"}


def run_manual_benchmark(config: ExperimentConfig, root: Path) -> list[dict[str, Any]]:
    """Benchmark an existing deployment, with telemetry if Ray is reachable."""
    (root / "benchmark").mkdir(parents=True, exist_ok=True)
    if config.benchmark.mode not in CONTINUOUS_MODES:
        return run_benchmarks(config, root)
    write_config(config, root / "resolved.yaml")
    telemetry = None
    try:
        RayServeBackend(config).connect()
        telemetry = TelemetrySession(config, root)
    except Exception as exc:
        print(
            f"WARNING: running without Serve telemetry; could not connect to Ray "
            f"({type(exc).__name__}: {exc})",
            file=sys.stderr,
        )
    summary = run_benchmarks(config, root, telemetry)
    for warning in (analysis := run_analysis(root)).get("warnings", []):
        print(f"WARNING: {warning}", file=sys.stderr)
    for key in ("analysis_error", "plot_error"):
        if analysis.get(key):
            print(f"WARNING: {key}: {analysis[key]}", file=sys.stderr)
    return summary


def run_experiment(
    config: ExperimentConfig,
    config_path: Path,
    *,
    replace: bool = False,
    keep_deployment: bool | None = None,
) -> Path:
    artifacts = RunArtifacts.create(config, config_path)
    backend = RayServeBackend(config, replace=replace)
    deployed = False
    stage = "environment_check"
    keep = config.runtime.keep_deployment if keep_deployment is None else keep_deployment
    try:
        env = environment_check(connect=True)
        artifacts.manifest["environment"] = env
        artifacts.flush_manifest()

        stage = "deploy"
        artifacts.record_event("deploy_started")
        write_json(
            artifacts.root / "deployment" / "config.json",
            backend.deployment_spec(),
        )
        status = backend.deploy()
        deployed = True
        write_json(artifacts.root / "deployment" / "initial_status.json", status)

        stage = "readiness"
        ready_started = time.monotonic()
        ready = backend.wait_healthy()
        artifacts.manifest["readiness_s"] = time.monotonic() - ready_started
        artifacts.flush_manifest()
        write_json(artifacts.root / "deployment" / "ready_status.json", ready)
        artifacts.record_event("deployment_ready")

        stage = "benchmark"
        continuous = config.benchmark.mode in CONTINUOUS_MODES
        # Start telemetry immediately before AIPerf; static sweeps collect none.
        telemetry = TelemetrySession(config, artifacts.root) if continuous else None
        try:
            summary = run_benchmarks(config, artifacts.root, telemetry)
        finally:
            if telemetry is not None:
                artifacts.manifest["telemetry"] = telemetry.summary()
                artifacts.flush_manifest()
        if continuous:
            stage = "analysis"
            artifacts.manifest["analysis"] = run_analysis(artifacts.root)
            artifacts.flush_manifest()
        if not any(row.get("error") is None for row in summary):
            raise RuntimeError("all benchmark points failed")
        artifacts.finish("succeeded")
        return artifacts.root
    except Exception as exc:
        artifacts.finish("failed", failure_stage=stage, error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if deployed and not keep:
            artifacts.record_event("teardown_started")
            try:
                backend.teardown()
                artifacts.record_event("teardown_completed")
            except Exception as exc:
                artifacts.record_event(
                    "teardown_failed", error=f"{type(exc).__name__}: {exc}"
                )
        elif deployed:
            artifacts.record_event("teardown_skipped")
        print(format_run_summary(artifacts.root), flush=True)
