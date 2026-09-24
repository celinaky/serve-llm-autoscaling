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
from .backend import RayServeBackend, check_routing_support, verify_topology
from .benchmark import AIPerfRunner, aiperf_command
from .config import (
    CONTINUOUS_MODE_DIRS,
    CONTINUOUS_MODES,
    AgentXWorkloadConfig,
    ExperimentConfig,
    write_config,
)
from .telemetry import TelemetrySession
from .windows import profiling_phase_ns

LIFECYCLE_SCHEMA_VERSION = 1


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


def _run_job(
    config: ExperimentConfig,
    root: Path,
    level: Any,
    job: Callable[[], dict[str, Any]],
    summary: list[dict[str, Any]],
) -> None:
    """Run one benchmark, recording a failure row unless ``fail_fast``."""
    workload = config.benchmark.workload
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


def lifecycle_record(
    config: ExperimentConfig,
    artifact_dir: Path,
    *,
    aiperf_launched_ns: int,
    aiperf_exited_ns: int,
    observation_end_ns: int | None = None,
    observation_skipped: str | None = None,
) -> dict[str, Any]:
    """Epoch-ns boundaries of a continuous run's load, drain and idle periods.

    Load issuance ends ``duration_s`` after profiling starts; AIPerf then
    drains in-flight requests (up to ``grace_period_s``) and exits; the
    harness then observes an idle deployment for ``post_load_observation_s``.
    """
    benchmark = config.benchmark
    workload = benchmark.workload
    start, phase_end = profiling_phase_ns(artifact_dir)

    def offset(seconds: float) -> int | None:
        return None if start is None else start + round(seconds * 1e9)

    return {
        "schema_version": LIFECYCLE_SCHEMA_VERSION,
        "clock": "epoch_ns",
        "aiperf_launched_ns": aiperf_launched_ns,
        "profiling_start_ns": start,
        "ramp_end_ns": (
            offset(workload.concurrency_ramp_duration_s)
            if isinstance(workload, AgentXWorkloadConfig) else None
        ),
        "load_end_ns": offset(benchmark.duration_s),
        "profiling_phase_end_ns": phase_end,
        "aiperf_exited_ns": aiperf_exited_ns,
        "post_load_observation_s": benchmark.post_load_observation_s,
        "observation_end_ns": observation_end_ns,
        "observation_skipped": observation_skipped,
    }


def _run_continuous(
    config: ExperimentConfig,
    root: Path,
    level: Any,
    job: Callable[[], dict[str, Any]],
    summary: list[dict[str, Any]],
    observe: bool,
) -> None:
    """Run the single AIPerf process, then observe the idle deployment."""
    artifact_dir = root / "benchmark" / CONTINUOUS_MODE_DIRS[config.benchmark.mode]
    path = root / "benchmark" / "lifecycle.json"
    launched = time.time_ns()
    try:
        _run_job(config, root, level, job, summary)
    finally:
        # Written even when fail_fast re-raises, which skips the observation.
        exited = time.time_ns()
        write_json(path, lifecycle_record(
            config, artifact_dir, aiperf_launched_ns=launched, aiperf_exited_ns=exited,
        ))
    observation_s = config.benchmark.post_load_observation_s
    if not observation_s:
        return
    if not observe:
        write_json(path, lifecycle_record(
            config, artifact_dir, aiperf_launched_ns=launched, aiperf_exited_ns=exited,
            observation_skipped="no telemetry",
        ))
        return
    time.sleep(observation_s)
    write_json(path, lifecycle_record(
        config, artifact_dir, aiperf_launched_ns=launched, aiperf_exited_ns=exited,
        observation_end_ns=time.time_ns(),
    ))


def run_benchmarks(
    config: ExperimentConfig, root: Path, telemetry: TelemetrySession | None = None
) -> list[dict[str, Any]]:
    """Run the configured benchmarks, collecting ``telemetry`` for their duration.

    For continuous modes, telemetry also spans the post-load observation.
    """
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
        if config.benchmark.mode in CONTINUOUS_MODES:
            (level, job), = jobs
            _run_continuous(config, root, level, job, summary, observe=telemetry is not None)
        else:
            for level, job in jobs:
                _run_job(config, root, level, job, summary)
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

        stage = "routing_preflight"
        routing = {**backend.routing_summary(), "resolved": check_routing_support(config)}
        routing_path = artifacts.root / "deployment" / "routing.json"
        write_json(routing_path, routing)
        artifacts.manifest["routing"] = routing["resolved"]
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

        stage = "topology"
        routing["verified"] = verify_topology(config, ready["serve_status"])
        write_json(routing_path, routing)
        artifacts.manifest["routing"] = {**routing["resolved"], **routing["verified"]}
        artifacts.flush_manifest()

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
