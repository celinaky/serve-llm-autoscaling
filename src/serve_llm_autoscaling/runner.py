from __future__ import annotations

import importlib.metadata
import platform
import sys
import time
from pathlib import Path
from typing import Any

from .artifacts import RunArtifacts, write_json
from .backend import RayServeBackend
from .benchmark import AIPerfRunner
from .config import ExperimentConfig


def environment_check(connect: bool = True) -> dict[str, Any]:
    result: dict[str, Any] = {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "aiperf_version": importlib.metadata.version("aiperf"),
    }
    try:
        import ray
    except ImportError as exc:
        raise RuntimeError(
            "Ray is not importable. In Anyscale, create the project environment "
            "with `uv venv --system-site-packages` before running `uv sync`."
        ) from exc
    result.update({"ray_version": ray.__version__, "ray_path": ray.__file__})
    if connect:
        ray.init(address="auto", ignore_reinit_error=True)
        result["cluster_resources"] = ray.cluster_resources()
        if result["cluster_resources"].get("GPU", 0) < 1:
            raise RuntimeError("the connected Ray cluster does not report any GPUs")
    return result


def run_benchmarks(config: ExperimentConfig, root: Path) -> list[dict[str, Any]]:
    runner = AIPerfRunner(config, root / "benchmark")
    summary: list[dict[str, Any]] = []
    for level in config.benchmark.levels:
        try:
            row = runner.run_point(level)
        except Exception as exc:
            row = {
                "mode": config.benchmark.mode,
                "level": int(level) if float(level).is_integer() else level,
                "error": f"{type(exc).__name__}: {exc}",
            }
            summary.append(row)
            write_json(root / "benchmark" / "sweep_summary.json", summary)
            if config.benchmark.fail_fast:
                raise
        else:
            summary.append(row)
            write_json(root / "benchmark" / "sweep_summary.json", summary)
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
        summary = run_benchmarks(config, artifacts.root)
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
