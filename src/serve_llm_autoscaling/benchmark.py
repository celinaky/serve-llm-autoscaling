from __future__ import annotations

import json
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import ExperimentConfig
from .windows import windowed_summary

# AIPerf 0.12 needs Python 3.11+, which the Ray image's driver may not have.
AIPERF_COMMAND = ["uvx", "--python", "3.11", "--from", "aiperf==0.12.0", "aiperf"]


def aiperf_command() -> list[str]:
    if shutil.which("uvx") is None:
        raise RuntimeError("uvx not on PATH; install uv to run AIPerf")
    return list(AIPERF_COMMAND)


def _metric_value(data: dict[str, Any], key: str, stat: str = "avg") -> Any:
    value = data.get(key)
    if isinstance(value, dict):
        return value.get(stat)
    return value


def normalize_aiperf(data: dict[str, Any]) -> dict[str, Any]:
    errors = data.get("error_summary") or []
    failed_requests = _metric_value(data, "error_request_count")
    if failed_requests is None:
        failed_requests = sum(
            int(item.get("count", 0)) for item in errors if isinstance(item, dict)
        )
    return {
        "request_count": _metric_value(data, "request_count"),
        "failed_requests": failed_requests,
        "request_throughput": _metric_value(data, "request_throughput"),
        "output_token_throughput": _metric_value(data, "output_token_throughput"),
        "total_token_throughput": _metric_value(data, "total_token_throughput"),
        "mean_ttft_ms": _metric_value(data, "time_to_first_token", "avg"),
        "p50_ttft_ms": _metric_value(data, "time_to_first_token", "p50"),
        "p90_ttft_ms": _metric_value(data, "time_to_first_token", "p90"),
        "p99_ttft_ms": _metric_value(data, "time_to_first_token", "p99"),
        "mean_tpot_ms": _metric_value(data, "inter_token_latency", "avg"),
        "p50_tpot_ms": _metric_value(data, "inter_token_latency", "p50"),
        "p90_tpot_ms": _metric_value(data, "inter_token_latency", "p90"),
        "p99_tpot_ms": _metric_value(data, "inter_token_latency", "p99"),
        "mean_e2el_ms": _metric_value(data, "request_latency", "avg"),
        "p99_e2el_ms": _metric_value(data, "request_latency", "p99"),
    }


@dataclass
class AIPerfRunner:
    config: ExperimentConfig
    benchmark_root: Path

    def _base_command(self, artifact_dir: Path) -> list[str]:
        benchmark = self.config.benchmark
        workload = benchmark.workload
        deployment = self.config.deployment
        cmd = [
            *aiperf_command(),
            "profile",
            "--model", deployment.model_id,
            "--tokenizer", str(deployment.tokenizer),
            "--url", self.config.runtime.endpoint_url,
            "--endpoint-type", "chat",
            "--isl", str(workload.input_tokens),
            "--isl-stddev", str(workload.input_tokens_stddev),
            "--osl", str(workload.output_tokens),
            "--osl-stddev", str(workload.output_tokens_stddev),
            "--random-seed", str(workload.seed),
            "--benchmark-duration", str(benchmark.duration_s),
            "--benchmark-grace-period", str(benchmark.grace_period_s),
            "--artifact-dir", str(artifact_dir),
            "--export-level", "records",
            "--ui", "none",
        ]
        if benchmark.streaming:
            cmd.append("--streaming")
        if benchmark.use_server_token_count:
            cmd.append("--use-server-token-count")
        if workload.ignore_eos:
            cmd.extend(["--extra-inputs", "ignore_eos:true"])
        if benchmark.warmup.enabled:
            cmd.extend(["--warmup-duration", str(benchmark.warmup.duration_s)])
        return cmd

    def _arrival_args(self) -> list[str]:
        benchmark = self.config.benchmark
        args = ["--arrival-pattern", benchmark.arrival_pattern]
        if benchmark.arrival_smoothness is not None:
            args.extend(["--arrival-smoothness", str(benchmark.arrival_smoothness)])
        return args

    def build_command(self, level: float, artifact_dir: Path) -> list[str]:
        benchmark = self.config.benchmark
        cmd = self._base_command(artifact_dir)
        if benchmark.mode == "concurrency":
            cmd.extend(["--concurrency", str(int(level))])
        else:
            cmd.extend(["--request-rate", str(level), *self._arrival_args()])
        cmd.extend(benchmark.extra_args)
        return cmd

    def build_series_command(self, series_path: Path, artifact_dir: Path) -> list[str]:
        cmd = self._base_command(artifact_dir)
        # AIPerf refuses series paths with symlinked components; resolve them.
        cmd.extend(
            ["--request-rate-series", str(series_path.resolve()), *self._arrival_args()]
        )
        cmd.extend(self.config.benchmark.extra_args)
        return cmd

    def write_rate_series(self, path: Path) -> None:
        points = self.config.benchmark.rate_series or []
        with path.open("w") as fh:
            json.dump({"points": [point.model_dump() for point in points]}, fh, indent=2)
            fh.write("\n")

    def run_point(self, level: float) -> dict[str, Any]:
        mode = self.config.benchmark.mode
        label_value = (
            str(int(level))
            if float(level).is_integer()
            else str(level).replace(".", "p")
        )
        point_dir = self.benchmark_root / f"{mode}-{label_value}"
        point_dir.mkdir(parents=True, exist_ok=False)
        cmd = self.build_command(level, point_dir)
        return self._execute(
            cmd, point_dir, int(level) if float(level).is_integer() else level
        )

    def run_series(self) -> dict[str, Any]:
        series_dir = self.benchmark_root / "request-rate-series"
        series_dir.mkdir(parents=True, exist_ok=False)
        series_path = series_dir / "rate_series.json"
        self.write_rate_series(series_path)
        cmd = self.build_series_command(series_path, series_dir)
        row = self._execute(cmd, series_dir, "series")
        try:
            windows = windowed_summary(series_dir, self.config.benchmark.duration_s)
        except Exception as exc:  # Keep AIPerf's result if windowing fails.
            row["windows_error"] = f"{type(exc).__name__}: {exc}"
        else:
            with (series_dir / "windows.json").open("w") as fh:
                json.dump(windows, fh, indent=2)
        return row

    def _execute(self, cmd: list[str], point_dir: Path, level: Any) -> dict[str, Any]:
        with (point_dir / "command.json").open("w") as fh:
            json.dump({"argv": cmd, "display": shlex.join(cmd)}, fh, indent=2)
            fh.write("\n")

        started = time.monotonic()
        with (point_dir / "stdout.log").open("w") as stdout, (
            point_dir / "stderr.log"
        ).open("w") as stderr:
            completed = subprocess.run(cmd, stdout=stdout, stderr=stderr, check=False)
        elapsed = time.monotonic() - started
        if completed.returncode != 0:
            raise RuntimeError(
                f"AIPerf exited {completed.returncode}; see {point_dir / 'stderr.log'}"
            )

        result_path = point_dir / "profile_export_aiperf.json"
        if not result_path.exists():
            candidates = list(point_dir.rglob("profile_export_aiperf.json"))
            if len(candidates) != 1:
                raise RuntimeError(
                    f"expected one profile_export_aiperf.json below {point_dir}, "
                    f"found {len(candidates)}"
                )
            result_path = candidates[0]
        with result_path.open() as fh:
            raw = json.load(fh)
        normalized = normalize_aiperf(raw)
        normalized.update(
            {
                "mode": self.config.benchmark.mode,
                "level": level,
                "elapsed_s": elapsed,
                "artifact_dir": str(point_dir),
                "error": None,
            }
        )
        return normalized
