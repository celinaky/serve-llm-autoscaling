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

    def executable(self) -> str:
        executable = shutil.which("aiperf")
        if executable is None:
            raise RuntimeError(
                "aiperf is not available on PATH; run `uv sync --extra dev` first"
            )
        return executable

    def build_command(self, level: float, artifact_dir: Path) -> list[str]:
        benchmark = self.config.benchmark
        workload = benchmark.workload
        deployment = self.config.deployment
        cmd = [
            self.executable(),
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
        if benchmark.mode == "concurrency":
            cmd.extend(["--concurrency", str(int(level))])
        else:
            cmd.extend(
                [
                    "--request-rate", str(level),
                    "--arrival-pattern", benchmark.arrival_pattern,
                ]
            )
        cmd.extend(benchmark.extra_args)
        return cmd

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
                "mode": mode,
                "level": int(level) if float(level).is_integer() else level,
                "elapsed_s": elapsed,
                "artifact_dir": str(point_dir),
                "error": None,
            }
        )
        return normalized
