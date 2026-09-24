from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .config import ExperimentConfig, write_config


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_revision() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(value, fh, indent=2, sort_keys=True, default=str)
        fh.write("\n")


def _read_json(path: Path) -> Any:
    try:
        with path.open() as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _fmt(value: Any, spec: str = ".1f", suffix: str = "") -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:{spec}}{suffix}"
    return f"{value}{suffix}"


def _duration(started: str | None, finished: str | None) -> str | None:
    if not started or not finished:
        return None
    seconds = (
        datetime.fromisoformat(finished) - datetime.fromisoformat(started)
    ).total_seconds()
    minutes, seconds = divmod(int(round(seconds)), 60)
    return f"{minutes}m {seconds:02d}s" if minutes else f"{seconds}s"


def _teardown_state(root: Path) -> str:
    state = "not run"
    try:
        lines = (root / "deployment" / "events.jsonl").read_text().splitlines()
    except OSError:
        return state
    for line in lines:
        event = json.loads(line)
        if event["event"] == "teardown_completed":
            state = "completed"
        elif event["event"] == "teardown_skipped":
            state = "skipped (deployment kept running)"
        elif event["event"] == "teardown_failed":
            state = f"FAILED ({event.get('error')})"
    return state


def format_run_summary(root: Path) -> str:
    """Render a human-readable summary of a run directory for the terminal."""
    manifest = _read_json(root / "manifest.json") or {}
    points = _read_json(root / "benchmark" / "sweep_summary.json") or []

    failed_points = [p for p in points if p.get("error")]
    failed_requests = sum(p.get("failed_requests") or 0 for p in points)
    status = manifest.get("status")
    if status == "succeeded" and not failed_points and not failed_requests:
        verdict = "SUCCEEDED (no errors)"
    elif status == "succeeded":
        problems = []
        if failed_points:
            problems.append(f"{len(failed_points)} of {len(points)} points failed")
        if failed_requests:
            problems.append(f"{failed_requests} failed requests")
        verdict = f"SUCCEEDED WITH ERRORS ({', '.join(problems)})"
    elif status == "failed":
        verdict = f"FAILED at stage '{manifest.get('failure_stage')}'"
    else:
        verdict = "INCOMPLETE (interrupted)"

    duration = _duration(manifest.get("started_at"), manifest.get("finished_at"))
    readiness = manifest.get("readiness_s")
    timing = _fmt(duration)
    if readiness is not None:
        timing += f" (deployment ready after {readiness:.1f}s)"

    lines = [
        "",
        f"=== Run summary: {manifest.get('name', root.name)} ===",
        f"Status:    {verdict}",
    ]
    if manifest.get("error"):
        lines.append(f"Error:     {manifest['error']}")
    lines += [
        f"Duration:  {timing}",
        f"Teardown:  {_teardown_state(root)}",
        f"Artifacts: {root}",
    ]

    if points:
        header = (
            f"{'mode':<20}{'level':>7}{'requests':>10}{'failed':>8}{'req/s':>8}"
            f"{'TTFT p50':>11}{'TTFT p99':>11}{'TPOT p50':>11}{'E2E p99':>11}"
        )
        lines += ["", header]
        for p in points:
            if p.get("error"):
                lines.append(
                    f"{p.get('mode', ''):<20}{_fmt(p.get('level')):>7}  ERROR: {p['error']}"
                )
                continue
            lines.append(
                f"{p.get('mode', ''):<20}{_fmt(p.get('level')):>7}"
                f"{_fmt(p.get('request_count'), '.0f'):>10}{_fmt(p.get('failed_requests'), '.0f'):>8}"
                f"{_fmt(p.get('request_throughput'), '.2f'):>8}"
                f"{_fmt(p.get('p50_ttft_ms'), '.1f', 'ms'):>11}"
                f"{_fmt(p.get('p99_ttft_ms'), '.1f', 'ms'):>11}"
                f"{_fmt(p.get('p50_tpot_ms'), '.2f', 'ms'):>11}"
                f"{_fmt(p.get('p99_e2el_ms'), '.1f', 'ms'):>11}"
            )
    timeseries = _read_json(root / "analysis" / "request_timeseries.json")
    if timeseries and timeseries.get("windows"):
        lines += ["", f"{'window':>9}{'offered/s':>11}{'ok/s':>7}{'failed/s':>10}"
                      f"{'inflight':>10}{'TTFT p50':>11}{'TTFT p99':>11}"]
        for w in timeseries["windows"]:
            lines.append(
                f"{w['window_start_s']:>8g}s{w['offered_rps']:>11.1f}"
                f"{w['successful_completed_rps']:>7.1f}{w['failed_completed_rps']:>10.1f}"
                f"{w['in_flight_at_end']:>10}"
                f"{_fmt(w['p50_ttft_ms'], '.0f', 'ms'):>11}"
                f"{_fmt(w['p99_ttft_ms'], '.0f', 'ms'):>11}"
            )
    analysis = manifest.get("analysis") or {}
    if analysis:
        lines.append("")
        for key in ("analysis_error", "plot_error"):
            if analysis.get(key):
                lines.append(f"{key.replace('_', ' ').capitalize()}: {analysis[key]}")
        for warning in analysis.get("warnings", []):
            lines.append(f"Warning: {warning}")
        plot_data = _read_json(root / "analysis" / "plot_data.json") or {}
        periods = (plot_data.get("lifecycle") or {}).get("periods") or []
        if periods:
            lines.append("Lifecycle: " + " · ".join(
                f"{p['name']} {p['start_s']:.0f}-{p['end_s']:.0f}s" for p in periods
            ))
        timeline = root / "analysis" / "autoscaling_timeline.png"
        if timeline.exists():
            lines.append(f"Timeline:  {timeline}")
    return "\n".join(lines) + "\n"


@dataclass
class RunArtifacts:
    root: Path
    manifest: dict[str, Any]

    @classmethod
    def create(cls, config: ExperimentConfig, input_path: Path) -> "RunArtifacts":
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        safe_name = "".join(
            char if char.isalnum() or char in "-_" else "-" for char in config.name
        )
        root = config.runtime.results_dir / f"{stamp}-{safe_name}"
        root.mkdir(parents=True, exist_ok=False)
        (root / "deployment").mkdir()
        (root / "benchmark").mkdir()
        with input_path.open() as src, (root / "input.yaml").open("w") as dst:
            dst.write(src.read())
        write_config(config, root / "resolved.yaml")
        manifest = {
            "name": config.name,
            "started_at": utc_now(),
            "status": "running",
            "failure_stage": None,
            "git_revision": _git_revision(),
        }
        obj = cls(root=root, manifest=manifest)
        obj.flush_manifest()
        return obj

    def flush_manifest(self) -> None:
        write_json(self.root / "manifest.json", self.manifest)

    def finish(self, status: str, failure_stage: str | None = None, error: str | None = None) -> None:
        self.manifest.update(
            {
                "status": status,
                "failure_stage": failure_stage,
                "error": error,
                "finished_at": utc_now(),
            }
        )
        self.flush_manifest()

    def record_event(self, event: str, **fields: Any) -> None:
        payload = {"timestamp": utc_now(), "event": event, **fields}
        with (self.root / "deployment" / "events.jsonl").open("a") as fh:
            fh.write(json.dumps(payload, default=str) + "\n")

