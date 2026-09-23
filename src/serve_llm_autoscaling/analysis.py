"""Offline analysis of a request-rate-series run directory.

Everything here reads saved artifacts only, so it runs without Ray or a GPU
and can be repeated with different windows after the experiment.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

import yaml

from .artifacts import write_json
from .config import AnalysisConfig, ExperimentConfig
from .telemetry import SELECTED_METRICS, normalize_metric_name
from .windows import request_timeseries

SCHEMA_VERSION = 1
SERIES_DIR = Path("benchmark") / "request-rate-series"

# key -> (normalized sample name, aggregation)
METRIC_SERIES = {
    "ongoing_requests": ("serve_replica_processing_queries", "sum"),
    "handle_ongoing_requests": ("serve_num_ongoing_requests_at_replicas", "sum"),
    "router_queue": ("serve_request_router_queue_len", "sum"),
    "queued_queries": ("serve_deployment_queued_queries", "sum"),
    "healthy_replicas": ("serve_deployment_replica_healthy", "healthy"),
    "desired_replicas": ("serve_autoscaling_desired_replicas", "unique"),
    "target_replicas": ("serve_autoscaling_target_replicas", "unique"),
    "autoscaling_total_requests": ("serve_autoscaling_total_requests", "unique"),
    "target_ongoing_requests": ("serve_autoscaling_target_ongoing_requests", "unique"),
    "replica_startups": ("serve_replica_startup_latency_ms_count", "counter_delta"),
}


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open() as fh:
        for line in fh:
            try:
                record = json.loads(line)
            except ValueError:
                continue  # e.g. a line truncated by an interrupted run
            if isinstance(record, dict):
                yield record


def load_run_config(root: Path) -> ExperimentConfig:
    with (root / "resolved.yaml").open() as fh:
        return ExperimentConfig.model_validate(yaml.safe_load(fh))


# --- Metrics inventory ------------------------------------------------------


def build_metrics_inventory(metrics_path: Path) -> dict[str, Any]:
    metrics: dict[str, dict[str, Any]] = {}
    other: set[str] = set()
    endpoints: set[str] = set()
    scrapes = errors = 0
    if metrics_path.exists():
        for record in read_jsonl(metrics_path):
            scrapes += 1
            if record.get("endpoint"):
                endpoints.add(record["endpoint"])
            if record.get("error"):
                errors += 1
            other.update(record.get("available_serve_metrics") or [])
            ts = record.get("timestamp_ns")
            for sample in record.get("samples") or []:
                entry = metrics.setdefault(sample["normalized_name"], {
                    "original_names": set(), "sample_names": set(), "types": set(),
                    "labels": set(), "source_nodes": set(), "sample_count": 0,
                    "first_timestamp_ns": ts, "last_timestamp_ns": ts,
                })
                family = sample["name"]
                if sample.get("type") in ("histogram", "summary", "counter"):
                    family = family.removesuffix("_bucket").removesuffix("_count")
                    family = family.removesuffix("_sum").removesuffix("_total")
                entry["original_names"].add(family)
                entry["sample_names"].add(sample["name"])
                entry["types"].add(sample.get("type"))
                entry["labels"].update(sample.get("labels") or {})
                entry["source_nodes"].add(record.get("node_id"))
                entry["sample_count"] += 1
                entry["first_timestamp_ns"] = min(entry["first_timestamp_ns"], ts)
                entry["last_timestamp_ns"] = max(entry["last_timestamp_ns"], ts)
    for entry in metrics.values():
        for key, value in entry.items():
            if isinstance(value, set):
                entry[key] = sorted(v for v in value if v is not None)
    return {
        "schema_version": SCHEMA_VERSION,
        "collected": metrics_path.exists(),
        "scrape_records": scrapes,
        "scrape_errors": errors,
        "endpoints": sorted(endpoints),
        "metrics": dict(sorted(metrics.items())),
        "missing_recommended_metrics": [m for m in SELECTED_METRICS if m not in metrics],
        "other_available_serve_metrics": sorted(other - set(SELECTED_METRICS)),
    }


# --- Serve status -----------------------------------------------------------


def relevant_deployments(names: set[str]) -> set[str]:
    """The LLM server deployments; the ingress does not run the model."""
    llm = {name for name in names if name.startswith("LLMServer")}
    return llm or set(names)


def status_samples(
    records: list[dict[str, Any]], deployments: set[str], t0: int
) -> list[dict[str, Any]]:
    samples = []
    for record in records:
        sample = {
            "relative_time_s": (record["timestamp_ns"] - t0) / 1e9,
            "timestamp_ns": record["timestamp_ns"],
            "error": record.get("error"),
        }
        if not record.get("error"):
            chosen = {
                name: d for name, d in (record.get("deployments") or {}).items()
                if name in deployments
            }
            states: dict[str, int] = defaultdict(int)
            for d in chosen.values():
                for state, n in (d.get("replicas_by_state") or {}).items():
                    states[state] += n
            targets = [d.get("target_num_replicas") for d in chosen.values()]
            sample.update({
                "application_status": record.get("application_status"),
                "target_replicas": (
                    sum(targets) if targets and None not in targets else None
                ),
                "live_replicas": sum(d.get("live_replicas", 0) for d in chosen.values()),
                "running_replicas": states.get("RUNNING", 0),
                "starting_replicas": states.get("STARTING", 0),
                "replicas_by_state": dict(states),
            })
        samples.append(sample)
    return samples


# --- Prometheus aggregation -------------------------------------------------


def _label_key(labels: dict[str, str]) -> tuple:
    return tuple(sorted(labels.items()))


def aggregate_metric(
    ticks: list[tuple[int, list[tuple[str | None, dict[str, Any]]]]],
    aggregation: str,
    t0: int,
    warnings: list[str],
    metric: str,
) -> list[dict[str, Any]]:
    """Aggregate one metric per scrape tick.

    ``ticks`` holds (timestamp_ns, [(node_id, sample), ...]) in time order.
    Ticks without samples produce no point, so gaps are not zeros.
    """
    points = []
    previous: dict[tuple, float] = {}
    conflicting = False
    for timestamp_ns, samples in ticks:
        value: float | None
        series = {}
        for node_id, sample in samples:
            if sample.get("value") is not None:
                series[(node_id, _label_key(sample["labels"]))] = sample
        if aggregation == "counter_delta":
            # Node-local deltas first; a drop means the counter reset.
            delta, seen = 0.0, False
            for key, sample in series.items():
                if key in previous:
                    prior = previous[key]
                    delta += sample["value"] - prior if sample["value"] >= prior else sample["value"]
                    seen = True
                previous[key] = sample["value"]
            if not seen:
                continue
            value = delta
        elif not series:
            continue
        elif aggregation == "sum":
            # A series exported by two nodes is still one series.
            unique = {labels: s["value"] for (_, labels), s in series.items()}
            value = sum(unique.values())
        elif aggregation == "healthy":
            per_replica = [s for s in series.values() if "replica" in s["labels"]]
            if per_replica:
                value = sum(
                    1 for v in {_label_key(s["labels"]): s["value"] for s in per_replica}.values()
                    if v >= 1
                )
            else:
                value = _unique_deployment_sum(series.values())[0]
        elif aggregation == "unique":
            value, conflict = _unique_deployment_sum(series.values())
            conflicting = conflicting or conflict
        else:
            raise ValueError(f"unknown aggregation {aggregation!r}")
        points.append({
            "relative_time_s": (timestamp_ns - t0) / 1e9,
            "value": value,
            "series_count": len(series),
        })
    if conflicting:
        warnings.append(
            f"{metric}: nodes exported conflicting values for one deployment; used the maximum"
        )
    return points


def _unique_deployment_sum(samples) -> tuple[float, bool]:
    """Controller gauges: one value per application/deployment, never summed across nodes."""
    by_deployment: dict[tuple, set[float]] = defaultdict(set)
    for s in samples:
        labels = s["labels"]
        by_deployment[(labels.get("application"), labels.get("deployment"))].add(s["value"])
    conflict = any(len(values) > 1 for values in by_deployment.values())
    return sum(max(values) for values in by_deployment.values()), conflict


def metric_series(
    records: list[dict[str, Any]],
    application: str,
    deployments: set[str],
    interval_s: float,
    t0: int,
    warnings: list[str],
) -> dict[str, dict[str, Any]]:
    ticks: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("error") or record.get("timestamp_ns") is None:
            continue
        tick = record.get("tick")
        if tick is None:
            tick = round(record["timestamp_ns"] / 1e9 / interval_s)
        ticks[tick].append(record)

    unlabeled: set[str] = set()
    by_metric: dict[str, list] = {name: [] for name, _ in METRIC_SERIES.values()}
    for tick in sorted(ticks):
        tick_records = ticks[tick]
        timestamp_ns = max(r["timestamp_ns"] for r in tick_records)
        selected: dict[str, list] = defaultdict(list)
        for record in tick_records:
            for sample in record.get("samples") or []:
                name = normalize_metric_name(sample["name"])
                if name not in by_metric:
                    continue
                labels = sample.get("labels") or {}
                if "application" in labels and labels["application"] != application:
                    continue
                if "deployment" in labels and labels["deployment"] not in deployments:
                    continue
                if "application" not in labels or "deployment" not in labels:
                    unlabeled.add(name)
                selected[name].append((record.get("node_id"), sample))
        for name in by_metric:
            by_metric[name].append((timestamp_ns, selected.get(name, [])))

    for name in sorted(unlabeled):
        warnings.append(
            f"{name}: some series lack application/deployment labels and were kept; "
            "they may include other deployments"
        )
    result = {}
    for key, (name, aggregation) in METRIC_SERIES.items():
        points = aggregate_metric(by_metric[name], aggregation, t0, warnings, name)
        if points:
            result[key] = {"metric": name, "aggregation": aggregation, "points": points}
    return result


# --- Plot data --------------------------------------------------------------


def build_plot_data(
    root: Path,
    config: ExperimentConfig,
    analysis: AnalysisConfig,
    timeseries: dict[str, Any],
) -> dict[str, Any]:
    series_dir = root / SERIES_DIR
    warnings: list[str] = []
    t0 = timeseries["profiling_start_ns"]
    if t0 is None:
        raise ValueError("no profiling records or phase manifest; cannot align timestamps")
    if timeseries["time_origin_source"] != "phase_manifest":
        warnings.append("phase_manifest.json unavailable; time origin is the first profiling credit")
    if timeseries["malformed_record_count"]:
        warnings.append(
            f"{timeseries['malformed_record_count']} malformed AIPerf records were skipped"
        )

    try:
        rate_curve = json.loads((series_dir / "rate_series.json").read_text())["points"]
    except (OSError, ValueError, KeyError):
        rate_curve = [p.model_dump() for p in config.benchmark.rate_series or []]

    application = config.deployment.application_name
    status_path = root / "telemetry" / "serve_status.jsonl"
    metrics_path = root / "telemetry" / "serve_metrics.jsonl"
    status_records = list(read_jsonl(status_path)) if status_path.exists() else []
    metric_records = list(read_jsonl(metrics_path)) if metrics_path.exists() else []

    names = {n for r in status_records for n in (r.get("deployments") or {})}
    names |= {
        s["labels"]["deployment"]
        for r in metric_records for s in r.get("samples") or []
        if s.get("labels", {}).get("application") == application
        and "deployment" in s.get("labels", {})
    }
    deployments = relevant_deployments(names)

    status = status_samples(status_records, deployments, t0)
    if not status_path.exists():
        warnings.append("Serve status telemetry was not collected; replica panel is empty")
    elif not any(s.get("error") is None for s in status):
        warnings.append("every Serve status sample failed; replica panel is empty")
    elif errors := sum(1 for s in status if s.get("error")):
        warnings.append(f"{errors} of {len(status)} Serve status samples failed")

    if not metrics_path.exists():
        warnings.append("Prometheus metrics were not collected; Serve-side signals are omitted")
    series = metric_series(
        metric_records, application, deployments, analysis.telemetry_interval_s, t0, warnings
    )
    if metrics_path.exists():
        for key in ("ongoing_requests", "router_queue", "desired_replicas"):
            if key not in series:
                warnings.append(f"{METRIC_SERIES[key][0]} was not exported; omitted from the plot")

    manifest_end = None
    try:
        phases = json.loads((series_dir / "phase_manifest.json").read_text())["phases"]
        manifest_end = next(
            p.get("end_ns") for p in phases if p.get("phase_kind") == "profiling"
        )
    except (OSError, ValueError, KeyError, StopIteration):
        pass

    autoscaling = config.deployment.autoscaling
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": {
            "name": config.name,
            "model_id": config.deployment.model_id,
            "application": application,
            "deployments": sorted(deployments),
            "min_replicas": autoscaling.min_replicas,
            "initial_replicas": autoscaling.initial_replicas,
            "max_replicas": autoscaling.max_replicas,
            "target_ongoing_requests": autoscaling.target_ongoing_requests,
            "upscale_delay_s": autoscaling.upscale_delay_s,
            "downscale_delay_s": autoscaling.downscale_delay_s,
            "duration_s": config.benchmark.duration_s,
            "grace_period_s": config.benchmark.grace_period_s,
            "window_s": analysis.window_s,
            "tail_window_s": analysis.tail_window_s,
            "telemetry_interval_s": analysis.telemetry_interval_s,
            "ttft_slo_ms": analysis.ttft_slo_ms,
        },
        "time_origin": {
            "profiling_start_ns": t0,
            "source": timeseries["time_origin_source"],
            "profiling_end_s": (manifest_end - t0) / 1e9 if manifest_end else None,
        },
        "request_rate_curve": rate_curve,
        "request_windows": timeseries["windows"],
        "serve_status_samples": status,
        "serve_metric_series": series,
        "availability": {
            "replica_status": any(s.get("error") is None for s in status),
            "router_queue": "router_queue" in series,
            "ongoing_requests": "ongoing_requests" in series or "handle_ongoing_requests" in series,
            "desired_replicas": "desired_replicas" in series,
            "target_replicas_metric": "target_replicas" in series,
            "healthy_replicas": "healthy_replicas" in series,
        },
        "warnings": warnings,
    }


def analyze_run(
    root: Path,
    *,
    window_s: float | None = None,
    tail_window_s: float | None = None,
    plot: bool | None = None,
) -> dict[str, Any]:
    """Regenerate every derived analysis artifact from a run's raw files."""
    config = load_run_config(root)
    overrides = {k: v for k, v in (("window_s", window_s), ("tail_window_s", tail_window_s))
                 if v is not None}
    analysis = AnalysisConfig.model_validate({**config.analysis.model_dump(), **overrides})
    out = root / "analysis"
    series_dir = root / SERIES_DIR

    timeseries = request_timeseries(
        series_dir,
        window_s=analysis.window_s,
        tail_window_s=analysis.tail_window_s,
        ttft_slo_ms=analysis.ttft_slo_ms,
        duration_s=config.benchmark.duration_s,
    )
    write_json(out / "request_timeseries.json", timeseries)
    write_json(
        out / "metrics_inventory.json",
        build_metrics_inventory(root / "telemetry" / "serve_metrics.jsonl"),
    )
    plot_data = build_plot_data(root, config, analysis, timeseries)
    write_json(out / "plot_data.json", plot_data)

    result: dict[str, Any] = {"warnings": plot_data["warnings"], "plot_error": None}
    if analysis.generate_plots if plot is None else plot:
        try:
            from .plotting import plot_timeline

            plot_timeline(out / "plot_data.json", out / "autoscaling_timeline.png")
        except Exception as exc:  # raw and derived data stay valid without the figure
            result["plot_error"] = f"{type(exc).__name__}: {exc}"
    return result
