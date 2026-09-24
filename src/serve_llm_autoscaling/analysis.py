"""Offline analysis of a continuous (request-rate-series or AgentX) run directory.

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
from .config import (
    CONTINUOUS_MODE_DIRS,
    AgentXWorkloadConfig,
    AnalysisConfig,
    ExperimentConfig,
)
from .telemetry import SELECTED_METRICS, normalize_metric_name
from .windows import configured_concurrency_at, request_timeseries

# 2: ``request_rate_curve`` became the dimension-tagged ``configured_load``.
SCHEMA_VERSION = 2

# key -> (normalized sample name, aggregation, identity labels)
#
# A series is identified by application, deployment and the identity labels:
# the replica for replica metrics, the handle or router actor for router
# metrics, nothing more for controller gauges. Copies of one series exported by
# several nodes are deduplicated; samples missing an identity label are
# excluded, since they cannot be told apart.
METRIC_SERIES = {
    "ongoing_requests": ("serve_replica_processing_queries", "sum", ("replica",)),
    "handle_ongoing_requests": (
        "serve_num_ongoing_requests_at_replicas", "sum", ("handle", "actor_id")
    ),
    "router_queue": (
        "serve_request_router_queue_len", "sum", ("actor_id", "handle_source", "replica_id")
    ),
    "queued_queries": ("serve_deployment_queued_queries", "sum", ("handle", "actor_id")),
    "healthy_replicas": ("serve_deployment_replica_healthy", "healthy", ("replica",)),
    "desired_replicas": ("serve_autoscaling_desired_replicas", "sum", ()),
    "target_replicas": ("serve_autoscaling_target_replicas", "sum", ()),
    "autoscaling_total_requests": ("serve_autoscaling_total_requests", "sum", ()),
    "target_ongoing_requests": ("serve_autoscaling_target_ongoing_requests", "sum", ()),
    "replica_startups": ("serve_replica_startup_latency_ms_count", "counter_delta", ()),
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


def workload_artifact_dir(root: Path, config: ExperimentConfig) -> Path:
    """The raw AIPerf artifact directory of a continuous run."""
    mode = config.benchmark.mode
    if mode not in CONTINUOUS_MODE_DIRS:
        raise ValueError(f"mode {mode!r} is not a continuous workload; nothing to analyze")
    return root / "benchmark" / CONTINUOUS_MODE_DIRS[mode]


# --- Configured load ----------------------------------------------------------


def configured_load(config: ExperimentConfig, artifact_dir: Path) -> dict[str, Any]:
    """The load the harness asked for, tagged with what it measures.

    For AgentX this is session-tree concurrency, not a request rate: the
    request rate it produces is an observed outcome.
    """
    benchmark = config.benchmark
    workload = benchmark.workload
    if isinstance(workload, AgentXWorkloadConfig):
        ramp = workload.concurrency_ramp_duration_s
        points = [{"time_s": 0.0, "value": 1.0},
                  {"time_s": ramp, "value": float(workload.target_concurrency)}]
        if benchmark.duration_s > ramp:
            points.append({"time_s": benchmark.duration_s,
                           "value": float(workload.target_concurrency)})
        return {"dimension": "session_concurrency", "unit": "session trees",
                "label": "configured session concurrency", "points": points,
                "markers": [ramp]}
    try:
        curve = json.loads((artifact_dir / "rate_series.json").read_text())["points"]
    except (OSError, ValueError, KeyError):
        curve = [p.model_dump() for p in benchmark.rate_series or []]
    return {"dimension": "request_rate", "unit": "requests/s", "label": "configured QPS",
            "points": [{"time_s": p["time_s"], "value": p["qps"]} for p in curve],
            "markers": [p["time_s"] for p in curve[1:]]}


def concurrency_qps(
    workload: AgentXWorkloadConfig, timeseries: dict[str, Any]
) -> dict[str, Any]:
    """Configured session concurrency beside the request rates it generated.

    Concurrency is the configured ramp at each window's midpoint; the live
    session count can briefly differ because of scheduling and tree turnover.
    """
    rows = []
    for w in timeseries["windows"]:
        mid = (w["window_start_s"] + w["window_end_s"]) / 2
        sessions = configured_concurrency_at(
            mid, workload.target_concurrency, workload.concurrency_ramp_duration_s
        )
        row = {
            "window_start_s": w["window_start_s"],
            "window_end_s": w["window_end_s"],
            "configured_session_concurrency": sessions,
            "offered_qps": w["offered_rps"],
            "started_qps": w["started_rps"],
            "successful_completed_qps": w["successful_completed_rps"],
            "failed_completed_qps": w["failed_completed_rps"],
            "in_flight_at_end": w["in_flight_at_end"],
            # A diagnostic only: it varies with trace mix, think time,
            # latency and subagent activity.
            "requests_per_second_per_configured_session": w["offered_rps"] / sessions,
        }
        if "offered_root_rps" in w:
            row["offered_root_qps"] = w["offered_root_rps"]
            row["offered_subagent_qps"] = w["offered_subagent_rps"]
        rows.append(row)
    return {
        "schema_version": SCHEMA_VERSION,
        "target_concurrency": workload.target_concurrency,
        "ramp_duration_s": workload.concurrency_ramp_duration_s,
        "window_s": timeseries["window_s"],
        "summary": timeseries["summary"],
        "windows": rows,
    }


# --- Metrics inventory ------------------------------------------------------


def build_metrics_inventory(metrics_path: Path) -> dict[str, Any]:
    metrics: dict[str, dict[str, Any]] = {}
    other: set[str] = set()
    endpoints: set[str] = set()
    scrapes = errors = 0
    durations: list[float] = []
    sizes: list[int] = []
    overran: set[Any] = set()
    if metrics_path.exists():
        for record in read_jsonl(metrics_path):
            scrapes += 1
            if record.get("endpoint"):
                endpoints.add(record["endpoint"])
            if record.get("error"):
                errors += 1
            if record.get("scrape_duration_s") is not None:
                durations.append(record["scrape_duration_s"])
            if record.get("bytes") is not None:
                sizes.append(record["bytes"])
            if record.get("cycle_overran"):
                overran.add(record.get("tick"))
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
        # Collector overhead, to check that scraping did not perturb the run.
        "max_scrape_duration_s": max(durations, default=None),
        "max_scrape_bytes": max(sizes, default=None),
        "overran_cycles": len(overran),
        "metrics": dict(sorted(metrics.items())),
        "missing_recommended_metrics": [m for m in SELECTED_METRICS if m not in metrics],
        "other_available_serve_metrics": sorted(other - set(SELECTED_METRICS)),
    }


# --- Serve status -----------------------------------------------------------


def relevant_deployments(names: set[str]) -> set[str]:
    """The LLM server deployments; the ingress does not run the model."""
    llm = {name for name in names if name.startswith("LLMServer")}
    return llm or set(names)


STARTING_STATES = ("STARTING", "RECOVERING")


def status_samples(
    records: list[dict[str, Any]], deployments: set[str], t0: int
) -> list[dict[str, Any]]:
    """Replica counts per status sample; unavailable samples carry only an error.

    An absent application or deployment is a gap, never zero replicas.
    """
    samples = []
    for record in records:
        sample = {
            "relative_time_s": (record["timestamp_ns"] - t0) / 1e9,
            "timestamp_ns": record["timestamp_ns"],
            "error": record.get("error"),
        }
        chosen = [
            d for name, d in (record.get("deployments") or {}).items() if name in deployments
        ]
        if not sample["error"] and not chosen:
            sample["error"] = "no matching deployment in Serve status"
        if not sample["error"]:
            states: dict[str, int] = defaultdict(int)
            for d in chosen:
                # ``replicas_by_state`` is the key in runs recorded before serve.status().
                for state, n in (d.get("replica_states") or d.get("replicas_by_state") or {}).items():
                    states[state] += n
            sample.update({
                "application_status": record.get("application_status"),
                "live_replicas": sum(n for state, n in states.items() if state != "STOPPED"),
                "running_replicas": states.get("RUNNING", 0),
                "starting_replicas": sum(states.get(state, 0) for state in STARTING_STATES),
                "stopping_replicas": states.get("STOPPING", 0),
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
    identity: tuple[str, ...],
    t0: int,
    warnings: list[str],
    metric: str,
) -> list[dict[str, Any]]:
    """Aggregate one metric per scrape tick.

    ``ticks`` holds (timestamp_ns, [(node_id, sample), ...]) in time order.
    Ticks without samples produce no point, so gaps are not zeros. A tick in
    which copies of one series disagree is omitted rather than guessed.
    """
    points = []
    previous: dict[tuple, float] = {}
    unidentified = conflicting = 0
    for timestamp_ns, samples in ticks:
        series: dict[tuple, set[float]] = defaultdict(set)
        for node_id, sample in samples:
            labels = sample["labels"]
            if sample.get("value") is None:
                continue
            if any(key not in labels for key in identity):
                unidentified += 1
                continue
            key = (labels.get("application"), labels.get("deployment"),
                   *(labels[k] for k in identity))
            if aggregation == "counter_delta":
                # Counters are node-local; deltas are taken per node series.
                key = (node_id, _label_key(labels))
            series[key].add(sample["value"])
        if any(len(values) > 1 for values in series.values()):
            conflicting += 1
            continue
        current = {key: values.pop() for key, values in series.items()}
        value: float
        if aggregation == "counter_delta":
            # A drop means the counter reset.
            delta, seen = 0.0, False
            for key, v in current.items():
                if key in previous:
                    delta += v - previous[key] if v >= previous[key] else v
                    seen = True
                previous[key] = v
            if not seen:
                continue
            value = delta
        elif not current:
            continue
        elif aggregation == "sum":
            value = sum(current.values())
        elif aggregation == "healthy":
            value = sum(1 for v in current.values() if v >= 1)
        else:
            raise ValueError(f"unknown aggregation {aggregation!r}")
        points.append({
            "relative_time_s": (timestamp_ns - t0) / 1e9,
            "value": value,
            "series_count": len(current),
        })
    if unidentified:
        warnings.append(
            f"{metric}: excluded {unidentified} samples missing identity labels "
            f"{list(identity)}"
        )
    if conflicting:
        warnings.append(
            f"{metric}: omitted {conflicting} scrapes where copies of one series disagreed"
        )
    return points


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

    unlabeled: dict[str, int] = defaultdict(int)
    by_metric: dict[str, list] = {name: [] for name, _, _ in METRIC_SERIES.values()}
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
                if "application" not in labels or "deployment" not in labels:
                    unlabeled[name] += 1  # cannot be attributed to the deployment
                    continue
                if labels["application"] != application:
                    continue
                if labels["deployment"] not in deployments:
                    continue
                selected[name].append((record.get("node_id"), sample))
        for name in by_metric:
            by_metric[name].append((timestamp_ns, selected.get(name, [])))

    for name in sorted(unlabeled):
        warnings.append(
            f"{name}: excluded {unlabeled[name]} samples without application/deployment labels"
        )
    result = {}
    for key, (name, aggregation, identity) in METRIC_SERIES.items():
        points = aggregate_metric(by_metric[name], aggregation, identity, t0, warnings, name)
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
    artifact_dir = workload_artifact_dir(root, config)
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
        metric_records, application, deployments, analysis.metrics_interval_s, t0, warnings
    )
    if metrics_path.exists():
        for key in ("ongoing_requests", "router_queue", "desired_replicas"):
            if key not in series:
                warnings.append(f"{METRIC_SERIES[key][0]} was not exported; omitted from the plot")

    manifest_end = None
    try:
        phases = json.loads((artifact_dir / "phase_manifest.json").read_text())["phases"]
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
            "status_interval_s": analysis.status_interval_s,
            "metrics_interval_s": analysis.metrics_interval_s,
            "ttft_slo_ms": analysis.ttft_slo_ms,
            "mode": config.benchmark.mode,
        },
        "time_origin": {
            "profiling_start_ns": t0,
            "source": timeseries["time_origin_source"],
            "profiling_end_s": (manifest_end - t0) / 1e9 if manifest_end else None,
        },
        "configured_load": configured_load(config, artifact_dir),
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

    timeseries = request_timeseries(
        workload_artifact_dir(root, config),
        window_s=analysis.window_s,
        tail_window_s=analysis.tail_window_s,
        ttft_slo_ms=analysis.ttft_slo_ms,
        duration_s=config.benchmark.duration_s,
    )
    write_json(out / "request_timeseries.json", timeseries)
    if isinstance(config.benchmark.workload, AgentXWorkloadConfig):
        write_json(out / "concurrency_qps.json",
                   concurrency_qps(config.benchmark.workload, timeseries))
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
