"""Background collectors for Serve status and Ray's Prometheus exports.

Both collectors poll on a ``time.monotonic()`` schedule but stamp samples with
``time.time_ns()`` so they share AIPerf's epoch clock. Collection failures are
written as error samples and never propagate to the benchmark.
"""
from __future__ import annotations

import json
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

import requests

from .config import ExperimentConfig

SELECTED_METRICS = (
    "serve_autoscaling_desired_replicas",
    "serve_autoscaling_target_replicas",
    "serve_autoscaling_total_requests",
    "serve_autoscaling_target_ongoing_requests",
    "serve_num_ongoing_requests_at_replicas",
    "serve_request_router_queue_len",
    "serve_deployment_queued_queries",
    "serve_replica_processing_queries",
    "serve_deployment_replica_healthy",
    "serve_replica_startup_latency_ms",
)
SCRAPE_TIMEOUT_S = 2.0


def _error(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


class JsonlCollector:
    """Calls ``sample`` every ``interval_s`` on a thread, appending JSONL records."""

    def __init__(
        self,
        path: Path,
        interval_s: float,
        sample: Callable[[int, int], list[dict[str, Any]]],
        error_record: Callable[[int, str], dict[str, Any]],
        name: str,
        close: Callable[[], None] | None = None,
    ):
        self.path = path
        self.interval_s = interval_s
        self._sample = sample
        self._error_record = error_record
        self._close = close
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._fh = None
        self.records_written = 0
        self.error_records = 0

    def start(self) -> None:
        self._fh = self.path.open("w")
        self._thread.start()

    def stop(self, timeout_s: float | None = None) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout_s)
        if self._thread.is_alive():
            return  # a hung sample still owns the file; the daemon thread dies with us
        if self._close is not None:
            self._close()
            self._close = None
        if self._fh is not None:
            self._fh.close()

    def __enter__(self) -> "JsonlCollector":
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    def _write(self, record: dict[str, Any]) -> None:
        try:
            self._fh.write(json.dumps(record, default=str) + "\n")
            self._fh.flush()
        except Exception:  # never let a disk problem kill the thread
            return
        self.records_written += 1
        if record.get("error"):
            self.error_records += 1

    def _run(self) -> None:
        tick = 0
        next_tick = time.monotonic()
        while not self._stop.is_set():
            timestamp_ns = time.time_ns()
            try:
                records = self._sample(timestamp_ns, tick)
            except Exception as exc:
                records = [self._error_record(timestamp_ns, _error(exc))]
            for record in records:
                record.setdefault("tick", tick)
                self._write(record)
            tick += 1
            next_tick += self.interval_s
            now = time.monotonic()
            if next_tick < now:  # a slow poll skips the ticks it overran
                next_tick += math.ceil((now - next_tick) / self.interval_s) * self.interval_s
            self._stop.wait(next_tick - now)


# --- Serve status -----------------------------------------------------------


def _value(value: Any) -> Any:
    """Serve's status enums are str enums; record their plain values."""
    return getattr(value, "value", value)


def fetch_serve_status() -> dict[str, Any]:
    """The public ``serve.status()`` as plain JSON-compatible data.

    It reports replica counts by state but no target replica count; the
    desired/target counts come from the autoscaling metrics instead.
    """
    from ray import serve

    status = serve.status()
    return {"applications": {
        name: {
            "status": _value(app.status),
            "deployments": {
                dep_name: {
                    "status": _value(dep.status),
                    "status_trigger": _value(dep.status_trigger),
                    "replica_states": {
                        _value(state): n for state, n in (dep.replica_states or {}).items()
                    },
                }
                for dep_name, dep in (app.deployments or {}).items()
            },
        }
        for name, app in (status.applications or {}).items()
    }}


def serve_status_sample(
    status: dict[str, Any], application: str, timestamp_ns: int
) -> dict[str, Any]:
    app = (status.get("applications") or {}).get(application)
    if not app:
        # Not zero replicas: the application is not there to be measured.
        return {
            "timestamp_ns": timestamp_ns, "application": application,
            "application_status": None, "deployments": {}, "available": False,
            "error": "configured application not present",
        }
    deployments = {
        name: {
            "status": _value(d.get("status")),
            "status_trigger": _value(d.get("status_trigger")),
            "replica_states": {
                _value(state): n for state, n in (d.get("replica_states") or {}).items()
            },
        }
        for name, d in (app.get("deployments") or {}).items()
    }
    return {
        "timestamp_ns": timestamp_ns,
        "application": application,
        "application_status": _value(app.get("status")),
        "deployments": deployments,
        "available": True,
        "error": None,
    }


def serve_status_collector(
    path: Path,
    application: str,
    interval_s: float,
    fetch: Callable[[], dict[str, Any]] = fetch_serve_status,
) -> JsonlCollector:
    return JsonlCollector(
        path,
        interval_s,
        lambda ts, tick: [serve_status_sample(fetch(), application, ts)],
        lambda ts, error: {
            "timestamp_ns": ts, "application": application, "application_status": None,
            "deployments": {}, "available": False, "error": error,
        },
        name="serve-status-collector",
    )


# --- Prometheus -------------------------------------------------------------


def normalize_metric_name(name: str) -> str:
    return name[len("ray_"):] if name.startswith("ray_") else name


def ray_nodes() -> list[dict[str, Any]]:
    import ray

    return ray.nodes()


def discover_endpoints(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    endpoints: dict[str, dict[str, Any]] = {}
    for node in nodes:
        ip, port = node.get("NodeManagerAddress"), node.get("MetricsExportPort")
        if not node.get("Alive") or not ip or not port:
            continue
        url = f"http://{ip}:{port}/metrics"
        endpoints.setdefault(
            url, {"node_id": node.get("NodeID"), "node_ip": ip, "endpoint": url}
        )
    return list(endpoints.values())


def parse_metrics(text: str) -> tuple[list[dict[str, Any]], list[str]]:
    """Selected samples plus the names of every Serve metric family present."""
    from prometheus_client.parser import text_string_to_metric_families

    # Ray exports ~2MB per node; parse only lines that can be Serve metrics.
    serve_lines = "\n".join(line for line in text.splitlines() if "serve_" in line)
    samples: list[dict[str, Any]] = []
    families: set[str] = set()
    for family in text_string_to_metric_families(serve_lines + "\n"):
        normalized = normalize_metric_name(family.name)
        if not normalized.startswith("serve_"):
            continue
        families.add(normalized)
        if normalized not in SELECTED_METRICS:
            continue
        for sample in family.samples:
            value = sample.value
            samples.append({
                "name": sample.name,
                "normalized_name": normalized,
                "type": family.type,
                "labels": dict(sample.labels),
                "value": value if math.isfinite(value) else None,
            })
    return samples, sorted(families)


def fetch_metrics_text(url: str) -> str:
    # Ray's exporter ignores Prometheus' ``name[]`` filter, so this is the full export.
    response = requests.get(url, timeout=SCRAPE_TIMEOUT_S)
    response.raise_for_status()
    return response.text


class _PrometheusSampler:
    def __init__(self, discover: Callable[[], list[dict[str, Any]]],
                 fetch: Callable[[str], str], interval_s: float):
        self._discover = discover
        self._fetch = fetch
        self._interval_s = interval_s
        self._pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="metrics-scrape")
        self._families: dict[str, list[str]] = {}

    def _scrape(self, endpoint: dict[str, Any]) -> dict[str, Any]:
        url = endpoint["endpoint"]
        started = time.time_ns()
        record = {**endpoint, "scrape_started_at_ns": started, "samples": [], "error": None}
        try:
            text = self._fetch(url)
            record["bytes"] = len(text)
            record["samples"], families = parse_metrics(text)
        except Exception as exc:
            record["error"] = _error(exc)
            families = None
        # A gauge is read once the response arrives, so align on completion.
        finished = time.time_ns()
        record.update(scrape_finished_at_ns=finished, timestamp_ns=finished,
                      scrape_duration_s=(finished - started) / 1e9)
        # The full Serve metric list is only written when it changes.
        if families is not None and self._families.get(url) != families:
            self._families[url] = families
            record["available_serve_metrics"] = families
        return record

    def __call__(self, timestamp_ns: int, tick: int) -> list[dict[str, Any]]:
        started = time.monotonic()
        endpoints = discover_endpoints(self._discover())
        if not endpoints:
            raise RuntimeError("No live Ray metrics endpoints discovered")
        records = list(self._pool.map(self._scrape, endpoints))
        cycle_s = time.monotonic() - started
        for record in records:
            record["cycle_duration_s"] = cycle_s
            record["cycle_overran"] = cycle_s > self._interval_s
        return records

    def close(self) -> None:
        self._pool.shutdown(wait=True, cancel_futures=True)


def prometheus_collector(
    path: Path,
    interval_s: float,
    discover: Callable[[], list[dict[str, Any]]] = ray_nodes,
    fetch: Callable[[str], str] = fetch_metrics_text,
) -> JsonlCollector:
    sampler = _PrometheusSampler(discover, fetch, interval_s)
    return JsonlCollector(
        path,
        interval_s,
        sampler,
        lambda ts, error: {
            "timestamp_ns": ts, "node_id": None, "node_ip": None,
            "endpoint": None, "samples": [], "error": error,
        },
        name="prometheus-collector",
        close=sampler.close,
    )


class TelemetrySession:
    """Runs the status and Prometheus collectors for the lifetime of a block."""

    def __init__(
        self,
        config: ExperimentConfig,
        root: Path,
        *,
        status_fetch: Callable[[], dict[str, Any]] = fetch_serve_status,
        discover: Callable[[], list[dict[str, Any]]] = ray_nodes,
        fetch_metrics: Callable[[str], str] = fetch_metrics_text,
    ):
        analysis = config.analysis
        telemetry_dir = root / "telemetry"
        telemetry_dir.mkdir(parents=True, exist_ok=True)
        self.collectors = [
            serve_status_collector(
                telemetry_dir / "serve_status.jsonl",
                config.deployment.application_name,
                analysis.status_interval_s,
                status_fetch,
            )
        ]
        if analysis.prometheus_enabled:
            self.collectors.append(prometheus_collector(
                telemetry_dir / "serve_metrics.jsonl",
                analysis.metrics_interval_s,
                discover,
                fetch_metrics,
            ))
        self._stop_timeout_s = (
            max(analysis.status_interval_s, analysis.metrics_interval_s) + SCRAPE_TIMEOUT_S + 5
        )

    def __enter__(self) -> "TelemetrySession":
        started = []
        try:
            for collector in self.collectors:
                collector.start()
                started.append(collector)
        except BaseException:
            for collector in started:
                collector.stop(self._stop_timeout_s)
            raise
        return self

    def __exit__(self, *exc: Any) -> None:
        for collector in self.collectors:
            collector._stop.set()
        for collector in self.collectors:
            collector.stop(self._stop_timeout_s)

    def summary(self) -> dict[str, Any]:
        return {
            Path(c.path).name: {"records": c.records_written, "error_records": c.error_records}
            for c in self.collectors
        }
