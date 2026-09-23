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
from collections import Counter
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
    ):
        self.path = path
        self.interval_s = interval_s
        self._sample = sample
        self._error_record = error_record
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
        if self._fh is not None and not self._thread.is_alive():
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


def fetch_serve_details() -> dict[str, Any]:
    """Serve instance details, keyed as in the public ``ServeInstanceDetails`` schema.

    ``serve.status()`` is built from these details but drops per-replica
    identities and target replica counts, which the timeline needs.
    """
    from ray.serve.context import _get_global_client

    client = _get_global_client(raise_if_no_controller_running=False)
    if client is None:
        return {"applications": {}}
    return client.get_serve_details()


def _value(value: Any) -> Any:
    """Serve's status enums are str enums; record their plain values."""
    return getattr(value, "value", value)


def serve_status_sample(
    details: dict[str, Any], application: str, timestamp_ns: int
) -> dict[str, Any]:
    app = (details.get("applications") or {}).get(application) or {}
    deployments = {}
    for name, deployment in (app.get("deployments") or {}).items():
        # Only ``replicas`` are live; ``recent_dead_replicas`` are history.
        replicas = [
            {key: _value(r.get(key)) for key in ("replica_id", "state", "node_id", "actor_id")}
            for r in deployment.get("replicas") or []
        ]
        deployments[name] = {
            "status": _value(deployment.get("status")),
            "status_trigger": _value(deployment.get("status_trigger")),
            "target_num_replicas": deployment.get("target_num_replicas"),
            "live_replicas": len(replicas),
            "replicas_by_state": dict(Counter(r["state"] for r in replicas)),
            "replicas": replicas,
        }
    return {
        "timestamp_ns": timestamp_ns,
        "application": application,
        "application_status": _value(app.get("status")),
        "deployments": deployments,
        "error": None,
    }


def serve_status_collector(
    path: Path,
    application: str,
    interval_s: float,
    fetch: Callable[[], dict[str, Any]] = fetch_serve_details,
) -> JsonlCollector:
    return JsonlCollector(
        path,
        interval_s,
        lambda ts, tick: [serve_status_sample(fetch(), application, ts)],
        lambda ts, error: {
            "timestamp_ns": ts, "application": application,
            "application_status": None, "deployments": {}, "error": error,
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
    response = requests.get(url, timeout=SCRAPE_TIMEOUT_S)
    response.raise_for_status()
    return response.text


class _PrometheusSampler:
    def __init__(self, discover: Callable[[], list[dict[str, Any]]],
                 fetch: Callable[[str], str]):
        self._discover = discover
        self._fetch = fetch
        self._pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="metrics-scrape")
        self._families: dict[str, list[str]] = {}

    def _scrape(self, endpoint: dict[str, Any]) -> dict[str, Any]:
        record = {"timestamp_ns": time.time_ns(), **endpoint, "samples": [], "error": None}
        try:
            record["samples"], families = parse_metrics(self._fetch(endpoint["endpoint"]))
        except Exception as exc:
            record["error"] = _error(exc)
            return record
        # The full Serve metric list is only written when it changes.
        if self._families.get(endpoint["endpoint"]) != families:
            self._families[endpoint["endpoint"]] = families
            record["available_serve_metrics"] = families
        return record

    def __call__(self, timestamp_ns: int, tick: int) -> list[dict[str, Any]]:
        endpoints = discover_endpoints(self._discover())
        return list(self._pool.map(self._scrape, endpoints))


def prometheus_collector(
    path: Path,
    interval_s: float,
    discover: Callable[[], list[dict[str, Any]]] = ray_nodes,
    fetch: Callable[[str], str] = fetch_metrics_text,
) -> JsonlCollector:
    return JsonlCollector(
        path,
        interval_s,
        _PrometheusSampler(discover, fetch),
        lambda ts, error: {
            "timestamp_ns": ts, "node_id": None, "node_ip": None,
            "endpoint": None, "samples": [], "error": error,
        },
        name="prometheus-collector",
    )


class TelemetrySession:
    """Runs the status and Prometheus collectors for the lifetime of a block."""

    def __init__(
        self,
        config: ExperimentConfig,
        root: Path,
        *,
        status_fetch: Callable[[], dict[str, Any]] = fetch_serve_details,
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
                analysis.telemetry_interval_s,
                status_fetch,
            )
        ]
        if analysis.prometheus_enabled:
            self.collectors.append(prometheus_collector(
                telemetry_dir / "serve_metrics.jsonl",
                analysis.telemetry_interval_s,
                discover,
                fetch_metrics,
            ))
        self._stop_timeout_s = analysis.telemetry_interval_s + SCRAPE_TIMEOUT_S + 5

    def __enter__(self) -> "TelemetrySession":
        for collector in self.collectors:
            collector.start()
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
