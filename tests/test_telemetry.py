import json
import threading
from enum import Enum
from pathlib import Path

import pytest

from serve_llm_autoscaling.config import load_config
from serve_llm_autoscaling.telemetry import (
    TelemetrySession,
    discover_endpoints,
    parse_metrics,
    prometheus_collector,
    serve_status_collector,
    serve_status_sample,
)

APP = "ttft-benchmark"
LLM = "LLMServer:Qwen--Qwen3-0_6B-FP8"


def _replica(replica_id, state, node="node-1"):
    return {"replica_id": replica_id, "state": state, "node_id": node,
            "actor_id": f"actor-{replica_id}", "pid": 1, "start_time_s": 0.0}


def serve_details(target=3, replicas=(("r1", "RUNNING"), ("r2", "STARTING")), dead=("r0",)):
    return {"applications": {APP: {
        "status": "RUNNING",
        "deployments": {LLM: {
            "status": "UPSCALING",
            "status_trigger": "AUTOSCALING",
            "target_num_replicas": target,
            "replicas": [_replica(r, s) for r, s in replicas],
            "recent_dead_replicas": [_replica(r, "STOPPED") for r in dead],
        }},
    }}}


def _read(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def _wait_for_records(collector, count, timeout=5):
    event = threading.Event()
    original = collector._write

    def write(record):
        original(record)
        if collector.records_written >= count:
            event.set()

    collector._write = write
    collector.start()
    assert event.wait(timeout)
    collector.stop(timeout)


def test_status_sample_counts_live_replicas():
    sample = serve_status_sample(serve_details(), APP, 123)
    deployment = sample["deployments"][LLM]
    assert sample["timestamp_ns"] == 123 and sample["error"] is None
    assert sample["application_status"] == "RUNNING"
    assert deployment["target_num_replicas"] == 3
    assert deployment["live_replicas"] == 2  # r0 is a recent dead replica
    assert deployment["replicas_by_state"] == {"RUNNING": 1, "STARTING": 1}
    assert [r["replica_id"] for r in deployment["replicas"]] == ["r1", "r2"]
    assert deployment["replicas"][0] == {
        "replica_id": "r1", "state": "RUNNING", "node_id": "node-1", "actor_id": "actor-r1"
    }


def test_status_sample_records_enum_values():
    class State(str, Enum):
        RUNNING = "RUNNING"

    details = serve_details()
    deployment = details["applications"][APP]["deployments"][LLM]
    deployment["status"] = State.RUNNING
    deployment["replicas"][0]["state"] = State.RUNNING
    deployment = serve_status_sample(details, APP, 1)["deployments"][LLM]
    assert type(deployment["status"]) is str
    assert [type(k) for k in deployment["replicas_by_state"]] == [str, str]


def test_status_sample_missing_application():
    sample = serve_status_sample({"applications": {}}, APP, 1)
    assert sample["application_status"] is None and sample["deployments"] == {}


def test_status_collector_recovers_after_error(tmp_path: Path):
    calls = []

    def fetch():
        calls.append(1)
        if len(calls) == 1:
            raise ConnectionError("controller unavailable")
        return serve_details()

    path = tmp_path / "serve_status.jsonl"
    _wait_for_records(serve_status_collector(path, APP, 0.01, fetch), 2)
    first, second = _read(path)[:2]
    assert first["error"] == "ConnectionError: controller unavailable"
    assert first["deployments"] == {} and first["timestamp_ns"] > 0
    assert second["error"] is None
    assert second["deployments"][LLM]["live_replicas"] == 2


def _session(tmp_path, **overrides):
    config = load_config("experiments/step_rate_autoscaling.yaml")
    config.analysis.telemetry_interval_s = 0.01
    kwargs = {
        "status_fetch": serve_details,
        "discover": lambda: [{"NodeID": "n1", "NodeManagerAddress": "10.0.0.1",
                              "MetricsExportPort": 8085, "Alive": True}],
        "fetch_metrics": lambda url: "ray_serve_replica_processing_queries 1\n",
    }
    return TelemetrySession(config, tmp_path, **{**kwargs, **overrides})


def test_session_stops_and_flushes_after_exception(tmp_path: Path):
    session = _session(tmp_path)
    with pytest.raises(RuntimeError, match="aiperf failed"):
        with session:
            threading.Event().wait(0.1)
            raise RuntimeError("aiperf failed")
    assert all(not c._thread.is_alive() for c in session.collectors)
    assert all(c._fh.closed for c in session.collectors)
    status = _read(tmp_path / "telemetry" / "serve_status.jsonl")
    metrics = _read(tmp_path / "telemetry" / "serve_metrics.jsonl")
    assert len(status) == session.summary()["serve_status.jsonl"]["records"] > 0
    assert metrics and metrics[0]["samples"][0]["value"] == 1


def test_session_without_prometheus(tmp_path: Path):
    config = load_config("experiments/step_rate_autoscaling.yaml")
    config.analysis.prometheus_enabled = False
    session = TelemetrySession(config, tmp_path, status_fetch=serve_details)
    assert [Path(c.path).name for c in session.collectors] == ["serve_status.jsonl"]


def test_discover_endpoints():
    nodes = [
        {"NodeID": "a", "NodeManagerAddress": "10.0.0.1", "MetricsExportPort": 8085,
         "Alive": True},
        {"NodeID": "a2", "NodeManagerAddress": "10.0.0.1", "MetricsExportPort": 8085,
         "Alive": True},
        {"NodeID": "b", "NodeManagerAddress": "10.0.0.2", "MetricsExportPort": 8085,
         "Alive": False},
        {"NodeID": "c", "NodeManagerAddress": "10.0.0.3", "Alive": True},
        {"NodeID": "d", "NodeManagerAddress": "10.0.0.4", "MetricsExportPort": 9000,
         "Alive": True},
    ]
    assert discover_endpoints(nodes) == [
        {"node_id": "a", "node_ip": "10.0.0.1", "endpoint": "http://10.0.0.1:8085/metrics"},
        {"node_id": "d", "node_ip": "10.0.0.4", "endpoint": "http://10.0.0.4:9000/metrics"},
    ]


EXPOSITION = """\
# HELP ray_serve_num_ongoing_requests_at_replicas Ongoing requests.
# TYPE ray_serve_num_ongoing_requests_at_replicas gauge
ray_serve_num_ongoing_requests_at_replicas{application="ttft-benchmark",deployment="LLMServer:x",handle="h1",actor_id="a1"} 4.0
# TYPE serve_autoscaling_desired_replicas gauge
serve_autoscaling_desired_replicas{application="ttft-benchmark",deployment="LLMServer:x"} 2.0
# TYPE ray_serve_http_request_latency_ms histogram
ray_serve_http_request_latency_ms_count{route="/"} 3.0
# TYPE ray_tasks gauge
ray_tasks{State="RUNNING"} 5.0
"""


def test_parse_metrics_selects_and_normalizes():
    samples, families = parse_metrics(EXPOSITION)
    assert [(s["name"], s["normalized_name"]) for s in samples] == [
        ("ray_serve_num_ongoing_requests_at_replicas",
         "serve_num_ongoing_requests_at_replicas"),
        ("serve_autoscaling_desired_replicas", "serve_autoscaling_desired_replicas"),
    ]
    assert samples[0]["labels"] == {"application": "ttft-benchmark",
                                    "deployment": "LLMServer:x", "handle": "h1",
                                    "actor_id": "a1"}
    assert samples[0]["value"] == 4 and samples[0]["type"] == "gauge"
    # Unselected Serve families are listed, non-Serve families are not.
    assert "serve_http_request_latency_ms" in families
    assert not any("tasks" in f for f in families)


def test_parse_metrics_without_serve_metrics():
    assert parse_metrics("# TYPE ray_tasks gauge\nray_tasks 1\n") == ([], [])


def test_prometheus_collector_records_node_failure(tmp_path: Path):
    nodes = [{"NodeID": "good", "NodeManagerAddress": "10.0.0.1", "MetricsExportPort": 1,
              "Alive": True},
             {"NodeID": "bad", "NodeManagerAddress": "10.0.0.2", "MetricsExportPort": 1,
              "Alive": True}]

    def fetch(url):
        if "10.0.0.2" in url:
            raise TimeoutError("scrape timed out")
        return EXPOSITION

    path = tmp_path / "serve_metrics.jsonl"
    _wait_for_records(prometheus_collector(path, 0.01, lambda: nodes, fetch), 4)
    records = _read(path)
    by_node = {r["node_id"]: r for r in records[:2]}
    assert by_node["bad"]["error"] == "TimeoutError: scrape timed out"
    assert by_node["bad"]["samples"] == []
    assert by_node["good"]["error"] is None
    assert by_node["good"]["endpoint"] == "http://10.0.0.1:1/metrics"
    assert len(by_node["good"]["samples"]) == 2
    assert "available_serve_metrics" in by_node["good"]
    # The family list is only repeated when it changes.
    later = [r for r in records[2:] if r["node_id"] == "good"]
    assert later and "available_serve_metrics" not in later[0]
    assert records[0]["tick"] == records[1]["tick"] == 0


def test_prometheus_collector_records_discovery_failure(tmp_path: Path):
    def discover():
        raise RuntimeError("ray not connected")

    path = tmp_path / "serve_metrics.jsonl"
    _wait_for_records(prometheus_collector(path, 0.01, discover, lambda url: ""), 1)
    record = _read(path)[0]
    assert record["error"] == "RuntimeError: ray not connected"
    assert record["endpoint"] is None and record["samples"] == []
