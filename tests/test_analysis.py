import json
from pathlib import Path

import pytest

from serve_llm_autoscaling.analysis import (
    aggregate_metric,
    analyze_run,
    build_metrics_inventory,
    build_plot_data,
    load_run_config,
    metric_series,
)
from serve_llm_autoscaling.artifacts import write_json
from serve_llm_autoscaling.cli import main
from serve_llm_autoscaling.config import load_config, write_config
from serve_llm_autoscaling.plotting import primary_points
from serve_llm_autoscaling.windows import request_timeseries

S = 1_000_000_000
T0 = 1_790_000_000 * S  # an epoch timestamp, as AIPerf and time.time_ns() use
APP = "ttft-benchmark"
LLM = "LLMServer:Qwen--Qwen3-0_6B-FP8"
INGRESS = "OpenAiIngress"


def _request(credit_s, end_s, ttft=100.0):
    return {"metadata": {"credit_issued_ns": T0 + int(credit_s * S),
                         "request_start_ns": T0 + int(credit_s * S),
                         "request_end_ns": T0 + int(end_s * S),
                         "benchmark_phase": "profiling"},
            "metrics": {"time_to_first_token": {"value": ttft}}}


def _status(t_s, target, states, error=None):
    if error:
        return {"timestamp_ns": T0 + int(t_s * S), "application": APP,
                "application_status": None, "deployments": {}, "error": error}
    replicas = [{"replica_id": f"r{i}", "state": state, "node_id": "n1", "actor_id": f"a{i}"}
                for i, state in enumerate(states)]
    deployment = {"status": "HEALTHY", "status_trigger": "AUTOSCALING",
                  "target_num_replicas": target, "live_replicas": len(states),
                  "replicas_by_state": {s: states.count(s) for s in set(states)},
                  "replicas": replicas}
    ingress = {**deployment, "target_num_replicas": 1, "live_replicas": 1,
               "replicas_by_state": {"RUNNING": 1}, "replicas": []}
    return {"timestamp_ns": T0 + int(t_s * S), "application": APP,
            "application_status": "RUNNING", "error": None,
            "deployments": {LLM: deployment, INGRESS: ingress}}


def _sample(name, value, **labels):
    normalized = name.removeprefix("ray_")
    for suffix in ("_count", "_sum", "_bucket"):
        if normalized.startswith("serve_replica_startup_latency_ms"):
            normalized = normalized.removesuffix(suffix)
    return {"name": name, "normalized_name": normalized, "type": "gauge",
            "labels": labels, "value": value}


def _scrape(t_s, tick, samples, node="n1", error=None):
    return {"timestamp_ns": T0 + int(t_s * S), "tick": tick, "node_id": node,
            "node_ip": "10.0.0.1", "endpoint": f"http://{node}:8085/metrics",
            "samples": [] if error else samples, "error": error}


def _jsonl(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


def make_run(root: Path, *, status=True, metrics=True) -> Path:
    config = load_config("experiments/step_rate_autoscaling.yaml")
    root.mkdir(parents=True, exist_ok=True)
    write_config(config, root / "resolved.yaml")
    series = root / "benchmark" / "request-rate-series"
    series.mkdir(parents=True)
    write_json(series / "rate_series.json",
               {"points": [p.model_dump() for p in config.benchmark.rate_series]})
    write_json(series / "phase_manifest.json", {"phases": [
        {"phase_kind": "warmup", "start_ns": T0 - 30 * S, "end_ns": T0},
        {"phase_kind": "profiling", "start_ns": T0, "end_ns": T0 + 300 * S},
    ]})
    _jsonl(series / "profile_export.jsonl",
           [_request(t, t + 0.5, ttft=100 + t) for t in range(0, 300, 2)])
    if status:
        _jsonl(root / "telemetry" / "serve_status.jsonl", [
            _status(-10, 1, ["RUNNING"]),
            _status(2, 1, ["RUNNING"]),
            _status(3, None, [], error="ConnectionError: controller"),
            _status(90, 3, ["RUNNING", "STARTING", "STARTING"]),
            _status(150, 3, ["RUNNING", "RUNNING", "RUNNING"]),
        ])
    if metrics:
        labels = {"application": APP, "deployment": LLM}
        _jsonl(root / "telemetry" / "serve_metrics.jsonl", [
            _scrape(t, i, [
                _sample("ray_serve_replica_processing_queries", 4, replica="r0", **labels),
                _sample("ray_serve_replica_processing_queries", 3, replica="r1", **labels),
                _sample("ray_serve_replica_processing_queries", 9, replica="x",
                        application=APP, deployment=INGRESS),
                _sample("ray_serve_request_router_queue_len", 2, replica_id="r0",
                        actor_id="router", **labels),
                _sample("ray_serve_autoscaling_desired_replicas", 1 + (t >= 90) * 2, **labels),
                _sample("ray_serve_autoscaling_target_replicas", 1 + (t >= 90) * 2, **labels),
            ])
            for i, t in enumerate(range(-5, 200, 5))
        ] + [_scrape(200, 99, [], node="n2", error="TimeoutError: slow node")])
    return root


def test_plot_data_aligns_epoch_timestamps(tmp_path: Path):
    root = make_run(tmp_path / "run")
    analyze_run(root, plot=False)
    data = json.loads((root / "analysis" / "plot_data.json").read_text())
    assert data["schema_version"] == 1
    assert data["time_origin"] == {"profiling_start_ns": T0, "source": "phase_manifest",
                                   "profiling_end_s": 300.0}
    times = [s["relative_time_s"] for s in data["serve_status_samples"]]
    assert times == [-10.0, 2.0, 3.0, 90.0, 150.0]
    points = data["serve_metric_series"]["ongoing_requests"]["points"]
    assert points[0]["relative_time_s"] == -5.0 and points[1]["relative_time_s"] == 0.0
    assert data["request_rate_curve"][2] == {"time_s": 60.1, "qps": 18.0}
    assert data["request_windows"][0]["ttft_sample_count"] == 3
    assert data["experiment"]["deployments"] == [LLM]


def test_replica_counts_remain_step_data(tmp_path: Path):
    root = make_run(tmp_path / "run")
    analyze_run(root, plot=False)
    data = json.loads((root / "analysis" / "plot_data.json").read_text())
    samples = data["serve_status_samples"]
    # Kept at their own timestamps, not averaged into request windows.
    assert len(samples) == 5
    upscaling = samples[3]
    assert upscaling["target_replicas"] == 3  # the LLM deployment only, not the ingress
    assert (upscaling["running_replicas"], upscaling["starting_replicas"]) == (1, 2)
    assert samples[2]["error"] and "target_replicas" not in samples[2]
    assert "1 of 5 Serve status samples failed" in data["warnings"]


def test_primary_plot_excludes_pre_profile_samples():
    points = [{"relative_time_s": -10, "v": 1}, {"relative_time_s": 5, "v": 3},
              {"relative_time_s": 9, "v": None}, {"relative_time_s": 12, "v": 2}]
    assert primary_points(points, "v") == ([5, 12], [3, 2])
    # A step series starts at t=0 with the state carried from before profiling.
    assert primary_points(points, "v", step=True) == ([0.0, 5, 12], [1, 3, 2])


def test_metric_filters_and_sums_replicas(tmp_path: Path):
    root = make_run(tmp_path / "run")
    analyze_run(root, plot=False)
    data = json.loads((root / "analysis" / "plot_data.json").read_text())
    series = data["serve_metric_series"]
    assert series["ongoing_requests"]["points"][1]["value"] == 7  # ingress excluded
    assert series["router_queue"]["points"][0]["value"] == 2
    assert [p["value"] for p in series["desired_replicas"]["points"]][18:20] == [1, 3]
    assert data["availability"] == {
        "replica_status": True, "router_queue": True, "ongoing_requests": True,
        "desired_replicas": True, "target_replicas_metric": True, "healthy_replicas": False,
    }


def test_duplicate_controller_metrics_are_not_summed():
    labels = {"application": APP, "deployment": LLM}
    samples = [("n1", _sample("serve_autoscaling_desired_replicas", 3, **labels)),
               ("n2", _sample("serve_autoscaling_desired_replicas", 3, **labels))]
    warnings = []
    points = aggregate_metric([(T0, samples)], "unique", T0, warnings, "desired")
    assert points == [{"relative_time_s": 0.0, "value": 3, "series_count": 2}]
    assert warnings == []
    conflicting = [samples[0], ("n2", _sample("serve_autoscaling_desired_replicas", 2,
                                              **labels))]
    points = aggregate_metric([(T0, conflicting)], "unique", T0, warnings, "desired")
    assert points[0]["value"] == 3 and "conflicting" in warnings[0]


def test_per_replica_summation():
    labels = {"application": APP, "deployment": LLM}
    samples = [("n1", _sample("serve_replica_processing_queries", 4, replica="a", **labels)),
               ("n2", _sample("serve_replica_processing_queries", 5, replica="b", **labels)),
               ("n1", _sample("serve_replica_processing_queries", None, replica="c", **labels))]
    points = aggregate_metric([(T0, samples), (T0 + S, [])], "sum", T0, [], "ongoing")
    assert points == [{"relative_time_s": 0.0, "value": 9, "series_count": 2}]  # no zero gap


def test_healthy_replica_count():
    per_replica = [("n1", _sample("serve_deployment_replica_healthy", v, replica=r,
                                  application=APP, deployment=LLM))
                   for r, v in (("a", 1), ("b", 0), ("c", 1))]
    assert aggregate_metric([(T0, per_replica)], "healthy", T0, [], "h")[0]["value"] == 2
    aggregate = [("n1", _sample("serve_deployment_replica_healthy", 3, application=APP,
                                deployment=LLM))]
    assert aggregate_metric([(T0, aggregate)], "healthy", T0, [], "h")[0]["value"] == 3


def test_counter_deltas_are_node_local():
    def tick(t, a, b):
        return (T0 + t * S, [
            ("n1", _sample("serve_replica_startup_latency_ms_count", a, deployment=LLM)),
            ("n2", _sample("serve_replica_startup_latency_ms_count", b, deployment=LLM)),
        ])

    # n2 resets from 7 to 1 between the second and third scrape.
    points = aggregate_metric(
        [tick(0, 2, 5), tick(1, 3, 7), tick(2, 3, 1)], "counter_delta", T0, [], "c"
    )
    assert [(p["relative_time_s"], p["value"]) for p in points] == [(1.0, 3), (2.0, 1)]


def test_unlabeled_series_are_kept_with_warning():
    records = [_scrape(0, 0, [_sample("ray_serve_replica_processing_queries", 2)])]
    warnings = []
    series = metric_series(records, APP, {LLM}, 1, T0, warnings)
    assert series["ongoing_requests"]["points"][0]["value"] == 2
    assert "lack application/deployment labels" in warnings[0]


def test_metrics_inventory(tmp_path: Path):
    root = make_run(tmp_path / "run")
    inventory = build_metrics_inventory(root / "telemetry" / "serve_metrics.jsonl")
    entry = inventory["metrics"]["serve_replica_processing_queries"]
    assert entry["original_names"] == ["ray_serve_replica_processing_queries"]
    assert entry["labels"] == ["application", "deployment", "replica"]
    assert entry["source_nodes"] == ["n1"]
    assert entry["sample_count"] == 41 * 3
    assert entry["first_timestamp_ns"] == T0 - 5 * S
    assert entry["last_timestamp_ns"] == T0 + 195 * S
    assert "serve_deployment_replica_healthy" in inventory["missing_recommended_metrics"]
    assert "serve_autoscaling_desired_replicas" not in inventory["missing_recommended_metrics"]
    assert inventory["scrape_errors"] == 1


def test_plot_with_all_metrics(tmp_path: Path):
    root = make_run(tmp_path / "run")
    result = analyze_run(root)
    assert result["plot_error"] is None
    assert (root / "analysis" / "autoscaling_timeline.png").stat().st_size > 10_000
    for name in ("request_timeseries.json", "metrics_inventory.json", "plot_data.json"):
        assert (root / "analysis" / name).exists()


def test_plot_without_prometheus(tmp_path: Path):
    root = make_run(tmp_path / "run", metrics=False)
    result = analyze_run(root)
    assert result["plot_error"] is None
    assert any("Prometheus metrics were not collected" in w for w in result["warnings"])
    data = json.loads((root / "analysis" / "plot_data.json").read_text())
    assert data["serve_metric_series"] == {}
    assert data["availability"]["replica_status"] is True
    inventory = json.loads((root / "analysis" / "metrics_inventory.json").read_text())
    assert inventory["collected"] is False
    assert (root / "analysis" / "autoscaling_timeline.png").exists()


def test_plot_with_only_status_telemetry(tmp_path: Path):
    root = make_run(tmp_path / "run", metrics=False)
    _jsonl(root / "telemetry" / "serve_metrics.jsonl",
           [_scrape(0, 0, []), _scrape(1, 1, [])])
    result = analyze_run(root)
    assert result["plot_error"] is None
    # Missing Serve metrics are warnings, not zero-valued series.
    assert any("serve_request_router_queue_len was not exported" in w
               for w in result["warnings"])
    data = json.loads((root / "analysis" / "plot_data.json").read_text())
    assert data["serve_metric_series"] == {}
    assert data["availability"]["router_queue"] is False


def test_plot_without_any_telemetry(tmp_path: Path):
    root = make_run(tmp_path / "run", status=False, metrics=False)
    result = analyze_run(root)
    assert result["plot_error"] is None
    assert any("Serve status telemetry was not collected" in w for w in result["warnings"])


def test_plot_error_keeps_derived_data(tmp_path: Path, monkeypatch):
    root = make_run(tmp_path / "run")

    def broken(*args):
        raise RuntimeError("no display")

    monkeypatch.setattr("serve_llm_autoscaling.plotting.plot_timeline", broken)
    result = analyze_run(root)
    assert result["plot_error"] == "RuntimeError: no display"
    assert (root / "analysis" / "plot_data.json").exists()


def test_analyze_command_regenerates_with_new_window(tmp_path: Path, capsys):
    root = make_run(tmp_path / "run")
    assert main(["analyze", str(root), "--window-s", "10", "--tail-window-s", "60"]) == 0
    timeseries = json.loads((root / "analysis" / "request_timeseries.json").read_text())
    assert timeseries["window_s"] == 10 and timeseries["tail_window_s"] == 60
    assert timeseries["windows"][1]["window_start_s"] == 10
    data = json.loads((root / "analysis" / "plot_data.json").read_text())
    assert data["experiment"]["window_s"] == 10
    assert (root / "analysis" / "autoscaling_timeline.png").exists()


def test_analyze_command_no_plot(tmp_path: Path):
    root = make_run(tmp_path / "run")
    assert main(["analyze", str(root), "--no-plot"]) == 0
    assert (root / "analysis" / "plot_data.json").exists()
    assert not (root / "analysis" / "autoscaling_timeline.png").exists()


def test_analyze_command_rejects_short_tail_window(tmp_path: Path, capsys):
    root = make_run(tmp_path / "run")
    assert main(["analyze", str(root), "--window-s", "60"]) == 1
    assert "tail_window_s must be >= window_s" in capsys.readouterr().err


def test_old_runs_without_analysis_config_load(tmp_path: Path):
    root = make_run(tmp_path / "run")
    resolved = (root / "resolved.yaml").read_text()
    (root / "resolved.yaml").write_text(resolved.split("analysis:")[0])
    assert load_run_config(root).analysis.window_s == 5


def test_build_plot_data_requires_time_origin(tmp_path: Path):
    root = make_run(tmp_path / "run")
    series = root / "benchmark" / "request-rate-series"
    (series / "phase_manifest.json").unlink()
    (series / "profile_export.jsonl").write_text("")
    config = load_run_config(root)
    with pytest.raises(ValueError, match="cannot align"):
        build_plot_data(root, config, config.analysis, request_timeseries(series))
