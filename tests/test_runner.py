import json
from pathlib import Path

import pytest

from serve_llm_autoscaling.config import load_config
from serve_llm_autoscaling.runner import run_benchmarks


def test_sweep_continues_after_failure(monkeypatch, tmp_path: Path):
    config = load_config("experiments/baseline.yaml")
    config.benchmark.levels = [1.0, 2.0]

    def fake_run(self, level):
        if level == 1:
            raise RuntimeError("boom")
        return {"level": level, "error": None}

    monkeypatch.setattr(
        "serve_llm_autoscaling.runner.AIPerfRunner.run_point", fake_run
    )
    (tmp_path / "benchmark").mkdir()
    summary = run_benchmarks(config, tmp_path)
    assert "boom" in summary[0]["error"]
    assert summary[1]["error"] is None
    assert (tmp_path / "benchmark" / "sweep_summary.json").exists()



def _fail_on_run_point(self, level):
    raise AssertionError("series mode must not run static points")


def test_series_runs_once(monkeypatch, tmp_path: Path):
    config = load_config("experiments/smoke_rate_series.yaml")
    calls = []

    def fake_series(self):
        calls.append(1)
        return {"mode": "request_rate_series", "level": "series",
                "request_count": 40, "error": None}

    monkeypatch.setattr(
        "serve_llm_autoscaling.runner.AIPerfRunner.run_series", fake_series
    )
    monkeypatch.setattr(
        "serve_llm_autoscaling.runner.AIPerfRunner.run_point", _fail_on_run_point
    )
    summary = run_benchmarks(config, tmp_path)
    assert calls == [1]
    assert summary == json.loads(
        (tmp_path / "benchmark" / "sweep_summary.json").read_text()
    )
    assert len(summary) == 1
    assert summary[0]["level"] == "series"
    assert summary[0]["error"] is None


def test_series_failure_is_recorded(monkeypatch, tmp_path: Path):
    config = load_config("experiments/smoke_rate_series.yaml")
    config.benchmark.fail_fast = False

    def fake_series(self):
        raise RuntimeError("AIPerf exited 1; see x/stderr.log")

    monkeypatch.setattr(
        "serve_llm_autoscaling.runner.AIPerfRunner.run_series", fake_series
    )
    summary = run_benchmarks(config, tmp_path)
    saved = json.loads((tmp_path / "benchmark" / "sweep_summary.json").read_text())
    assert saved == summary
    assert summary == [{
        "mode": "request_rate_series",
        "level": "series",
        "error": "RuntimeError: AIPerf exited 1; see x/stderr.log",
    }]


def test_series_failure_raises_with_fail_fast(monkeypatch, tmp_path: Path):
    config = load_config("experiments/smoke_rate_series.yaml")
    assert config.benchmark.fail_fast

    def fake_series(self):
        raise RuntimeError("boom")

    monkeypatch.setattr(
        "serve_llm_autoscaling.runner.AIPerfRunner.run_series", fake_series
    )
    with pytest.raises(RuntimeError, match="boom"):
        run_benchmarks(config, tmp_path)
    assert (tmp_path / "benchmark" / "sweep_summary.json").exists()


def test_static_mode_runs_each_level(monkeypatch, tmp_path: Path):
    config = load_config("experiments/baseline_rate.yaml")
    seen = []

    def fake_run(self, level):
        seen.append(level)
        return {"level": level, "error": None}

    monkeypatch.setattr(
        "serve_llm_autoscaling.runner.AIPerfRunner.run_point", fake_run
    )
    summary = run_benchmarks(config, tmp_path)
    assert seen == config.benchmark.levels
    assert len(summary) == len(config.benchmark.levels)


def _fake_cluster(monkeypatch, tmp_path: Path):
    from serve_llm_autoscaling import runner
    from serve_llm_autoscaling.telemetry import TelemetrySession

    monkeypatch.setattr(runner, "environment_check", lambda connect: {})
    for name, value in {"deploy": {}, "wait_healthy": {}, "teardown": None}.items():
        monkeypatch.setattr(runner.RayServeBackend, name, lambda self, v=value: v)
    sessions = []

    def session(config, root):
        sessions.append(TelemetrySession(
            config, root, status_fetch=lambda: {"applications": {}},
            discover=lambda: [], fetch_metrics=lambda url: "",
        ))
        return sessions[-1]

    monkeypatch.setattr(runner, "TelemetrySession", session)
    return sessions


def test_series_experiment_collects_telemetry_and_analyzes(monkeypatch, tmp_path: Path):
    from serve_llm_autoscaling.runner import run_experiment

    sessions = _fake_cluster(monkeypatch, tmp_path)
    config = load_config("experiments/smoke_rate_series.yaml")
    config.runtime.results_dir = tmp_path / "runs"
    order = []

    def fake_series(self):
        order.append([c._thread.is_alive() for c in sessions[0].collectors])
        return {"mode": "request_rate_series", "level": "series", "error": None}

    monkeypatch.setattr(
        "serve_llm_autoscaling.runner.AIPerfRunner.run_series", fake_series
    )
    monkeypatch.setattr("serve_llm_autoscaling.runner.run_analysis",
                        lambda root: {"warnings": ["w"], "plot_error": None})
    root = run_experiment(config, Path("experiments/smoke_rate_series.yaml"))
    assert order == [[True, True]]  # collectors ran during AIPerf
    assert all(not c._thread.is_alive() for c in sessions[0].collectors)
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["status"] == "succeeded"
    assert manifest["analysis"] == {"warnings": ["w"], "plot_error": None}
    assert set(manifest["telemetry"]) == {"serve_status.jsonl", "serve_metrics.jsonl"}
    assert (root / "telemetry" / "serve_status.jsonl").exists()


def test_analysis_failure_does_not_fail_run(monkeypatch, tmp_path: Path):
    from serve_llm_autoscaling.runner import run_experiment

    _fake_cluster(monkeypatch, tmp_path)
    config = load_config("experiments/smoke_rate_series.yaml")
    config.runtime.results_dir = tmp_path / "runs"
    monkeypatch.setattr(
        "serve_llm_autoscaling.runner.AIPerfRunner.run_series",
        lambda self: {"mode": "request_rate_series", "level": "series", "error": None},
    )
    # No AIPerf artifacts exist, so the real analysis fails.
    root = run_experiment(config, Path("experiments/smoke_rate_series.yaml"))
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["status"] == "succeeded"
    assert "analysis_error" in manifest["analysis"]


def test_static_experiment_is_unchanged(monkeypatch, tmp_path: Path):
    from serve_llm_autoscaling.runner import run_experiment

    sessions = _fake_cluster(monkeypatch, tmp_path)
    config = load_config("experiments/baseline.yaml")
    config.runtime.results_dir = tmp_path / "runs"
    monkeypatch.setattr("serve_llm_autoscaling.runner.AIPerfRunner.run_point",
                        lambda self, level: {"level": level, "error": None})
    root = run_experiment(config, Path("experiments/baseline.yaml"))
    assert sessions == []
    assert not (root / "telemetry").exists() and not (root / "analysis").exists()
    manifest = json.loads((root / "manifest.json").read_text())
    assert "telemetry" not in manifest and "analysis" not in manifest


def test_manual_series_benchmark_without_ray_warns(monkeypatch, tmp_path: Path, capsys):
    from serve_llm_autoscaling import runner

    def no_ray(self):
        raise ConnectionError("no cluster")

    monkeypatch.setattr(runner.RayServeBackend, "connect", no_ray)
    monkeypatch.setattr(
        "serve_llm_autoscaling.runner.AIPerfRunner.run_series",
        lambda self: {"mode": "request_rate_series", "level": "series", "error": None},
    )
    config = load_config("experiments/smoke_rate_series.yaml")
    runner.run_manual_benchmark(config, tmp_path)
    err = capsys.readouterr().err
    assert "running without Serve telemetry" in err
    assert not (tmp_path / "telemetry").exists()
    assert (tmp_path / "resolved.yaml").exists()


# --- AgentX concurrency ramp ------------------------------------------------


def _agentx_row(self):
    return {"mode": "agentx_concurrency_ramp", "level": 16, "target_concurrency": 16,
            "ramp_duration_s": 1200, "error": None}


def test_agentx_ramp_runs_once(agentx_config, monkeypatch, tmp_path: Path):
    calls = []

    def fake_ramp(self):
        calls.append(1)
        return _agentx_row(self)

    monkeypatch.setattr("serve_llm_autoscaling.runner.AIPerfRunner.run_agentx_ramp",
                        fake_ramp)
    monkeypatch.setattr("serve_llm_autoscaling.runner.AIPerfRunner.run_point",
                        _fail_on_run_point)
    monkeypatch.setattr("serve_llm_autoscaling.runner.AIPerfRunner.run_series",
                        _fail_on_run_point)
    summary = run_benchmarks(agentx_config(), tmp_path)
    assert calls == [1]
    assert summary == [_agentx_row(None)]


def test_agentx_ramp_failure_row(agentx_config, monkeypatch, tmp_path: Path):
    def fail(self):
        raise RuntimeError("AIPerf exited 1")

    monkeypatch.setattr("serve_llm_autoscaling.runner.AIPerfRunner.run_agentx_ramp", fail)
    summary = run_benchmarks(agentx_config(fail_fast=False), tmp_path)
    assert summary == [{"mode": "agentx_concurrency_ramp", "level": 16,
                        "target_concurrency": 16, "ramp_duration_s": 1200,
                        "error": "RuntimeError: AIPerf exited 1"}]


def test_agentx_experiment_collects_telemetry_and_analyzes(
    agentx_config, monkeypatch, tmp_path: Path
):
    from serve_llm_autoscaling.runner import run_experiment

    sessions = _fake_cluster(monkeypatch, tmp_path)
    config = agentx_config()
    config.runtime.results_dir = tmp_path / "runs"
    input_path = tmp_path / "agentx.yaml"
    input_path.write_text("name: agentx-ramp\n")
    alive = []

    def fake_ramp(self):
        alive.append([c._thread.is_alive() for c in sessions[0].collectors])
        return _agentx_row(self)

    analyzed = []
    monkeypatch.setattr("serve_llm_autoscaling.runner.AIPerfRunner.run_agentx_ramp",
                        fake_ramp)
    monkeypatch.setattr("serve_llm_autoscaling.runner.run_analysis",
                        lambda root: analyzed.append(root) or {"warnings": []})
    root = run_experiment(config, input_path)
    assert alive == [[True, True]]  # collectors ran during AIPerf
    assert analyzed == [root]
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["status"] == "succeeded"
    assert set(manifest["telemetry"]) == {"serve_status.jsonl", "serve_metrics.jsonl"}


def test_manual_agentx_benchmark_attempts_telemetry(
    agentx_config, monkeypatch, tmp_path: Path, capsys
):
    from serve_llm_autoscaling import runner

    def no_ray(self):
        raise ConnectionError("no cluster")

    monkeypatch.setattr(runner.RayServeBackend, "connect", no_ray)
    monkeypatch.setattr("serve_llm_autoscaling.runner.AIPerfRunner.run_agentx_ramp",
                        _agentx_row)
    runner.run_manual_benchmark(agentx_config(), tmp_path)
    assert "running without Serve telemetry" in capsys.readouterr().err
    assert (tmp_path / "resolved.yaml").exists()
