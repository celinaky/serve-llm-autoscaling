import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from serve_llm_autoscaling import runner
from serve_llm_autoscaling.analysis import analyze_run, lifecycle_periods
from serve_llm_autoscaling.artifacts import format_run_summary, write_json
from serve_llm_autoscaling.config import ExperimentConfig, load_config
from test_analysis import S, T0, make_agentx_run
from test_runner import _agentx_row, _fake_cluster


def _observed(agentx_config, **benchmark):
    return agentx_config({"concurrency_ramp_duration_s": 120, "target_concurrency": 4,
                          "unsafe_override": True},
                         duration_s=200, grace_period_s=30, **benchmark)


# --- Config -----------------------------------------------------------------


def test_observation_defaults_to_zero():
    assert load_config("experiments/baseline.yaml").benchmark.post_load_observation_s == 0


@pytest.mark.parametrize("mode", ["concurrency", "request_rate"])
def test_observation_requires_continuous_mode(mode):
    with pytest.raises(ValidationError, match="post_load_observation_s requires"):
        ExperimentConfig.model_validate({
            "name": "x", "deployment": {"model_id": "m"},
            "benchmark": {"mode": mode, "post_load_observation_s": 30},
        })


def test_observation_accepted_for_agentx(agentx_config):
    assert _observed(agentx_config, post_load_observation_s=90).benchmark \
        .post_load_observation_s == 90


# --- Runner -----------------------------------------------------------------


def _phase_manifest(root: Path, end_s: float = 225):
    ramp = root / "benchmark" / "agentx-concurrency-ramp"
    ramp.mkdir(parents=True, exist_ok=True)
    write_json(ramp / "phase_manifest.json", {"phases": [
        {"phase_kind": "profiling", "start_ns": T0, "end_ns": T0 + int(end_s * S)},
    ]})


def _clock(monkeypatch, sleeps):
    """A fake epoch clock that advances only when the harness sleeps."""
    now = [T0 - 20 * S]
    monkeypatch.setattr(runner.time, "time_ns", lambda: now[0])

    def sleep(seconds):
        sleeps.append(seconds)
        now[0] += int(seconds * S)

    monkeypatch.setattr(runner.time, "sleep", sleep)
    return now


def test_observation_runs_with_telemetry_after_aiperf(agentx_config, monkeypatch, tmp_path):
    sessions = _fake_cluster(monkeypatch, tmp_path)
    config = _observed(agentx_config, post_load_observation_s=90)
    config.runtime.results_dir = tmp_path / "runs"
    sleeps = []
    now = _clock(monkeypatch, sleeps)
    alive_during_sleep = []

    def fake_ramp(self):
        _phase_manifest(self.benchmark_root.parent)
        now[0] = T0 + 240 * S  # AIPerf exits after its drain and export
        return _agentx_row(self)

    real_sleep = runner.time.sleep

    def sleep(seconds):
        alive_during_sleep.append([c._thread.is_alive() for c in sessions[0].collectors])
        real_sleep(seconds)

    monkeypatch.setattr(runner.time, "sleep", sleep)
    monkeypatch.setattr(runner.AIPerfRunner, "run_agentx_ramp", fake_ramp)
    monkeypatch.setattr(runner, "run_analysis", lambda root: {"warnings": []})
    input_path = tmp_path / "agentx.yaml"
    input_path.write_text("name: agentx-ramp\n")
    root = runner.run_experiment(config, input_path)

    assert sleeps == [90]
    assert alive_during_sleep == [[True, True]]
    lifecycle = json.loads((root / "benchmark" / "lifecycle.json").read_text())
    assert lifecycle == {
        "schema_version": 1,
        "clock": "epoch_ns",
        "aiperf_launched_ns": T0 - 20 * S,
        "profiling_start_ns": T0,
        "ramp_end_ns": T0 + 120 * S,
        "load_end_ns": T0 + 200 * S,
        "profiling_phase_end_ns": T0 + 225 * S,
        "aiperf_exited_ns": T0 + 240 * S,
        "post_load_observation_s": 90,
        "observation_end_ns": T0 + 330 * S,
        "observation_skipped": None,
    }
    routing = json.loads((root / "deployment" / "routing.json").read_text())
    assert routing["verified"]["topology"] == "openai_ingress"
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["routing"]["topology"] == "openai_ingress"


def test_no_observation_by_default(agentx_config, monkeypatch, tmp_path):
    sleeps = []
    _clock(monkeypatch, sleeps)
    monkeypatch.setattr(runner.AIPerfRunner, "run_agentx_ramp", _agentx_row)
    runner.run_benchmarks(_observed(agentx_config), tmp_path)
    assert sleeps == []
    lifecycle = json.loads((tmp_path / "benchmark" / "lifecycle.json").read_text())
    assert lifecycle["observation_end_ns"] is None
    assert lifecycle["profiling_start_ns"] is None  # AIPerf wrote no manifest


def test_fail_fast_skips_observation_but_records_exit(agentx_config, monkeypatch, tmp_path):
    sleeps = []
    _clock(monkeypatch, sleeps)

    def fail(self):
        raise RuntimeError("AIPerf exited 1")

    monkeypatch.setattr(runner.AIPerfRunner, "run_agentx_ramp", fail)
    config = _observed(agentx_config, post_load_observation_s=90)
    with pytest.raises(RuntimeError, match="exited 1"):
        runner.run_benchmarks(config, tmp_path)
    assert sleeps == []
    lifecycle = json.loads((tmp_path / "benchmark" / "lifecycle.json").read_text())
    assert lifecycle["aiperf_exited_ns"] is not None
    assert lifecycle["observation_end_ns"] is None


def test_failure_without_fail_fast_still_observes(agentx_config, monkeypatch, tmp_path):
    sleeps = []
    _clock(monkeypatch, sleeps)

    def fail(self):
        raise RuntimeError("AIPerf exited 1")

    monkeypatch.setattr(runner.AIPerfRunner, "run_agentx_ramp", fail)
    config = _observed(agentx_config, post_load_observation_s=45, fail_fast=False)
    summary = runner.run_benchmarks(config, tmp_path, telemetry=_NullTelemetry())
    assert summary[0]["error"] == "RuntimeError: AIPerf exited 1"
    assert sleeps == [45]


class _NullTelemetry:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None


def test_observation_skipped_without_telemetry(agentx_config, monkeypatch, tmp_path):
    sleeps = []
    _clock(monkeypatch, sleeps)
    monkeypatch.setattr(runner.AIPerfRunner, "run_agentx_ramp", _agentx_row)
    runner.run_benchmarks(_observed(agentx_config, post_load_observation_s=90), tmp_path)
    assert sleeps == []
    lifecycle = json.loads((tmp_path / "benchmark" / "lifecycle.json").read_text())
    assert lifecycle["observation_skipped"] == "no telemetry"


def test_series_mode_observes_too(monkeypatch, tmp_path):
    sleeps = []
    _clock(monkeypatch, sleeps)
    config = load_config("experiments/smoke_rate_series.yaml")
    config.benchmark.post_load_observation_s = 20
    monkeypatch.setattr(runner.AIPerfRunner, "run_series", lambda self: {
        "mode": "request_rate_series", "level": "series", "error": None})
    runner.run_benchmarks(config, tmp_path, telemetry=_NullTelemetry())
    assert sleeps == [20]
    lifecycle = json.loads((tmp_path / "benchmark" / "lifecycle.json").read_text())
    assert lifecycle["ramp_end_ns"] is None


def test_static_sweep_writes_no_lifecycle(monkeypatch, tmp_path):
    monkeypatch.setattr(runner.AIPerfRunner, "run_point",
                        lambda self, level: {"level": level, "error": None})
    runner.run_benchmarks(load_config("experiments/baseline.yaml"), tmp_path)
    assert not (tmp_path / "benchmark" / "lifecycle.json").exists()


# --- Analysis ---------------------------------------------------------------


def _lifecycle(root: Path, **fields):
    record = {
        "schema_version": 1, "clock": "epoch_ns", "aiperf_launched_ns": T0 - 20 * S,
        "profiling_start_ns": T0, "ramp_end_ns": T0 + 120 * S, "load_end_ns": T0 + 200 * S,
        "profiling_phase_end_ns": T0 + 225 * S, "aiperf_exited_ns": T0 + 240 * S,
        "post_load_observation_s": 90, "observation_end_ns": T0 + 330 * S,
        "observation_skipped": None, **fields,
    }
    write_json(root / "benchmark" / "lifecycle.json", record)


@pytest.fixture
def observed_run(agentx_config, tmp_path: Path):
    config = _observed(agentx_config, post_load_observation_s=90)
    root = make_agentx_run(tmp_path / "run", config)
    # AIPerf closed the profiling phase after draining for 25s.
    _phase_manifest(root, end_s=225)
    _lifecycle(root)
    return root


def test_plot_data_separates_drain_from_idle(observed_run: Path):
    analyze_run(observed_run, plot=False)
    data = json.loads((observed_run / "analysis" / "plot_data.json").read_text())
    lifecycle = data["lifecycle"]
    assert lifecycle["source"] == "lifecycle.json"
    assert lifecycle["markers"] == {
        "ramp_end_s": 120, "load_end_s": 200, "drain_end_s": 225.0,
        "aiperf_exit_s": 240.0, "observation_end_s": 330.0,
    }
    assert lifecycle["periods"] == [
        {"name": "load", "start_s": 0.0, "end_s": 200},
        {"name": "drain", "start_s": 200, "end_s": 225.0},
        {"name": "idle", "start_s": 225.0, "end_s": 330.0},
    ]
    assert data["experiment"]["post_load_observation_s"] == 90
    assert data["experiment"]["routing_policy"] == "power_of_two"


def test_request_windows_cover_idle_observation(observed_run: Path):
    analyze_run(observed_run, plot=False)
    timeseries = json.loads(
        (observed_run / "analysis" / "request_timeseries.json").read_text())
    windows = timeseries["windows"]
    assert windows[-1]["window_end_s"] == 330
    idle = [w for w in windows if w["window_start_s"] >= 240]
    assert idle and all(w["offered_requests"] == 0 and w["in_flight_at_end"] == 0
                        for w in idle)


def test_concurrency_is_zero_after_load_ends(observed_run: Path):
    analyze_run(observed_run, plot=False)
    result = json.loads((observed_run / "analysis" / "concurrency_qps.json").read_text())
    after = [w for w in result["windows"] if w["window_start_s"] >= 200]
    assert after and all(w["configured_session_concurrency"] == 0 for w in after)
    assert all(w["requests_per_second_per_configured_session"] is None for w in after)


def test_lifecycle_falls_back_to_config_without_record(agentx_config):
    config = _observed(agentx_config)
    periods = lifecycle_periods(config, T0, T0 + 215 * S, None)
    assert periods["source"] == "config"
    assert periods["periods"] == [
        {"name": "load", "start_s": 0.0, "end_s": 200},
        {"name": "drain", "start_s": 200, "end_s": 215.0},
    ]


def test_observed_plot_renders_and_summary_lists_periods(observed_run: Path):
    result = analyze_run(observed_run)
    assert result["plot_error"] is None
    assert (observed_run / "analysis" / "autoscaling_timeline.png").stat().st_size > 10_000
    write_json(observed_run / "manifest.json",
               {"name": "agentx-ramp", "status": "succeeded", "analysis": result})
    summary = format_run_summary(observed_run)
    assert "Lifecycle: load 0-200s · drain 200-225s · idle 225-330s" in summary
