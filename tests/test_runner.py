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
