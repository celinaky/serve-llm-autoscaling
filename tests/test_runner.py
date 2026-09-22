from pathlib import Path

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

