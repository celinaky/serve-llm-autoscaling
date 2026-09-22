import json
from pathlib import Path

from serve_llm_autoscaling.benchmark import AIPerfRunner, normalize_aiperf
from serve_llm_autoscaling.config import load_config


def test_build_concurrency_command(monkeypatch, tmp_path: Path):
    config = load_config("experiments/baseline.yaml")
    runner = AIPerfRunner(config, tmp_path)
    monkeypatch.setattr(runner, "executable", lambda: "/bin/aiperf")
    command = runner.build_command(4.0, tmp_path / "point")
    assert command[:2] == ["/bin/aiperf", "profile"]
    assert command[command.index("--concurrency") + 1] == "4"
    assert command[command.index("--isl") + 1] == "8000"
    assert command[command.index("--osl") + 1] == "50"
    assert "--request-rate" not in command


def test_build_request_rate_command(monkeypatch, tmp_path: Path):
    config = load_config("experiments/baseline.yaml")
    config.benchmark.mode = "request_rate"
    runner = AIPerfRunner(config, tmp_path)
    monkeypatch.setattr(runner, "executable", lambda: "/bin/aiperf")
    command = runner.build_command(2.5, tmp_path / "point")
    assert command[command.index("--request-rate") + 1] == "2.5"
    assert command[command.index("--arrival-pattern") + 1] == "poisson"
    assert "--concurrency" not in command


def test_normalize_aiperf():
    raw = {
        "request_count": {"avg": 10},
        "request_throughput": {"avg": 2.5},
        "time_to_first_token": {"avg": 11, "p50": 9, "p90": 15, "p99": 20},
        "inter_token_latency": {"avg": 3},
        "request_latency": {"avg": 100, "p99": 150},
        "error_summary": [{"count": 2}],
    }
    result = normalize_aiperf(raw)
    assert result["request_count"] == 10
    assert result["request_throughput"] == 2.5
    assert result["p99_ttft_ms"] == 20
    assert result["failed_requests"] == 2

