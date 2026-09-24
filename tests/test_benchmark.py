import json
from pathlib import Path

import pytest

from serve_llm_autoscaling import benchmark
from serve_llm_autoscaling.benchmark import AIPerfRunner, normalize_aiperf
from serve_llm_autoscaling.config import load_config


def test_build_concurrency_command(monkeypatch, tmp_path: Path):
    config = load_config("experiments/baseline.yaml")
    runner = AIPerfRunner(config, tmp_path)
    monkeypatch.setattr(benchmark.shutil, "which", lambda name: f"/bin/{name}")
    command = runner.build_command(4.0, tmp_path / "point")
    assert command[:7] == [
        "uvx", "--python", "3.11", "--from", "aiperf==0.12.0", "aiperf", "profile"
    ]
    assert command[command.index("--concurrency") + 1] == "4"
    assert command[command.index("--isl") + 1] == "8000"
    assert command[command.index("--osl") + 1] == "50"
    assert "--request-rate" not in command


def test_build_request_rate_command(monkeypatch, tmp_path: Path):
    config = load_config("experiments/baseline.yaml")
    config.benchmark.mode = "request_rate"
    runner = AIPerfRunner(config, tmp_path)
    monkeypatch.setattr(benchmark.shutil, "which", lambda name: f"/bin/{name}")
    command = runner.build_command(2.5, tmp_path / "point")
    assert command[command.index("--request-rate") + 1] == "2.5"
    assert command[command.index("--arrival-pattern") + 1] == "poisson"
    assert "--concurrency" not in command


def _series_config():
    config = load_config("experiments/smoke_rate_series.yaml")
    config.benchmark.extra_args = []
    return config


def test_write_rate_series(tmp_path: Path):
    runner = AIPerfRunner(_series_config(), tmp_path)
    path = tmp_path / "rate_series.json"
    runner.write_rate_series(path)
    assert json.loads(path.read_text()) == {
        "points": [
            {"time_s": 0.0, "qps": 1.0},
            {"time_s": 10.0, "qps": 1.0},
            {"time_s": 10.1, "qps": 2.0},
            {"time_s": 20.0, "qps": 2.0},
            {"time_s": 20.1, "qps": 1.0},
            {"time_s": 30.0, "qps": 1.0},
        ]
    }


def test_build_series_command(monkeypatch, tmp_path: Path):
    runner = AIPerfRunner(_series_config(), tmp_path)
    monkeypatch.setattr(benchmark.shutil, "which", lambda name: f"/bin/{name}")
    series_path = tmp_path / "rate_series.json"
    command = runner.build_series_command(series_path, tmp_path)
    assert command[command.index("--request-rate-series") + 1] == str(
        series_path.resolve()
    )
    assert command[command.index("--arrival-pattern") + 1] == "poisson"
    assert command[command.index("--benchmark-duration") + 1] == "30.0"
    assert "--arrival-smoothness" not in command
    assert "--request-rate" not in command
    assert "--concurrency" not in command


def test_build_series_command_gamma(monkeypatch, tmp_path: Path):
    config = _series_config()
    config.benchmark.arrival_pattern = "gamma"
    config.benchmark.arrival_smoothness = 0.5
    runner = AIPerfRunner(config, tmp_path)
    monkeypatch.setattr(benchmark.shutil, "which", lambda name: f"/bin/{name}")
    command = runner.build_series_command(tmp_path / "rate_series.json", tmp_path)
    assert command[command.index("--arrival-pattern") + 1] == "gamma"
    assert command[command.index("--arrival-smoothness") + 1] == "0.5"


def test_run_series(monkeypatch, tmp_path: Path):
    runner = AIPerfRunner(_series_config(), tmp_path)
    monkeypatch.setattr(benchmark.shutil, "which", lambda name: f"/bin/{name}")
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        artifact_dir = Path(cmd[cmd.index("--artifact-dir") + 1])
        series = json.loads(
            Path(cmd[cmd.index("--request-rate-series") + 1]).read_text()
        )
        assert len(series["points"]) == 6
        (artifact_dir / "profile_export_aiperf.json").write_text(
            json.dumps({"request_count": {"avg": 40}})
        )
        record = {
            "metadata": {"credit_issued_ns": 1, "request_end_ns": 2,
                         "benchmark_phase": "profiling"},
            "metrics": {},
        }
        (artifact_dir / "profile_export.jsonl").write_text(json.dumps(record) + "\n")
        return benchmark.subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(benchmark.subprocess, "run", fake_run)
    row = runner.run_series()
    series_dir = tmp_path / "request-rate-series"
    assert len(calls) == 1
    assert row["mode"] == "request_rate_series"
    assert row["level"] == "series"
    assert row["request_count"] == 40
    assert row["error"] is None
    assert row["artifact_dir"] == str(series_dir)
    for name in ["rate_series.json", "command.json", "stdout.log", "stderr.log",
                 "profile_export.jsonl"]:
        assert (series_dir / name).exists()
    # Windowing is analysis, not part of running AIPerf.
    assert not (series_dir / "windows.json").exists()
    assert "--request-rate-series" in json.loads(
        (series_dir / "command.json").read_text()
    )["argv"]


def test_run_series_failure_points_to_stderr(monkeypatch, tmp_path: Path):
    runner = AIPerfRunner(_series_config(), tmp_path)
    monkeypatch.setattr(benchmark.shutil, "which", lambda name: f"/bin/{name}")
    monkeypatch.setattr(
        benchmark.subprocess,
        "run",
        lambda cmd, **kwargs: benchmark.subprocess.CompletedProcess(cmd, 2),
    )
    with pytest.raises(RuntimeError, match="request-rate-series/stderr.log"):
        runner.run_series()


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



# --- AgentX concurrency ramp ------------------------------------------------

SYNTHETIC_OR_RATE_FLAGS = [
    "--isl", "--isl-stddev", "--osl", "--osl-stddev", "--request-rate",
    "--request-rate-series", "--arrival-pattern", "--arrival-smoothness",
    "--warmup-duration", "--extra-inputs",
]


def _arg(command, flag):
    return command[command.index(flag) + 1]


def test_build_agentx_ramp_command(agentx_config, monkeypatch, tmp_path: Path):
    runner = AIPerfRunner(agentx_config(), tmp_path)
    monkeypatch.setattr(benchmark.shutil, "which", lambda name: f"/bin/{name}")
    command = runner.build_agentx_ramp_command(tmp_path / "ramp")
    assert command[:7] == [
        "uvx", "--python", "3.11", "--from", "aiperf==0.12.0", "aiperf", "profile"
    ]
    assert _arg(command, "--scenario") == "inferencex-agentx-mvp"
    assert _arg(command, "--public-dataset") == "semianalysis_cc_traces_weka_062126_256k"
    assert _arg(command, "--concurrency") == "16"
    assert _arg(command, "--concurrency-ramp-duration") == "1200.0"
    assert _arg(command, "--trajectory-start-min-ratio") == "0.0"
    assert _arg(command, "--trajectory-start-max-ratio") == "0.0"
    assert _arg(command, "--benchmark-duration") == "1800.0"
    assert _arg(command, "--benchmark-grace-period") == "300.0"
    assert _arg(command, "--random-seed") == "42"
    assert _arg(command, "--endpoint-type") == "chat"
    assert _arg(command, "--export-level") == "records"
    assert _arg(command, "--ui") == "none"
    assert _arg(command, "--artifact-dir") == str(tmp_path / "ramp")
    assert "--streaming" in command and "--use-server-token-count" in command
    assert "--unsafe-override" not in command


def test_agentx_command_has_no_synthetic_or_rate_flags(
    agentx_config, monkeypatch, tmp_path: Path
):
    runner = AIPerfRunner(agentx_config(), tmp_path)
    monkeypatch.setattr(benchmark.shutil, "which", lambda name: f"/bin/{name}")
    command = runner.build_agentx_ramp_command(tmp_path)
    assert not set(command) & set(SYNTHETIC_OR_RATE_FLAGS)
    assert not any("ignore_eos" in arg for arg in command)


def test_agentx_command_unsafe_override(agentx_config, monkeypatch, tmp_path: Path):
    config = agentx_config(
        {"concurrency_ramp_duration_s": 60, "unsafe_override": True}, duration_s=120
    )
    monkeypatch.setattr(benchmark.shutil, "which", lambda name: f"/bin/{name}")
    command = AIPerfRunner(config, tmp_path).build_agentx_ramp_command(tmp_path)
    assert command.count("--unsafe-override") == 1


def test_agentx_command_appends_extra_args(agentx_config, monkeypatch, tmp_path: Path):
    config = agentx_config(extra_args=["--request-timeout-seconds", "600"])
    monkeypatch.setattr(benchmark.shutil, "which", lambda name: f"/bin/{name}")
    command = AIPerfRunner(config, tmp_path).build_agentx_ramp_command(tmp_path)
    assert command[-2:] == ["--request-timeout-seconds", "600"]


def test_synthetic_commands_reject_agentx_workload(agentx_config, monkeypatch, tmp_path):
    monkeypatch.setattr(benchmark.shutil, "which", lambda name: f"/bin/{name}")
    with pytest.raises(TypeError, match="synthetic workload"):
        AIPerfRunner(agentx_config(), tmp_path).build_command(4, tmp_path)


def _agentx_record(credit_s, end_s, t0, error=None):
    record = {
        "metadata": {"credit_issued_ns": t0 + int(credit_s * 1e9),
                     "request_start_ns": t0 + int(credit_s * 1e9),
                     "request_end_ns": t0 + int(end_s * 1e9),
                     "benchmark_phase": "profiling", "agent_depth": 0},
        "metrics": {},
    }
    if error:
        record["error"] = {"message": error}
    return record


def test_run_agentx_ramp(agentx_config, monkeypatch, tmp_path: Path):
    config = agentx_config(
        {"concurrency_ramp_duration_s": 5, "unsafe_override": True}, duration_s=10
    )
    runner = AIPerfRunner(config, tmp_path)
    monkeypatch.setattr(benchmark.shutil, "which", lambda name: f"/bin/{name}")
    calls = []
    t0 = 1_000 * 10**9

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        artifact_dir = Path(cmd[cmd.index("--artifact-dir") + 1])
        (artifact_dir / "profile_export_aiperf.json").write_text(json.dumps({
            "request_count": {"avg": 4}, "request_throughput": {"avg": 0.35},
        }))
        (artifact_dir / "phase_manifest.json").write_text(json.dumps({"phases": [
            {"phase_kind": "profiling", "start_ns": t0, "end_ns": t0 + 12 * 10**9},
        ]}))
        records = [_agentx_record(1, 2, t0), _agentx_record(3, 4, t0),
                   _agentx_record(5, 6, t0, error="boom"),
                   # Completes in the grace period: outside the profiling interval.
                   _agentx_record(9, 11, t0)]
        (artifact_dir / "profile_export.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in records)
        )
        return benchmark.subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(benchmark.subprocess, "run", fake_run)
    row = runner.run_agentx_ramp()
    ramp_dir = tmp_path / "agentx-concurrency-ramp"
    assert len(calls) == 1
    assert row["artifact_dir"] == str(ramp_dir)
    assert {k: row[k] for k in ("mode", "level", "target_concurrency", "ramp_duration_s",
                                "error")} == {
        "mode": "agentx_concurrency_ramp", "level": 16, "target_concurrency": 16,
        "ramp_duration_s": 5, "error": None,
    }
    assert row["request_count"] == 4
    # AIPerf's number and the harness-derived rates are distinct fields.
    assert row["aiperf_request_throughput"] == 0.35
    assert row["profiling_interval_s"] == 10
    assert row["mean_offered_qps"] == pytest.approx(0.4)
    assert row["mean_started_qps"] == pytest.approx(0.4)
    assert row["mean_successful_qps"] == pytest.approx(0.2)
    assert row["mean_failed_qps"] == pytest.approx(0.1)
    for name in ["command.json", "stdout.log", "stderr.log", "profile_export.jsonl",
                 "profile_export_aiperf.json", "phase_manifest.json"]:
        assert (ramp_dir / name).exists()
    argv = json.loads((ramp_dir / "command.json").read_text())["argv"]
    assert "--scenario" in argv and "--unsafe-override" in argv


def test_run_agentx_ramp_failure_points_to_stderr(agentx_config, monkeypatch, tmp_path):
    runner = AIPerfRunner(agentx_config(), tmp_path)
    monkeypatch.setattr(benchmark.shutil, "which", lambda name: f"/bin/{name}")
    monkeypatch.setattr(
        benchmark.subprocess, "run",
        lambda cmd, **kwargs: benchmark.subprocess.CompletedProcess(cmd, 2),
    )
    with pytest.raises(RuntimeError, match="agentx-concurrency-ramp/stderr.log"):
        runner.run_agentx_ramp()
