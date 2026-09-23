import json
from pathlib import Path

from serve_llm_autoscaling.artifacts import format_run_summary, write_json
from serve_llm_autoscaling.windows import windowed_summary

S = 1_000_000_000


def _record(sent, end, ttft=10.0, phase="profiling", error=None):
    record = {
        "metadata": {"credit_issued_ns": int(sent * S), "request_end_ns": int(end * S),
                     "benchmark_phase": phase},
        "metrics": {"time_to_first_token": {"value": ttft}},
    }
    if error:
        record["error"] = {"message": error}
    return record


def _write_records(directory: Path, records):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "profile_export.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records)
    )
    write_json(directory / "phase_manifest.json",
               {"phases": [{"phase_kind": "warmup", "start_ns": 0},
                           {"phase_kind": "profiling", "start_ns": 100 * S}]})


def test_windowed_summary(tmp_path: Path):
    _write_records(tmp_path, [
        _record(95, 96, phase="warmup"),
        _record(101, 102, ttft=10),
        _record(103, 104, ttft=30),
        _record(109, 112, ttft=500),
        _record(111, 112, error="boom"),
        _record(119, 124, ttft=40),
    ])
    first, second, drain = windowed_summary(tmp_path, duration_s=20)
    assert first == {"window_start_s": 0, "sent_rps": 0.3, "completed_rps": 0.2,
                     "in_flight_at_end": 1, "failed_requests": 0,
                     "p50_ttft_ms": 30, "p99_ttft_ms": 500}
    assert second["failed_requests"] == 1
    assert second["p50_ttft_ms"] == 40  # failed requests have no latency
    assert drain["sent_rps"] == 0
    assert drain["completed_rps"] == 0.1


def test_run_summary_prints_windows(tmp_path: Path):
    series_dir = tmp_path / "benchmark" / "request-rate-series"
    _write_records(series_dir, [_record(101, 102)])
    write_json(series_dir / "windows.json", windowed_summary(series_dir, 10))
    write_json(tmp_path / "manifest.json", {"name": "demo", "status": "succeeded"})
    write_json(tmp_path / "benchmark" / "sweep_summary.json", [])
    assert "TTFT p50" in format_run_summary(tmp_path)
