import json
from pathlib import Path

import numpy as np
import pytest

from serve_llm_autoscaling.artifacts import format_run_summary, write_json
from serve_llm_autoscaling.windows import percentile, request_timeseries

S = 1_000_000_000
T0 = 100 * S


def _record(credit, end, start=None, ttft=10.0, phase="profiling", error=None):
    """A record with times in seconds since the profiling start."""
    start = credit if start is None else start
    record = {
        "metadata": {
            "credit_issued_ns": T0 + int(credit * S),
            "request_start_ns": T0 + int(start * S),
            "request_end_ns": T0 + int(end * S),
            "benchmark_phase": phase,
        },
        "metrics": {"time_to_first_token": {"value": ttft}} if ttft is not None else {},
    }
    if error:
        record["error"] = {"message": error}
    return record


def _write(directory: Path, records, manifest=True, extra_lines=()):
    directory.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(r) for r in records] + list(extra_lines)
    (directory / "profile_export.jsonl").write_text("".join(l + "\n" for l in lines))
    if manifest:
        write_json(directory / "phase_manifest.json",
                   {"phases": [{"phase_kind": "warmup", "start_ns": 0},
                               {"phase_kind": "profiling", "start_ns": T0}]})
    return directory


def test_default_window_is_five_seconds(tmp_path: Path):
    result = request_timeseries(_write(tmp_path, [_record(1, 2), _record(7, 8)]))
    assert result["schema_version"] == 1
    assert result["window_s"] == 5
    assert result["profiling_start_ns"] == T0
    assert result["time_origin_source"] == "phase_manifest"
    assert [(w["window_start_s"], w["window_end_s"]) for w in result["windows"]] == [
        (0, 5), (5, 10)
    ]


def test_configurable_window(tmp_path: Path):
    result = request_timeseries(_write(tmp_path, [_record(1, 2), _record(17, 18)]),
                                window_s=10)
    assert [w["window_start_s"] for w in result["windows"]] == [0, 10]
    assert result["windows"][1]["offered_rps"] == pytest.approx(0.1)


@pytest.mark.parametrize("q", [0.5, 0.9, 0.99, 0.0, 1.0])
def test_percentile_matches_numpy_linear(q):
    values = sorted([12.0, 3.0, 7.5, 100.0, 41.0, 8.0, 9.0])
    assert percentile(values, q) == pytest.approx(np.percentile(values, q * 100))


def test_percentile_interpolates():
    assert percentile([1, 2, 3, 4], 0.5) == 2.5
    assert percentile([10, 20], 0.99) == pytest.approx(19.9)
    assert percentile([], 0.5) is None


def test_offered_versus_started_and_client_queue(tmp_path: Path):
    # Credit issued at 4.9s, but the request only starts at 5.2s.
    result = request_timeseries(_write(tmp_path, [_record(4.9, 6, start=5.2)]))
    first, second = result["windows"]
    assert (first["offered_requests"], first["started_requests"]) == (1, 0)
    assert (second["offered_requests"], second["started_requests"]) == (0, 1)
    assert second["started_rps"] == 0.2
    assert second["p50_client_queue_ms"] == pytest.approx(300)
    assert first["p50_client_queue_ms"] is None


def test_successful_versus_failed_completions(tmp_path: Path):
    result = request_timeseries(_write(tmp_path, [
        _record(1, 2), _record(1, 3, error="boom", ttft=None), _record(2, 4),
    ]))
    window = result["windows"][0]
    assert window["successful_completions"] == 2
    assert window["failed_completions"] == 1
    assert window["failed_completed_rps"] == 0.2
    assert window["ttft_sample_count"] == 2  # failed requests have no TTFT


def test_ttft_grouped_by_request_start(tmp_path: Path):
    result = request_timeseries(_write(tmp_path, [_record(1, 12, ttft=900)]))
    first, second, third = result["windows"]
    assert first["p50_ttft_ms"] == 900
    assert first["successful_completions"] == 0
    assert third["successful_completions"] == 1 and third["ttft_sample_count"] == 0


def test_in_flight_at_boundary(tmp_path: Path):
    result = request_timeseries(_write(tmp_path, [
        _record(1, 6), _record(2, 5), _record(4, 4.5), _record(6, 7),
    ]))
    # Started before 5s and not ended by it: only 1->6; 2->5 completed at 5s.
    first, second = result["windows"]
    assert first["in_flight_at_end"] == 1
    assert second["in_flight_at_end"] == 0
    # Completions fall in (start, end]: 2->5 completes in the first window.
    assert (first["successful_completions"], second["successful_completions"]) == (2, 2)


def test_rejects_out_of_order_and_nonfinite_records(tmp_path: Path):
    result = request_timeseries(_write(tmp_path, [
        _record(1, 2), _record(3, 2), _record(2, 4, start=1),
        _record(1, 2, ttft=float("inf")), _record(1, 2, ttft=float("nan")),
    ]))
    assert result["malformed_record_count"] == 4
    assert result["windows"][0]["ttft_sample_count"] == 1


def test_empty_windows(tmp_path: Path):
    result = request_timeseries(_write(tmp_path, [_record(1, 2), _record(11, 12)]))
    empty = result["windows"][1]
    assert empty["offered_requests"] == empty["started_requests"] == 0
    assert empty["ttft_sample_count"] == 0
    assert empty["p50_ttft_ms"] is None and empty["p99_client_queue_ms"] is None


def test_drain_windows_continue_until_last_request(tmp_path: Path):
    result = request_timeseries(_write(tmp_path, [_record(1, 23)]), duration_s=10)
    assert result["windows"][-1]["window_end_s"] == 25
    assert result["windows"][-1]["successful_completions"] == 1
    assert result["windows"][2]["in_flight_at_end"] == 1


def test_windows_cover_duration_without_late_requests(tmp_path: Path):
    result = request_timeseries(_write(tmp_path, [_record(1, 2)]), duration_s=20)
    assert len(result["windows"]) == 4


def test_slo_attainment(tmp_path: Path):
    records = [_record(1, 2, ttft=v) for v in (50, 100, 150, 200)]
    result = request_timeseries(_write(tmp_path, records), ttft_slo_ms=100)
    assert result["windows"][0]["ttft_slo_attainment"] == 0.5
    assert request_timeseries(tmp_path)["windows"][0]["ttft_slo_attainment"] is None


def test_missing_manifest_falls_back_to_first_credit(tmp_path: Path):
    _write(tmp_path, [_record(3, 4, phase="warmup"), _record(7, 8), _record(9, 10)],
           manifest=False)
    result = request_timeseries(tmp_path)
    assert result["time_origin_source"] == "first_credit"
    assert result["profiling_start_ns"] == T0 + 7 * S
    assert result["windows"][0]["offered_requests"] == 2


def test_malformed_records_are_counted(tmp_path: Path):
    missing_start = _record(1, 2)
    del missing_start["metadata"]["request_start_ns"]
    null_end = _record(1, 2)
    null_end["metadata"]["request_end_ns"] = None
    _write(tmp_path, [_record(1, 2), missing_start, null_end, _record(0, 1, phase="warmup")],
           extra_lines=["{not json", json.dumps({"metrics": {}})])
    result = request_timeseries(tmp_path)
    assert result["malformed_record_count"] == 4
    assert result["windows"][0]["offered_requests"] == 1


def test_rolling_tail_latency(tmp_path: Path):
    records = [_record(1, 2, ttft=100), _record(6, 7, ttft=1000), _record(11, 12, ttft=10)]
    result = request_timeseries(_write(tmp_path, records), tail_window_s=10)
    rolling = [(w["rolling_ttft_sample_count"], w["rolling_p99_ttft_ms"])
               for w in result["windows"]]
    assert rolling[0] == (1, 100)
    assert rolling[1][0] == 2 and rolling[1][1] == pytest.approx(percentile([100, 1000], 0.99))
    assert rolling[2][0] == 2  # the 1s request has left the 10s tail window


def test_run_summary_prints_windows(tmp_path: Path):
    series_dir = _write(tmp_path / "benchmark" / "request-rate-series", [_record(1, 2)])
    write_json(tmp_path / "analysis" / "request_timeseries.json",
               request_timeseries(series_dir))
    write_json(tmp_path / "manifest.json", {"name": "demo", "status": "succeeded"})
    write_json(tmp_path / "benchmark" / "sweep_summary.json", [])
    assert "TTFT p50" in format_run_summary(tmp_path)
