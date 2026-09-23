import json
from pathlib import Path

from serve_llm_autoscaling.artifacts import format_run_summary, write_json


def _make_run(root: Path, status: str, points: list, **manifest) -> Path:
    write_json(
        root / "manifest.json",
        {
            "name": "demo",
            "status": status,
            "started_at": "2026-09-23T16:37:36+00:00",
            "finished_at": "2026-09-23T16:39:51+00:00",
            **manifest,
        },
    )
    write_json(root / "benchmark" / "sweep_summary.json", points)
    (root / "deployment").mkdir()
    (root / "deployment" / "events.jsonl").write_text(
        json.dumps({"event": "teardown_completed"}) + "\n"
    )
    return root


def _point(level, **fields):
    return {"mode": "concurrency", "level": level, "error": None,
            "request_count": 280.0, "failed_requests": 0, "p50_ttft_ms": 20.5,
            **fields}


def test_summary_clean_success(tmp_path: Path):
    root = _make_run(tmp_path, "succeeded", [_point(1)], readiness_s=85.25)
    text = format_run_summary(root)
    assert "SUCCEEDED (no errors)" in text
    assert "2m 15s (deployment ready after 85.2s)" in text
    assert "Teardown:  completed" in text
    assert "280" in text and "20.5ms" in text


def test_summary_flags_partial_failures(tmp_path: Path):
    points = [_point(1, failed_requests=3), {"mode": "concurrency", "level": 2,
                                             "error": "RuntimeError: boom"}]
    text = format_run_summary(_make_run(tmp_path, "succeeded", points))
    assert "SUCCEEDED WITH ERRORS (1 of 2 points failed, 3 failed requests)" in text
    assert "ERROR: RuntimeError: boom" in text


def test_summary_failed_run(tmp_path: Path):
    root = _make_run(tmp_path, "failed", [], failure_stage="deploy",
                     error="RuntimeError: Deploying application failed")
    text = format_run_summary(root)
    assert "FAILED at stage 'deploy'" in text
    assert "Error:     RuntimeError: Deploying application failed" in text
