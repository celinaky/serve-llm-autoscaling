from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

WINDOW_S = 10


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * pct / 100))]


def windowed_summary(artifact_dir: Path, duration_s: float) -> list[dict[str, Any]]:
    """Summarize a series run's profiling records in windows of send time."""
    with (artifact_dir / "profile_export.jsonl").open() as fh:
        records = [json.loads(line) for line in fh if line.strip()]
    records = [r for r in records if r["metadata"]["benchmark_phase"] == "profiling"]
    # AIPerf only writes the phase manifest when the run has a warmup phase.
    manifest = artifact_dir / "phase_manifest.json"
    if manifest.exists():
        t0 = next(
            p["start_ns"] for p in json.loads(manifest.read_text())["phases"]
            if p["phase_kind"] == "profiling"
        )
    else:
        t0 = min(r["metadata"]["credit_issued_ns"] for r in records)
    rows = [
        {
            "sent": (r["metadata"]["credit_issued_ns"] - t0) / 1e9,
            "end": (r["metadata"]["request_end_ns"] - t0) / 1e9,
            "failed": r.get("error") is not None,
            "ttft": r["metrics"].get("time_to_first_token", {}).get("value"),
        }
        for r in records
    ]
    # Run past duration_s until the last response so the queue drain shows.
    horizon = max([duration_s] + [r["end"] for r in rows])
    windows = []
    for index in range(math.ceil(horizon / WINDOW_S)):
        start, end = index * WINDOW_S, (index + 1) * WINDOW_S
        sent = [r for r in rows if start <= r["sent"] < end]
        ttft = [r["ttft"] for r in sent if not r["failed"] and r["ttft"] is not None]
        windows.append(
            {
                "window_start_s": start,
                "sent_rps": len(sent) / WINDOW_S,
                "completed_rps": sum(start <= r["end"] < end for r in rows) / WINDOW_S,
                "in_flight_at_end": sum(r["sent"] < end <= r["end"] for r in rows),
                "failed_requests": sum(r["failed"] for r in sent),
                "p50_ttft_ms": _percentile(ttft, 50),
                "p99_ttft_ms": _percentile(ttft, 99),
            }
        )
    return windows
