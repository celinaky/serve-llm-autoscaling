from __future__ import annotations

import json
import math
from bisect import bisect_left, bisect_right
from pathlib import Path
from typing import Any, NamedTuple

SCHEMA_VERSION = 1


class _Request(NamedTuple):
    credit_ns: int
    start_ns: int
    end_ns: int
    failed: bool
    ttft_ms: float | None
    # 0 for root agents, > 0 for subagents; None if the record does not say.
    agent_depth: int | None


def percentile(ordered: list[float], q: float) -> float | None:
    """Linear interpolation between closest ranks (NumPy's default method).

    ``ordered`` must already be sorted; ``q`` is a fraction in [0, 1].
    """
    if not ordered:
        return None
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def profiling_phase_ns(artifact_dir: Path) -> tuple[int | None, int | None]:
    """Profiling (start, end) from AIPerf's phase manifest, where usable.

    The end is when AIPerf closed the phase, after draining in-flight requests.
    """
    try:
        manifest = json.loads((artifact_dir / "phase_manifest.json").read_text())
        phase = next(p for p in manifest["phases"] if p.get("phase_kind") == "profiling")
        start = int(phase["start_ns"])
    except (OSError, ValueError, KeyError, TypeError, StopIteration):
        return None, None
    try:
        end = int(phase["end_ns"])
    except (KeyError, ValueError, TypeError):
        end = None
    return start, end


def profiling_start_ns(artifact_dir: Path) -> int | None:
    """Profiling start from AIPerf's phase manifest, if it is usable."""
    return profiling_phase_ns(artifact_dir)[0]


def _parse(line: str) -> _Request | None:
    """Return one profiling request, or None if it belongs to another phase.

    Raises ValueError for malformed records.
    """
    try:
        record = json.loads(line)
        meta = record["metadata"]
        phase = meta["benchmark_phase"]
        credit, start, end = (
            meta["credit_issued_ns"], meta["request_start_ns"], meta["request_end_ns"]
        )
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError("malformed record") from exc
    if phase != "profiling":
        return None
    if not all(isinstance(v, int) and not isinstance(v, bool) for v in (credit, start, end)):
        raise ValueError("timestamps must be integers")
    if not credit <= start <= end:
        raise ValueError("timestamps out of order")
    depth = meta.get("agent_depth")
    if isinstance(depth, bool) or not isinstance(depth, int):
        depth = None
    ttft = (record.get("metrics") or {}).get("time_to_first_token") or {}
    ttft_ms = ttft.get("value") if isinstance(ttft, dict) else None
    if isinstance(ttft_ms, bool) or not isinstance(ttft_ms, (int, float)):
        ttft_ms = None
    elif not math.isfinite(ttft_ms):
        raise ValueError("TTFT is not finite")
    return _Request(
        credit, start, end, record.get("error") is not None,
        float(ttft_ms) if ttft_ms is not None else None, depth,
    )


def _read_requests(path: Path) -> tuple[list[_Request], int]:
    requests: list[_Request] = []
    malformed = 0
    with path.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                request = _parse(line)
            except ValueError:
                malformed += 1
                continue
            if request is not None:
                requests.append(request)
    return requests, malformed


def configured_concurrency_at(
    time_s: float, target: int, ramp_duration_s: float
) -> float:
    """The configured linear session-concurrency ramp from 1 to ``target``."""
    if time_s >= ramp_duration_s:
        return float(target)
    return 1 + (target - 1) * max(time_s, 0.0) / ramp_duration_s


def _qps_summary(requests: list[_Request], t0: int | None, duration_s: float) -> dict[str, Any]:
    """Mean request rates over the profiling send interval ``[t0, t0 + duration_s)``.

    Only profiling records are counted, and the interval starts at the
    profiling start, so dataset preparation and warmup do not dilute the
    rates. Completions during the grace period fall outside the interval.
    """
    counts = {"offered": 0, "started": 0, "ok": 0, "failed": 0}
    if t0 is not None:
        end = t0 + round(duration_s * 1e9)
        for r in requests:
            counts["offered"] += t0 <= r.credit_ns < end
            counts["started"] += t0 <= r.start_ns < end
            if t0 < r.end_ns <= end:
                counts["failed" if r.failed else "ok"] += 1
    return {
        "profiling_interval_s": duration_s,
        "request_count": len(requests),
        "failed_requests": sum(r.failed for r in requests),
        "mean_offered_qps": counts["offered"] / duration_s,
        "mean_started_qps": counts["started"] / duration_s,
        "mean_successful_qps": counts["ok"] / duration_s,
        "mean_failed_qps": counts["failed"] / duration_s,
    }


def _time_origin(artifact_dir: Path, requests: list[_Request]) -> tuple[int | None, str]:
    t0 = profiling_start_ns(artifact_dir)
    if t0 is not None:
        return t0, "phase_manifest"
    return min((r.credit_ns for r in requests), default=None), "first_credit"


def profiling_qps_summary(artifact_dir: Path, duration_s: float) -> dict[str, Any]:
    """Harness-derived mean QPS of a run, from its per-request records."""
    requests, _ = _read_requests(artifact_dir / "profile_export.jsonl")
    t0, _ = _time_origin(artifact_dir, requests)
    return _qps_summary(requests, t0, duration_s)


def request_timeseries(
    artifact_dir: Path,
    *,
    window_s: float = 5,
    tail_window_s: float | None = None,
    ttft_slo_ms: float | None = None,
    duration_s: float | None = None,
    horizon_s: float | None = None,
) -> dict[str, Any]:
    """Summarize a series run's profiling records in fixed windows.

    Offered load uses credit issue time, starts use request start, completions
    use request end, and TTFT belongs to the window in which the request
    started. Credits and starts fall in ``[start, end)`` windows, completions in
    ``(start, end]``, so a request ending exactly on a boundary completes in the
    earlier window and is no longer in flight at that boundary. Windows continue
    until the last request finishes, or until ``horizon_s`` if later (the end
    of post-load observation, so idle windows read as zero traffic).
    """
    requests, malformed = _read_requests(artifact_dir / "profile_export.jsonl")
    t0, source = _time_origin(artifact_dir, requests)
    # Subagent requests are real requests and count toward the totals; the
    # split is reported only when the records carry agent_depth.
    by_depth = any(r.agent_depth is not None for r in requests)

    def rel(ns: int) -> float:
        return (ns - t0) / 1e9

    count = 0
    if t0 is not None:
        horizon = max([duration_s or 0.0, horizon_s or 0.0] + [rel(r.end_ns) for r in requests])
        count = math.ceil(horizon / window_s)

    def index(ns: int, *, closed_right: bool = False) -> int | None:
        position = rel(ns) / window_s
        i = math.ceil(position) - 1 if closed_right else math.floor(position)
        return i if 0 <= i < count else None

    buckets = [
        {"offered": 0, "offered_root": 0, "started": 0, "ok": 0, "failed": 0,
         "ttft": [], "queue": []}
        for _ in range(count)
    ]
    for r in requests:
        if (i := index(r.credit_ns)) is not None:
            buckets[i]["offered"] += 1
            buckets[i]["offered_root"] += r.agent_depth == 0
        if (i := index(r.start_ns)) is not None:
            buckets[i]["started"] += 1
            buckets[i]["queue"].append((r.start_ns - r.credit_ns) / 1e6)
            if not r.failed and r.ttft_ms is not None:
                buckets[i]["ttft"].append(r.ttft_ms)
        if (i := index(r.end_ns, closed_right=True)) is not None:
            buckets[i]["failed" if r.failed else "ok"] += 1

    starts = sorted(r.start_ns for r in requests)
    ends = sorted(r.end_ns for r in requests)
    # (start time, TTFT) for the rolling tail, ordered by start.
    ttft_by_start = sorted(
        (r.start_ns, r.ttft_ms) for r in requests if not r.failed and r.ttft_ms is not None
    )
    ttft_starts = [ns for ns, _ in ttft_by_start]

    windows = []
    for i, b in enumerate(buckets):
        start_s, end_s = i * window_s, (i + 1) * window_s
        boundary = t0 + round(end_s * 1e9)
        ttft = sorted(b["ttft"])
        queue = sorted(b["queue"])
        window = {
            "window_start_s": start_s,
            "window_end_s": end_s,
            "offered_requests": b["offered"],
            "started_requests": b["started"],
            "successful_completions": b["ok"],
            "failed_completions": b["failed"],
            "offered_rps": b["offered"] / window_s,
            "started_rps": b["started"] / window_s,
            "successful_completed_rps": b["ok"] / window_s,
            "failed_completed_rps": b["failed"] / window_s,
            # Started before the boundary and not ended by it.
            "in_flight_at_end": bisect_left(starts, boundary) - bisect_right(ends, boundary),
            "ttft_sample_count": len(ttft),
            "p50_ttft_ms": percentile(ttft, 0.50),
            "p90_ttft_ms": percentile(ttft, 0.90),
            "p99_ttft_ms": percentile(ttft, 0.99),
            "p50_client_queue_ms": percentile(queue, 0.50),
            "p99_client_queue_ms": percentile(queue, 0.99),
            "ttft_slo_attainment": (
                sum(v <= ttft_slo_ms for v in ttft) / len(ttft)
                if ttft_slo_ms is not None and ttft else None
            ),
        }
        if by_depth:
            subagent = b["offered"] - b["offered_root"]
            window.update({
                "offered_root_requests": b["offered_root"],
                "offered_subagent_requests": subagent,
                "offered_root_rps": b["offered_root"] / window_s,
                "offered_subagent_rps": subagent / window_s,
            })
        if tail_window_s is not None:
            lo = bisect_left(ttft_starts, boundary - round(tail_window_s * 1e9))
            hi = bisect_left(ttft_starts, boundary)
            tail = sorted(v for _, v in ttft_by_start[lo:hi])
            window["rolling_ttft_sample_count"] = len(tail)
            window["rolling_p99_ttft_ms"] = percentile(tail, 0.99)
        windows.append(window)

    return {
        "schema_version": SCHEMA_VERSION,
        "window_s": window_s,
        "tail_window_s": tail_window_s,
        "ttft_slo_ms": ttft_slo_ms,
        "profiling_start_ns": t0,
        "time_origin_source": source,
        "malformed_record_count": malformed,
        "agent_depth_available": by_depth,
        "summary": _qps_summary(requests, t0, duration_s) if duration_s else None,
        "windows": windows,
    }
