"""Render the autoscaling timeline from ``analysis/plot_data.json`` alone."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def primary_points(
    points: list[dict[str, Any]], key: str, *, step: bool = False
) -> tuple[list[float], list[float]]:
    """Points on the profiling clock (t >= 0).

    Pre-profile samples are excluded; a step series instead carries its last
    pre-profile value to t=0 so the state at the start is still visible.
    """
    xs: list[float] = []
    ys: list[float] = []
    carried = None
    for p in points:
        value = p.get(key)
        if value is None:
            continue
        if p["relative_time_s"] < 0:
            carried = value
            continue
        if step and carried is not None and not xs and p["relative_time_s"] > 0:
            xs.append(0.0)
            ys.append(carried)
        xs.append(p["relative_time_s"])
        ys.append(value)
    return xs, ys


def _nan(value: Any) -> float:
    return math.nan if value is None else value


def _title(experiment: dict[str, Any]) -> str:
    return (
        f"{experiment['name']} · {experiment['model_id']}\n"
        f"replicas {experiment['min_replicas']}-{experiment['max_replicas']} "
        f"(initial {experiment['initial_replicas']}) · "
        f"target_ongoing_requests={experiment['target_ongoing_requests']:g} · "
        f"window {experiment['window_s']:g}s · "
        f"status every {experiment['status_interval_s']:g}s · "
        f"metrics every {experiment['metrics_interval_s']:g}s"
    )


def plot_timeline(plot_data_path: Path, out_path: Path) -> None:
    data = json.loads(Path(plot_data_path).read_text())
    experiment = data["experiment"]
    windows = data["request_windows"]
    status = data["serve_status_samples"]
    metrics = data["serve_metric_series"]

    fig, axes = plt.subplots(4, 1, figsize=(14, 15), sharex=True)
    traffic, latency, pressure, replicas = axes
    twins = {}

    edges = [w["window_start_s"] for w in windows[:1]] + [w["window_end_s"] for w in windows]
    mids = [(w["window_start_s"] + w["window_end_s"]) / 2 for w in windows]
    ends = [w["window_end_s"] for w in windows]

    # Panel 1: traffic. Observed request rates are on the left axis; AgentX
    # configures session-tree concurrency, drawn on its own right axis.
    load = data["configured_load"]
    load_xs = [p["time_s"] for p in load["points"]]
    load_ys = [p["value"] for p in load["points"]]
    if load["dimension"] == "request_rate":
        traffic.plot(load_xs, load_ys, "k--", lw=1.5, label=load["label"])
    else:
        sessions = twins[traffic] = traffic.twinx()
        sessions.plot(load_xs, load_ys, "k--", lw=1.5, label=load["label"])
        sessions.set_ylim(bottom=0, top=max(load_ys) * 1.1)
        sessions.set_ylabel(load["unit"])
    for key, label, lw in (("offered_rps", "offered", 3), ("started_rps", "started", 1.5),
                           ("successful_completed_rps", "completed (ok)", 1.5)):
        if windows:
            traffic.stairs([w[key] for w in windows], edges, label=label, lw=lw)
    if any(w["failed_completed_rps"] for w in windows):
        traffic.stairs([w["failed_completed_rps"] for w in windows], edges,
                       label="completed (failed)", lw=1.5, color="red")
    if any(w.get("offered_subagent_rps") for w in windows):
        traffic.stairs([w["offered_root_rps"] for w in windows], edges,
                       label="offered (root agents)", lw=1, ls=":")
        traffic.stairs([w["offered_subagent_rps"] for w in windows], edges,
                       label="offered (subagents)", lw=1, ls=":")
    traffic.set_ylabel("requests / s")
    traffic.set_title("Traffic", loc="left", fontsize=10)

    # Panel 2: user experience. Empty windows plot as gaps, not points.
    for key, label in (("p50_ttft_ms", "TTFT p50"), ("p90_ttft_ms", "TTFT p90"),
                       ("p99_ttft_ms", "TTFT p99")):
        latency.plot(mids, [_nan(w[key]) for w in windows], marker=".", ms=4, label=label)
    if windows and "rolling_p99_ttft_ms" in windows[0]:
        latency.plot(ends, [_nan(w["rolling_p99_ttft_ms"]) for w in windows], "k-", lw=2,
                     label=f"rolling {experiment['tail_window_s']:g}s p99")
    values = [w[k] for w in windows for k in ("p50_ttft_ms", "p99_ttft_ms") if w[k]]
    if values and max(values) / max(min(values), 1e-9) >= 20:
        latency.set_yscale("log")
    slo = experiment.get("ttft_slo_ms")
    if slo:
        latency.axhline(slo, color="red", ls=":", label=f"SLO {slo:g}ms")
        attainment = [w["ttft_slo_attainment"] for w in windows]
        if any(a is not None for a in attainment):
            right = twins[latency] = latency.twinx()
            right.plot(mids, [_nan(a) * 100 for a in attainment], color="red", alpha=0.4,
                       label="SLO attainment")
            right.set_ylim(0, 105)
            right.set_ylabel("SLO attainment %")
    latency.set_ylabel("TTFT (ms)")
    latency.set_title("User experience (by request start)", loc="left", fontsize=10)

    # Panel 3: pressure. Missing Serve signals are omitted, never drawn as zero.
    pressure.plot(ends, [w["in_flight_at_end"] for w in windows], label="client in-flight")
    for key, label in (("ongoing_requests", "Serve ongoing (replicas)"),
                       ("handle_ongoing_requests", "Serve ongoing (handles)"),
                       ("router_queue", "router queue len"),
                       ("queued_queries", "handle queued queries")):
        if key in metrics:
            xs, ys = primary_points(metrics[key]["points"], "value")
            pressure.plot(xs, ys, lw=1, label=label)
    pressure.set_ylabel("requests")
    queue = twins[pressure] = pressure.twinx()
    queue.plot(mids, [_nan(w["p99_client_queue_ms"]) for w in windows], color="gray",
               ls="--", lw=1, label="client queue p99")
    queue.set_ylabel("client queue delay (ms)")
    pressure.set_title("Pressure", loc="left", fontsize=10)

    # Panel 4: autoscaling, as step functions at the original sample times.
    # Unavailable status samples have no counts and break the line.
    observed = [experiment["max_replicas"], 1]
    for key, label, style in (("target_replicas", "target", {"lw": 2.5}),
                              ("desired_replicas", "desired", {"lw": 1, "ls": "--"})):
        if key in metrics:
            xs, ys = primary_points(metrics[key]["points"], "value", step=True)
            replicas.step(xs, ys, where="post", label=label, **style)
            observed += ys
    for key, label in (("running_replicas", "running"), ("starting_replicas", "starting"),
                       ("stopping_replicas", "stopping")):
        xs, ys = primary_points(
            [s if s.get("error") is None else {**s, key: math.nan} for s in status],
            key, step=True,
        )
        if any(not math.isnan(y) for y in ys):
            replicas.step(xs, ys, where="post", label=label, lw=1.5)
            observed += [y for y in ys if not math.isnan(y)]
    if not replicas.lines:
        replicas.text(0.5, 0.5, "no replica telemetry", transform=replicas.transAxes,
                      ha="center", va="center", color="gray")
    replicas.set_ylim(bottom=0, top=max(observed) + 0.5)
    replicas.yaxis.get_major_locator().set_params(integer=True)
    replicas.set_ylabel("replicas")
    replicas.set_xlabel("seconds since profiling start")
    replicas.set_title("Autoscaling", loc="left", fontsize=10)

    for ax in axes:
        for marker in load["markers"]:
            ax.axvline(marker, color="gray", alpha=0.3, lw=0.8)
        ax.grid(alpha=0.2)
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        if ax in twins:
            extra_handles, extra_labels = twins[ax].get_legend_handles_labels()
            handles, labels = handles + extra_handles, labels + extra_labels
        if handles:
            ax.legend(handles, labels, loc="upper right", fontsize=8)

    end = max(ends + [s["relative_time_s"] for s in status] + [1.0])
    replicas.set_xlim(0, end)
    fig.suptitle(_title(experiment), fontsize=11)
    if data["warnings"]:
        fig.text(0.01, 0.005, "\n".join(data["warnings"][:6]), fontsize=7, color="dimgray",
                 va="bottom")
    fig.tight_layout(rect=(0, 0.02 + 0.012 * min(len(data["warnings"]), 6), 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
