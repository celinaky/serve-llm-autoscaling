# Serve LLM Autoscaling Harness

An experiment harness for benchmarking Ray Serve LLM deployments and
autoscaling policies in a dedicated Anyscale workspace.

## Workflow

The harness deploys a controlled Ray Serve LLM application, waits for its
OpenAI-compatible endpoint, generates traffic with AIPerf, saves the results,
and tears Serve down. It supports two kinds of load tests:

- **Sweeps** run a separate benchmark at each configured concurrency or request
  rate. Use these to characterize steady-state performance and locate the
  system's capacity limit.
- **Request-rate series** run one uninterrupted benchmark while the offered
  request rate changes over time. Use these to study autoscaling, overload, and
  recovery behavior.

The checked-in baseline uses one real `Qwen/Qwen3-0.6B-FP8` replica and a
synthetic prefill-heavy workload (8,000 input tokens and 50 output tokens).

## Setup in an Anyscale workspace

Run the harness on the Anyscale image's Python so the Ray driver matches the
workers. AIPerf 0.12 runs through `uvx` with an isolated Python 3.11 environment
because its dependencies and Python requirement may differ from the image's.

```bash
pip install -e ".[dev]"
pip install "quack-kernels==0.6.3" "nvidia-cutlass-dsl==4.6.0"
autoscale-harness check
```

The second `pip install` works around an image bug: its vLLM 0.26.0 pins
`nvidia-cutlass-dsl==4.6.0`, but the bundled `quack-kernels` 0.4.1 is
incompatible with it and fails engine startup with
`AttributeError: module 'cutlass.cute.core' has no attribute 'ThrMma'`.

## Run

Validate the experiment without touching the cluster:

```bash
autoscale-harness validate experiments/baseline.yaml
```

Run the complete lifecycle:

```bash
autoscale-harness run experiments/baseline.yaml
```

If the dedicated workspace already has a Serve application, explicitly allow
replacement:

```bash
autoscale-harness run experiments/baseline.yaml --replace
```

Keep the deployment alive for debugging:

```bash
autoscale-harness run experiments/baseline.yaml --keep-deployment
```

## Run changing traffic over time

Set `benchmark.mode` to `request_rate_series` when the request rate needs to
change during one experiment. Unlike a request-rate sweep, this starts only one
AIPerf process, so requests and server state continue across every rate change.

Describe the load curve with `rate_series`. Each point contains:

- `time_s`: seconds from the start of AIPerf's measured profiling period
- `qps`: the offered request rate at that time

AIPerf linearly interpolates between consecutive points. A pair of points with
the same QPS creates a plateau; two points close together create an approximate
step. For example, this configuration sends 8 requests/second for 60 seconds,
steps up to 18 requests/second, stays there until 180 seconds, and then returns
to 8 requests/second for the remainder of the 300-second run:

```yaml
benchmark:
  mode: request_rate_series
  duration_s: 300
  arrival_pattern: poisson   # or constant, or gamma with arrival_smoothness
  rate_series:
    - {time_s: 0, qps: 8}
    - {time_s: 60, qps: 8}
    - {time_s: 60.1, qps: 18}
    - {time_s: 180, qps: 18}
    - {time_s: 180.1, qps: 8}
    - {time_s: 300, qps: 8}
```

The series must contain at least two points. The first must have `time_s: 0`,
times must be strictly increasing, all QPS values must be positive, and the
last point cannot be later than `duration_s`. The `levels` setting used by
static sweeps is ignored in this mode.

`arrival_pattern` controls how arrivals are distributed around the
instantaneous QPS target:

- `constant` spaces requests evenly and is useful for deterministic tests.
- `poisson` introduces random variation and is the default.
- `gamma` supports configurable burstiness through `arrival_smoothness`.

Start with the short 1 -> 2 -> 1 requests/second smoke test:

```bash
autoscale-harness run experiments/smoke_rate_series.yaml
```

Then run the full 8 -> 18 -> 8 requests/second experiment, first against one
fixed replica (the control) and then with autoscaling enabled (1-4 replicas,
`target_ongoing_requests: 4`, `downscale_delay_s: 120` so recovery shows during
the drain). Both use the same model, workload and load curve:

```bash
autoscale-harness run experiments/step_rate.yaml
autoscale-harness run experiments/step_rate_autoscaling.yaml
```

The harness translates the YAML curve into the JSON format expected by AIPerf.

## Autoscaling telemetry and timeline

While a request-rate series runs, two background collectors sample the cluster
from immediately before AIPerf starts through its grace/drain period:

- **Serve status** (public `serve.status()`, every `analysis.status_interval_s`,
  default 1s) records realized capacity: replica counts by state (`STARTING`,
  `RUNNING`, `STOPPING`, ...). A missing application is recorded as unavailable,
  never as zero replicas.
- **Prometheus metrics** (every `analysis.metrics_interval_s`, default 2s)
  record what the autoscaler and routers see: desired/target replicas, ongoing
  requests at replicas, router and handle queues, replica health and startup
  latency. The harness scrapes each Ray node's metrics export port directly;
  no Prometheus server is required.

Both are stamped with the same epoch clock as AIPerf and aligned to AIPerf's
profiling start. Collection failures are recorded as error samples and never
stop the benchmark. Static sweeps collect no telemetry.

After the run, the harness derives the analysis from the raw artifacts:

- **Request windows** (default 5s, `analysis.window_s`): offered load by credit
  issue time, actual starts, successful and failed completions, in-flight
  requests at each boundary, client queue delay, and TTFT percentiles grouped by
  request start. Windows continue until the last request finishes.
- **Rolling tail latency**: TTFT p99 over the preceding `analysis.tail_window_s`
  (default 30s), which is stable even when a 5s window has few samples. Set
  `analysis.ttft_slo_ms` to also report SLO attainment.
- **Metrics inventory**: every Serve metric the workspace exported, with labels,
  source nodes and sample counts, plus recommended metrics that were absent.
- **Timeline** (`autoscaling_timeline.png`), four panels on one time axis:
  1. Traffic: configured, offered, started and completed QPS.
  2. User experience: TTFT p50/p90/p99, rolling p99, optional SLO.
  3. Pressure: client in-flight requests and queue delay, Serve ongoing
     requests and router queue length.
  4. Autoscaling: desired and target replicas (metrics) with running,
     starting and stopping replicas (status) as step functions.

  Vertical lines mark the load curve's control points. Signals that were not
  collected are omitted with a warning rather than drawn as zero.

Regenerate the analysis with different windows without rerunning the
experiment; this needs no Ray cluster or GPU:

```bash
autoscale-harness analyze runs/<run-directory> --window-s 10 --tail-window-s 60
autoscale-harness analyze runs/<run-directory> --no-plot
```

A request-rate-series run directory contains:

```text
benchmark/
├── request-rate-series/          # raw AIPerf artifacts
│   ├── rate_series.json          # curve passed to AIPerf
│   ├── command.json              # exact command used
│   ├── stdout.log
│   ├── stderr.log
│   ├── phase_manifest.json       # profiling start time (the time origin)
│   ├── profile_export.jsonl      # per-request AIPerf records
│   └── profile_export_aiperf.json
└── sweep_summary.json            # aggregate metrics for the complete series
telemetry/                        # raw samples, never modified by analysis
├── serve_status.jsonl
└── serve_metrics.jsonl
analysis/                         # derived; regenerated by `analyze`
├── request_timeseries.json
├── metrics_inventory.json
├── plot_data.json                # aligned data the timeline is drawn from
└── autoscaling_timeline.png
```

The summary contains one row with `mode: "request_rate_series"` and
`level: "series"`. It summarizes the complete run; use the analysis artifacts
for behavior over time.

## Results

Results are written below `runs/<UTC timestamp>-<experiment name>/`. Every run
contains the original and resolved configuration, environment metadata,
deployment status snapshots, AIPerf logs and native exports, and a normalized
`sweep_summary.json`.

## Commands

```text
autoscale-harness check
autoscale-harness validate <config>
autoscale-harness deploy <config> [--replace]
autoscale-harness benchmark <config>
autoscale-harness status <config>
autoscale-harness teardown <config>
autoscale-harness run <config> [--replace] [--keep-deployment]
autoscale-harness analyze <run-directory> [--window-s S] [--tail-window-s S] [--no-plot]
```

`benchmark` runs against an existing deployment. For a request-rate series it
collects telemetry when it can connect to Ray, and otherwise runs without it and
warns that the analysis has no Serve telemetry.

The workspace is assumed to be dedicated to the experiment. Teardown calls
`serve.shutdown()` and therefore removes the complete Serve instance, while
leaving the Ray cluster running.
