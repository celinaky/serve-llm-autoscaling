# Serve LLM Autoscaling Harness

An experiment harness for benchmarking Ray Serve LLM deployments and
autoscaling policies in a dedicated Anyscale workspace.

## MVP workflow

The initial harness deploys a controlled Ray Serve LLM application, waits for
its OpenAI-compatible endpoint, runs a fixed concurrency or request-rate sweep, or one continuous run along a request-rate curve with AIPerf, saves all
artifacts, and tears Serve down.

The checked-in baseline uses one real `Qwen/Qwen3-0.6B-FP8` replica and a
synthetic prefill-heavy workload (8,000 input tokens and 50 output tokens).

## Setup in an Anyscale workspace

Run the harness on the Anyscale image's Python, so the
Ray driver matches the workers. AIPerf runs through `uvx` in its own cached
environment because its dependencies conflict with the image's.

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

## Request-rate series

`mode: request_rate_series` runs a single, continuous AIPerf process whose
request rate follows a curve, instead of several independent benchmark points.
The curve is a list of `(time_s, qps)` points; AIPerf interpolates linearly
between them and holds the last rate afterwards. The first point must be at
`time_s: 0`, times must be strictly increasing, and the last point must not
exceed `duration_s`. `levels` is ignored in this mode.

Because of the interpolation, approximate a step with two points 0.1s apart.
This curve holds 8 req/s, steps to 18 req/s at 60s, and back to 8 at 180s:

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

A short 1 -> 2 -> 1 req/s smoke test and the full step experiment:

```bash
autoscale-harness run experiments/smoke_rate_series.yaml
autoscale-harness run experiments/step_rate.yaml
```

The run writes `benchmark/request-rate-series/` with the generated
`rate_series.json`, the AIPerf command and logs, and AIPerf's native exports
(including per-request records in `profile_export.jsonl`). `sweep_summary.json`
holds a single row with `level: "series"` summarizing the whole run.

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
```

The workspace is assumed to be dedicated to the experiment. Teardown calls
`serve.shutdown()` and therefore removes the complete Serve instance, while
leaving the Ray cluster running.
