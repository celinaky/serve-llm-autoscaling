# Serve LLM Autoscaling Harness

An experiment harness for benchmarking Ray Serve LLM deployments and
autoscaling policies in a dedicated Anyscale workspace.

## MVP workflow

The initial harness deploys a controlled Ray Serve LLM application, waits for
its OpenAI-compatible endpoint, runs a fixed concurrency or request-rate sweep
with AIPerf, saves all artifacts, and tears Serve down.

The checked-in baseline uses one real `Qwen/Qwen3-0.6B-FP8` replica and a
synthetic prefill-heavy workload (8,000 input tokens and 50 output tokens).

## Setup in an Anyscale workspace

The environment must reuse the Ray installation provided by Anyscale. Do not
install a second Ray wheel into the project environment.

```bash
uv venv --system-site-packages
uv sync --extra dev
uv run autoscale-harness check
```

Manual activation is not required. If `.venv` was originally created without
`--system-site-packages`, recreate it with the command above before syncing.

## Run

Validate the experiment without touching the cluster:

```bash
uv run autoscale-harness validate experiments/baseline.yaml
```

Run the complete lifecycle:

```bash
uv run autoscale-harness run experiments/baseline.yaml
```

If the dedicated workspace already has a Serve application, explicitly allow
replacement:

```bash
uv run autoscale-harness run experiments/baseline.yaml --replace
```

Keep the deployment alive for debugging:

```bash
uv run autoscale-harness run experiments/baseline.yaml --keep-deployment
```

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
