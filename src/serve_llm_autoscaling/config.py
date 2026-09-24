from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal, Union

import yaml
from pydantic import BaseModel, ConfigDict, Discriminator, Field, Tag, model_validator

# Modes that run one uninterrupted load-generator process with telemetry,
# mapped to their artifact directory below ``benchmark/``.
CONTINUOUS_MODE_DIRS = {
    "request_rate_series": "request-rate-series",
    "agentx_concurrency_ramp": "agentx-concurrency-ramp",
}
CONTINUOUS_MODES = tuple(CONTINUOUS_MODE_DIRS)

# The inferencex-agentx-mvp scenario rejects shorter runs unless overridden.
AGENTX_MIN_DURATION_S = 900

# Options the harness always sets on the AIPerf command line.
COMMON_OWNED_ARGS = {
    "--model", "--model-names", "--tokenizer", "--url", "--streaming",
    "--isl", "--osl", "--random-seed", "--benchmark-duration",
    "--artifact-dir", "--output-artifact-dir", "--concurrency",
    "--request-rate", "--arrival-pattern", "--warmup-duration",
    "--request-rate-series", "--arrival-smoothness",
}
AGENTX_OWNED_ARGS = {
    "--scenario", "--public-dataset", "--concurrency-ramp-duration",
    "--trajectory-start-min-ratio", "--trajectory-start-max-ratio",
    "--unsafe-override",
}


class EngineConfig(BaseModel):
    """vLLM engine settings forwarded to Ray Serve LLM."""

    model_config = ConfigDict(extra="allow")

    tensor_parallel_size: int = Field(default=1, ge=1)
    max_model_len: int = Field(default=10_000, ge=1)
    gpu_memory_utilization: float = Field(default=0.95, gt=0, le=1)
    max_num_seqs: int = Field(default=128, ge=1)
    max_num_batched_tokens: int = Field(default=16_384, ge=1)
    enable_prefix_caching: bool = False
    kv_cache_dtype: str = "fp8"


class AutoscalingConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    min_replicas: int = Field(default=1, ge=0)
    initial_replicas: int | None = Field(default=1, ge=0)
    max_replicas: int = Field(default=1, ge=1)
    target_ongoing_requests: float = Field(default=4, gt=0)
    upscale_delay_s: float = Field(default=30, ge=0)
    downscale_delay_s: float = Field(default=600, ge=0)

    @model_validator(mode="after")
    def validate_bounds(self) -> "AutoscalingConfig":
        if self.min_replicas > self.max_replicas:
            raise ValueError("min_replicas must be <= max_replicas")
        if self.initial_replicas is not None and not (
            self.min_replicas <= self.initial_replicas <= self.max_replicas
        ):
            raise ValueError(
                "initial_replicas must be between min_replicas and max_replicas"
            )
        return self


class DeploymentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    application_name: str = "ttft-benchmark"
    model_id: str
    model_source: str | None = None
    tokenizer: str | None = None
    accelerator_type: str | None = None
    route_prefix: str = "/"
    direct_streaming: bool = False
    engine: EngineConfig = Field(default_factory=EngineConfig)
    autoscaling: AutoscalingConfig = Field(default_factory=AutoscalingConfig)
    max_ongoing_requests: int = Field(default=8192, ge=1)
    experimental_configs: dict[str, Any] = Field(
        default_factory=lambda: {"stream_batching_interval_ms": 0}
    )

    @model_validator(mode="after")
    def fill_defaults(self) -> "DeploymentConfig":
        if self.model_source is None:
            self.model_source = self.model_id
        if self.tokenizer is None:
            self.tokenizer = self.model_source
        return self


class WarmupConfig(BaseModel):
    enabled: bool = True
    duration_s: float = Field(default=30, gt=0)


class SyntheticWorkloadConfig(BaseModel):
    type: Literal["synthetic"] = "synthetic"
    input_tokens: int = Field(default=8000, ge=1)
    output_tokens: int = Field(default=50, ge=1)
    input_tokens_stddev: float = Field(default=0, ge=0)
    output_tokens_stddev: float = Field(default=0, ge=0)
    seed: int = 42
    ignore_eos: bool = True


class AgentXWorkloadConfig(BaseModel):
    """AgentX trace replay; concurrency counts live session trees, not requests."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["agentx"] = "agentx"
    public_dataset: str = "semianalysis_cc_traces_weka_062126_256k"
    target_concurrency: int = Field(ge=1, strict=True)
    concurrency_ramp_duration_s: float = Field(gt=0)
    # Used as both the min and max start ratio; 0 starts every trajectory cold
    # at turn zero.
    trajectory_start_ratio: float = Field(default=0, ge=0, le=1)
    seed: int = 42
    unsafe_override: bool = False


def _workload_type(value: Any) -> str:
    # ``type`` predates AgentX, so configs may omit it for synthetic workloads.
    if isinstance(value, dict):
        return value.get("type", "synthetic")
    return getattr(value, "type", "synthetic")


WorkloadConfig = Annotated[
    Union[
        Annotated[SyntheticWorkloadConfig, Tag("synthetic")],
        Annotated[AgentXWorkloadConfig, Tag("agentx")],
    ],
    Discriminator(_workload_type),
]


class RequestRatePoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    time_s: float = Field(ge=0)
    qps: float = Field(gt=0)


class BenchmarkConfig(BaseModel):
    generator: Literal["aiperf"] = "aiperf"
    mode: Literal[
        "concurrency", "request_rate", "request_rate_series", "agentx_concurrency_ramp"
    ] = "concurrency"
    # Only used by the static sweep modes; ignored for request_rate_series.
    levels: list[float] = Field(default_factory=lambda: [1, 2, 4, 8], min_length=1)
    duration_s: float = Field(default=120, gt=0)
    grace_period_s: float = Field(default=30, ge=0)
    warmup: WarmupConfig = Field(default_factory=WarmupConfig)
    workload: WorkloadConfig = Field(default_factory=SyntheticWorkloadConfig)
    streaming: bool = True
    arrival_pattern: Literal["constant", "poisson", "gamma"] = "poisson"
    arrival_smoothness: float | None = Field(default=None, gt=0)
    # AIPerf interpolates linearly between points, so a step needs two points
    # close together (e.g. 60 -> 60.1).
    rate_series: list[RequestRatePoint] | None = None
    fail_fast: bool = False
    use_server_token_count: bool = True
    extra_args: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_levels(self) -> "BenchmarkConfig":
        agentx = isinstance(self.workload, AgentXWorkloadConfig)
        if self.mode == "agentx_concurrency_ramp":
            self._validate_agentx()
        elif agentx:
            raise ValueError("workload type agentx requires mode: agentx_concurrency_ramp")
        elif self.mode == "request_rate_series":
            self._validate_rate_series()
        else:
            if self.rate_series is not None:
                raise ValueError("rate_series requires mode: request_rate_series")
            self._validate_static_levels()
        if self.arrival_smoothness is not None and (
            self.arrival_pattern != "gamma" or self.mode == "concurrency"
        ):
            raise ValueError(
                "arrival_smoothness requires arrival_pattern: gamma in a "
                "request-rate mode"
            )
        owned = COMMON_OWNED_ARGS | (AGENTX_OWNED_ARGS if agentx else set())
        conflicts = sorted(set(self.extra_args) & owned)
        if conflicts:
            raise ValueError(
                "extra_args cannot override harness-owned options: "
                + ", ".join(conflicts)
            )
        return self

    def _validate_agentx(self) -> None:
        workload = self.workload
        if not isinstance(workload, AgentXWorkloadConfig):
            raise ValueError("agentx_concurrency_ramp mode requires workload type: agentx")
        # The trace replay owns arrivals and warmup; reject settings that would
        # otherwise be silently ignored. Defaults are accepted so that a
        # resolved config, which lists every field, still loads.
        unsupported = sorted(
            name
            for name in ("levels", "rate_series", "arrival_pattern",
                         "arrival_smoothness", "warmup")
            if getattr(self, name) != type(self).model_fields[name].get_default(
                call_default_factory=True
            )
        )
        if unsupported:
            raise ValueError(
                "agentx_concurrency_ramp mode does not accept: " + ", ".join(unsupported)
            )
        if not self.streaming:
            raise ValueError("agentx_concurrency_ramp mode requires streaming: true")
        if workload.concurrency_ramp_duration_s > self.duration_s:
            raise ValueError(
                f"concurrency_ramp_duration_s {workload.concurrency_ramp_duration_s} "
                f"exceeds duration_s {self.duration_s}"
            )
        if self.duration_s < AGENTX_MIN_DURATION_S and not workload.unsafe_override:
            raise ValueError(
                f"AgentX runs require duration_s >= {AGENTX_MIN_DURATION_S}; set "
                "workload.unsafe_override: true for a shorter development run"
            )

    def _validate_rate_series(self) -> None:
        points = self.rate_series
        if not points:
            raise ValueError("request_rate_series mode requires rate_series")
        if len(points) < 2:
            raise ValueError("rate_series requires at least two points")
        if points[0].time_s != 0:
            raise ValueError("rate_series must start at time_s 0")
        times = [point.time_s for point in points]
        if any(later <= earlier for earlier, later in zip(times, times[1:])):
            raise ValueError("rate_series time_s values must be strictly increasing")
        if times[-1] > self.duration_s:
            raise ValueError(
                f"rate_series ends at {times[-1]}s, after duration_s {self.duration_s}"
            )

    def _validate_static_levels(self) -> None:
        if any(level <= 0 for level in self.levels):
            raise ValueError("benchmark levels must all be positive")
        if len(set(self.levels)) != len(self.levels):
            raise ValueError("benchmark levels must be unique")
        if self.levels != sorted(self.levels):
            raise ValueError("benchmark levels must be ordered from low to high")
        if self.mode == "concurrency" and any(
            not float(level).is_integer() for level in self.levels
        ):
            raise ValueError("concurrency levels must be integers")


class RuntimeConfig(BaseModel):
    endpoint_url: str = "http://127.0.0.1:8000"
    readiness_timeout_s: float = Field(default=1200, gt=0)
    readiness_poll_s: float = Field(default=2, gt=0)
    keep_deployment: bool = False
    results_dir: Path = Path("runs")


class AnalysisConfig(BaseModel):
    window_s: float = Field(default=5, gt=0)
    # Rolling window for tail latency; smooths p99 over sparse request windows.
    tail_window_s: float = Field(default=30, gt=0)
    status_interval_s: float = Field(default=1, gt=0)
    metrics_interval_s: float = Field(default=2, gt=0)
    prometheus_enabled: bool = True
    generate_plots: bool = True
    ttft_slo_ms: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_tail_window(self) -> "AnalysisConfig":
        if self.tail_window_s < self.window_s:
            raise ValueError("tail_window_s must be >= window_s")
        return self


class ExperimentConfig(BaseModel):
    name: str
    deployment: DeploymentConfig
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)

    @model_validator(mode="after")
    def validate_context_length(self) -> "ExperimentConfig":
        workload = self.benchmark.workload
        if not isinstance(workload, SyntheticWorkloadConfig):
            return self  # trace lengths come from the dataset, not the config
        required = workload.input_tokens + workload.output_tokens
        if required > self.deployment.engine.max_model_len:
            raise ValueError(
                f"workload requires {required} tokens but max_model_len is "
                f"{self.deployment.engine.max_model_len}"
            )
        return self

    def resolved_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    with config_path.open() as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"{config_path} must contain a YAML mapping")
    return ExperimentConfig.model_validate(raw)


def write_config(config: ExperimentConfig, path: Path) -> None:
    with path.open("w") as fh:
        yaml.safe_dump(config.resolved_dict(), fh, sort_keys=False)
