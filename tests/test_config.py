from pathlib import Path

import pytest
from pydantic import ValidationError

from serve_llm_autoscaling.config import ExperimentConfig, load_config


def test_baseline_config_is_valid():
    config = load_config(Path("experiments/baseline.yaml"))
    assert config.deployment.model_id == "Qwen/Qwen3-0.6B-FP8"
    assert config.benchmark.levels == [1.0, 2.0, 4.0, 8.0]
    assert config.deployment.model_source == config.deployment.model_id


def test_rejects_bad_replica_bounds():
    with pytest.raises(ValidationError, match="min_replicas"):
        ExperimentConfig.model_validate(
            {
                "name": "bad",
                "deployment": {
                    "model_id": "model",
                    "autoscaling": {"min_replicas": 3, "max_replicas": 2},
                },
            }
        )


def test_rejects_fractional_concurrency():
    with pytest.raises(ValidationError, match="concurrency levels must be integers"):
        ExperimentConfig.model_validate(
            {
                "name": "bad",
                "deployment": {"model_id": "model"},
                "benchmark": {"mode": "concurrency", "levels": [1.5]},
            }
        )


def test_rejects_workload_larger_than_context():
    with pytest.raises(ValidationError, match="workload requires"):
        ExperimentConfig.model_validate(
            {
                "name": "bad",
                "deployment": {
                    "model_id": "model",
                    "engine": {"max_model_len": 100},
                },
                "benchmark": {
                    "workload": {"input_tokens": 90, "output_tokens": 20}
                },
            }
        )



SERIES = [
    {"time_s": 0, "qps": 1},
    {"time_s": 10, "qps": 2},
    {"time_s": 20, "qps": 1},
]


def _benchmark(**benchmark):
    return ExperimentConfig.model_validate(
        {"name": "series", "deployment": {"model_id": "model"}, "benchmark": benchmark}
    )


@pytest.mark.parametrize(
    "path",
    [
        "experiments/baseline.yaml",
        "experiments/baseline_rate.yaml",
        "experiments/smoke.yaml",
        "experiments/step_rate.yaml",
        "experiments/step_rate_autoscaling.yaml",
        "experiments/smoke_rate_series.yaml",
    ],
)
def test_checked_in_experiments_are_valid(path):
    load_config(Path(path))


def test_valid_request_rate_series():
    config = _benchmark(mode="request_rate_series", duration_s=20, rate_series=SERIES)
    assert [point.qps for point in config.benchmark.rate_series] == [1, 2, 1]


def test_step_rate_experiment():
    config = load_config(Path("experiments/step_rate.yaml"))
    assert config.benchmark.mode == "request_rate_series"
    assert config.benchmark.rate_series[-1].time_s == config.benchmark.duration_s


@pytest.mark.parametrize(
    ("rate_series", "match"),
    [
        (None, "requires rate_series"),
        ([], "requires rate_series"),
        ([{"time_s": 0, "qps": 1}], "at least two points"),
        ([{"time_s": 1, "qps": 1}, {"time_s": 2, "qps": 1}], "start at time_s 0"),
        (
            [{"time_s": 0, "qps": 1}, {"time_s": 5, "qps": 1}, {"time_s": 5, "qps": 2}],
            "strictly increasing",
        ),
        (
            [{"time_s": 0, "qps": 1}, {"time_s": 5, "qps": 1}, {"time_s": 4, "qps": 2}],
            "strictly increasing",
        ),
        ([{"time_s": 0, "qps": 1}, {"time_s": 30, "qps": 1}], "after duration_s"),
        ([{"time_s": 0, "qps": 0}, {"time_s": 5, "qps": 1}], "greater than 0"),
        ([{"time_s": 0, "qps": -1}, {"time_s": 5, "qps": 1}], "greater than 0"),
    ],
)
def test_rejects_bad_rate_series(rate_series, match):
    with pytest.raises(ValidationError, match=match):
        _benchmark(mode="request_rate_series", duration_s=20, rate_series=rate_series)


@pytest.mark.parametrize("mode", ["concurrency", "request_rate"])
def test_rejects_rate_series_in_static_mode(mode):
    with pytest.raises(ValidationError, match="requires mode: request_rate_series"):
        _benchmark(mode=mode, rate_series=SERIES)


def test_rejects_smoothness_without_gamma():
    with pytest.raises(ValidationError, match="arrival_smoothness requires"):
        _benchmark(
            mode="request_rate_series",
            duration_s=20,
            rate_series=SERIES,
            arrival_pattern="poisson",
            arrival_smoothness=0.5,
        )


def test_rejects_smoothness_in_concurrency_mode():
    with pytest.raises(ValidationError, match="arrival_smoothness requires"):
        _benchmark(mode="concurrency", arrival_pattern="gamma", arrival_smoothness=0.5)


def test_accepts_gamma_smoothness():
    config = _benchmark(
        mode="request_rate_series",
        duration_s=20,
        rate_series=SERIES,
        arrival_pattern="gamma",
        arrival_smoothness=0.5,
    )
    assert config.benchmark.arrival_smoothness == 0.5


@pytest.mark.parametrize("flag", ["--request-rate-series", "--arrival-smoothness"])
def test_rejects_series_flags_in_extra_args(flag):
    with pytest.raises(ValidationError, match="harness-owned"):
        _benchmark(
            mode="request_rate_series",
            duration_s=20,
            rate_series=SERIES,
            extra_args=[flag, "x"],
        )


def test_analysis_defaults():
    analysis = load_config(Path("experiments/step_rate.yaml")).analysis
    assert (analysis.window_s, analysis.tail_window_s) == (5, 30)
    assert (analysis.status_interval_s, analysis.metrics_interval_s) == (1, 2)
    assert analysis.prometheus_enabled and analysis.generate_plots
    assert analysis.ttft_slo_ms is None


def test_rejects_tail_window_shorter_than_window():
    with pytest.raises(ValidationError, match="tail_window_s must be >= window_s"):
        ExperimentConfig.model_validate({
            "name": "bad", "deployment": {"model_id": "model"},
            "analysis": {"window_s": 10, "tail_window_s": 5},
        })


def test_autoscaling_experiment_matches_control_load():
    control = load_config(Path("experiments/step_rate.yaml"))
    autoscaled = load_config(Path("experiments/step_rate_autoscaling.yaml"))
    assert autoscaled.benchmark == control.benchmark
    assert autoscaled.deployment.model_id == control.deployment.model_id
    assert control.deployment.autoscaling.max_replicas == 1
    scaling = autoscaled.deployment.autoscaling
    assert (scaling.min_replicas, scaling.initial_replicas, scaling.max_replicas) == (1, 1, 4)
    assert (scaling.target_ongoing_requests, scaling.upscale_delay_s,
            scaling.downscale_delay_s) == (4, 30, 120)
