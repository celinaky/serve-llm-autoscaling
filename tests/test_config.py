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

