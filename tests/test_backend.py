from serve_llm_autoscaling.backend import RayServeBackend
from serve_llm_autoscaling.config import load_config


def test_deployment_spec_matches_baseline():
    config = load_config("experiments/baseline.yaml")
    spec = RayServeBackend(config).deployment_spec()
    assert spec["model_loading_config"] == {
        "model_id": "Qwen/Qwen3-0.6B-FP8",
        "model_source": "Qwen/Qwen3-0.6B-FP8",
    }
    assert spec["engine_kwargs"]["max_model_len"] == 10000
    assert spec["engine_kwargs"]["enable_prefix_caching"] is False
    assert spec["deployment_config"]["autoscaling_config"]["min_replicas"] == 1
    assert spec["deployment_config"]["autoscaling_config"]["max_replicas"] == 1
