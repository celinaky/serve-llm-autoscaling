import copy

import pytest

from serve_llm_autoscaling.config import ExperimentConfig

# A long-context deployment; AgentX traces do not fit the 10K Qwen configs.
AGENTX_RAW = {
    "name": "agentx-ramp",
    "deployment": {
        "model_id": "long-context-model",
        "engine": {"max_model_len": 262_144},
        "autoscaling": {"min_replicas": 1, "initial_replicas": 1, "max_replicas": 4},
    },
    "benchmark": {
        "mode": "agentx_concurrency_ramp",
        "duration_s": 1800,
        "grace_period_s": 300,
        "fail_fast": True,
        "use_server_token_count": True,
        "workload": {
            "type": "agentx",
            "public_dataset": "semianalysis_cc_traces_weka_062126_256k",
            "target_concurrency": 16,
            "concurrency_ramp_duration_s": 1200,
            "trajectory_start_ratio": 0,
            "seed": 42,
        },
    },
}


@pytest.fixture
def agentx_config():
    """Build an AgentX ramp config, overriding benchmark or workload fields."""

    def build(workload=None, **benchmark) -> ExperimentConfig:
        raw = copy.deepcopy(AGENTX_RAW)
        raw["benchmark"].update(benchmark)
        raw["benchmark"]["workload"].update(workload or {})
        return ExperimentConfig.model_validate(raw)

    return build
