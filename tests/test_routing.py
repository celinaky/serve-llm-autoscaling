import sys
import types
from pathlib import Path

import pytest
from pydantic import ValidationError

from serve_llm_autoscaling import backend, benchmark
from serve_llm_autoscaling.backend import (
    RayServeBackend,
    apply_routing_environment,
    check_routing_support,
    verify_topology,
)
from serve_llm_autoscaling.benchmark import AIPerfRunner
from serve_llm_autoscaling.config import (
    ROUTER_CLASSES,
    ExperimentConfig,
    RoutingConfig,
    load_config,
)

AGENTX_PATH = "experiments/agentx_kv_aware_autoscaling.yaml"
ALL_ENV_VARS = (*backend.ROUTING_ENV_VARS, "RAY_SERVE_ENABLE_HA_PROXY")


def _config(routing=None, **deployment) -> ExperimentConfig:
    return ExperimentConfig.model_validate({
        "name": "routing",
        "deployment": {"model_id": "m", "routing": routing or {}, **deployment},
    })


# --- RoutingConfig ----------------------------------------------------------


def test_default_routing_is_power_of_two_through_ingress():
    routing = RoutingConfig()
    assert routing.policy == "power_of_two"
    assert (routing.direct_streaming, routing.forward_request_body) == (False, False)
    assert routing.environment() == {}
    assert routing.router_class.endswith("PowerOfTwoChoicesRequestRouter")


def test_kv_aware_enables_direct_streaming_and_body_forwarding():
    routing = RoutingConfig(policy="kv_aware")
    assert (routing.direct_streaming, routing.forward_request_body) == (True, True)
    assert routing.router_class == "ray.serve.llm.request_router.KVAwareRouter"
    assert routing.environment() == {
        "RAY_SERVE_LLM_ENABLE_DIRECT_STREAMING": "1",
        "RAY_SERVE_ENABLE_HA_PROXY": "1",
        "RAY_SERVE_INGRESS_REQUEST_ROUTER_FORWARD_BODY": "1",
    }


@pytest.mark.parametrize("routing, match", [
    ({"policy": "kv_aware", "direct_streaming": False}, "requires direct_streaming"),
    ({"policy": "kv_aware", "forward_request_body": False}, "requires forward_request_body"),
    ({"forward_request_body": True}, "forward_request_body requires direct_streaming"),
    ({"direct_streaming": True, "haproxy_request_buffer_bytes": 65536},
     "requires forward_request_body"),
    ({"policy": "random"}, "policy"),
])
def test_rejects_inconsistent_routing(routing, match):
    with pytest.raises(ValidationError, match=match):
        RoutingConfig.model_validate(routing)


def test_body_aware_policies_forward_body_only_with_direct_streaming():
    assert not RoutingConfig(policy="prefix_cache_affinity").forward_request_body
    assert RoutingConfig(policy="prefix_cache_affinity", direct_streaming=True).forward_request_body
    assert not RoutingConfig(policy="round_robin", direct_streaming=True).forward_request_body


def test_direct_streaming_environment_includes_buffer_size():
    env = RoutingConfig(
        policy="kv_aware", haproxy_request_buffer_bytes=8_388_608
    ).environment()
    assert env["RAY_SERVE_HAPROXY_INGRESS_REQUEST_ROUTER_BUFSIZE"] == "8388608"


def test_every_policy_maps_to_a_router_class():
    policies = RoutingConfig.model_fields["policy"].annotation.__args__
    assert set(policies) == set(ROUTER_CLASSES)


def test_top_level_direct_streaming_is_rejected():
    with pytest.raises(ValidationError, match="direct_streaming"):
        _config(direct_streaming=True)


def test_resolved_routing_round_trips():
    config = load_config(AGENTX_PATH)
    assert ExperimentConfig.model_validate(config.resolved_dict()) == config


def test_consistent_hash_owns_session_header():
    with pytest.raises(ValidationError, match="--session-header"):
        ExperimentConfig.model_validate({
            "name": "x",
            "deployment": {"model_id": "m", "routing": {"policy": "consistent_hash"}},
            "benchmark": {"extra_args": ["--session-header", "X-Other"]},
        })


# --- Checked-in AgentX experiment -------------------------------------------


def test_agentx_kv_aware_experiment():
    config = load_config(AGENTX_PATH)
    routing = config.deployment.routing
    assert routing.policy == "kv_aware" and routing.direct_streaming
    assert routing.haproxy_request_buffer_bytes == 8_388_608
    assert config.deployment.engine.max_model_len == 262_144
    assert config.deployment.engine.enable_prefix_caching
    benchmark_config = config.benchmark
    assert (benchmark_config.duration_s, benchmark_config.grace_period_s,
            benchmark_config.post_load_observation_s) == (210, 30, 90)
    workload = benchmark_config.workload
    assert (workload.target_concurrency, workload.concurrency_ramp_duration_s) == (32, 150)
    assert workload.unsafe_override


# --- AIPerf command ---------------------------------------------------------


@pytest.mark.parametrize("policy, header", [
    ("consistent_hash", True), ("kv_aware", False), ("power_of_two", False),
])
def test_session_header_only_for_consistent_hash(policy, header, monkeypatch, tmp_path):
    monkeypatch.setattr(benchmark.shutil, "which", lambda name: f"/bin/{name}")
    config = load_config(AGENTX_PATH)
    config.deployment.routing = RoutingConfig(policy=policy)
    command = AIPerfRunner(config, tmp_path).build_agentx_ramp_command(tmp_path / "a")
    if header:
        assert command[command.index("--session-header") + 1] == "X-Session-ID"
    else:
        assert "--session-header" not in command


# --- Deployment spec and environment ----------------------------------------


def test_deployment_spec_passes_request_router_config():
    config = _config({"policy": "consistent_hash",
                      "request_router_kwargs": {"num_virtual_nodes": 200}})
    spec = RayServeBackend(config).deployment_spec()
    assert spec["deployment_config"]["request_router_config"] == {
        "request_router_class": ROUTER_CLASSES["consistent_hash"],
        "request_router_kwargs": {"num_virtual_nodes": 200},
    }


def test_routing_summary_records_switches():
    summary = RayServeBackend(load_config(AGENTX_PATH)).routing_summary()
    assert summary["request_router_class"] == ROUTER_CLASSES["kv_aware"]
    assert summary["environment"]["RAY_SERVE_ENABLE_HA_PROXY"] == "1"


def test_apply_routing_environment_sets_and_clears(monkeypatch):
    for name in ALL_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    for module, _, _ in backend._SNAPSHOTTED_FLAGS:
        monkeypatch.delitem(sys.modules, module, raising=False)
    apply_routing_environment(load_config(AGENTX_PATH))
    import os

    assert os.environ["RAY_SERVE_ENABLE_HA_PROXY"] == "1"
    assert os.environ["RAY_SERVE_HAPROXY_INGRESS_REQUEST_ROUTER_BUFSIZE"] == "8388608"
    apply_routing_environment(_config())
    assert not any(name in os.environ for name in backend.ROUTING_ENV_VARS)
    assert os.environ["RAY_SERVE_ENABLE_HA_PROXY"] == "1"  # left to the user


def test_apply_routing_environment_detects_stale_import(monkeypatch):
    for name in ALL_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    stale = types.ModuleType("constants")
    stale.RAY_SERVE_ENABLE_HA_PROXY = False
    stale.RAY_SERVE_INGRESS_REQUEST_ROUTER_FORWARD_BODY = True
    monkeypatch.setitem(sys.modules, "ray.serve._private.constants", stale)
    monkeypatch.delitem(sys.modules, "ray.llm._internal.serve.constants", raising=False)
    with pytest.raises(RuntimeError, match="RAY_SERVE_ENABLE_HA_PROXY"):
        apply_routing_environment(load_config(AGENTX_PATH))


# --- Preflight --------------------------------------------------------------


def _fake_dynamo(monkeypatch, version):
    # Serve is already imported in this process; the ordering is covered by
    # test_preflight_applies_environment_before_first_serve_import.
    monkeypatch.setattr(backend, "apply_routing_environment", lambda config: {})

    def fake_version(name):
        if name != "ai-dynamo" or version is None:
            raise backend.importlib.metadata.PackageNotFoundError(name)
        return version

    monkeypatch.setattr(backend.importlib.metadata, "version", fake_version)
    llm = types.ModuleType("dynamo.llm")
    llm.SelectionService = object
    monkeypatch.setitem(sys.modules, "dynamo", types.ModuleType("dynamo"))
    monkeypatch.setitem(sys.modules, "dynamo.llm", llm)


@pytest.mark.parametrize("policy", sorted(set(ROUTER_CLASSES) - {"kv_aware"}))
def test_preflight_resolves_installed_routers(policy):
    result = check_routing_support(_config({"policy": policy}))
    assert result["policy"] == policy
    assert result["request_router_class"].rsplit(".", 1)[1] == (
        ROUTER_CLASSES[policy].replace(":", ".").rsplit(".", 1)[1]
    )


def test_preflight_kv_aware_requires_ai_dynamo(monkeypatch):
    _fake_dynamo(monkeypatch, None)
    with pytest.raises(RuntimeError, match=r"requires ai-dynamo>=1\.4\.0"):
        check_routing_support(load_config(AGENTX_PATH))


def test_preflight_rejects_old_ai_dynamo(monkeypatch):
    _fake_dynamo(monkeypatch, "1.3.2")
    with pytest.raises(RuntimeError, match="found 1.3.2"):
        check_routing_support(load_config(AGENTX_PATH))


def test_preflight_accepts_ai_dynamo(monkeypatch):
    _fake_dynamo(monkeypatch, "1.4.0.post1")
    result = check_routing_support(load_config(AGENTX_PATH))
    assert result == {
        "policy": "kv_aware",
        "request_router_class": "ray.serve.llm.request_router.KVAwareRouter",
        "ai_dynamo_version": "1.4.0.post1",
    }


def test_preflight_reports_missing_router(monkeypatch):
    monkeypatch.setitem(ROUTER_CLASSES, "round_robin", "ray.serve.nowhere.RoundRobinRouter")
    with pytest.raises(RuntimeError, match=r"does not provide"):
        check_routing_support(_config({"policy": "round_robin"}))


# --- Topology ---------------------------------------------------------------


def _status(config, *names):
    app = config.deployment.application_name
    return {"applications": {app: {"deployments": {n: {} for n in names}}}}


def test_verify_direct_streaming_topology():
    config = load_config(AGENTX_PATH)
    result = verify_topology(config, _status(config, "LLMServer:m", "LLMRouter"))
    assert result == {"topology": "direct_streaming",
                      "deployments": ["LLMRouter", "LLMServer:m"]}
    with pytest.raises(RuntimeError, match="expected LLMRouter"):
        verify_topology(config, _status(config, "LLMServer:m", "OpenAiIngress"))
    with pytest.raises(RuntimeError, match="expected LLMRouter"):
        verify_topology(config, _status(config, "LLMServer:m", "LLMRouter", "OpenAiIngress"))


def test_verify_ingress_topology():
    config = _config()
    assert verify_topology(config, _status(config, "LLMServer:m", "OpenAiIngress")) == {
        "topology": "openai_ingress", "deployments": ["LLMServer:m", "OpenAiIngress"],
    }
    with pytest.raises(RuntimeError, match="OpenAiIngress topology"):
        verify_topology(config, _status(config, "LLMServer:m", "LLMRouter"))
    with pytest.raises(RuntimeError, match="OpenAiIngress topology"):
        verify_topology(config, {"applications": {}})


def test_topology_failure_fails_run(monkeypatch, tmp_path: Path):
    import json

    from serve_llm_autoscaling import runner

    config = load_config("experiments/baseline.yaml")
    config.runtime.results_dir = tmp_path / "runs"
    monkeypatch.setattr(runner, "environment_check", lambda connect: {})
    monkeypatch.setattr(runner, "check_routing_support", lambda c: {"policy": "p"})
    torn_down = []
    monkeypatch.setattr(runner.RayServeBackend, "deploy", lambda self: {})
    monkeypatch.setattr(runner.RayServeBackend, "teardown", lambda self: torn_down.append(1))
    monkeypatch.setattr(runner.RayServeBackend, "wait_healthy",
                        lambda self: {"serve_status": _status(config, "LLMRouter")})
    with pytest.raises(RuntimeError, match="OpenAiIngress topology"):
        runner.run_experiment(config, Path("experiments/baseline.yaml"))
    (root,) = (tmp_path / "runs").iterdir()
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["failure_stage"] == "topology"
    assert torn_down == [1]
    routing = json.loads((root / "deployment" / "routing.json").read_text())
    assert routing["policy"] == "power_of_two" and "verified" not in routing


def test_preflight_applies_environment_before_first_serve_import(tmp_path):
    # A fresh interpreter: in this one, Ray Serve is already imported.
    import subprocess

    script = f"""
import os
for name in {ALL_ENV_VARS!r}:
    os.environ.pop(name, None)
from serve_llm_autoscaling import backend
from serve_llm_autoscaling.config import load_config
try:
    backend.check_routing_support(load_config({AGENTX_PATH!r}))
except RuntimeError:
    pass  # ai-dynamo is optional here; the flags are already snapshotted
from ray.serve._private import build_app, constants
print(build_app.RAY_SERVE_ENABLE_HA_PROXY,
      constants.RAY_SERVE_INGRESS_REQUEST_ROUTER_FORWARD_BODY)
"""
    src = str(Path(__file__).resolve().parents[1] / "src")
    completed = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120,
        env={**__import__("os").environ, "PYTHONPATH": src},
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.split()[-2:] == ["True", "True"]
