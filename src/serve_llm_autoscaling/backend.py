from __future__ import annotations

import importlib
import importlib.metadata
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urljoin

import requests

from .config import ExperimentConfig

# Ray's KVAwareRouter scores replicas with Dynamo's selection service.
MIN_AI_DYNAMO_VERSION = (1, 4, 0)
# Switches that only matter to the direct-streaming topology; unset ones are
# cleared so an earlier deploy in the same process cannot leak into this one.
# RAY_SERVE_ENABLE_HA_PROXY is set when needed but never cleared: HAProxy in
# front of OpenAiIngress is a valid, independent choice.
ROUTING_ENV_VARS = (
    "RAY_SERVE_LLM_ENABLE_DIRECT_STREAMING",
    "RAY_SERVE_INGRESS_REQUEST_ROUTER_FORWARD_BODY",
    "RAY_SERVE_HAPROXY_INGRESS_REQUEST_ROUTER_BUFSIZE",
)
# (module, constant, env var) that Ray reads once at import time.
_SNAPSHOTTED_FLAGS = (
    ("ray.serve._private.constants", "RAY_SERVE_ENABLE_HA_PROXY",
     "RAY_SERVE_ENABLE_HA_PROXY"),
    ("ray.serve._private.constants", "RAY_SERVE_INGRESS_REQUEST_ROUTER_FORWARD_BODY",
     "RAY_SERVE_INGRESS_REQUEST_ROUTER_FORWARD_BODY"),
    ("ray.llm._internal.serve.constants", "RAY_SERVE_LLM_ENABLE_DIRECT_STREAMING",
     "RAY_SERVE_LLM_ENABLE_DIRECT_STREAMING"),
)


def _release(version: str) -> tuple[int, ...]:
    match = re.match(r"\d+(\.\d+)*", version)
    if match is None:
        raise ValueError(f"unparseable version {version!r}")
    return tuple(int(part) for part in match.group().split("."))


def check_routing_support(config: ExperimentConfig) -> dict[str, Any]:
    """Fail before deploying if this environment cannot run the routing policy.

    Checked in the driver, which shares the Anyscale image with the workers.
    This is the first Ray Serve import, so it applies the routing environment.
    """
    routing = config.deployment.routing
    apply_routing_environment(config)
    from ray.serve.config import RequestRouterConfig

    import ray

    try:
        router = RequestRouterConfig(
            request_router_class=routing.router_class
        ).get_request_router_class()
    except Exception as exc:
        raise RuntimeError(
            f"routing policy {routing.policy!r} needs {routing.router_class}, which "
            f"Ray {ray.__version__} does not provide ({type(exc).__name__}: {exc})"
        ) from exc
    result: dict[str, Any] = {
        "policy": routing.policy,
        "request_router_class": f"{router.__module__}.{router.__qualname__}",
    }
    if routing.policy == "kv_aware":
        # Without Dynamo, Ray logs a warning and silently load-balances instead.
        wanted = ".".join(map(str, MIN_AI_DYNAMO_VERSION))
        try:
            version = importlib.metadata.version("ai-dynamo")
        except importlib.metadata.PackageNotFoundError:
            raise RuntimeError(
                f"routing policy kv_aware requires ai-dynamo>={wanted}; install it "
                f"on every node (pip install 'ai-dynamo>={wanted}')"
            ) from None
        if _release(version) < MIN_AI_DYNAMO_VERSION:
            raise RuntimeError(
                f"routing policy kv_aware requires ai-dynamo>={wanted}, found {version}"
            )
        try:
            importlib.import_module("dynamo.llm").SelectionService
        except (ImportError, AttributeError) as exc:
            raise RuntimeError(
                f"ai-dynamo {version} is installed but dynamo.llm.SelectionService "
                f"is not importable ({type(exc).__name__}: {exc})"
            ) from exc
        result["ai_dynamo_version"] = version
    return result


def apply_routing_environment(config: ExperimentConfig) -> dict[str, str]:
    """Set the topology switches in this process before Ray Serve is imported.

    Ray snapshots them into module constants on first import, so a module
    imported earlier with different values is an error rather than ignored.
    """
    env = config.deployment.routing.environment()
    for name in ROUTING_ENV_VARS:
        if name not in env:
            os.environ.pop(name, None)
    os.environ.update(env)
    stale = []
    for module_name, constant, name in _SNAPSHOTTED_FLAGS:
        module = sys.modules.get(module_name)
        if module is None or name not in env:
            continue
        if getattr(module, constant, None) is not True:
            stale.append(f"{module_name}.{constant}")
    if stale:
        raise RuntimeError(
            "Ray Serve was imported before the routing environment was applied "
            f"({', '.join(stale)}); export {', '.join(sorted(env))} before starting "
            "the harness"
        )
    return env


def verify_topology(config: ExperimentConfig, status: dict[str, Any]) -> dict[str, Any]:
    """Check that Serve built the ingress topology the routing config asks for."""
    application = config.deployment.application_name
    app = (status.get("applications") or {}).get(application) or {}
    names = sorted(app.get("deployments") or {})
    has_router = "LLMRouter" in names
    has_ingress = "OpenAiIngress" in names
    if config.deployment.routing.direct_streaming:
        if not has_router or has_ingress:
            raise RuntimeError(
                "direct streaming expected LLMRouter without OpenAiIngress, but "
                f"application {application!r} has deployments {names}"
            )
        topology = "direct_streaming"
    else:
        if has_router or not has_ingress:
            raise RuntimeError(
                "expected the OpenAiIngress topology, but application "
                f"{application!r} has deployments {names}"
            )
        topology = "openai_ingress"
    return {"topology": topology, "deployments": names}


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "dict"):
        return value.dict()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "__dict__"):
        return {k: _jsonable(v) for k, v in vars(value).items() if not k.startswith("_")}
    return value


@dataclass
class RayServeBackend:
    config: ExperimentConfig
    replace: bool = False

    @property
    def base_url(self) -> str:
        return self.config.runtime.endpoint_url.rstrip("/")

    def connect(self) -> dict[str, Any]:
        import ray

        ray.init(address="auto", ignore_reinit_error=True)
        return {
            "ray_version": ray.__version__,
            "ray_path": ray.__file__,
            "cluster_resources": ray.cluster_resources(),
            "available_resources": ray.available_resources(),
        }

    def status(self) -> dict[str, Any]:
        from ray import serve

        return _jsonable(serve.status())

    def deployment_spec(self) -> dict[str, Any]:
        """Return the serializable LLMConfig payload used for deployment."""
        deployment = self.config.deployment
        spec: dict[str, Any] = {
            "model_loading_config": {
                "model_id": deployment.model_id,
                "model_source": deployment.model_source,
            },
            "engine_kwargs": deployment.engine.model_dump(),
            "deployment_config": {
                "autoscaling_config": deployment.autoscaling.model_dump(
                    exclude_none=True
                ),
                "max_ongoing_requests": deployment.max_ongoing_requests,
                "request_router_config": {
                    "request_router_class": deployment.routing.router_class,
                    "request_router_kwargs": deployment.routing.request_router_kwargs,
                },
            },
            "experimental_configs": deployment.experimental_configs,
        }
        if deployment.accelerator_type:
            spec["accelerator_type"] = deployment.accelerator_type
        return spec

    def routing_summary(self) -> dict[str, Any]:
        """The requested routing and the Ray switches that implement it."""
        routing = self.config.deployment.routing
        return {
            **routing.model_dump(mode="json"),
            "request_router_class": routing.router_class,
            "environment": routing.environment(),
        }

    def deploy(self) -> dict[str, Any]:
        # Must precede the first Ray Serve import in this process.
        env = apply_routing_environment(self.config)
        from ray import serve

        current = serve.status()
        applications = getattr(current, "applications", {})
        if applications:
            if not self.replace:
                names = ", ".join(sorted(applications))
                raise RuntimeError(
                    f"Serve already has application(s): {names}. Pass --replace to replace them."
                )
            serve.shutdown()
        elif env:
            # Controller options apply only when the controller starts, so an
            # idle controller from an earlier run would keep its environment.
            serve.shutdown()

        # Import after setting the topology switch; Ray snapshots some LLM flags
        # at module import time.
        from ray.serve.llm import LLMConfig, build_openai_app

        deployment = self.config.deployment
        app = build_openai_app(
            {"llm_configs": [LLMConfig(**self.deployment_spec())]}
        )
        # HAProxy and its body forwarding are configured in the controller,
        # which does not inherit the driver's environment.
        controller_options = {"runtime_env": {"env_vars": env}} if env else None
        serve.run(
            app,
            name=deployment.application_name,
            route_prefix=deployment.route_prefix,
            controller_options=controller_options,
        )
        return self.status()

    def wait_healthy(self) -> dict[str, Any]:
        runtime = self.config.runtime
        model_id = self.config.deployment.model_id
        models_url = urljoin(self.base_url + "/", "v1/models")
        deadline = time.monotonic() + runtime.readiness_timeout_s
        last_error = "not checked"

        while time.monotonic() < deadline:
            try:
                status = self.status()
                response = requests.get(models_url, timeout=5)
                response.raise_for_status()
                body = response.json()
                ids = [item.get("id") for item in body.get("data", [])]
                if model_id in ids:
                    return {"serve_status": status, "models": body}
                last_error = f"registered models were {ids!r}"
            except Exception as exc:  # readiness must tolerate transient failures
                last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(runtime.readiness_poll_s)

        raise TimeoutError(
            f"{model_id!r} did not become ready at {models_url} within "
            f"{runtime.readiness_timeout_s}s; last observation: {last_error}"
        )

    def teardown(self) -> None:
        from ray import serve

        serve.shutdown()
