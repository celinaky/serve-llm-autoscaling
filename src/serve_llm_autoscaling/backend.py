from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urljoin

import httpx

from .config import ExperimentConfig


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
            },
            "experimental_configs": deployment.experimental_configs,
        }
        if deployment.accelerator_type:
            spec["accelerator_type"] = deployment.accelerator_type
        return spec

    def deploy(self) -> dict[str, Any]:
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

        # Body forwarding is only needed by the multi-model direct-streaming topology.
        direct_env = "RAY_SERVE_LLM_ENABLE_DIRECT_STREAMING"
        forward_body_env = "RAY_SERVE_INGRESS_REQUEST_ROUTER_FORWARD_BODY"
        if self.config.deployment.direct_streaming:
            os.environ[direct_env] = "1"
            os.environ[forward_body_env] = "1"
        else:
            os.environ.pop(direct_env, None)
            os.environ.pop(forward_body_env, None)

        # Import after setting the topology switch; Ray snapshots some LLM flags
        # at module import time.
        from ray.serve.llm import LLMConfig, build_openai_app

        deployment = self.config.deployment
        app = build_openai_app(
            {"llm_configs": [LLMConfig(**self.deployment_spec())]}
        )
        serve.run(
            app,
            name=deployment.application_name,
            route_prefix=deployment.route_prefix,
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
                response = httpx.get(models_url, timeout=5)
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
