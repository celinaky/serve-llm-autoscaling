from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .config import ExperimentConfig, write_config


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_revision() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(value, fh, indent=2, sort_keys=True, default=str)
        fh.write("\n")


@dataclass
class RunArtifacts:
    root: Path
    manifest: dict[str, Any]

    @classmethod
    def create(cls, config: ExperimentConfig, input_path: Path) -> "RunArtifacts":
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        safe_name = "".join(
            char if char.isalnum() or char in "-_" else "-" for char in config.name
        )
        root = config.runtime.results_dir / f"{stamp}-{safe_name}"
        root.mkdir(parents=True, exist_ok=False)
        (root / "deployment").mkdir()
        (root / "benchmark").mkdir()
        with input_path.open() as src, (root / "input.yaml").open("w") as dst:
            dst.write(src.read())
        write_config(config, root / "resolved.yaml")
        manifest = {
            "name": config.name,
            "started_at": utc_now(),
            "status": "running",
            "failure_stage": None,
            "git_revision": _git_revision(),
        }
        obj = cls(root=root, manifest=manifest)
        obj.flush_manifest()
        return obj

    def flush_manifest(self) -> None:
        write_json(self.root / "manifest.json", self.manifest)

    def finish(self, status: str, failure_stage: str | None = None, error: str | None = None) -> None:
        self.manifest.update(
            {
                "status": status,
                "failure_stage": failure_stage,
                "error": error,
                "finished_at": utc_now(),
            }
        )
        self.flush_manifest()

    def record_event(self, event: str, **fields: Any) -> None:
        payload = {"timestamp": utc_now(), "event": event, **fields}
        with (self.root / "deployment" / "events.jsonl").open("a") as fh:
            fh.write(json.dumps(payload, default=str) + "\n")

