from __future__ import annotations

import json
import tempfile
from dataclasses import replace
from pathlib import Path

from synthetic_workspace_gym.prime.verifier import verify_workspace
from synthetic_workspace_gym.runtime.environment import load_environment
from synthetic_workspace_gym.sandbox.schemas import SandboxCommand, SandboxConfig
from synthetic_workspace_gym.sandbox.runner import build_sandbox_backend


def verify_workspace_in_sandbox(
    environment_path: str | Path,
    workspace_path: str | Path,
    config: SandboxConfig,
) -> dict[str, object]:
    if config.backend == "local":
        return verify_workspace(environment_path, workspace_path)

    environment = load_environment(Path(environment_path))
    with tempfile.TemporaryDirectory(prefix="swg-eval-manifest-") as tmp_dir:
        tmp_path = Path(tmp_dir)
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps(environment.manifest.to_dict(), sort_keys=True), encoding="utf-8")
        command_config = replace(
            config,
            extra_docker_args=[
                *config.extra_docker_args,
                "--mount",
                f"type=bind,src={tmp_path.resolve()},dst=/environment,readonly",
            ],
        )
        backend = build_sandbox_backend(command_config)
        command = SandboxCommand(
            argv=[
                "python",
                "-m",
                "synthetic_workspace_gym.sandbox.evaluator_entrypoint",
                "--manifest",
                "/environment/manifest.json",
                "--workspace",
                config.workdir,
                "--hidden",
                config.hidden_dir,
            ],
            mode="evaluator",
            timeout_seconds=config.timeout_seconds,
        )
        try:
            result = backend.run(command, Path(workspace_path), hidden_path=environment.hidden_root)
        finally:
            close = getattr(backend, "close", None)
            if callable(close):
                close()
    if not result.success:
        return {
            "reward": 0.0,
            "success": False,
            "score": 0.0,
            "subscores": {},
            "failure_labels": ["sandbox_evaluator_failed"],
            "diagnostics": result.to_public_dict(),
            "runtime_seconds": result.duration_seconds,
        }
    try:
        return json.loads(result.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        return {
            "reward": 0.0,
            "success": False,
            "score": 0.0,
            "subscores": {},
            "failure_labels": ["sandbox_evaluator_bad_output"],
            "diagnostics": {"stdout": result.stdout, "stderr": result.stderr, "error": str(exc)},
            "runtime_seconds": result.duration_seconds,
        }
