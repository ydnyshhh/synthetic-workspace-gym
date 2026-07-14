from __future__ import annotations

import io
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_support import workspace_tempdir

from synthetic_workspace_gym.sandbox.errors import SandboxExecutionError
from synthetic_workspace_gym.sandbox.prime import PrimeSandboxBackend, _extract_safe_archive
from synthetic_workspace_gym.sandbox.runner import build_sandbox_backend
from synthetic_workspace_gym.sandbox.schemas import SandboxCommand, SandboxConfig


class _FakeClient:
    def __init__(self) -> None:
        self.commands: list[dict[str, object]] = []
        self.deleted: list[str] = []

    def execute_command(self, sandbox_id: str, command: str, **kwargs):
        self.commands.append({"sandbox_id": sandbox_id, "command": command, **kwargs})
        return SimpleNamespace(exit_code=0, stdout="ok\n", stderr="")

    def delete(self, sandbox_id: str) -> None:
        self.deleted.append(sandbox_id)


class _RecordingBackend(PrimeSandboxBackend):
    def __init__(self, config: SandboxConfig, client: _FakeClient) -> None:
        super().__init__(config, client=client)
        self.uploads: list[tuple[Path, str, str]] = []
        self.downloads: list[tuple[str, Path]] = []
        self.runtime_uploads = 0

    def _ensure_sandbox(self, network_enabled: bool) -> str:
        self._sandbox_id = "sandbox-test"
        self._network_enabled = network_enabled
        return self._sandbox_id

    def _replace_remote_tree(self, sandbox_id: str, source: Path, destination: str, label: str) -> None:
        self.uploads.append((source, destination, label))

    def _download_remote_tree(self, sandbox_id: str, source: str, destination: Path, timeout: int) -> None:
        self.downloads.append((source, destination))

    def _upload_runtime(self, sandbox_id: str) -> None:
        self.runtime_uploads += 1


def test_prime_tool_sandbox_never_uploads_hidden_or_evaluator_assets() -> None:
    with workspace_tempdir() as tmp_dir:
        root = Path(tmp_dir)
        workspace = root / "workspace"
        hidden = root / "hidden"
        environment = root / "environment"
        workspace.mkdir()
        hidden.mkdir()
        environment.mkdir()
        client = _FakeClient()
        backend = _RecordingBackend(
            SandboxConfig(
                backend="prime",
                extra_docker_args=[
                    "--mount",
                    f"type=bind,src={environment},dst=/environment,readonly",
                ],
            ),
            client,
        )

        result = backend.run(
            SandboxCommand(argv=["python", "probe.py"], mode="tool"),
            workspace,
            hidden_path=hidden,
        )

        assert result.success
        assert backend.uploads == [(workspace.resolve(), "/workspace", "workspace")]
        assert backend.runtime_uploads == 0
        assert backend.downloads == [("/workspace", workspace.resolve())]


def test_prime_evaluator_sandbox_uploads_hidden_manifest_and_runtime_separately() -> None:
    with workspace_tempdir() as tmp_dir:
        root = Path(tmp_dir)
        workspace = root / "workspace"
        hidden = root / "hidden"
        environment = root / "environment"
        for path in (workspace, hidden, environment):
            path.mkdir()
        client = _FakeClient()
        backend = _RecordingBackend(
            SandboxConfig(
                backend="prime",
                extra_docker_args=[
                    "--mount",
                    f"type=bind,src={environment},dst=/environment,readonly",
                ],
            ),
            client,
        )

        result = backend.run(
            SandboxCommand(
                argv=["python", "-m", "synthetic_workspace_gym.sandbox.evaluator_entrypoint"],
                mode="evaluator",
            ),
            workspace,
            hidden_path=hidden,
        )

        assert result.success
        assert backend.uploads == [
            (workspace.resolve(), "/workspace", "workspace"),
            (hidden.resolve(), "/hidden", "hidden"),
            (environment.resolve(), "/environment", "readonly"),
        ]
        assert backend.runtime_uploads == 1
        assert backend.downloads == []
        assert client.commands[-1]["env"] == {"PYTHONPATH": "/opt/swg-runtime"}


def test_prime_backend_rejects_non_readonly_or_unexpected_mounts() -> None:
    backend = PrimeSandboxBackend(
        SandboxConfig(
            backend="prime",
            extra_docker_args=["--mount", "type=bind,src=/tmp/data,dst=/hidden"],
        ),
        client=_FakeClient(),
    )
    with pytest.raises(SandboxExecutionError, match="readonly"):
        backend._readonly_mounts()


def test_prime_archive_extraction_rejects_traversal() -> None:
    with workspace_tempdir() as tmp_dir:
        root = Path(tmp_dir)
        archive = root / "unsafe.tar.gz"
        with tarfile.open(archive, "w:gz") as handle:
            payload = b"secret"
            member = tarfile.TarInfo("../escape.txt")
            member.size = len(payload)
            handle.addfile(member, io.BytesIO(payload))
        with pytest.raises(SandboxExecutionError, match="unsafe archive path"):
            _extract_safe_archive(archive, root / "extract")


def test_build_prime_backend_and_close_delete_remote_sandbox() -> None:
    built = build_sandbox_backend(SandboxConfig(backend="prime"))
    assert isinstance(built, PrimeSandboxBackend)

    client = _FakeClient()
    backend = PrimeSandboxBackend(SandboxConfig(backend="prime"), client=client)
    backend._sandbox_id = "sandbox-test"
    backend.close()
    assert client.deleted == ["sandbox-test"]
