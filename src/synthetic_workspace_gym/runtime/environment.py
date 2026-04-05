from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from synthetic_workspace_gym.schemas import EnvironmentManifest
from synthetic_workspace_gym.utils.io import read_json


@dataclass(slots=True)
class LoadedEnvironment:
    root: Path
    manifest: EnvironmentManifest
    visible_root: Path
    hidden_root: Path


def load_environment(root: Path) -> LoadedEnvironment:
    root = root.resolve()
    manifest = EnvironmentManifest.from_dict(read_json(root / "manifest.json"))
    return LoadedEnvironment(
        root=root,
        manifest=manifest,
        visible_root=root / manifest.workspace_root,
        hidden_root=root / manifest.hidden_root,
    )
