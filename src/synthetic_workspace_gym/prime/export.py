from __future__ import annotations

import json
import re
import shutil
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from synthetic_workspace_gym.generators.registry import get_generator, list_generators
from synthetic_workspace_gym.prime.dataset import SyntheticWorkspacePrimeDataset
from synthetic_workspace_gym.utils.io import read_json, write_json

TOOL_SCHEMA_VERSION = "swg-prime-tools-v1"
INTERACTION_TYPE = "multi_turn_tool_use"
REWARD_TYPE = "hidden_evaluator"
EXPORT_NAME = "synthetic-workspace-gym-prime-export"


def export_prime_pack(
    output_dir: str | Path,
    dataset: SyntheticWorkspacePrimeDataset | None = None,
    existing_environments_dir: str | Path | None = None,
    families: Sequence[str] | None = None,
    scenarios: dict[str, Sequence[str]] | None = None,
    difficulties: Sequence[int] = (1, 2, 3),
    seeds: Sequence[int] = range(10),
    export_name: str | None = None,
    overwrite: bool = False,
) -> dict[str, object]:
    export_root = _resolve_export_root(output_dir, export_name)
    if existing_environments_dir is not None:
        return export_existing_environments(existing_environments_dir, export_root, overwrite=overwrite)

    if dataset is not None:
        task_rows = dataset.to_list()
        task_families = sorted({str(row["family"]) for row in task_rows})
        task_difficulties = sorted({int(row["difficulty"]) for row in task_rows})
        task_seeds = sorted({int(row["seed"]) for row in task_rows})
        task_scenarios: dict[str, list[str | None]] = {}
        for row in task_rows:
            task_scenarios.setdefault(str(row["family"]), []).append(
                str(row["scenario"]) if row.get("scenario") is not None else None
            )
        return generate_and_export_prime_pack(
            export_root,
            families=task_families,
            scenarios={
                family: tuple(dict.fromkeys(values))
                for family, values in task_scenarios.items()
            },
            difficulties=task_difficulties,
            seeds=task_seeds,
            overwrite=overwrite,
        )

    return generate_and_export_prime_pack(
        export_root,
        families=tuple(families or list_generators()),
        scenarios=scenarios,
        difficulties=difficulties,
        seeds=seeds,
        overwrite=overwrite,
    )


def export_existing_environments(
    existing_environments_dir: str | Path,
    output_dir: str | Path,
    overwrite: bool = False,
) -> dict[str, object]:
    export_root = Path(output_dir).resolve()
    errors: list[dict[str, str]] = []
    _prepare_export_root(export_root, overwrite=overwrite)
    environments_root = export_root / "environments"
    environments_root.mkdir(parents=True, exist_ok=True)

    copied_paths: list[Path] = []
    for source in _find_environment_roots(Path(existing_environments_dir)):
        try:
            manifest = read_json(source / "manifest.json")
            env_id = str(manifest.get("env_id") or source.name)
            target = environments_root / env_id
            if target.exists():
                if overwrite:
                    shutil.rmtree(target)
                else:
                    raise FileExistsError(f"Environment already exists in export: {target}")
            shutil.copytree(source, target)
            copied_paths.append(target)
        except Exception as exc:
            errors.append({"environment_path": str(source), "error": str(exc)})

    rows = [build_manifest_row(path, export_root) for path in sorted(copied_paths, key=lambda item: item.name)]
    manifest_path = write_manifest_jsonl(export_root / "manifest.jsonl", rows)
    metadata_path = write_metadata_json(export_root / "metadata.json", rows)
    return _summary(export_root, manifest_path, metadata_path, rows, errors)


def generate_and_export_prime_pack(
    output_dir: str | Path,
    families: Sequence[str],
    scenarios: dict[str, Sequence[str | None]] | None,
    difficulties: Sequence[int],
    seeds: Sequence[int],
    overwrite: bool = False,
) -> dict[str, object]:
    export_root = Path(output_dir).resolve()
    errors: list[dict[str, str]] = []
    _prepare_export_root(export_root, overwrite=overwrite)
    generated_root = export_root / ".generated"
    environments_root = export_root / "environments"
    generated_root.mkdir(parents=True, exist_ok=True)
    environments_root.mkdir(parents=True, exist_ok=True)

    dataset = SyntheticWorkspacePrimeDataset(
        families=tuple(families),
        difficulties=tuple(int(item) for item in difficulties),
        scenarios=scenarios,
        seeds=tuple(int(item) for item in seeds),
        split="prime-export",
    )
    exported_paths: list[Path] = []
    for task in dataset:
        family = str(task["family"])
        scenario = str(task["scenario"]) if task.get("scenario") is not None else None
        difficulty = int(task["difficulty"])
        seed = int(task["seed"])
        try:
            generator = get_generator(family)
            spec = generator.sample_spec(
                difficulty=difficulty,
                seed=seed,
                scenario_id=scenario,
            )
            bundle = generator.generate_instance(spec, generated_root)
            target = environments_root / bundle.manifest.env_id
            if target.exists():
                if overwrite:
                    shutil.rmtree(target)
                else:
                    raise FileExistsError(f"Environment already exists in export: {target}")
            shutil.copytree(bundle.root, target)
            exported_paths.append(target)
        except Exception as exc:
            errors.append(
                {
                    "family": family,
                    "scenario": scenario or "",
                    "difficulty": str(difficulty),
                    "seed": str(seed),
                    "error": str(exc),
                }
            )

    if generated_root.exists():
        shutil.rmtree(generated_root, ignore_errors=True)

    rows = [build_manifest_row(path, export_root) for path in sorted(exported_paths, key=lambda item: item.name)]
    manifest_path = write_manifest_jsonl(export_root / "manifest.jsonl", rows)
    metadata_path = write_metadata_json(export_root / "metadata.json", rows)
    return _summary(export_root, manifest_path, metadata_path, rows, errors)


def write_manifest_jsonl(path: str | Path, rows: Sequence[dict[str, object]]) -> Path:
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")
    return manifest_path


def write_metadata_json(path: str | Path, rows: Sequence[dict[str, object]]) -> Path:
    payload = {
        "name": EXPORT_NAME,
        "version": "v1",
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        "environment_count": len(rows),
        "families": sorted({str(row["family"]) for row in rows if row.get("family") is not None}),
        "difficulties": sorted({int(row["difficulty"]) for row in rows if row.get("difficulty") is not None}),
        "seeds": sorted({int(row["seed"]) for row in rows if row.get("seed") is not None}),
        "tool_schema_version": TOOL_SCHEMA_VERSION,
        "interaction_type": INTERACTION_TYPE,
        "reward_type": REWARD_TYPE,
        "source": "synthetic-workspace-gym",
        "notes": "Portable export generated for Prime/verifiers-style infrastructure.",
    }
    metadata_path = Path(path)
    write_json(metadata_path, payload)
    return metadata_path


def build_manifest_row(
    environment_path: str | Path,
    export_root: str | Path,
) -> dict[str, object]:
    environment_root = Path(environment_path).resolve()
    export_root = Path(export_root).resolve()
    manifest_path = environment_root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest.json under {environment_root}")

    manifest = read_json(manifest_path)
    metadata = dict(manifest.get("metadata", {}) or {})
    env_id = str(manifest.get("env_id") or environment_root.name)
    family = _string_or_none(manifest.get("family") or metadata.get("family") or _parse_env_id(env_id).get("family"))
    scenario = _string_or_none(metadata.get("scenario_id") or metadata.get("scenario") or _parse_env_id(env_id).get("scenario"))
    difficulty = _int_or_none(manifest.get("difficulty") or metadata.get("difficulty") or _parse_env_id(env_id).get("difficulty"))
    seed = _int_or_none(manifest.get("seed") or metadata.get("seed") or _parse_env_id(env_id).get("seed"))

    environment_rel = _relative_posix(environment_root, export_root)
    manifest_rel = _relative_posix(manifest_path, export_root)
    visible_root = environment_root / str(manifest.get("workspace_root") or "visible")
    hidden_root = environment_root / str(manifest.get("hidden_root") or "hidden")
    task_scenario = scenario or "default"
    task_id = f"swg.{family or 'unknown'}.{task_scenario}.d{difficulty if difficulty is not None else 'unknown'}.s{seed if seed is not None else 'unknown'}"
    tool_permissions = _enabled_tools(manifest.get("tool_permissions", {}))
    tags = ["synthetic-workspace-gym", "tool-use", "hidden-verifier"]
    if family:
        tags.insert(1, family)

    return {
        "task_id": task_id,
        "env_id": env_id,
        "family": family,
        "scenario": scenario,
        "difficulty": difficulty,
        "seed": seed,
        "instruction": manifest.get("instruction"),
        "environment_path": environment_rel,
        "manifest_path": manifest_rel,
        "visible_path": _relative_posix(visible_root, export_root),
        "hidden_path": _relative_posix(hidden_root, export_root),
        "evaluator_entrypoint": manifest.get("evaluator_entrypoint"),
        "visible_files": list(manifest.get("visible_files", []) or []),
        "hidden_files": list(manifest.get("hidden_files", []) or []),
        "tool_permissions": tool_permissions,
        "tool_schema_version": TOOL_SCHEMA_VERSION,
        "reward_type": REWARD_TYPE,
        "interaction_type": INTERACTION_TYPE,
        "max_steps": manifest.get("max_steps"),
        "tags": tags,
        "metadata": metadata,
    }


def _summary(
    export_root: Path,
    manifest_path: Path,
    metadata_path: Path,
    rows: Sequence[dict[str, object]],
    errors: Sequence[dict[str, str]],
) -> dict[str, object]:
    return {
        "export_root": str(export_root),
        "environment_count": len(rows),
        "manifest_path": str(manifest_path),
        "metadata_path": str(metadata_path),
        "families": sorted({str(row["family"]) for row in rows if row.get("family") is not None}),
        "difficulties": sorted({int(row["difficulty"]) for row in rows if row.get("difficulty") is not None}),
        "seeds": sorted({int(row["seed"]) for row in rows if row.get("seed") is not None}),
        "errors": list(errors),
    }


def _resolve_export_root(output_dir: str | Path, export_name: str | None) -> Path:
    root = Path(output_dir)
    if export_name:
        root = root / export_name
    return root.resolve()


def _prepare_export_root(export_root: Path, *, overwrite: bool) -> None:
    if export_root.exists() and overwrite:
        shutil.rmtree(export_root)
    export_root.mkdir(parents=True, exist_ok=True)


def _find_environment_roots(root: Path) -> list[Path]:
    root = root.resolve()
    if (root / "manifest.json").exists():
        return [root]
    return sorted(path.parent for path in root.rglob("manifest.json"))


def _relative_posix(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _enabled_tools(payload: Any) -> list[str]:
    if isinstance(payload, list):
        return [str(item) for item in payload]
    if not isinstance(payload, dict):
        return []
    ordered = [
        "read_file",
        "write_file",
        "append_file",
        "list_directory",
        "run_shell",
        "run_python",
        "submit",
    ]
    return [name for name in ordered if bool(payload.get(name, False))]


def _parse_env_id(env_id: str) -> dict[str, object]:
    match = re.match(r"(?P<family>.+)-d(?P<difficulty>\d+)-s(?P<seed>-?\d+)", env_id)
    if not match:
        return {}
    return {
        "family": match.group("family"),
        "difficulty": int(match.group("difficulty")),
        "seed": int(match.group("seed")),
    }


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
