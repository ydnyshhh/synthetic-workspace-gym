from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

import pytest

from test_support import workspace_tempdir

from synthetic_workspace_gym.cli import build_parser
from synthetic_workspace_gym.counterfactual.hosted import (
    hash_branch_pack,
    inspect_hosted_wheel,
    package_hosted_branch_pack,
    validate_branch_pack,
)
from synthetic_workspace_gym.runtime.environment import load_environment
from synthetic_workspace_gym.utils.io import read_json

SWG_REF = "df0e0462de3c2c006ba4a4db69785e60ec8cccc4"


def _demo_pack() -> Path:
    return Path(__file__).parents[1] / "examples" / "counterfactual" / "demo-pack"


def _copy_pack(root: Path) -> Path:
    target = root / "branch-pack"
    shutil.copytree(_demo_pack(), target)
    return target


def test_package_hosted_generates_self_contained_smoke_package() -> None:
    with workspace_tempdir() as tmp_dir:
        root = Path(tmp_dir)
        source = _copy_pack(root)
        output = root / "swg-counterfactual-pilot"
        result = package_hosted_branch_pack(
            source,
            output,
            "swg-counterfactual-pilot",
            SWG_REF,
            pack_id="swg-cf-pilot-test",
            build_wheel=False,
            smoke_test=True,
        )

        assert result.task_count == len(validate_branch_pack(source))
        assert result.pack_sha256 == hash_branch_pack(source)
        assert len(result.pack_sha256) == 64
        assert (output / "environment.py").is_file()
        assert (output / "pyproject.toml").is_file()
        copied = output / "src" / "swg_counterfactual_pilot" / "branch_pack"
        assert (copied / "manifest.jsonl").is_file()
        assert hash_branch_pack(copied) == result.pack_sha256
        metadata = read_json(output / "src" / "swg_counterfactual_pilot" / "hosted_metadata.json")
        assert metadata["pack_id"] == "swg-cf-pilot-test"
        assert metadata["pack_sha256"] == result.pack_sha256
        assert metadata["source_swg_commit"] == SWG_REF
        assert "visibility PRIVATE" in (output / "README.md").read_text(encoding="utf-8")


def test_hosted_pack_rejects_path_escape_and_missing_hidden_assets() -> None:
    with workspace_tempdir() as tmp_dir:
        root = Path(tmp_dir)
        traversal_pack = _copy_pack(root / "traversal")
        manifest = traversal_pack / "manifest.jsonl"
        rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
        rows[0]["environment_path"] = "../../outside"
        manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="escapes the branch pack"):
            validate_branch_pack(traversal_pack)

        missing_pack = _copy_pack(root / "missing")
        first = json.loads((missing_pack / "manifest.jsonl").read_text(encoding="utf-8").splitlines()[0])
        environment_root = missing_pack / Path(*Path(first["environment_path"]).parts)
        loaded = load_environment(environment_root)
        shutil.rmtree(loaded.hidden_root)
        with pytest.raises(ValueError, match="missing hidden evaluator assets"):
            validate_branch_pack(missing_pack)


def test_hosted_wheel_inspection_requires_every_pack_file() -> None:
    with workspace_tempdir() as tmp_dir:
        root = Path(tmp_dir)
        source = _copy_pack(root)
        output = root / "generated"
        package_hosted_branch_pack(
            source,
            output,
            "swg-counterfactual-wheel-test",
            SWG_REF,
            build_wheel=False,
            smoke_test=False,
        )
        module_name = "swg_counterfactual_wheel_test"
        package_root = output / "src" / module_name
        wheel = root / "complete.whl"
        with zipfile.ZipFile(wheel, "w") as archive:
            for path in package_root.rglob("*"):
                if path.is_file():
                    archive.write(path, path.relative_to(output / "src").as_posix())
        inspect_hosted_wheel(wheel, package_root / "branch_pack", module_name)

        incomplete = root / "incomplete.whl"
        with zipfile.ZipFile(incomplete, "w") as archive:
            archive.writestr(f"{module_name}/__init__.py", "")
            archive.writestr(f"{module_name}/hosted_metadata.json", "{}")
        with pytest.raises(RuntimeError, match="missing"):
            inspect_hosted_wheel(incomplete, package_root / "branch_pack", module_name)


def test_package_hosted_cli_contract() -> None:
    args = build_parser().parse_args([
        "counterfactual",
        "package-hosted",
        "--branch-pack", "artifacts/pilot-pack",
        "--output-dir", "dist/swg-counterfactual-pilot",
        "--package-name", "swg-counterfactual-pilot",
        "--swg-ref", SWG_REF,
        "--pack-id", "swg-cf-pilot-test",
    ])
    assert args.counterfactual_command == "package-hosted"
    assert args.package_name == "swg-counterfactual-pilot"
    assert args.swg_ref == SWG_REF


def test_hosted_package_refuses_nonempty_output_without_force() -> None:
    with workspace_tempdir() as tmp_dir:
        root = Path(tmp_dir)
        source = _copy_pack(root)
        output = root / "existing"
        output.mkdir()
        (output / "keep.txt").write_text("user data", encoding="utf-8")
        with pytest.raises(FileExistsError, match="--force"):
            package_hosted_branch_pack(
                source,
                output,
                "swg-counterfactual-existing-test",
                SWG_REF,
                build_wheel=False,
                smoke_test=False,
            )
        assert (output / "keep.txt").read_text(encoding="utf-8") == "user data"



def test_hosted_package_rejects_nested_source_and_output_paths() -> None:
    with workspace_tempdir() as tmp_dir:
        root = Path(tmp_dir)
        source = _copy_pack(root)
        with pytest.raises(ValueError, match="must not contain one another"):
            package_hosted_branch_pack(
                source,
                source / "generated",
                "swg-counterfactual-nested-test",
                SWG_REF,
                build_wheel=False,
                smoke_test=False,
            )


def test_branch_pack_hash_changes_when_content_changes() -> None:
    with workspace_tempdir() as tmp_dir:
        source = _copy_pack(Path(tmp_dir))
        original = hash_branch_pack(source)
        branch_file = next(source.glob("environments/*/branch.json"))
        branch_file.write_text(branch_file.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        assert hash_branch_pack(source) != original
