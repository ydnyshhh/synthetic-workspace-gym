from __future__ import annotations

import hashlib
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from synthetic_workspace_gym.evaluators.registry import get_evaluator
from synthetic_workspace_gym.generators.d5_profiles import select_weighted_d5_profile
from synthetic_workspace_gym.generators.registry import get_generator
from synthetic_workspace_gym.generators.retrieval_profile_scenarios import (
    build_profiled_retrieval_scenario,
)
from synthetic_workspace_gym.provenance import ENVIRONMENT_VERSION, EVALUATOR_VERSION
from test_support import workspace_tempdir


PROFILE_SCENARIOS = {
    "d5_b": "client_adapter_policy_sync",
    "d5_c": "versioned_client_migration",
}


def _profile_seeds(profile: str, count: int) -> list[int]:
    seeds: list[int] = []
    seed = 0
    while len(seeds) < count:
        selected = select_weighted_d5_profile(5, seed)
        if selected is not None and selected.profile_id == profile:
            seeds.append(seed)
        seed += 1
    return seeds


def _scenario_for_seed(seed: int) -> dict[str, object]:
    generator = get_generator("retrieval_workspace")
    spec = generator.sample_spec(
        difficulty=5,
        seed=seed,
        generation_params={"composition_mode": "hard_atomic"},
    )
    return build_profiled_retrieval_scenario(
        random.Random(f"{seed}:profiled_retrieval"), spec
    )


def _content_fingerprint(files: dict[str, object]) -> str:
    digest = hashlib.sha256()
    for path, content in sorted(files.items()):
        digest.update(path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(content).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


@pytest.mark.parametrize("profile", ["d5_b", "d5_c"])
def test_retrieval_surface_is_complete_across_100_profile_seeds(profile: str) -> None:
    seeds = _profile_seeds(profile, 100)
    assert len(seeds) == 100
    fingerprints: set[str] = set()

    for seed in seeds:
        scenario = _scenario_for_seed(seed)
        assert scenario["scenario_id"] == PROFILE_SCENARIOS[profile]
        files = dict(scenario["files"])
        roots = set(scenario["document_roots"])
        assert {"policies", "release"}.issubset(roots)

        stages = dict(scenario["composition_spec"])["stages"]
        required = {
            str(path)
            for stage in stages
            for path in dict(stage)["required_inputs"]
            if str(path).split("/", 1)[0] in roots
        }
        assert required
        assert required.issubset(files)

        manifest = json.loads(str(files["release/current_manifest.json"]))
        assert manifest == {
            "active_api": "v3",
            "cutover": "2026-03-01",
            "policy_bundle": "warehouse-policy-2026-03",
            "region": "eu-west",
        }
        precedence = str(files["docs/authority_precedence.md"]).lower()
        assert "manifest" in precedence
        assert "archived" in precedence
        assert "outrank" in precedence or "resolve conflicts in this order" in precedence

        policy_path = (
            "policies/warehouse-policy-2026-03.md"
            if profile == "d5_b"
            else "policies/regional_override.md"
        )
        policy = str(files[policy_path])
        for fact in ("unknown", "eu-west", "ams-old", "eu-central", "dub-legacy"):
            assert fact in policy
        legacy = str(files["notes/legacy_rollout.md"]).lower()
        assert "archived" in legacy
        assert "superseded" in legacy
        fingerprints.add(_content_fingerprint(files))

    assert len(fingerprints) >= 4


@pytest.mark.parametrize("seed", [100, 102, 107])
def test_public_check_rejects_broken_surface_and_accepts_reference(seed: int) -> None:
    generator = get_generator("retrieval_workspace")
    spec = generator.sample_spec(
        difficulty=5,
        seed=seed,
        generation_params={"composition_mode": "hard_atomic"},
    )
    scenario = _scenario_for_seed(seed)

    with workspace_tempdir() as tmp:
        root = Path(tmp)
        bundle = generator.generate_instance(spec, root / "generated")
        broken = subprocess.run(
            [sys.executable, "run_example.py"],
            cwd=bundle.visible_root,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert broken.returncode != 0

        solved = root / "solved"
        shutil.copytree(bundle.visible_root, solved)
        for relative_path, content in dict(scenario["correct_files"]).items():
            target = solved / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(str(content), encoding="utf-8")
        public = subprocess.run(
            [sys.executable, "run_example.py"],
            cwd=solved,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert public.returncode == 0, public.stderr
        summary = json.loads(public.stdout)
        assert set(summary) == {
            "request_id",
            "next_cursor",
            "record_count",
            "total_quantity",
            "warehouses",
        }


@pytest.mark.parametrize("seed", [102, 107])
def test_every_profile_bug_independently_loses_hidden_capability_credit(seed: int) -> None:
    generator = get_generator("retrieval_workspace")
    spec = generator.sample_spec(
        difficulty=5,
        seed=seed,
        generation_params={"composition_mode": "hard_atomic"},
    )
    scenario = _scenario_for_seed(seed)

    with workspace_tempdir() as tmp:
        root = Path(tmp)
        bundle = generator.generate_instance(spec, root / "generated")
        evaluator = get_evaluator(
            "retrieval_workspace",
            evaluator_entrypoint=bundle.manifest.evaluator_entrypoint,
        )
        for index, bug in enumerate(scenario["bugs"]):
            workspace = root / f"bug-{index}"
            shutil.copytree(bundle.visible_root, workspace)
            files = dict(scenario["correct_files"])
            target_path = str(bug["target_path"])
            files[target_path] = bug["apply"](files[target_path])
            for relative_path, content in files.items():
                target = workspace / relative_path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(str(content), encoding="utf-8")
            result = evaluator.evaluate(workspace, bundle.manifest, bundle.hidden_root)
            assert result.score < 1.0, str(bug["label"])
            assert result.diagnostics["failed_capabilities"], str(bug["label"])


@pytest.mark.parametrize("seed", [102, 107])
def test_generated_retrieval_manifest_records_release_provenance(seed: int) -> None:
    generator = get_generator("retrieval_workspace")
    spec = generator.sample_spec(
        difficulty=5,
        seed=seed,
        generation_params={"composition_mode": "hard_atomic"},
    )
    with workspace_tempdir() as tmp:
        bundle = generator.generate_instance(spec, Path(tmp) / "generated")
    provenance = dict(bundle.manifest.metadata["release_provenance"])
    assert provenance["environment_version"] == ENVIRONMENT_VERSION
    assert provenance["evaluator_version"] == EVALUATOR_VERSION
    assert provenance["horizon_unit"] == "tool_steps"
    assert len(str(provenance["generation_fingerprint"])) == 64
