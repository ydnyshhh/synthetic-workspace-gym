from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

from test_support import workspace_tempdir
from synthetic_workspace_gym.evaluators.registry import get_evaluator
from synthetic_workspace_gym.generators.registry import get_generator
from synthetic_workspace_gym.hub import _build_rows, _resolve_frozen_manifest
from synthetic_workspace_gym.prime.env import SyntheticWorkspacePrimeEnv


def solve(bundle, destination: Path) -> Path:
    shutil.copytree(bundle.visible_root, destination)
    for relative, content in bundle.manifest.reference_solution["files"].items():
        path = destination / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return destination


def fingerprint(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


class TestCompositeWorkspace:
    def test_nested_difficulty_and_composition_metadata(self) -> None:
        generator = get_generator("composite_workspace")
        with workspace_tempdir() as raw:
            root = Path(raw)
            bundles = [
                generator.generate_instance(
                    generator.sample_spec(
                        difficulty=difficulty,
                        seed=41,
                        scenario_id="retrieval_guided_pipeline_repair",
                        split="training",
                    ),
                    root / str(difficulty),
                    validate=False,
                )
                for difficulty in range(1, 6)
            ]
            bug_sets = [
                set(
                    bundle.manifest.metadata["difficulty_realization"][
                        "applied_bug_ids"
                    ]
                )
                for bundle in bundles
            ]
            assert all(left < right for left, right in zip(bug_sets, bug_sets[1:]))
            evaluator = get_evaluator("composite_workspace")
            untouched_rewards = [
                evaluator.evaluate(
                    bundle.visible_root, bundle.manifest, bundle.hidden_root
                ).score
                for bundle in bundles
            ]
            assert all(
                harder < easier
                for easier, harder in zip(untouched_rewards, untouched_rewards[1:])
            )
            d5 = bundles[-1].manifest.metadata
            assert d5["source_families"] == ["retrieval_workspace", "pipeline"]
            assert d5["composition_spec"]["stage_count"] == 3
            assert d5["composition_spec"]["downstream_consumes_upstream_artifact"]
            assert d5["difficulty_realization"]["distractor_count"] == 3

    def test_training_and_evaluation_fixtures_are_disjoint_across_100_seeds(
        self,
    ) -> None:
        generator = get_generator("composite_workspace")
        training_ids: set[str] = set()
        evaluation_ids: set[str] = set()
        fingerprints: set[str] = set()
        with workspace_tempdir() as raw:
            root = Path(raw)
            for split, target in (
                ("training", training_ids),
                ("evaluation", evaluation_ids),
            ):
                for seed in range(100):
                    bundle = generator.generate_instance(
                        generator.sample_spec(difficulty=5, seed=seed, split=split),
                        root / split / str(seed),
                        validate=False,
                    )
                    target.add(str(bundle.manifest.metadata["split_fixture_id"]))
                    fingerprints.add(fingerprint(bundle.visible_root))
            assert training_ids.isdisjoint(evaluation_ids)
            assert len(training_ids) == len(evaluation_ids) == 100
            assert len(fingerprints) == 200

    def test_reference_public_and_hidden_checks_pass(self) -> None:
        generator = get_generator("composite_workspace")
        with workspace_tempdir() as raw:
            root = Path(raw)
            bundle = generator.generate_instance(
                generator.sample_spec(difficulty=5, seed=73, split="evaluation"),
                root / "generated",
            )
            evaluator = get_evaluator(bundle.manifest.family)
            solved = solve(bundle, root / "solved")
            public = subprocess.run(
                [sys.executable, "public_check.py"],
                cwd=solved,
                capture_output=True,
                text=True,
            )
            assert public.returncode == 0, public.stderr
            assert (
                evaluator.evaluate(solved, bundle.manifest, bundle.hidden_root).score
                == 1.0
            )
            untouched = subprocess.run(
                [sys.executable, "public_check.py"],
                cwd=bundle.visible_root,
                capture_output=True,
                text=True,
            )
            assert untouched.returncode != 0

    def test_each_injected_defect_independently_loses_capability_credit(self) -> None:
        generator = get_generator("composite_workspace")
        with workspace_tempdir() as raw:
            root = Path(raw)
            bundle = generator.generate_instance(
                generator.sample_spec(difficulty=5, seed=83, split="evaluation"),
                root / "generated",
            )
            evaluator = get_evaluator(bundle.manifest.family)
            solved = solve(bundle, root / "solved")
            assert (
                evaluator.evaluate(solved, bundle.manifest, bundle.hidden_root).score
                == 1.0
            )
            for index, defect in enumerate(generator.defects()):
                candidate = root / f"defect-{index}"
                shutil.copytree(solved, candidate)
                path = candidate / defect["path"]
                content = path.read_text(encoding="utf-8")
                assert defect["before"] in content
                path.write_text(
                    content.replace(defect["before"], defect["after"], 1),
                    encoding="utf-8",
                )
                result = evaluator.evaluate(
                    candidate, bundle.manifest, bundle.hidden_root
                )
                assert result.score < 1.0, defect["id"]

    def test_packaged_frozen_manifest_loads_exact_assignments(self) -> None:
        path = _resolve_frozen_manifest("train-all-family-seed-42", None)
        rows = _build_rows(
            split="train",
            family=None,
            scenario=None,
            difficulty=None,
            seed=None,
            families=None,
            difficulties=None,
            seeds=None,
            split_manifest_path=path,
            include_splits=None,
            exclude_splits=None,
            task_id=None,
            max_examples=-1,
            sample_strategy="first",
            shuffle=False,
            shuffle_seed=0,
        )
        assert len(rows) == 512
        assert len({row["task_id"] for row in rows}) == 512
        assert {row["family"] for row in rows} == {
            "tabular",
            "script_repair",
            "pipeline",
            "retrieval_workspace",
        }

    def test_prime_generation_propagates_split_and_task_identity(self) -> None:
        with workspace_tempdir() as raw:
            env = SyntheticWorkspacePrimeEnv(
                family="composite_workspace",
                scenario="retrieval_guided_pipeline_repair",
                difficulty=3,
                seed=11,
                split="heldout",
                task_id="frozen.composite.11",
                output_dir=Path(raw),
            )
            try:
                observation = env.reset()
                metadata = observation["metadata"]
                assert metadata["split"] == "heldout"
                assert metadata["task_id"] == "frozen.composite.11"
                assert str(metadata["split_fixture_id"]).startswith("heldout-11-")
            finally:
                env.close()
