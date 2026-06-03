from __future__ import annotations

import unittest
from dataclasses import replace

import test_support  # noqa: F401

from synthetic_workspace_gym.splits import build_split_manifest, default_split_policy
from synthetic_workspace_gym.splits.validation import validate_split_manifest


class SplitValidationTests(unittest.TestCase):
    def test_valid_default_split_passes(self) -> None:
        specs = default_split_policy(families=("script_repair",))
        manifest = build_split_manifest("valid", specs, max_per_split={split: 2 for split in specs})

        payload = validate_split_manifest(manifest)

        self.assertTrue(payload["valid"], payload["errors"])
        self.assertEqual(payload["error_count"], 0)

    def test_duplicate_task_id_fails(self) -> None:
        specs = default_split_policy(families=("script_repair",))
        manifest = build_split_manifest("duplicate-task", specs, max_per_split={"train": 2})
        manifest.assignments[1] = replace(manifest.assignments[1], task_id=manifest.assignments[0].task_id)

        payload = validate_split_manifest(manifest)

        self.assertFalse(payload["valid"])
        self.assertTrue(any("Duplicate task_id" in error for error in payload["errors"]))

    def test_overlapping_tuple_across_splits_fails(self) -> None:
        specs = default_split_policy(families=("script_repair",))
        manifest = build_split_manifest("overlap", specs, max_per_split={"train": 1, "validation": 1})
        train = manifest.assignments[0]
        manifest.assignments[1] = replace(
            manifest.assignments[1],
            family=train.family,
            scenario=train.scenario,
            difficulty=train.difficulty,
            seed=train.seed,
        )

        payload = validate_split_manifest(manifest)

        self.assertFalse(payload["valid"])
        self.assertTrue(any("Duplicate environment tuple" in error for error in payload["errors"]))

    def test_heldout_scenario_leakage_fails(self) -> None:
        specs = default_split_policy(families=("script_repair",))
        manifest = build_split_manifest("heldout-leak", specs, max_per_split={"train": 1, "heldout": 1})
        heldout = next(assignment for assignment in manifest.assignments if assignment.split == "heldout")
        train_index = next(index for index, assignment in enumerate(manifest.assignments) if assignment.split == "train")
        manifest.assignments[train_index] = replace(
            manifest.assignments[train_index],
            scenario=heldout.scenario,
            seed=0,
            difficulty=1,
            task_id="swg.train.script_repair.leaked_heldout.d1.s0",
        )

        payload = validate_split_manifest(manifest)

        self.assertFalse(payload["valid"])
        self.assertTrue(any("Heldout scenarios appear" in error for error in payload["errors"]))


if __name__ == "__main__":
    unittest.main()
