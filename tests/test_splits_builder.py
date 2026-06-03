from __future__ import annotations

import unittest
from pathlib import Path

from test_support import workspace_tempdir

from synthetic_workspace_gym.splits import build_split_assignments, build_split_manifest, default_split_policy
from synthetic_workspace_gym.splits.manifest import read_split_jsonl, read_split_manifest, write_split_jsonl, write_split_manifest
from synthetic_workspace_gym.splits.validation import validate_split_manifest


class SplitBuilderTests(unittest.TestCase):
    def test_build_assignments_are_split_prefixed_and_limited(self) -> None:
        specs = default_split_policy(families=("script_repair",))
        assignments = build_split_assignments(specs, max_per_split={"train": 2, "validation": 1})

        self.assertEqual([item.split for item in assignments[:2]], ["train", "train"])
        self.assertTrue(assignments[0].task_id.startswith("swg.train.script_repair."))
        self.assertEqual(sum(1 for item in assignments if item.split == "train"), 2)
        self.assertEqual(sum(1 for item in assignments if item.split == "validation"), 1)

    def test_build_manifest_validates_and_round_trips(self) -> None:
        specs = default_split_policy(families=("script_repair",))
        manifest = build_split_manifest("smoke", specs, max_per_split={split: 2 for split in specs})
        validation = validate_split_manifest(manifest)

        self.assertTrue(validation["valid"], validation["errors"])
        with workspace_tempdir() as tmp_dir:
            root = Path(tmp_dir)
            manifest_path = write_split_manifest(root / "split_manifest.json", manifest)
            assignments_path = write_split_jsonl(root / "split_assignments.jsonl", manifest.assignments)
            restored_manifest = read_split_manifest(manifest_path)
            restored_assignments = read_split_jsonl(assignments_path)

        self.assertEqual(restored_manifest.name, "smoke")
        self.assertEqual(len(restored_assignments), len(manifest.assignments))


if __name__ == "__main__":
    unittest.main()
