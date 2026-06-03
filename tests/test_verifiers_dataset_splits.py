from __future__ import annotations

import json
import unittest
from pathlib import Path

from test_support import workspace_tempdir

from synthetic_workspace_gym.splits import build_split_manifest, default_split_policy, write_split_manifest
from synthetic_workspace_gym.verifiers.dataset import SWGVerifiersDataset, load_from_prime_manifest


class VerifiersDatasetSplitTests(unittest.TestCase):
    def test_verifiers_dataset_loads_split_manifest(self) -> None:
        specs = default_split_policy(families=("script_repair",))
        manifest = build_split_manifest("smoke", specs, max_per_split={split: 1 for split in specs})
        with workspace_tempdir() as tmp_dir:
            path = write_split_manifest(Path(tmp_dir) / "split_manifest.json", manifest)
            rows = SWGVerifiersDataset(split_manifest_path=path, split="heldout").to_list()

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["split"], "heldout")
        self.assertTrue(rows[0]["question"])

    def test_verifiers_dataset_include_and_exclude_splits(self) -> None:
        specs = default_split_policy(families=("script_repair",))
        manifest = build_split_manifest("smoke", specs, max_per_split={split: 1 for split in specs})
        with workspace_tempdir() as tmp_dir:
            path = write_split_manifest(Path(tmp_dir) / "split_manifest.json", manifest)
            included = SWGVerifiersDataset(split_manifest_path=path, include_splits=("train", "heldout")).to_list()
            excluded = SWGVerifiersDataset(split_manifest_path=path, exclude_splits=("validation", "test")).to_list()

        self.assertEqual({row["split"] for row in included}, {"train", "heldout"})
        self.assertEqual({row["split"] for row in excluded}, {"train", "heldout"})

    def test_load_from_prime_manifest_preserves_split_and_task_id(self) -> None:
        with workspace_tempdir() as tmp_dir:
            root = Path(tmp_dir)
            manifest_path = root / "manifest.jsonl"
            manifest_path.write_text(
                json.dumps(
                    {
                        "task_id": "swg.train.script_repair.csv_schema_drift.d1.s7",
                        "split": "train",
                        "env_id": "env-1",
                        "family": "script_repair",
                        "scenario": "csv_schema_drift",
                        "difficulty": 1,
                        "seed": 7,
                        "instruction": "Fix it",
                        "environment_path": "environments/train/env-1",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            rows = load_from_prime_manifest(manifest_path).to_list()

        self.assertEqual(rows[0]["split"], "train")
        self.assertEqual(rows[0]["task_id"], "swg.train.script_repair.csv_schema_drift.d1.s7")


if __name__ == "__main__":
    unittest.main()
