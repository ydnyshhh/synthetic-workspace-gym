from __future__ import annotations

import unittest
from pathlib import Path

from test_support import workspace_tempdir

from synthetic_workspace_gym.prime.dataset import SyntheticWorkspacePrimeDataset
from synthetic_workspace_gym.splits import build_split_manifest, default_split_policy, write_split_manifest


class PrimeDatasetSplitTests(unittest.TestCase):
    def test_default_split_dataset_uses_policy(self) -> None:
        dataset = SyntheticWorkspacePrimeDataset(families=("script_repair",), split="heldout")
        rows = dataset.to_list()

        self.assertTrue(rows)
        self.assertEqual({row["split"] for row in rows}, {"heldout"})
        self.assertTrue(all(str(row["task_id"]).startswith("swg.heldout.") for row in rows))
        self.assertEqual({row["scenario"] for row in rows}, {"team_roster_export"})

    def test_dataset_loads_and_filters_split_manifest(self) -> None:
        specs = default_split_policy(families=("script_repair",))
        manifest = build_split_manifest("smoke", specs, max_per_split={split: 1 for split in specs})
        with workspace_tempdir() as tmp_dir:
            path = write_split_manifest(Path(tmp_dir) / "split_manifest.json", manifest)
            rows = SyntheticWorkspacePrimeDataset(
                split_manifest_path=path,
                include_splits=("train", "test"),
            ).to_list()

        self.assertEqual({row["split"] for row in rows}, {"train", "test"})
        self.assertTrue(all(row["task_id"] and row["split"] for row in rows))

    def test_exclude_splits_filters_split_manifest(self) -> None:
        specs = default_split_policy(families=("script_repair",))
        manifest = build_split_manifest("smoke", specs, max_per_split={split: 1 for split in specs})
        with workspace_tempdir() as tmp_dir:
            path = write_split_manifest(Path(tmp_dir) / "split_manifest.json", manifest)
            rows = SyntheticWorkspacePrimeDataset(
                split_manifest_path=path,
                exclude_splits=("heldout", "validation"),
            ).to_list()

        self.assertEqual({row["split"] for row in rows}, {"train", "test"})


if __name__ == "__main__":
    unittest.main()
