from __future__ import annotations

import unittest
from pathlib import Path

from test_support import workspace_tempdir

from synthetic_workspace_gym.prime.dataset import SyntheticWorkspacePrimeDataset
from synthetic_workspace_gym.splits import (
    build_split_manifest,
    default_split_policy,
    write_split_manifest,
)


class PrimeDatasetSplitTests(unittest.TestCase):
    def test_default_split_dataset_uses_policy(self) -> None:
        dataset = SyntheticWorkspacePrimeDataset(
            families=("script_repair",), split="heldout"
        )
        rows = dataset.to_list()

        self.assertTrue(rows)
        self.assertEqual({row["split"] for row in rows}, {"heldout"})
        self.assertTrue(
            all(str(row["task_id"]).startswith("swg.heldout.") for row in rows)
        )
        self.assertEqual({row["scenario"] for row in rows}, {"team_roster_export"})

    def test_d5_retrieval_resolves_client_adapter_sync(self) -> None:
        rows = SyntheticWorkspacePrimeDataset(
            families=("retrieval_workspace",),
            difficulties=(5,),
            seeds=(100,),
            split="test",
        ).to_list()
        self.assertEqual(rows[0]["scenario"], "client_adapter_sync")

    def test_d1_discovery_does_not_control_d5_scenario(self) -> None:
        rows = SyntheticWorkspacePrimeDataset(
            families=("retrieval_workspace",),
            difficulties=(1, 5),
            seeds=(101,),
            split="test",
        ).to_list()
        by_difficulty = {int(row["difficulty"]): row["scenario"] for row in rows}
        self.assertEqual(by_difficulty[1], "service_config_reconciliation")
        self.assertEqual(by_difficulty[5], "client_adapter_sync")

    def test_explicit_retrieval_scenario_is_preserved(self) -> None:
        rows = SyntheticWorkspacePrimeDataset(
            families=("retrieval_workspace",),
            scenarios={"retrieval_workspace": ("service_config_reconciliation",)},
            difficulties=(5,),
            seeds=(100,),
            split="test",
        ).to_list()
        self.assertEqual(rows[0]["scenario"], "service_config_reconciliation")

    def test_max_examples_head_samples_seeds_not_first_scenario(self) -> None:
        rows = SyntheticWorkspacePrimeDataset(
            families=("retrieval_workspace",),
            difficulties=(5,),
            seeds=(100, 101, 102, 103, 104),
            split="test",
        ).to_list()[:5]
        self.assertEqual([row["seed"] for row in rows], [100, 101, 102, 103, 104])
        self.assertEqual(
            {row["scenario"] for row in rows},
            {"client_adapter_sync", "client_adapter_policy_sync"},
        )

    def test_task_id_contains_resolved_scenario(self) -> None:
        row = SyntheticWorkspacePrimeDataset(
            families=("retrieval_workspace",),
            difficulties=(5,),
            seeds=(104,),
            split="test",
        ).to_list()[0]
        self.assertEqual(
            row["task_id"],
            "swg.test.retrieval_workspace.client_adapter_policy_sync.d5.d5_b.s104",
        )

    def test_dataset_loads_and_filters_split_manifest(self) -> None:
        specs = default_split_policy(families=("script_repair",))
        manifest = build_split_manifest(
            "smoke", specs, max_per_split={split: 1 for split in specs}
        )
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
        manifest = build_split_manifest(
            "smoke", specs, max_per_split={split: 1 for split in specs}
        )
        with workspace_tempdir() as tmp_dir:
            path = write_split_manifest(Path(tmp_dir) / "split_manifest.json", manifest)
            rows = SyntheticWorkspacePrimeDataset(
                split_manifest_path=path,
                exclude_splits=("heldout", "validation"),
            ).to_list()

        self.assertEqual({row["split"] for row in rows}, {"train", "test"})


if __name__ == "__main__":
    unittest.main()
