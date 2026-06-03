from __future__ import annotations

import json
import unittest
from pathlib import Path

from test_support import workspace_tempdir

from synthetic_workspace_gym.verifiers.dataset import SWGVerifiersDataset, load_from_prime_manifest


class VerifiersDatasetTests(unittest.TestCase):
    def test_dataset_yields_rows(self) -> None:
        dataset = SWGVerifiersDataset(
            families=("script_repair",),
            scenarios={"script_repair": ("csv_schema_drift",)},
            difficulties=(1,),
            seeds=(7,),
            split="eval",
        )
        rows = dataset.to_list()

        self.assertEqual(len(dataset), 1)
        self.assertEqual(rows[0]["task_id"], "swg.script_repair.csv_schema_drift.d1.s7")
        self.assertEqual(rows[0]["split"], "eval")
        self.assertIsNone(rows[0]["environment_path"])

    def test_load_from_prime_manifest_resolves_relative_environment_paths(self) -> None:
        with workspace_tempdir() as tmp_dir:
            root = Path(tmp_dir)
            manifest = root / "manifest.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "task_id": "swg.script_repair.csv_schema_drift.d1.s7",
                        "env_id": "env-1",
                        "family": "script_repair",
                        "scenario": "csv_schema_drift",
                        "difficulty": 1,
                        "seed": 7,
                        "instruction": "Fix it",
                        "environment_path": "environments/env-1",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            rows = load_from_prime_manifest(manifest).to_list()

        self.assertEqual(rows[0]["env_id"], "env-1")
        self.assertTrue(rows[0]["environment_path"].endswith(str(Path("environments") / "env-1")))
        self.assertEqual(rows[0]["instruction"], "Fix it")


if __name__ == "__main__":
    unittest.main()
