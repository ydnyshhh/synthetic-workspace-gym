from __future__ import annotations

import json
import unittest
from pathlib import Path

from test_support import workspace_tempdir

from synthetic_workspace_gym.prime.export import build_manifest_row, export_split_pack
from synthetic_workspace_gym.splits import build_split_manifest, default_split_policy


class SplitExportTests(unittest.TestCase):
    def test_export_split_pack_writes_split_artifacts(self) -> None:
        specs = default_split_policy(families=("script_repair",))
        manifest = build_split_manifest(
            "split-smoke",
            specs,
            max_per_split={"train": 1, "validation": 1, "test": 1, "heldout": 1},
        )
        with workspace_tempdir() as tmp_dir:
            root = Path(tmp_dir)
            summary = export_split_pack(root / "pack", split_manifest=manifest, overwrite=True)
            rows = [
                json.loads(line)
                for line in Path(summary["manifest_path"]).read_text(encoding="utf-8").splitlines()
            ]
            metadata = json.loads(Path(summary["metadata_path"]).read_text(encoding="utf-8"))

        self.assertEqual(summary["environment_count"], 4)
        self.assertEqual(summary["splits"], {"train": 1, "validation": 1, "test": 1, "heldout": 1})
        self.assertEqual(metadata["splits"]["heldout"], 1)
        self.assertTrue(all(row["split"] in {"train", "validation", "test", "heldout"} for row in rows))
        self.assertTrue(all(str(row["environment_path"]).startswith(f"environments/{row['split']}/") for row in rows))

    def test_build_manifest_row_preserves_split_metadata_task_id(self) -> None:
        specs = default_split_policy(families=("script_repair",))
        manifest = build_split_manifest("split-smoke", specs, max_per_split={"train": 1})
        with workspace_tempdir() as tmp_dir:
            root = Path(tmp_dir)
            summary = export_split_pack(root / "pack", split_manifest=manifest, overwrite=True)
            env_path = next((root / "pack" / "environments" / "train").iterdir())
            row = build_manifest_row(env_path, root / "pack")

        self.assertEqual(row["split"], "train")
        self.assertTrue(str(row["task_id"]).startswith("swg.train.script_repair."))


if __name__ == "__main__":
    unittest.main()
