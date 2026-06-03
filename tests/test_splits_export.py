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
            pack_root = root / "pack"
            layout_paths = {
                "metadata": pack_root / "metadata.json",
                "manifest": pack_root / "manifest.jsonl",
                "split_manifest": pack_root / "split_manifest.json",
                "split_assignments": pack_root / "split_assignments.jsonl",
                "train": pack_root / "environments" / "train",
                "validation": pack_root / "environments" / "validation",
                "test": pack_root / "environments" / "test",
                "heldout": pack_root / "environments" / "heldout",
            }
            rows = [
                json.loads(line)
                for line in Path(summary["manifest_path"]).read_text(encoding="utf-8").splitlines()
            ]
            metadata = json.loads(Path(summary["metadata_path"]).read_text(encoding="utf-8"))
            layout_exists = {name: path.exists() for name, path in layout_paths.items()}

        self.assertEqual(summary["environment_count"], 4)
        self.assertTrue(all(layout_exists.values()), layout_exists)
        self.assertEqual(summary["splits"], {"train": 1, "validation": 1, "test": 1, "heldout": 1})
        self.assertEqual(metadata["splits"]["heldout"], 1)
        self.assertTrue(all(row["split"] in {"train", "validation", "test", "heldout"} for row in rows))
        self.assertTrue(all(str(row["task_id"]).startswith(f"swg.{row['split']}.") for row in rows))
        self.assertTrue(all(str(row["environment_path"]).startswith(f"environments/{row['split']}/") for row in rows))

    def test_build_manifest_row_preserves_split_metadata_task_id(self) -> None:
        specs = default_split_policy(families=("script_repair",))
        manifest = build_split_manifest("split-smoke", specs, max_per_split={"train": 1})
        with workspace_tempdir() as tmp_dir:
            root = Path(tmp_dir)
            summary = export_split_pack(root / "pack", split_manifest=manifest, overwrite=True)
            env_path = next((root / "pack" / "environments" / "train").iterdir())
            row = build_manifest_row(env_path, root / "pack")
            env_manifest = json.loads((env_path / "manifest.json").read_text(encoding="utf-8"))

        self.assertEqual(row["split"], "train")
        self.assertTrue(str(row["task_id"]).startswith("swg.train.script_repair."))
        self.assertEqual(env_manifest["metadata"]["split"], "train")
        self.assertEqual(env_manifest["metadata"]["task_id"], row["task_id"])


if __name__ == "__main__":
    unittest.main()
