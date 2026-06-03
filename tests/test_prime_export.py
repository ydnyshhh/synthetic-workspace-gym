from __future__ import annotations

import argparse
import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from test_support import workspace_tempdir

from synthetic_workspace_gym.cli import (
    build_parser,
    command_prime_verify,
    parse_comma_separated,
    parse_difficulty_spec,
    parse_int_list_or_range,
)
from synthetic_workspace_gym.generators.registry import get_generator
from synthetic_workspace_gym.prime.export import (
    TOOL_SCHEMA_VERSION,
    build_manifest_row,
    export_existing_environments,
    export_prime_pack,
    write_manifest_jsonl,
    write_metadata_json,
)


class PrimeExportTests(unittest.TestCase):
    def test_parse_helpers(self) -> None:
        self.assertEqual(parse_comma_separated("tabular, script_repair"), ["tabular", "script_repair"])
        self.assertEqual(parse_int_list_or_range("0:3"), [0, 1, 2])
        self.assertEqual(parse_int_list_or_range("1,3,5"), [1, 3, 5])
        self.assertEqual(parse_difficulty_spec("1:3"), [1, 2, 3])
        self.assertEqual(parse_difficulty_spec("2,4"), [2, 4])

    def test_build_manifest_row_on_generated_environment(self) -> None:
        with workspace_tempdir() as tmp_dir:
            root = Path(tmp_dir)
            env_path = self._generate_environment(root / "generated")
            export_root = root / "export"
            target = export_root / "environments" / env_path.name
            self._copytree(env_path, target)

            row = build_manifest_row(target, export_root)

        self.assertEqual(row["family"], "script_repair")
        self.assertEqual(row["scenario"], "csv_schema_drift")
        self.assertEqual(row["difficulty"], 1)
        self.assertEqual(row["seed"], 7)
        self.assertTrue(str(row["environment_path"]).startswith("environments/"))
        self.assertEqual(row["tool_schema_version"], TOOL_SCHEMA_VERSION)
        self.assertIn("read_file", row["tool_permissions"])
        self.assertIn("hidden-verifier", row["tags"])
        self.assertIn("visible_files", row)
        self.assertIn("hidden_files", row)

    def test_write_manifest_jsonl_writes_valid_jsonl(self) -> None:
        with workspace_tempdir() as tmp_dir:
            path = Path(tmp_dir) / "manifest.jsonl"
            write_manifest_jsonl(path, [{"task_id": "a"}, {"task_id": "b"}])

            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(rows, [{"task_id": "a"}, {"task_id": "b"}])

    def test_write_metadata_json_writes_expected_fields(self) -> None:
        with workspace_tempdir() as tmp_dir:
            path = Path(tmp_dir) / "metadata.json"
            write_metadata_json(
                path,
                [
                    {
                        "family": "tabular",
                        "difficulty": 2,
                        "seed": 3,
                    }
                ],
            )

            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(payload["name"], "synthetic-workspace-gym-prime-export")
        self.assertEqual(payload["environment_count"], 1)
        self.assertEqual(payload["families"], ["tabular"])
        self.assertEqual(payload["difficulties"], [2])
        self.assertEqual(payload["seeds"], [3])
        self.assertEqual(payload["tool_schema_version"], TOOL_SCHEMA_VERSION)

    def test_export_existing_environments_copies_env_dirs(self) -> None:
        with workspace_tempdir() as tmp_dir:
            root = Path(tmp_dir)
            env_path = self._generate_environment(root / "generated")
            summary = export_existing_environments(
                existing_environments_dir=root / "generated",
                output_dir=root / "prime_export",
                overwrite=True,
            )

            exported_env = Path(summary["export_root"]) / "environments" / env_path.name

        self.assertEqual(summary["environment_count"], 1)
        self.assertEqual(summary["errors"], [])
        self.assertTrue((exported_env / "manifest.json").exists())
        self.assertTrue((exported_env / "visible").is_dir())
        self.assertTrue((exported_env / "hidden").is_dir())

    def test_export_prime_pack_returns_summary_and_rows(self) -> None:
        with workspace_tempdir() as tmp_dir:
            root = Path(tmp_dir)
            summary = export_prime_pack(
                output_dir=root / "pack",
                families=("script_repair",),
                scenarios={"script_repair": ("csv_schema_drift",)},
                difficulties=(1,),
                seeds=(7,),
                overwrite=True,
            )
            manifest_path = Path(summary["manifest_path"])
            rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(summary["environment_count"], 1)
        self.assertEqual(summary["errors"], [])
        self.assertEqual(rows[0]["family"], "script_repair")
        self.assertEqual(rows[0]["scenario"], "csv_schema_drift")
        self.assertEqual(rows[0]["reward_type"], "hidden_evaluator")
        self.assertEqual(rows[0]["interaction_type"], "multi_turn_tool_use")

    def test_prime_cli_subcommands_are_registered(self) -> None:
        parser = build_parser()

        export_args = parser.parse_args(
            [
                "prime",
                "export",
                "--output-dir",
                "prime_exports/smoke",
                "--families",
                "tabular",
                "--difficulties",
                "1",
                "--seeds",
                "0:1",
            ]
        )
        verify_args = parser.parse_args(
            [
                "prime",
                "verify",
                "--environment",
                "env",
                "--workspace",
                "env/visible",
            ]
        )
        manifest_args = parser.parse_args(
            [
                "prime",
                "manifest",
                "--environments",
                "pack/environments",
                "--output",
                "pack/manifest.jsonl",
            ]
        )
        smoke_args = parser.parse_args(
            [
                "prime",
                "smoke-test",
                "--environment",
                "pack/environments/env",
            ]
        )

        self.assertEqual(export_args.command, "prime")
        self.assertEqual(export_args.prime_command, "export")
        self.assertEqual(verify_args.prime_command, "verify")
        self.assertEqual(manifest_args.prime_command, "manifest")
        self.assertEqual(smoke_args.prime_command, "smoke-test")

    def test_verify_command_uses_prime_verifier_adapter(self) -> None:
        with patch(
            "synthetic_workspace_gym.cli.verify_workspace",
            return_value={"reward": 1.0, "success": True},
        ) as verifier:
            with redirect_stdout(io.StringIO()) as stdout:
                exit_code = command_prime_verify(
                    argparse.Namespace(
                        environment=Path("environment"),
                        workspace=Path("workspace"),
                    )
                )

        self.assertEqual(exit_code, 0)
        verifier.assert_called_once_with(Path("environment"), Path("workspace"))
        self.assertEqual(json.loads(stdout.getvalue()), {"reward": 1.0, "success": True})

    def _generate_environment(self, output_dir: Path) -> Path:
        generator = get_generator("script_repair")
        spec = generator.sample_spec(difficulty=1, seed=7, scenario_id="csv_schema_drift")
        bundle = generator.generate_instance(spec, output_dir)
        return bundle.root

    def _copytree(self, source: Path, target: Path) -> None:
        import shutil

        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target)


if __name__ == "__main__":
    unittest.main()
