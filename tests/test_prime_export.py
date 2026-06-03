from __future__ import annotations

import argparse
import io
import json
import subprocess
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from test_support import workspace_tempdir

from synthetic_workspace_gym.cli import (
    build_parser,
    command_prime_manifest,
    command_prime_rollout,
    command_prime_rollout_batch,
    command_prime_verify,
    command_sandbox_check,
    command_verifiers_check,
    command_verifiers_export_registry,
    command_verifiers_list,
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
        self.assertTrue(row["sandbox_compatible"])
        self.assertEqual(row["recommended_sandbox_backend"], "docker")
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
        self.assertEqual(payload["recommended_sandbox"]["backend"], "docker")
        self.assertEqual(payload["recommended_sandbox"]["hidden_mount_policy"], "evaluator_only_read_only")

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
            manifest_exists = (exported_env / "manifest.json").exists()
            visible_exists = (exported_env / "visible").is_dir()
            hidden_exists = (exported_env / "hidden").is_dir()

        self.assertEqual(summary["environment_count"], 1)
        self.assertEqual(summary["errors"], [])
        self.assertTrue(manifest_exists)
        self.assertTrue(visible_exists)
        self.assertTrue(hidden_exists)

    def test_export_existing_environments_rejects_source_inside_output_when_overwriting(self) -> None:
        with workspace_tempdir() as tmp_dir:
            root = Path(tmp_dir)

            with self.assertRaisesRegex(ValueError, "existing_environments_dir cannot be inside output_dir"):
                export_existing_environments(
                    existing_environments_dir=root / "pack" / "environments",
                    output_dir=root / "pack",
                    overwrite=True,
                )

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

    def test_export_prime_pack_uses_exact_custom_dataset_rows(self) -> None:
        class CustomDataset:
            def to_list(self) -> list[dict[str, object]]:
                return [
                    {
                        "family": "script_repair",
                        "scenario": "csv_schema_drift",
                        "difficulty": 1,
                        "seed": 7,
                    },
                    {
                        "family": "script_repair",
                        "scenario": "csv_schema_drift",
                        "difficulty": 2,
                        "seed": 8,
                    },
                ]

        with workspace_tempdir() as tmp_dir:
            root = Path(tmp_dir)
            summary = export_prime_pack(
                output_dir=root / "pack",
                dataset=CustomDataset(),
                overwrite=True,
            )
            manifest_path = Path(summary["manifest_path"])
            rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(summary["environment_count"], 2)
        self.assertEqual(
            {row["task_id"] for row in rows},
            {
                "swg.script_repair.csv_schema_drift.d1.s7",
                "swg.script_repair.csv_schema_drift.d2.s8",
            },
        )

    def test_prime_manifest_uses_environment_pack_root_for_relative_paths(self) -> None:
        with workspace_tempdir() as tmp_dir:
            root = Path(tmp_dir)
            env_path = self._generate_environment(root / "generated")
            pack_root = root / "pack"
            target = pack_root / "environments" / env_path.name
            self._copytree(env_path, target)
            output_path = root / "outside" / "manifest.jsonl"

            with redirect_stdout(io.StringIO()):
                exit_code = command_prime_manifest(
                    argparse.Namespace(
                        environments=pack_root / "environments",
                        output=output_path,
                    )
                )
            row = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])

        self.assertEqual(exit_code, 0)
        self.assertTrue(row["environment_path"].startswith("environments/"))
        self.assertFalse(Path(row["environment_path"]).is_absolute())

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
        rollout_args = parser.parse_args(
            [
                "prime",
                "rollout",
                "--family",
                "script_repair",
                "--client",
                "scripted",
                "--sandbox",
                "local",
                "--sandbox-user",
                "123:456",
            ]
        )
        rollout_batch_args = parser.parse_args(
            [
                "prime",
                "rollout-batch",
                "--manifest",
                "pack/manifest.jsonl",
                "--client",
                "scripted",
                "--sandbox",
                "local",
            ]
        )
        sandbox_check_args = parser.parse_args(
            ["sandbox", "check", "--image", "swg:test", "--sandbox-user", "123:456"]
        )
        sandbox_run_args = parser.parse_args(
            ["sandbox", "run", "--command", "python --version", "--sandbox-user", "123:456"]
        )
        verifiers_list_args = parser.parse_args(["verifiers", "list"])
        verifiers_smoke_args = parser.parse_args(
            [
                "verifiers",
                "smoke-test",
                "--env-id",
                "swg.script_repair.csv_schema_drift",
                "--difficulty",
                "1",
                "--seed",
                "7",
            ]
        )

        self.assertEqual(export_args.command, "prime")
        self.assertEqual(export_args.prime_command, "export")
        self.assertEqual(verify_args.prime_command, "verify")
        self.assertEqual(manifest_args.prime_command, "manifest")
        self.assertEqual(smoke_args.prime_command, "smoke-test")
        self.assertEqual(rollout_args.prime_command, "rollout")
        self.assertEqual(rollout_args.sandbox_user, "123:456")
        self.assertEqual(rollout_batch_args.prime_command, "rollout-batch")
        self.assertEqual(sandbox_check_args.command, "sandbox")
        self.assertEqual(sandbox_check_args.sandbox_command, "check")
        self.assertEqual(sandbox_check_args.sandbox_user, "123:456")
        self.assertEqual(sandbox_run_args.command, "sandbox")
        self.assertEqual(sandbox_run_args.sandbox_command, "run")
        self.assertEqual(sandbox_run_args.sandbox_command_text, "python --version")
        self.assertEqual(sandbox_run_args.sandbox_user, "123:456")
        self.assertEqual(verifiers_list_args.command, "verifiers")
        self.assertEqual(verifiers_list_args.verifiers_command, "list")
        self.assertEqual(verifiers_smoke_args.verifiers_command, "smoke-test")
        self.assertEqual(verifiers_smoke_args.env_id, "swg.script_repair.csv_schema_drift")

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

    def test_sandbox_check_fails_when_image_is_missing(self) -> None:
        with patch("synthetic_workspace_gym.cli.docker_available", return_value=True):
            with patch(
                "synthetic_workspace_gym.cli.subprocess.run",
                return_value=subprocess.CompletedProcess(args=["docker"], returncode=1, stdout="", stderr="missing"),
            ):
                with redirect_stdout(io.StringIO()) as stdout:
                    exit_code = command_sandbox_check(argparse.Namespace(image="missing:image", sandbox_user=None))

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 1)
        self.assertTrue(payload["docker_available"])
        self.assertFalse(payload["image_available"])
        self.assertFalse(payload["sandbox_smoke_test"])

    def test_verifiers_list_and_check_commands_do_not_require_verifiers(self) -> None:
        with redirect_stdout(io.StringIO()) as list_stdout:
            list_exit = command_verifiers_list(argparse.Namespace())
        with redirect_stdout(io.StringIO()) as check_stdout:
            check_exit = command_verifiers_check(argparse.Namespace())

        self.assertEqual(list_exit, 0)
        self.assertEqual(check_exit, 0)
        self.assertIn("swg.script_repair.csv_schema_drift", json.loads(list_stdout.getvalue())["environments"])
        self.assertIn("verifiers_available", json.loads(check_stdout.getvalue()))

    def test_verifiers_export_registry_writes_metadata(self) -> None:
        with workspace_tempdir() as tmp_dir:
            output = Path(tmp_dir) / "registry.json"
            with redirect_stdout(io.StringIO()):
                exit_code = command_verifiers_export_registry(argparse.Namespace(output=output))
            payload = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertGreaterEqual(len(payload["environments"]), 4)

    def test_rollout_command_writes_artifact(self) -> None:
        with workspace_tempdir() as tmp_dir:
            root = Path(tmp_dir)
            with redirect_stdout(io.StringIO()) as stdout:
                exit_code = command_prime_rollout(
                    argparse.Namespace(
                        family="script_repair",
                        scenario="csv_schema_drift",
                        difficulty=1,
                        seed=7,
                        environment=None,
                        client="scripted",
                        action_json=[
                            '{"tool":"list_directory","args":{"path":"."}}',
                            '{"tool":"submit","args":{"path_or_answer":"done"}}',
                        ],
                        output_dir=root / "rollouts",
                        max_turns=None,
                        rollout_id="cli-rollout",
                    )
            )
            payload = json.loads(stdout.getvalue())
            artifact_exists = Path(payload["prime_rollout_path"]).exists()

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["rollout_id"], "cli-rollout")
        self.assertTrue(artifact_exists)

    def test_rollout_batch_command_writes_summary(self) -> None:
        with workspace_tempdir() as tmp_dir:
            root = Path(tmp_dir)
            manifest_path = root / "manifest.jsonl"
            manifest_path.write_text(
                json.dumps({"env_id": "missing-env", "environment_path": "environments/missing-env"}) + "\n",
                encoding="utf-8",
            )

            with redirect_stdout(io.StringIO()) as stdout:
                exit_code = command_prime_rollout_batch(
                    argparse.Namespace(
                        manifest=manifest_path,
                        client="scripted",
                        action_json=[],
                        limit=1,
                        output_dir=root / "rollouts",
                        max_turns=None,
                    )
                )
            payload = json.loads(stdout.getvalue())
            summary_exists = (root / "rollouts" / "batch_summary.json").exists()

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["count"], 1)
        self.assertTrue(summary_exists)
        self.assertFalse(payload["rollouts"][0]["success"])

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
