from __future__ import annotations

import json
import hashlib
import subprocess
import shutil
import sys
import unittest
from pathlib import Path

from test_support import workspace_tempdir
from synthetic_workspace_gym.evaluators.registry import get_evaluator

from synthetic_workspace_gym.generators.base import BaseGenerator, GeneratedPayload
from synthetic_workspace_gym.generators.common import (
    build_d5_composition_profile,
    select_visible_hints,
)
from synthetic_workspace_gym.generators.registry import get_generator
from synthetic_workspace_gym.generators.script_repair_quality import (
    evaluate_fix_lattice,
    validate_partial_solution_lattice,
)
from synthetic_workspace_gym.schemas import EnvironmentFamily, EnvironmentSpec


class GeneratorValidityTests(unittest.TestCase):
    def test_each_family_generates_structurally_valid_environment(self) -> None:
        for family in EnvironmentFamily:
            for difficulty in (1, 3, 5):
                with self.subTest(family=family.value, difficulty=difficulty):
                    with workspace_tempdir() as tmp_dir:
                        generator = get_generator(family)
                        spec = generator.sample_spec(difficulty=difficulty, seed=17)
                        bundle = generator.generate_instance(spec, Path(tmp_dir))
                        self.assertTrue((bundle.root / "manifest.json").exists())
                        self.assertTrue(bundle.manifest.visible_files)
                        self.assertTrue(bundle.manifest.hidden_files)
                        self.assertTrue(bundle.manifest.reference_solution["files"])
                        self.assertIn("complexity_profile", bundle.manifest.metadata)
                        for relative_path in bundle.manifest.visible_files:
                            self.assertTrue(
                                (bundle.visible_root / relative_path).exists()
                            )
                        for relative_path in bundle.manifest.hidden_files:
                            self.assertTrue(
                                (bundle.hidden_root / relative_path).exists()
                            )

    def test_d5_distribution_is_exactly_half_compositional_across_families(
        self,
    ) -> None:
        atomic_families = [
            family
            for family in EnvironmentFamily
            if family != EnvironmentFamily.COMPOSITE_WORKSPACE
        ]
        for family in atomic_families:
            profiles = [
                build_d5_composition_profile(family, 5, seed) for seed in range(100)
            ]
            self.assertEqual(
                sum(
                    profile["composition_mode"] == "hard_atomic" for profile in profiles
                ),
                50,
            )
            self.assertEqual(
                sum(
                    profile["composition_mode"] == "compositional"
                    for profile in profiles
                ),
                50,
            )
            self.assertTrue(
                all(
                    len(profile["source_families"]) == int(profile["composition_depth"])
                    for profile in profiles
                )
            )
            self.assertEqual(build_d5_composition_profile(family, 4, 91), {})

        composite = build_d5_composition_profile(
            EnvironmentFamily.COMPOSITE_WORKSPACE, 5, 91
        )
        self.assertEqual(composite["composition_mode"], "compositional")
        self.assertEqual(
            composite["source_families"], ["retrieval_workspace", "pipeline"]
        )
        self.assertEqual(composite["composition_depth"], 2)
        self.assertEqual(
            build_d5_composition_profile(EnvironmentFamily.COMPOSITE_WORKSPACE, 4, 91),
            {},
        )

        scenario_ids = {
            EnvironmentFamily.TABULAR: "monthly_segment_report",
            EnvironmentFamily.SCRIPT_REPAIR: "team_roster_export",
            EnvironmentFamily.PIPELINE: "team_hours_pipeline",
            EnvironmentFamily.RETRIEVAL_WORKSPACE: "migration_plan_bundle",
        }
        with workspace_tempdir() as tmp_dir:
            root = Path(tmp_dir)
            for family, scenario_id in scenario_ids.items():
                generator = get_generator(family)
                for seed, expected_mode in ((90, "hard_atomic"), (91, "compositional")):
                    with self.subTest(
                        family=family.value,
                        seed=seed,
                        expected_mode=expected_mode,
                    ):
                        bundle = generator.generate_instance(
                            generator.sample_spec(
                                difficulty=5,
                                seed=seed,
                                scenario_id=scenario_id,
                            ),
                            root / family.value / str(seed),
                        )
                        task = dict(bundle.manifest.metadata["task_descriptor"])
                        realized = dict(
                            bundle.manifest.metadata["difficulty_realization"]
                        )
                        self.assertEqual(task["composition_mode"], expected_mode)
                        self.assertEqual(realized["composition_mode"], expected_mode)
                        self.assertEqual(
                            task["source_families"], realized["source_families"]
                        )
                        self.assertEqual(
                            int(task["composition_depth"]),
                            len(task["source_families"]),
                        )
                        evidence_paths = list(
                            task.get("composition_evidence_paths", [])
                        )
                        if expected_mode == "compositional":
                            self.assertEqual(len(task["source_families"]), 2)
                            self.assertTrue(evidence_paths)
                            self.assertTrue(
                                all(
                                    (bundle.visible_root / path).is_file()
                                    for path in evidence_paths
                                )
                            )
                        else:
                            self.assertEqual(task["source_families"], [family.value])
                            self.assertFalse(evidence_paths)

    def test_level_five_removes_solution_guidance_and_records_realized_complexity(
        self,
    ) -> None:
        with workspace_tempdir() as tmp_dir:
            root = Path(tmp_dir)
            for family in EnvironmentFamily:
                with self.subTest(family=family.value):
                    generator = get_generator(family)
                    bundle = generator.generate_instance(
                        generator.sample_spec(difficulty=5, seed=29),
                        root / family.value,
                    )
                    task = dict(bundle.manifest.metadata["task_descriptor"])
                    realization = dict(
                        bundle.manifest.metadata["difficulty_realization"]
                    )

                    self.assertEqual(task["hints"], [])
                    self.assertEqual(realization["level"], 5)
                    self.assertEqual(realization["guidance"], "none")
                    self.assertEqual(realization["hint_count"], 0)
                    self.assertTrue(realization["discovery_required"])
                    self.assertGreater(int(realization["candidate_file_count"]), 0)

                    if family in {
                        EnvironmentFamily.SCRIPT_REPAIR,
                        EnvironmentFamily.PIPELINE,
                    }:
                        disclosed = set(task["target_files"])
                        actual = set(bundle.manifest.reference_solution["files"])
                        self.assertTrue(actual.issubset(disclosed))
                        self.assertEqual(realization["touched_file_count"], len(actual))
                        self.assertGreaterEqual(
                            realization["candidate_file_count"], len(actual)
                        )

    def test_hint_schedule_preserves_lower_levels(self) -> None:
        hints = ["one", "two", "three"]
        self.assertEqual(select_visible_hints(hints, 1), hints)
        self.assertEqual(select_visible_hints(hints, 2), hints)
        self.assertEqual(select_visible_hints(hints, 3), hints[:2])
        self.assertEqual(select_visible_hints(hints, 4), hints[:1])
        self.assertEqual(select_visible_hints(hints, 5), [])

    def test_generator_subclasses_must_define_family(self) -> None:
        with self.assertRaises(TypeError):

            class MissingFamilyGenerator(BaseGenerator):
                def build_environment(
                    self,
                    spec: EnvironmentSpec,
                    *,
                    root: Path,
                    visible_root: Path,
                    hidden_root: Path,
                ) -> GeneratedPayload:
                    raise NotImplementedError

    def test_generators_support_explicit_scenario_selection(self) -> None:
        expected = {
            "tabular": {
                "monthly_segment_report",
                "channel_status_pivot",
                "weekly_refund_rollup",
                "supplier_restock_summary",
            },
            "script_repair": {
                "inventory_report",
                "path_batch",
                "csv_schema_drift",
                "timestamp_normalization",
                "team_roster_export",
            },
            "pipeline": {
                "team_hours_pipeline",
                "sales_csv_pipeline",
                "artifact_stitch_pipeline",
                "quality_gate_pipeline",
            },
            "retrieval_workspace": {
                "service_config_reconciliation",
                "migration_plan_bundle",
                "incident_report_bundle",
                "client_adapter_sync",
            },
        }
        with workspace_tempdir() as tmp_dir:
            root = Path(tmp_dir)
            for family, scenario_ids in expected.items():
                generator = get_generator(family)
                for scenario_id in scenario_ids:
                    for difficulty in (3, 5):
                        with self.subTest(
                            family=family,
                            scenario_id=scenario_id,
                            difficulty=difficulty,
                        ):
                            spec = generator.sample_spec(
                                difficulty=difficulty,
                                seed=99,
                                scenario_id=scenario_id,
                            )
                            bundle = generator.generate_instance(
                                spec, root / family / f"d{difficulty}"
                            )
                            self.assertEqual(
                                str(bundle.manifest.metadata["scenario_id"]),
                                scenario_id,
                            )
                            self.assertEqual(
                                bundle.manifest.metadata["scenario_selection"][
                                    "selection_mode"
                                ],
                                "explicit",
                            )
                            if difficulty == 5:
                                task = dict(bundle.manifest.metadata["task_descriptor"])
                                realization = dict(
                                    bundle.manifest.metadata["difficulty_realization"]
                                )
                                self.assertEqual(task["hints"], [])
                                self.assertTrue(realization["discovery_required"])
                                if family in {"script_repair", "pipeline"}:
                                    self.assertTrue(
                                        set(
                                            bundle.manifest.reference_solution["files"]
                                        ).issubset(set(task["target_files"]))
                                    )

    def test_retrieval_workspace_metadata_contains_retrieval_fields(self) -> None:
        with workspace_tempdir() as tmp_dir:
            generator = get_generator("retrieval_workspace")
            spec = generator.sample_spec(
                difficulty=4,
                seed=31,
                scenario_id="service_config_reconciliation",
            )
            bundle = generator.generate_instance(spec, Path(tmp_dir))
            profile = dict(bundle.manifest.metadata["scenario_profile"])
            for key in (
                "task_type",
                "content_variant_id",
                "document_count",
                "retrieval_hops",
                "evidence_distribution",
                "distractor_count",
                "staleness_pattern",
                "output_style",
            ):
                self.assertIn(key, profile)
            document_files = [
                path
                for path in bundle.manifest.visible_files
                if path.startswith(("docs/", "notes/", "specs/", "logs/", "changelog/"))
            ]
            self.assertEqual(profile["document_count"], len(document_files))
            self.assertEqual(profile["staleness_pattern"], "stale_note")

    def test_retrieval_workspace_difficulty_increases_retrieval_complexity(
        self,
    ) -> None:
        with workspace_tempdir() as tmp_dir:
            root = Path(tmp_dir)
            generator = get_generator("retrieval_workspace")
            low = generator.generate_instance(
                generator.sample_spec(
                    difficulty=1,
                    seed=44,
                    scenario_id="service_config_reconciliation",
                ),
                root / "low",
            )
            high = generator.generate_instance(
                generator.sample_spec(
                    difficulty=5,
                    seed=44,
                    scenario_id="service_config_reconciliation",
                ),
                root / "high",
            )
            low_profile = dict(low.manifest.metadata["scenario_profile"])
            high_profile = dict(high.manifest.metadata["scenario_profile"])
            self.assertLess(
                int(low_profile["document_count"]), int(high_profile["document_count"])
            )
            self.assertLess(
                int(low_profile["retrieval_hops"]), int(high_profile["retrieval_hops"])
            )
            self.assertLess(
                int(low_profile["distractor_count"]),
                int(high_profile["distractor_count"]),
            )
            self.assertEqual(low_profile["staleness_pattern"], "none")
            self.assertEqual(high_profile["staleness_pattern"], "superseded_changelog")

    def test_retrieval_workspace_seed_changes_core_fixture_outputs(self) -> None:
        scenarios = (
            "service_config_reconciliation",
            "migration_plan_bundle",
            "incident_report_bundle",
        )
        with workspace_tempdir() as tmp_dir:
            root = Path(tmp_dir)
            generator = get_generator("retrieval_workspace")
            for scenario_id in scenarios:
                with self.subTest(scenario_id=scenario_id):
                    first = generator.generate_instance(
                        generator.sample_spec(
                            difficulty=3, seed=1, scenario_id=scenario_id
                        ),
                        root / f"{scenario_id}-seed-1",
                    )
                    second = generator.generate_instance(
                        generator.sample_spec(
                            difficulty=3, seed=2, scenario_id=scenario_id
                        ),
                        root / f"{scenario_id}-seed-2",
                    )
                    first_profile = dict(first.manifest.metadata["scenario_profile"])
                    second_profile = dict(second.manifest.metadata["scenario_profile"])
                    self.assertNotEqual(
                        first_profile["content_variant_id"],
                        second_profile["content_variant_id"],
                    )
                    self.assertNotEqual(
                        first.manifest.reference_solution["files"],
                        second.manifest.reference_solution["files"],
                    )

    def test_weekly_refund_contract_names_lowercase_region_normalization(self) -> None:
        with workspace_tempdir() as tmp_dir:
            generator = get_generator("tabular")
            spec = generator.sample_spec(
                difficulty=4, seed=102, scenario_id="weekly_refund_rollup"
            )
            bundle = generator.generate_instance(spec, Path(tmp_dir))

            readme = (bundle.visible_root / "README.md").read_text(encoding="utf-8")
            task = (bundle.visible_root / "task.json").read_text(encoding="utf-8")
            expected_rows = json.loads(
                bundle.manifest.reference_solution["files"][
                    "outputs/weekly_rollup.json"
                ]
            )
            expected_regions = {row["region"] for row in expected_rows}

            self.assertIn("lowercasing account lookup values", readme)
            self.assertIn("strip_lowercase", task)
            self.assertTrue(expected_regions)
            self.assertTrue(
                all(region == region.lower() for region in expected_regions)
            )

    def test_migration_plan_schema_spec_path_matches_selected_schema(self) -> None:
        with workspace_tempdir() as tmp_dir:
            generator = get_generator("retrieval_workspace")
            for seed in (1, 2, 3):
                with self.subTest(seed=seed):
                    spec = generator.sample_spec(
                        difficulty=5, seed=seed, scenario_id="migration_plan_bundle"
                    )
                    bundle = generator.generate_instance(
                        spec, Path(tmp_dir) / f"seed-{seed}"
                    )
                    task = bundle.manifest.metadata["task_descriptor"]
                    schema_version = str(task["schema_version"])
                    schema_path = str(task["schema_spec_path"])
                    readme = (bundle.visible_root / "README.md").read_text(
                        encoding="utf-8"
                    )

                    self.assertEqual(schema_path, f"specs/schema_{schema_version}.md")
                    self.assertTrue((bundle.visible_root / schema_path).exists())
                    self.assertIn(f"current schema `{schema_version}`", readme)
                    self.assertNotIn("current v3 schema", readme)

    def test_csv_schema_drift_readme_includes_repair_contract(self) -> None:
        with workspace_tempdir() as tmp_dir:
            generator = get_generator("script_repair")
            spec = generator.sample_spec(
                difficulty=2, seed=7, scenario_id="csv_schema_drift"
            )
            bundle = generator.generate_instance(spec, Path(tmp_dir))
            readme = (bundle.visible_root / "README.md").read_text(encoding="utf-8")
            task = bundle.manifest.metadata["task_descriptor"]

            self.assertIn("## Expected behavior", readme)
            self.assertIn("account_id", readme)
            self.assertIn("sorted lexicographically", readme)
            self.assertIn("repair_contract", task)

    def test_team_roster_d5_is_plausibly_wrong_but_publicly_executable(self) -> None:
        with workspace_tempdir() as tmp_dir:
            generator = get_generator("script_repair")
            spec = generator.sample_spec(
                difficulty=5, seed=91, scenario_id="team_roster_export"
            )
            bundle = generator.generate_instance(spec, Path(tmp_dir))

            broken = subprocess.run(
                [sys.executable, "run_example.py"],
                cwd=bundle.visible_root,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(broken.returncode, 0, broken.stderr)
            self.assertIsInstance(json.loads(broken.stdout), list)
            self.assertEqual(
                bundle.manifest.metadata["scenario_profile"]["smoke_test_quality"],
                "structural",
            )
            self.assertIn(
                "authoritative",
                (bundle.visible_root / "docs/roster-v2.md")
                .read_text(encoding="utf-8")
                .casefold(),
            )

            evaluator = get_evaluator(EnvironmentFamily.SCRIPT_REPAIR)
            partial = evaluator.evaluate(
                bundle.visible_root, bundle.manifest, bundle.hidden_root
            )
            self.assertGreaterEqual(partial.score, 0.0)
            self.assertLess(partial.score, 1.0)

            for relative_path, content in bundle.manifest.reference_solution[
                "files"
            ].items():
                target = bundle.visible_root / relative_path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")
            repaired = evaluator.evaluate(
                bundle.visible_root, bundle.manifest, bundle.hidden_root
            )
            self.assertEqual(repaired.score, 1.0)

    def test_script_repair_d5_uses_semantic_bundles_across_100_seeds(self) -> None:
        generator = get_generator("script_repair")
        scenario_ids = (
            "csv_schema_drift",
            "timestamp_normalization",
            "team_roster_export",
        )
        scenarios = {
            str(scenario["scenario_id"]): scenario
            for scenario in generator.scenario_pool()
            if scenario["scenario_id"] in scenario_ids
        }

        for scenario_id in scenario_ids:
            fingerprints: set[str] = set()
            base = scenarios[scenario_id]
            for seed in range(100):
                selected_by_difficulty: dict[int, list[dict[str, object]]] = {}
                realized_by_difficulty: dict[int, dict[str, object]] = {}
                selection_metadata: dict[int, dict[str, object]] = {}
                for difficulty in (4, 5):
                    spec = generator.sample_spec(
                        difficulty=difficulty,
                        seed=seed,
                        scenario_id=scenario_id,
                    )
                    realized = dict(base)
                    realized.update(dict(base["materialize"](spec)))
                    selected, metadata = generator.select_bugs(realized, spec)
                    selected_by_difficulty[difficulty] = selected
                    realized_by_difficulty[difficulty] = realized
                    selection_metadata[difficulty] = metadata

                d5_labels = {str(bug["label"]) for bug in selected_by_difficulty[5]}
                self.assertNotIn("syntax_error", d5_labels)
                self.assertEqual(selection_metadata[4]["core_bug_count"], 3)
                self.assertEqual(selection_metadata[4]["advanced_bug_count"], 0)
                self.assertIsNotNone(selection_metadata[5]["bug_bundle_id"])
                self.assertGreaterEqual(len(d5_labels), 4)
                self.assertGreaterEqual(
                    len(selection_metadata[5]["dependency_edges"]), 3
                )
                self.assertEqual(len(selection_metadata[5]["capabilities"]), 5)
                self.assertGreaterEqual(
                    selection_metadata[5]["semantic_dependency_depth"], 4
                )
                self.assertGreaterEqual(
                    len({str(bug["target_path"]) for bug in selected_by_difficulty[5]}),
                    3,
                )
                profile = dict(realized_by_difficulty[5]["structure"])
                self.assertEqual(profile["dependency_depth"], 3)
                self.assertEqual(profile["distractor_count"], 3)
                self.assertGreaterEqual(profile["hidden_capability_count"], 6)

                data_payload = "\n".join(
                    str(content)
                    for path, content in sorted(
                        dict(realized_by_difficulty[5]["files"]).items()
                    )
                    if str(path).startswith("data/") and "legacy" not in str(path)
                )
                fingerprints.add(
                    hashlib.sha256(data_payload.encode("utf-8")).hexdigest()
                )

                if scenario_id == "team_roster_export":
                    d4_runner = str(
                        dict(realized_by_difficulty[4]["files"])["run_example.py"]
                    )
                    d5_runner = str(
                        dict(realized_by_difficulty[5]["files"])["run_example.py"]
                    )
                    self.assertIn("sorted lexicographically by team", d4_runner)
                    self.assertNotIn("sorted lexicographically by team", d5_runner)

            self.assertEqual(len(fingerprints), 100)

        with workspace_tempdir() as tmp_dir:
            legacy = generator.generate_instance(
                generator.sample_spec(
                    difficulty=5,
                    seed=17,
                    scenario_id="inventory_report",
                ),
                Path(tmp_dir),
            )
            realization = dict(legacy.manifest.metadata["difficulty_realization"])
            self.assertFalse(realization["nested_bug_selection"])
            self.assertEqual(realization["advanced_bug_count"], 0)

    def test_each_quality_bug_fails_an_isolated_hidden_capability(self) -> None:
        generator = get_generator("script_repair")
        scenario_ids = (
            "csv_schema_drift",
            "timestamp_normalization",
            "team_roster_export",
        )
        scenarios = {
            str(scenario["scenario_id"]): scenario
            for scenario in generator.scenario_pool()
            if scenario["scenario_id"] in scenario_ids
        }
        evaluator = get_evaluator(EnvironmentFamily.SCRIPT_REPAIR)

        with workspace_tempdir() as tmp_dir:
            root = Path(tmp_dir)
            for scenario_id in scenario_ids:
                with self.subTest(scenario_id=scenario_id):
                    spec = generator.sample_spec(
                        difficulty=5,
                        seed=91,
                        scenario_id=scenario_id,
                    )
                    bundle = generator.generate_instance(
                        spec,
                        root / "bundles" / scenario_id,
                    )
                    realized = dict(scenarios[scenario_id])
                    realized.update(dict(scenarios[scenario_id]["materialize"](spec)))
                    all_bugs, _ = generator.select_bugs(realized, spec)
                    correct_files = dict(realized["files"])
                    for bug in all_bugs:
                        workspace = root / "isolated" / scenario_id / str(bug["label"])
                        shutil.copytree(bundle.visible_root, workspace)
                        for relative_path, content in correct_files.items():
                            target = workspace / str(relative_path)
                            target.parent.mkdir(parents=True, exist_ok=True)
                            target.write_text(str(content), encoding="utf-8")
                        target_path = str(bug["target_path"])
                        target = workspace / target_path
                        target.write_text(
                            bug["apply"](target.read_text(encoding="utf-8")),
                            encoding="utf-8",
                        )
                        result = evaluator.evaluate(
                            workspace,
                            bundle.manifest,
                            bundle.hidden_root,
                        )
                        self.assertLess(result.score, 1.0, str(bug["label"]))
                        self.assertGreater(
                            result.subscores["tests_passed"],
                            0.0,
                            str(bug["label"]),
                        )

    def test_script_repair_d5_partial_solution_lattices(self) -> None:
        generator = get_generator("script_repair")
        evaluator = get_evaluator(EnvironmentFamily.SCRIPT_REPAIR)
        scenarios = {
            str(scenario["scenario_id"]): scenario
            for scenario in generator.scenario_pool()
        }
        with workspace_tempdir() as tmp_dir:
            root = Path(tmp_dir)
            for scenario_id in (
                "csv_schema_drift",
                "timestamp_normalization",
                "team_roster_export",
            ):
                with self.subTest(scenario_id=scenario_id):
                    spec = generator.sample_spec(
                        difficulty=5,
                        seed=91,
                        scenario_id=scenario_id,
                        generation_params={"composition_mode": "hard_atomic"},
                    )
                    bundle = generator.generate_instance(
                        spec, root / "bundles" / scenario_id
                    )
                    realized = dict(scenarios[scenario_id])
                    realized.update(dict(realized["materialize"](spec)))
                    selected_bugs, _ = generator.select_bugs(realized, spec)
                    scores = evaluate_fix_lattice(
                        correct_files=dict(realized["files"]),
                        selected_bugs=selected_bugs,
                        visible_template=bundle.visible_root,
                        scratch_root=root / "lattices" / scenario_id,
                        evaluator=evaluator,
                        manifest=bundle.manifest,
                        hidden_root=bundle.hidden_root,
                    )
                    profile = validate_partial_solution_lattice(
                        scores,
                        [str(bug["label"]) for bug in selected_bugs],
                        thresholds=dict(realized["partial_solution_lattice"]),
                    )
                    self.assertTrue(profile["valid"])
                    self.assertEqual(profile["full_solution_score"], 1.0)
                    self.assertLessEqual(profile["single_fix_max_score"], 0.4)
                    self.assertLessEqual(profile["pair_fix_max_score"], 0.65)

    def test_retrieval_d5_conflicts_require_authority_chain(self) -> None:
        scenarios = (
            "service_config_reconciliation",
            "migration_plan_bundle",
            "incident_report_bundle",
            "client_adapter_sync",
        )
        with workspace_tempdir() as tmp_dir:
            generator = get_generator("retrieval_workspace")
            for scenario_id in scenarios:
                with self.subTest(scenario_id=scenario_id):
                    bundle = generator.generate_instance(
                        generator.sample_spec(
                            difficulty=5, seed=91, scenario_id=scenario_id
                        ),
                        Path(tmp_dir) / scenario_id,
                    )
                    files = bundle.manifest.visible_files
                    profile = bundle.manifest.metadata["scenario_profile"]
                    index = (
                        bundle.visible_root / "changelog/evidence_index.md"
                    ).read_text(encoding="utf-8")
                    active_bundle = next(
                        line.split("`")[1]
                        for line in index.splitlines()
                        if "active evidence bundle" in line
                    )
                    archived_bundle = next(
                        line.split("`")[1]
                        for line in index.splitlines()
                        if "archived evidence bundle" in line
                    )
                    tagged_active = [
                        path
                        for path in files
                        if (bundle.visible_root / path).is_file()
                        and f"Evidence bundle: `{active_bundle}`"
                        in (bundle.visible_root / path).read_text(encoding="utf-8")
                    ]
                    tagged_archived = [
                        path
                        for path in files
                        if (bundle.visible_root / path).is_file()
                        and f"Evidence bundle: `{archived_bundle}`"
                        in (bundle.visible_root / path).read_text(encoding="utf-8")
                    ]

                    self.assertIn("docs/evidence_precedence.md", files)
                    self.assertEqual(profile["authority_chain_depth"], 4)
                    self.assertEqual(profile["conflict_count"], 1)
                    self.assertTrue(profile["conflict_resolution_required"])
                    self.assertGreaterEqual(len(tagged_active), 3)
                    self.assertEqual(len(tagged_archived), 1)

        with workspace_tempdir() as tmp_dir:
            generator = get_generator("retrieval_workspace")
            bundle = generator.generate_instance(
                generator.sample_spec(
                    difficulty=4,
                    seed=91,
                    scenario_id="service_config_reconciliation",
                ),
                Path(tmp_dir),
            )
            self.assertNotIn(
                "changelog/evidence_index.md", bundle.manifest.visible_files
            )
            self.assertEqual(
                bundle.manifest.metadata["scenario_profile"]["authority_chain_depth"],
                2,
            )


if __name__ == "__main__":
    unittest.main()
