from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

from synthetic_workspace_gym.evaluators.registry import get_evaluator
from synthetic_workspace_gym.generators.registry import get_generator
from synthetic_workspace_gym.generators.retrieval_profile_scenarios import (
    build_profiled_retrieval_scenario,
)
from synthetic_workspace_gym.generators.script_repair_quality import (
    evaluate_fix_lattice,
    validate_partial_solution_lattice,
)
from test_support import workspace_tempdir


def _evaluate_reference_subset(bundle, family: str, root: Path, paths: set[str]):
    workspace = root / ("case-" + str(len(list(root.glob("case-*")))))
    shutil.copytree(bundle.visible_root, workspace)
    for relative_path in paths:
        target = workspace / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            bundle.manifest.reference_solution["files"][relative_path],
            encoding="utf-8",
        )
    return get_evaluator(
        family, evaluator_entrypoint=bundle.manifest.evaluator_entrypoint
    ).evaluate(workspace, bundle.manifest, bundle.hidden_root)


def test_default_d5_distribution_uses_quality_scenarios() -> None:
    expected = {
        "tabular": "account_event_program_synthesis",
        "pipeline": "team_hours_pipeline",
        "retrieval_workspace": "client_adapter_sync",
    }
    with workspace_tempdir() as tmp_dir:
        root = Path(tmp_dir)
        for family, scenario_id in expected.items():
            bundle = get_generator(family).generate_instance(
                get_generator(family).sample_spec(difficulty=5, seed=90),
                root / family,
            )
            assert bundle.manifest.metadata["scenario_id"] == scenario_id

        script = get_generator("script_repair").generate_instance(
            get_generator("script_repair").sample_spec(difficulty=5, seed=90),
            root / "script",
        )
        realization = script.manifest.metadata["difficulty_realization"]
        assert realization["bug_bundle_id"]
        assert realization["semantic_dependency_depth"] >= 4


def test_tabular_composition_requires_both_stages() -> None:
    generator = get_generator("tabular")
    with workspace_tempdir() as tmp_dir:
        root = Path(tmp_dir)
        bundle = generator.generate_instance(
            generator.sample_spec(
                difficulty=5,
                seed=90,
                generation_params={"composition_mode": "compositional"},
            ),
            root / "generated",
        )
        files = set(bundle.manifest.reference_solution["files"])
        stage_a = _evaluate_reference_subset(
            bundle, "tabular", root, {"artifacts/account_map.json"}
        )
        stage_b = _evaluate_reference_subset(
            bundle,
            "tabular",
            root,
            {"transform.py", "outputs/account_report.json"},
        )
        full = _evaluate_reference_subset(bundle, "tabular", root, files)
        assert stage_a.score <= 0.40
        assert stage_b.score <= 0.30
        assert not stage_a.success and not stage_b.success
        assert full.success and full.score == 1.0
        spec = bundle.manifest.metadata["task_descriptor"]["composition_spec"]
        assert spec["downstream_consumes_upstream_artifact"] is True


def test_retrieval_composition_requires_config_and_code() -> None:
    generator = get_generator("retrieval_workspace")
    with workspace_tempdir() as tmp_dir:
        root = Path(tmp_dir)
        bundle = generator.generate_instance(
            generator.sample_spec(
                difficulty=5,
                seed=90,
                scenario_id="client_adapter_sync",
                generation_params={"composition_mode": "compositional"},
            ),
            root / "generated",
        )
        stage_a = _evaluate_reference_subset(
            bundle, "retrieval_workspace", root, {"config/adapter_contract.json"}
        )
        stage_b = _evaluate_reference_subset(
            bundle, "retrieval_workspace", root, {"src/client_adapter.py"}
        )
        full = _evaluate_reference_subset(
            bundle,
            "retrieval_workspace",
            root,
            set(bundle.manifest.reference_solution["files"]),
        )
        assert stage_a.score <= 0.40
        assert stage_b.score <= 0.35
        assert full.success and full.score == 1.0
        assert not (bundle.visible_root / "evidence/document_inventory.csv").exists()


def test_script_composition_does_not_publish_direct_repair_guide() -> None:
    generator = get_generator("script_repair")
    with workspace_tempdir() as tmp_dir:
        bundle = generator.generate_instance(
            generator.sample_spec(
                difficulty=5,
                seed=91,
                scenario_id="team_roster_export",
                generation_params={"composition_mode": "compositional"},
            ),
            Path(tmp_dir),
        )
        assert not (bundle.visible_root / "docs/current_repair_contract.md").exists()
        assert (bundle.visible_root / "docs/api_contract.md").exists()
        assert (bundle.visible_root / "changelog/schema_v4.md").exists()
        readme = (bundle.visible_root / "README.md").read_text(encoding="utf-8")
        assert "## Expected behavior" not in readme
        reference_files = set(bundle.manifest.reference_solution["files"])
        stage_a = _evaluate_reference_subset(
            bundle, "script_repair", Path(tmp_dir), {"artifacts/resolved_contract.json"}
        )
        stage_b = _evaluate_reference_subset(
            bundle,
            "script_repair",
            Path(tmp_dir),
            reference_files - {"artifacts/resolved_contract.json"},
        )
        full = _evaluate_reference_subset(
            bundle, "script_repair", Path(tmp_dir), reference_files
        )
        assert stage_a.score <= 0.40
        assert stage_b.score <= 0.30
        assert full.success and full.score == 1.0
        spec = bundle.manifest.metadata["task_descriptor"]["composition_spec"]
        assert spec["downstream_consumes_upstream_artifact"] is True


def test_pipeline_composition_reloads_the_intermediate_artifact() -> None:
    generator = get_generator("pipeline")
    with workspace_tempdir() as tmp_dir:
        root = Path(tmp_dir)
        bundle = generator.generate_instance(
            generator.sample_spec(
                difficulty=5,
                seed=90,
                scenario_id="team_hours_pipeline",
                generation_params={"composition_mode": "compositional"},
            ),
            root / "generated",
        )
        full = _evaluate_reference_subset(
            bundle,
            "pipeline",
            root,
            set(bundle.manifest.reference_solution["files"]),
        )
        assert full.success and full.score == 1.0

        workspace = root / "corrupted-stage-boundary"
        shutil.copytree(bundle.visible_root, workspace)
        for relative_path, content in bundle.manifest.reference_solution[
            "files"
        ].items():
            target = workspace / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        runner = workspace / "run_pipeline.py"
        source = runner.read_text(encoding="utf-8")
        source = source.replace(
            '    summary = build_summary(load_json(normalized_path), config["exclude_states"])\n',
            '    write_json(normalized_path, {"corrupt": True})\n'
            '    summary = build_summary(load_json(normalized_path), config["exclude_states"])\n',
            1,
        )
        runner.write_text(source, encoding="utf-8")
        corrupted = get_evaluator(
            "pipeline", evaluator_entrypoint=bundle.manifest.evaluator_entrypoint
        ).evaluate(workspace, bundle.manifest, bundle.hidden_root)
        assert not corrupted.success
        assert corrupted.score <= 0.30
        spec = bundle.manifest.metadata["task_descriptor"]["composition_spec"]
        assert spec["downstream_consumes_upstream_artifact"] is True


def test_retrieval_d5_full_oracle_lattice_passes_quality_limits() -> None:
    generator = get_generator("retrieval_workspace")
    spec = generator.sample_spec(
        difficulty=5,
        seed=90,
        scenario_id="client_adapter_sync",
        generation_params={"composition_mode": "hard_atomic"},
    )
    scenario = build_profiled_retrieval_scenario(
        random.Random("90:profiled_retrieval"), spec
    )
    with workspace_tempdir() as tmp_dir:
        root = Path(tmp_dir)
        bundle = generator.generate_instance(spec, root / "generated")
        scores = evaluate_fix_lattice(
            correct_files=scenario["correct_files"],
            selected_bugs=scenario["bugs"],
            visible_template=bundle.visible_root,
            scratch_root=root / "lattice",
            evaluator=get_evaluator("retrieval_workspace"),
            manifest=bundle.manifest,
            hidden_root=bundle.hidden_root,
        )
        profile = validate_partial_solution_lattice(
            scores,
            [bug["label"] for bug in scenario["bugs"]],
            thresholds={
                "no_fixes_max": 0.35,
                "single_fix_max": 0.60,
                "pair_fix_max": 0.85,
                "all_but_one_max": 0.95,
            },
        )
        assert profile["valid"] is True
        assert profile["no_fix_score"] == 0.15
        assert profile["single_fix_max_score"] > profile["no_fix_score"]
        assert profile["pair_fix_max_score"] > profile["single_fix_max_score"]
        assert profile["full_solution_score"] == 1.0


def test_tabular_d5_program_requires_reusable_hidden_fixture_logic() -> None:
    generator = get_generator("tabular")
    spec = generator.sample_spec(
        difficulty=5,
        seed=90,
        generation_params={"composition_mode": "hard_atomic"},
    )
    with workspace_tempdir() as tmp_dir:
        root = Path(tmp_dir)
        bundle = generator.generate_instance(spec, root / "generated")
        assert (
            bundle.manifest.metadata["scenario_id"] == "account_event_program_synthesis"
        )
        descriptor = bundle.manifest.metadata["task_descriptor"]
        assert descriptor["required_files"] == [
            "process_report.py",
            "artifacts/report.json",
        ]
        assert descriptor["entrypoint"] == (
            "python process_report.py --input-dir data --output artifacts/report.json"
        )
        assert set(descriptor["input_files"]) == {
            "data/events.csv",
            "data/account_aliases.json",
            "data/status_history.csv",
        }
        evaluator = get_evaluator(
            "tabular", evaluator_entrypoint=bundle.manifest.evaluator_entrypoint
        )
        untouched = evaluator.evaluate(
            bundle.visible_root, bundle.manifest, bundle.hidden_root
        )
        assert untouched.score < 1.0
        assert untouched.subscores["capability_determinism"] == 1.0

        visible_expected = json.loads(
            (bundle.hidden_root / "expected_output.json").read_text(encoding="utf-8")
        )
        hardcoded = root / "hardcoded"
        shutil.copytree(bundle.visible_root, hardcoded)
        payload = json.dumps(visible_expected, sort_keys=True)
        (hardcoded / "process_report.py").write_text(
            "import argparse, json\n"
            "from pathlib import Path\n"
            "parser = argparse.ArgumentParser()\n"
            "parser.add_argument('--input-dir', required=True)\n"
            "parser.add_argument('--output', required=True)\n"
            "args = parser.parse_args()\n"
            f"payload = json.loads({payload!r})\n"
            "target = Path(args.output)\n"
            "target.parent.mkdir(parents=True, exist_ok=True)\n"
            "target.write_text(json.dumps(payload, sort_keys=True) + '\\n', encoding='utf-8')\n",
            encoding="utf-8",
        )
        hardcoded_result = evaluator.evaluate(
            hardcoded, bundle.manifest, bundle.hidden_root
        )
        assert hardcoded_result.score <= 0.30
        assert hardcoded_result.subscores["capability_determinism"] == 1.0
        assert hardcoded_result.subscores["capability_hidden_end_to_end"] == 0.0

        solved = root / "solved-program"
        shutil.copytree(bundle.visible_root, solved)
        for relative_path, content in bundle.manifest.reference_solution[
            "files"
        ].items():
            target = solved / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        solved_result = evaluator.evaluate(solved, bundle.manifest, bundle.hidden_root)
        assert solved_result.success
        assert solved_result.score == 1.0
        assert solved_result.subscores["capability_determinism"] == 1.0


def test_pipeline_d5_full_oracle_lattice_passes_quality_limits() -> None:
    generator = get_generator("pipeline")
    spec = generator.sample_spec(
        difficulty=5,
        seed=90,
        scenario_id="team_hours_pipeline",
        generation_params={"composition_mode": "hard_atomic"},
    )
    scenario = generator.team_hours_pipeline_scenario(random.Random(spec.seed), spec)
    with workspace_tempdir() as tmp_dir:
        root = Path(tmp_dir)
        bundle = generator.generate_instance(spec, root / "generated")
        scores = evaluate_fix_lattice(
            correct_files=scenario["files"],
            selected_bugs=scenario["bugs"],
            visible_template=bundle.visible_root,
            scratch_root=root / "lattice",
            evaluator=get_evaluator(
                "pipeline",
                evaluator_entrypoint=bundle.manifest.evaluator_entrypoint,
            ),
            manifest=bundle.manifest,
            hidden_root=bundle.hidden_root,
        )
        profile = validate_partial_solution_lattice(
            scores,
            [bug["label"] for bug in scenario["bugs"]],
            thresholds={
                "no_fixes_max": 0.60,
                "single_fix_max": 0.95,
                "pair_fix_max": 1.0,
                "all_but_one_max": 0.95,
            },
        )
        assert profile["valid"] is True
        assert profile["no_fix_score"] < profile["full_solution_score"]
        assert profile["single_fix_max_score"] > profile["no_fix_score"]
        assert profile["full_solution_score"] == 1.0
