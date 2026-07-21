from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

from synthetic_workspace_gym.evaluators.registry import get_evaluator
from synthetic_workspace_gym.generators.registry import get_generator
from synthetic_workspace_gym.generators.retrieval_workspace_scenarios import (
    build_client_adapter_sync_scenario,
)
from synthetic_workspace_gym.generators.script_repair_quality import (
    evaluate_fix_lattice,
)
from synthetic_workspace_gym.generators.tabular_program_synthesis import (
    build_account_event_program_scenario,
)
from test_support import workspace_tempdir


def test_retrieval_capabilities_are_uncapped_and_independent() -> None:
    generator = get_generator("retrieval_workspace")
    spec = generator.sample_spec(
        difficulty=5,
        seed=90,
        scenario_id="client_adapter_sync",
        generation_params={"composition_mode": "hard_atomic"},
    )
    scenario = build_client_adapter_sync_scenario(
        random.Random("90:client_adapter_sync"), spec
    )
    with workspace_tempdir() as tmp:
        root = Path(tmp)
        bundle = generator.generate_instance(spec, root / "generated")
        evaluator = get_evaluator(
            "retrieval_workspace",
            evaluator_entrypoint=bundle.manifest.evaluator_entrypoint,
        )
        scores = evaluate_fix_lattice(
            correct_files=scenario["correct_files"],
            selected_bugs=scenario["bugs"],
            visible_template=bundle.visible_root,
            scratch_root=root / "lattice",
            evaluator=evaluator,
            manifest=bundle.manifest,
            hidden_root=bundle.hidden_root,
        )
        collection = frozenset({"legacy_collection_field"})
        collection_quantity = frozenset(
            {"legacy_collection_field", "legacy_quantity_field"}
        )
        assert scores[collection] > scores[frozenset()]
        assert scores[collection_quantity] > scores[collection]
        assert len(set(scores.values())) >= 5
        assert scores[frozenset(str(bug["label"]) for bug in scenario["bugs"])] == 1.0

        missing = root / "missing-target"
        shutil.copytree(bundle.visible_root, missing)
        (missing / "src/client_adapter.py").unlink()
        assert (
            evaluator.evaluate(missing, bundle.manifest, bundle.hidden_root).score
            == 0.0
        )


def test_pipeline_hidden_fixture_and_determinism_are_independent() -> None:
    generator = get_generator("pipeline")
    spec = generator.sample_spec(
        difficulty=5,
        seed=93,
        scenario_id="team_hours_pipeline",
        generation_params={"composition_mode": "hard_atomic"},
    )
    scenario = generator.team_hours_pipeline_scenario(random.Random(93), spec)
    with workspace_tempdir() as tmp:
        root = Path(tmp)
        bundle = generator.generate_instance(spec, root / "generated")
        evaluator = get_evaluator(
            "pipeline", evaluator_entrypoint=bundle.manifest.evaluator_entrypoint
        )

        visible_only = root / "visible-only"
        shutil.copytree(bundle.visible_root, visible_only)
        visible_expected = json.loads(
            (bundle.hidden_root / "expected_output.json").read_text(encoding="utf-8")
        )
        (visible_only / "run_pipeline.py").write_text(
            "import json\nfrom pathlib import Path\n"
            f"payload = {visible_expected!r}\n"
            "target = Path('artifacts/summary.json')\n"
            "target.parent.mkdir(parents=True, exist_ok=True)\n"
            "target.write_text(json.dumps(payload, sort_keys=True) + '\\n', encoding='utf-8')\n",
            encoding="utf-8",
        )
        visible_result = evaluator.evaluate(
            visible_only, bundle.manifest, bundle.hidden_root
        )
        assert visible_result.diagnostics["visible_exact"] is True
        assert visible_result.diagnostics["hidden_exact"] is False
        assert 0.0 < visible_result.score < 1.0

        nondeterministic = root / "nondeterministic"
        shutil.copytree(bundle.visible_root, nondeterministic)
        for relative_path, content in scenario["files"].items():
            target = nondeterministic / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        io_path = nondeterministic / "src/pipeline_app/io_utils.py"
        io_text = io_path.read_text(encoding="utf-8")
        io_text = io_text.replace(
            'temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n", encoding="utf-8")',
            "marker = target.with_suffix(target.suffix + '.toggle')\n"
            "    indent = 3 if marker.exists() else 2\n"
            "    marker.write_text('toggle', encoding='utf-8')\n"
            '    temporary.write_text(json.dumps(payload, indent=indent, sort_keys=True) + "\\n", encoding="utf-8")',
        )
        io_path.write_text(io_text, encoding="utf-8")
        nondeterministic_result = evaluator.evaluate(
            nondeterministic, bundle.manifest, bundle.hidden_root
        )
        assert nondeterministic_result.subscores["capability_determinism"] == 0.0
        assert nondeterministic_result.score == 0.9


def test_tabular_focused_capabilities_and_independent_determinism() -> None:
    generator = get_generator("tabular")
    spec = generator.sample_spec(
        difficulty=5,
        seed=98,
        scenario_id="account_event_program_synthesis",
        generation_params={"composition_mode": "hard_atomic"},
    )
    scenario = build_account_event_program_scenario(
        random.Random("98:account_event_program_synthesis"), spec
    )
    with workspace_tempdir() as tmp:
        root = Path(tmp)
        bundle = generator.generate_instance(spec, root / "generated")
        evaluator = get_evaluator(
            "tabular", evaluator_entrypoint=bundle.manifest.evaluator_entrypoint
        )
        untouched = evaluator.evaluate(
            bundle.visible_root, bundle.manifest, bundle.hidden_root
        )
        assert untouched.subscores["capability_determinism"] == 1.0
        assert untouched.score < 1.0
        assert (
            len(
                {
                    untouched.subscores[name]
                    for name in untouched.subscores
                    if name.startswith("capability_")
                }
            )
            >= 2
        )

        solved = root / "solved"
        shutil.copytree(bundle.visible_root, solved)
        for relative_path, content in scenario["reference_solution_files"].items():
            target = solved / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        result = evaluator.evaluate(solved, bundle.manifest, bundle.hidden_root)
        assert result.success
        assert result.score == 1.0
