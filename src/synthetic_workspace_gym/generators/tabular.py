from __future__ import annotations

import json
import random
from pathlib import Path

from synthetic_workspace_gym.generators.base import BaseGenerator, GeneratedPayload
from synthetic_workspace_gym.generators.tabular_scenarios import (
    build_channel_status_pivot_scenario,
    build_monthly_segment_report_scenario,
    build_supplier_restock_summary_scenario,
    build_weekly_refund_rollup_scenario,
)
from synthetic_workspace_gym.schemas import EnvironmentFamily, EnvironmentSpec
from synthetic_workspace_gym.utils.io import write_json, write_text


class TabularTransformationGenerator(BaseGenerator):
    family = EnvironmentFamily.TABULAR

    def build_environment(self, spec: EnvironmentSpec, *, root: Path, visible_root: Path, hidden_root: Path) -> GeneratedPayload:
        scenario = self.select_scenario(spec, self.scenario_pool(spec))
        task_descriptor = {
            "family": "tabular",
            "scenario_id": scenario["scenario_id"],
            **dict(scenario["task_descriptor"]),
            "hints": self.visible_hints(list(scenario["hints"]), spec.difficulty),
        }

        for relative_path, content in dict(scenario["files"]).items():
            write_text(visible_root / relative_path, str(content))
        write_text(visible_root / "README.md", self.build_readme(scenario, task_descriptor))
        write_json(visible_root / "task.json", task_descriptor)

        expected_output = scenario["expected_output"]
        output_path = str(task_descriptor["output_path"])
        write_json(hidden_root / "expected_output.json", expected_output)
        write_json(
            hidden_root / "evaluator_config.json",
            {
                "output_path": output_path,
                "comparison_mode": "exact_json",
            },
        )
        reference_solution = {
            "files": {
                output_path: json.dumps(expected_output, indent=2, sort_keys=True) + "\n",
            },
            "seed": spec.seed,
            "scenario_id": scenario["scenario_id"],
            "task_descriptor": task_descriptor,
        }
        write_json(hidden_root / "reference_solution.json", reference_solution)

        metadata = {
            "task_descriptor": task_descriptor,
            "complexity_profile": spec.complexity_profile.to_dict() if spec.complexity_profile else {},
            "scenario_id": scenario["scenario_id"],
            "scenario_profile": scenario["structure"],
            "scenario_selection": {
                "requested_scenario_id": spec.scenario_id,
                "selection_mode": "explicit" if spec.scenario_id else "seed_modulo",
            },
            "visible_artifact_layout": {
                "workspace_root": "visible",
                "output_path": output_path,
            },
        }
        return GeneratedPayload(
            instruction=self.build_instruction(task_descriptor),
            metadata=metadata,
            reference_solution=reference_solution,
            evaluator_entrypoint="synthetic_workspace_gym.evaluators.tabular:TabularEvaluator",
        )

    def scenario_pool(self, spec: EnvironmentSpec) -> list[dict[str, object]]:
        return [
            build_monthly_segment_report_scenario(random.Random(f"{spec.seed}:monthly_segment_report"), spec),
            build_channel_status_pivot_scenario(random.Random(f"{spec.seed}:channel_status_pivot"), spec),
            build_weekly_refund_rollup_scenario(random.Random(f"{spec.seed}:weekly_refund_rollup"), spec),
            build_supplier_restock_summary_scenario(random.Random(f"{spec.seed}:supplier_restock_summary"), spec),
        ]

    def visible_hints(self, hints: list[str], difficulty: int) -> list[str]:
        if difficulty <= 2:
            return hints
        if difficulty == 3:
            return hints[:2]
        return hints[:1]

    def build_instruction(self, task_descriptor: dict[str, object]) -> str:
        return (
            "Complete the tabular transformation described in README.md. "
            f"Write the final artifact to {task_descriptor['output_path']}."
        )

    def build_readme(self, scenario: dict[str, object], task_descriptor: dict[str, object]) -> str:
        inputs = "\n".join(f"- `{item}`" for item in task_descriptor["input_files"])
        operations = "\n".join(
            f"{index}. `{operation}`"
            for index, operation in enumerate(task_descriptor["operations"], start=1)
        )
        contract_lines = "\n".join(f"- {line}" for line in scenario["output_contract"])
        hints = "\n".join(f"- {hint}" for hint in task_descriptor["hints"])
        return (
            f"# {scenario['title']}\n\n"
            f"{scenario['description']}\n\n"
            "## Inputs\n"
            f"{inputs}\n\n"
            "## Required operations\n"
            f"{operations}\n\n"
            "## Output contract\n"
            f"{contract_lines}\n\n"
            "## Hints\n"
            f"{hints}\n"
        )
