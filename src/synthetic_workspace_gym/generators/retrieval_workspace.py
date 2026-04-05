from __future__ import annotations

import random
from pathlib import Path

from synthetic_workspace_gym.generators.base import BaseGenerator, GeneratedPayload
from synthetic_workspace_gym.generators.retrieval_workspace_scenarios import (
    build_client_adapter_sync_scenario,
    build_incident_report_bundle_scenario,
    build_migration_plan_bundle_scenario,
    build_service_config_reconciliation_scenario,
)
from synthetic_workspace_gym.schemas import EnvironmentFamily, EnvironmentSpec
from synthetic_workspace_gym.utils.io import write_json, write_text


class RetrievalWorkspaceGenerator(BaseGenerator):
    family = EnvironmentFamily.RETRIEVAL_WORKSPACE

    def build_environment(self, spec: EnvironmentSpec, *, root: Path, visible_root: Path, hidden_root: Path) -> GeneratedPayload:
        scenario = self.select_scenario(spec, self.scenario_pool(spec))
        task_descriptor = {
            "family": "retrieval_workspace",
            "scenario_id": scenario["scenario_id"],
            "task_type": scenario["task_type"],
            "target_path": scenario["target_path"],
            "output_style": scenario["output_style"],
            "document_roots": scenario["document_roots"],
            "hints": self.visible_hints(list(scenario["hints"]), spec.difficulty),
        }
        if scenario.get("entrypoint"):
            task_descriptor["entrypoint"] = scenario["entrypoint"]

        for relative_path, content in dict(scenario["files"]).items():
            write_text(visible_root / relative_path, str(content))
        write_text(visible_root / "README.md", self.build_readme(scenario, task_descriptor))
        write_json(visible_root / "task.json", task_descriptor)

        if scenario.get("expected_output") is not None:
            write_json(hidden_root / "expected_output.json", scenario["expected_output"])
        for relative_path, payload in dict(scenario.get("hidden_json_assets", {})).items():
            write_json(hidden_root / relative_path, payload)
        for relative_path, content in dict(scenario.get("hidden_text_assets", {})).items():
            write_text(hidden_root / relative_path, str(content))

        reference_solution = {
            "files": dict(scenario["reference_solution_files"]),
            "seed": spec.seed,
            "scenario_id": scenario["scenario_id"],
            "task_descriptor": task_descriptor,
        }
        write_json(hidden_root / "reference_solution.json", reference_solution)
        write_json(hidden_root / "evaluator_config.json", scenario["evaluator_config"])

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
                "target_path": scenario["target_path"],
                "document_roots": scenario["document_roots"],
            },
        }
        return GeneratedPayload(
            instruction=self.build_instruction(task_descriptor),
            metadata=metadata,
            reference_solution=reference_solution,
            evaluator_entrypoint="synthetic_workspace_gym.evaluators.retrieval_workspace:RetrievalWorkspaceEvaluator",
        )

    def scenario_pool(self, spec: EnvironmentSpec) -> list[dict[str, object]]:
        return [
            build_service_config_reconciliation_scenario(
                random.Random(f"{spec.seed}:service_config_reconciliation"),
                spec,
            ),
            build_migration_plan_bundle_scenario(
                random.Random(f"{spec.seed}:migration_plan_bundle"),
                spec,
            ),
            build_incident_report_bundle_scenario(
                random.Random(f"{spec.seed}:incident_report_bundle"),
                spec,
            ),
            build_client_adapter_sync_scenario(
                random.Random(f"{spec.seed}:client_adapter_sync"),
                spec,
            ),
        ]

    def visible_hints(self, hints: list[str], difficulty: int) -> list[str]:
        if difficulty <= 2:
            return hints
        if difficulty == 3:
            return hints[:2]
        return hints[:1]

    def build_instruction(self, task_descriptor: dict[str, object]) -> str:
        target_path = str(task_descriptor["target_path"])
        return (
            "Inspect the local document set, retrieve the relevant evidence, and update the target artifact. "
            f"Ground the final change in the visible workspace documents and write the result to {target_path}."
        )

    def build_readme(self, scenario: dict[str, object], task_descriptor: dict[str, object]) -> str:
        document_roots = "\n".join(f"- `{root}/`" for root in task_descriptor["document_roots"])
        output_contract = "\n".join(f"- {line}" for line in scenario["output_contract"])
        hints = "\n".join(f"- {hint}" for hint in task_descriptor["hints"])
        smoke_test = ""
        if task_descriptor.get("entrypoint"):
            smoke_test = f"## Optional smoke test\n- `{task_descriptor['entrypoint']}`\n\n"
        return (
            f"# {scenario['title']}\n\n"
            f"{scenario['description']}\n\n"
            "## Objective\n"
            f"- Update or create `{task_descriptor['target_path']}` using evidence from the local document set.\n"
            "- Ignore irrelevant or stale documents when they conflict with newer authoritative sources.\n\n"
            "## Document roots\n"
            f"{document_roots}\n\n"
            f"{smoke_test}"
            "## Output contract\n"
            f"{output_contract}\n\n"
            "## Hints\n"
            f"{hints}\n"
        )
