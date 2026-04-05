from __future__ import annotations

import json
import random
from pathlib import Path
from textwrap import dedent

from synthetic_workspace_gym.generators.base import BaseGenerator, GeneratedPayload
from synthetic_workspace_gym.generators.pipeline_scenarios import (
    build_artifact_stitch_pipeline_scenario,
    build_quality_gate_pipeline_scenario,
    build_sales_csv_pipeline_scenario,
)
from synthetic_workspace_gym.schemas import EnvironmentFamily, EnvironmentSpec
from synthetic_workspace_gym.utils.io import write_json, write_text


class PipelineCompletionGenerator(BaseGenerator):
    family = EnvironmentFamily.PIPELINE

    def _replace_once(self, old: str, new: str, *, label: str, target_path: str):
        def apply(content: str) -> str:
            updated = content.replace(old, new, 1)
            if updated == content:
                raise ValueError(
                    f"Bug application '{label}' did not modify {target_path!r}; canonical source drifted."
                )
            return updated

        return apply

    def _build_environment(self, spec: EnvironmentSpec, *, root: Path, visible_root: Path, hidden_root: Path) -> GeneratedPayload:
        rng = random.Random(spec.seed)
        scenarios = self._scenario_pool(rng, spec)
        scenario = scenarios[(spec.seed - 1) % len(scenarios)]
        expected_output = scenario["expected_output"]
        correct_files = dict(scenario["files"])

        bug_budget = min(len(scenario["bugs"]), {1: 1, 2: 2, 3: 2, 4: 3, 5: 4}[spec.difficulty])
        bug_candidates = list(scenario["bugs"])
        selected_bugs = bug_candidates[:bug_budget]
        if len(bug_candidates) > bug_budget:
            selected_bugs = rng.sample(bug_candidates, k=bug_budget)

        buggy_files = dict(correct_files)
        touched_files: set[str] = set()
        bug_labels: list[str] = []
        for bug in selected_bugs:
            target_path = bug["target_path"]
            buggy_files[target_path] = bug["apply"](buggy_files[target_path])
            touched_files.add(target_path)
            bug_labels.append(bug["label"])

        for relative_path, content in buggy_files.items():
            write_text(visible_root / relative_path, content)

        if spec.difficulty >= 4:
            write_text(
                visible_root / "config" / "README.txt",
                "Only `config/pipeline_config.json` is authoritative. Other config snippets are legacy leftovers.\n",
            )
            write_text(visible_root / "notes" / "handoff.md", str(scenario["debug_note"]))

        task_descriptor = {
            "family": "pipeline",
            "scenario_id": scenario["scenario_id"],
            "entrypoint": "python run_pipeline.py",
            "required_output_path": scenario["required_output_path"],
            "target_files": sorted(touched_files),
            "hints": self._visible_hints(list(scenario["hints"]), spec.difficulty),
        }
        write_text(visible_root / "README.md", self._build_readme(scenario, task_descriptor))
        write_json(visible_root / "task.json", task_descriptor)

        reference_solution = {
            "files": {path: correct_files[path] for path in sorted(touched_files)},
            "bug_labels": bug_labels,
            "scenario_id": scenario["scenario_id"],
        }
        write_json(hidden_root / "expected_output.json", expected_output)
        write_json(
            hidden_root / "evaluator_config.json",
            {
                "entrypoint": "run_pipeline.py",
                "required_output_path": scenario["required_output_path"],
            },
        )
        write_json(hidden_root / "solution_files.json", reference_solution)

        metadata = {
            "task_descriptor": task_descriptor,
            "complexity_profile": spec.complexity_profile.to_dict() if spec.complexity_profile else {},
            "bug_labels": bug_labels,
            "scenario_id": scenario["scenario_id"],
            "scenario_profile": scenario["structure"],
        }
        return GeneratedPayload(
            instruction="Repair the mini-project so running the pipeline produces the required final artifact.",
            metadata=metadata,
            reference_solution=reference_solution,
            evaluator_entrypoint="synthetic_workspace_gym.evaluators.pipeline:PipelineEvaluator",
        )

    def _visible_hints(self, hints: list[str], difficulty: int) -> list[str]:
        if difficulty <= 2:
            return hints
        if difficulty == 3:
            return hints[: max(2, min(len(hints), 2))]
        return hints[:1]

    def _scenario_pool(self, rng: random.Random, spec: EnvironmentSpec) -> list[dict[str, object]]:
        return [
            self._team_hours_pipeline_scenario(rng, spec),
            build_sales_csv_pipeline_scenario(self),
            build_artifact_stitch_pipeline_scenario(self),
            build_quality_gate_pipeline_scenario(self),
        ]

    def _team_hours_pipeline_scenario(self, rng: random.Random, spec: EnvironmentSpec) -> dict[str, object]:
        jobs = self._build_jobs(rng, count=6 + spec.difficulty)
        return {
            "scenario_id": "team_hours_pipeline",
            "title": "Team Hours Pipeline",
            "required_output_path": "artifacts/summary.json",
            "debug_note": "The final artifact must land in `artifacts/summary.json` even if the current config disagrees.\n",
            "hints": [
                "The config file and the code need to agree on both input and output paths.",
                "Cancelled jobs should not count toward the final summary.",
                "The final artifact must be JSON and sorted by team.",
            ],
            "structure": {
                "repair_surface": "config_and_aggregation",
                "bug_scope": "cross_file",
                "failure_mode": "semantic_and_formatting",
                "smoke_test_quality": "informative",
            },
            "files": self._correct_files(jobs),
            "expected_output": self._build_expected_output(jobs),
            "bugs": self._bug_candidates(),
        }

    def _io_utils_module(self, *, include_loader: bool = False) -> str:
        loader = ""
        if include_loader:
            loader = dedent(
                """\

                def load_rows(path: Path) -> list[dict[str, object]]:
                    return json.loads(Path(path).read_text(encoding="utf-8"))
                """
            )
        return (
            "from __future__ import annotations\n\n"
            "import json\n"
            "from pathlib import Path\n"
            f"{loader}\n"
            "def write_json(path: Path, payload: object) -> None:\n"
            "    Path(path).parent.mkdir(parents=True, exist_ok=True)\n"
            '    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n", encoding="utf-8")\n'
        )

    def _build_jobs(self, rng: random.Random, *, count: int) -> list[dict[str, object]]:
        teams = ["platform", "research", "ops"]
        states = ["ready", "complete", "cancelled"]
        jobs = []
        for index in range(count):
            jobs.append(
                {
                    "job_id": f"J{index + 1:03d}",
                    "team": rng.choice(teams).upper() if index % 2 == 0 else rng.choice(teams).title(),
                    "state": rng.choices(states, weights=(0.2, 0.6, 0.2), k=1)[0],
                    "hours": round(rng.uniform(1.5, 8.0), 1),
                }
            )
        return jobs

    def _build_expected_output(self, jobs: list[dict[str, object]]) -> list[dict[str, object]]:
        summary: dict[str, dict[str, object]] = {}
        for row in jobs:
            team = str(row["team"]).lower()
            state = str(row["state"]).lower()
            if state == "cancelled":
                continue
            if team not in summary:
                summary[team] = {"team": team, "job_count": 0, "total_hours": 0.0}
            summary[team]["job_count"] = int(summary[team]["job_count"]) + 1
            summary[team]["total_hours"] = round(float(summary[team]["total_hours"]) + float(row["hours"]), 1)
        return sorted(summary.values(), key=lambda item: str(item["team"]))

    def _correct_files(self, jobs: list[dict[str, object]]) -> dict[str, str]:
        config = {
            "input_path": "data/jobs.json",
            "output_path": "artifacts/summary.json",
            "exclude_states": ["cancelled"],
        }
        io_utils = self._io_utils_module(include_loader=True)
        steps = """from __future__ import annotations


def normalize_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    normalized = []
    for row in rows:
        normalized.append(
            {
                "team": str(row["team"]).lower(),
                "state": str(row["state"]).lower(),
                "hours": float(row["hours"]),
            }
        )
    return normalized


def build_summary(rows: list[dict[str, object]], *, exclude_states: list[str]) -> list[dict[str, object]]:
    summary: dict[str, dict[str, object]] = {}
    for row in rows:
        if row["state"] in exclude_states:
            continue
        team = str(row["team"])
        if team not in summary:
            summary[team] = {"team": team, "job_count": 0, "total_hours": 0.0}
        summary[team]["job_count"] = int(summary[team]["job_count"]) + 1
        summary[team]["total_hours"] = round(float(summary[team]["total_hours"]) + float(row["hours"]), 1)
    return sorted(summary.values(), key=lambda item: str(item["team"]))
"""
        runner = """from __future__ import annotations

import json
import sys
from pathlib import Path

workspace = Path(__file__).resolve().parent
sys.path.insert(0, str(workspace / "src"))

from pipeline_app.io_utils import load_rows, write_json
from pipeline_app.steps import build_summary, normalize_rows


def main() -> None:
    config = json.loads((workspace / "config" / "pipeline_config.json").read_text(encoding="utf-8"))
    rows = load_rows(workspace / config["input_path"])
    normalized = normalize_rows(rows)
    summary = build_summary(normalized, exclude_states=config["exclude_states"])
    write_json(workspace / config["output_path"], summary)


if __name__ == "__main__":
    main()
"""
        return {
            "src/pipeline_app/__init__.py": "",
            "src/pipeline_app/io_utils.py": io_utils,
            "src/pipeline_app/steps.py": steps,
            "run_pipeline.py": runner,
            "config/pipeline_config.json": json.dumps(config, indent=2, sort_keys=True) + "\n",
            "data/jobs.json": json.dumps(jobs, indent=2, sort_keys=True) + "\n",
        }

    def _bug_candidates(self) -> list[dict[str, object]]:
        return [
            {
                "label": "wrong_input_path",
                "target_path": "config/pipeline_config.json",
                "apply": self._replace_once(
                    "data/jobs.json",
                    "data/job.json",
                    label="wrong_input_path",
                    target_path="config/pipeline_config.json",
                ),
            },
            {
                "label": "wrong_output_path",
                "target_path": "config/pipeline_config.json",
                "apply": self._replace_once(
                    "artifacts/summary.json",
                    "artifacts/result.json",
                    label="wrong_output_path",
                    target_path="config/pipeline_config.json",
                ),
            },
            {
                "label": "missing_normalization_step",
                "target_path": "run_pipeline.py",
                "apply": self._replace_once(
                    "normalized = normalize_rows(rows)",
                    "normalized = rows",
                    label="missing_normalization_step",
                    target_path="run_pipeline.py",
                ),
            },
            {
                "label": "aggregation_bug",
                "target_path": "src/pipeline_app/steps.py",
                "apply": self._replace_once(
                    'summary[team]["total_hours"] = round(float(summary[team]["total_hours"]) + float(row["hours"]), 1)',
                    'summary[team]["total_hours"] = round(float(summary[team]["total_hours"]) + 1, 1)',
                    label="aggregation_bug",
                    target_path="src/pipeline_app/steps.py",
                ),
            },
            {
                "label": "output_format_bug",
                "target_path": "src/pipeline_app/io_utils.py",
                "apply": self._replace_once(
                    'Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n", encoding="utf-8")',
                    'Path(path).write_text(str(payload), encoding="utf-8")',
                    label="output_format_bug",
                    target_path="src/pipeline_app/io_utils.py",
                ),
            },
        ]

    def _build_readme(self, scenario: dict[str, object], task_descriptor: dict[str, object]) -> str:
        hints = "\n".join(f"- {hint}" for hint in task_descriptor["hints"])
        targets = "\n".join(f"- `{item}`" for item in task_descriptor["target_files"])
        return (
            f"# {scenario['title']}\n\n"
            "This mini-project is almost complete, but one or more files are inconsistent or incomplete.\n"
            "Repair the pipeline so the final artifact is produced correctly.\n\n"
            "## Smoke test\n"
            f"- `{task_descriptor['entrypoint']}`\n\n"
            "## Output contract\n"
            f"- The final artifact must be written to `{task_descriptor['required_output_path']}`.\n"
            "- The artifact must be valid JSON.\n"
            "- Rows must be sorted by `team`.\n"
            "- Cancelled jobs must be excluded.\n\n"
            "## Likely target files\n"
            f"{targets}\n\n"
            "## Hints\n"
            f"{hints}\n"
        )
