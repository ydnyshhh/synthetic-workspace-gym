from __future__ import annotations

import json

from synthetic_workspace_gym.agents.base import BaseAgent, solve_tabular_task
from synthetic_workspace_gym.schemas import Action, ActionType, ToolObservation, ToolState


class ReActBaselineAgent(BaseAgent):
    name = "react"

    def __init__(self) -> None:
        super().__init__()
        self.plan: list[Action] = []
        self.edits_applied = False
        self.smoke_test_attempts = 0
        self.submitted = False

    def reset(self, manifest, initial_observation):
        super().reset(manifest, initial_observation)
        self.plan = [
            Action(ActionType.LIST_DIRECTORY, {"path": "."}),
            Action(ActionType.READ_FILE, {"path": "README.md"}),
            Action(ActionType.READ_FILE, {"path": "task.json"}),
        ]
        self.edits_applied = False
        self.smoke_test_attempts = 0
        self.submitted = False

    def act(self, observation: ToolObservation | dict[str, object], tool_state: ToolState) -> Action:
        self._consume_observation(observation)
        if self.plan:
            return self._set_last_action(self.plan.pop(0))

        assert self.manifest is not None
        assert self.task is not None

        if self.manifest.family.value == "tabular":
            return self._tabular_action()
        if self.manifest.family.value == "script_repair":
            return self._script_repair_action(observation)
        return self._pipeline_action(observation)

    def _tabular_action(self) -> Action:
        assert self.task is not None
        missing_inputs = [path for path in self.task["input_files"] if path not in self.file_cache]
        if missing_inputs:
            return self._set_last_action(Action(ActionType.READ_FILE, {"path": missing_inputs[0]}))
        if self.task["output_path"] not in self.file_cache:
            content = solve_tabular_task(self.task, self.file_cache)
            return self._set_last_action(
                Action(ActionType.WRITE_FILE, {"path": self.task["output_path"], "content": content})
            )
        return self._set_last_action(Action(ActionType.SUBMIT, {"path_or_answer": self.task["output_path"]}))

    def _script_repair_action(self, observation: ToolObservation | dict[str, object]) -> Action:
        assert self.task is not None
        missing_targets = [path for path in self.task["target_files"] if path not in self.file_cache]
        if missing_targets:
            return self._set_last_action(Action(ActionType.READ_FILE, {"path": missing_targets[0]}))

        if not self.edits_applied:
            edits = self._script_repair_edits()
            if edits:
                self.edits_applied = True
                path, content = edits[0]
                self.plan.extend(Action(ActionType.WRITE_FILE, {"path": p, "content": c}) for p, c in edits[1:])
                self.plan.append(Action(ActionType.RUN_SHELL, {"command": self.task["entrypoint"]}))
                self.plan.append(Action(ActionType.SUBMIT, {"path_or_answer": "hidden-tests"}))
                return self._set_last_action(Action(ActionType.WRITE_FILE, {"path": path, "content": content}))

        if self.smoke_test_attempts == 0:
            self.smoke_test_attempts += 1
            return self._set_last_action(Action(ActionType.RUN_SHELL, {"command": self.task["entrypoint"]}))
        if not self.submitted:
            self.submitted = True
            return self._set_last_action(Action(ActionType.SUBMIT, {"path_or_answer": "hidden-tests"}))
        return self._set_last_action(Action(ActionType.SUBMIT, {"path_or_answer": "done"}))

    def _pipeline_action(self, observation: ToolObservation | dict[str, object]) -> Action:
        assert self.task is not None
        missing_targets = [path for path in self.task["target_files"] if path not in self.file_cache]
        if missing_targets:
            return self._set_last_action(Action(ActionType.READ_FILE, {"path": missing_targets[0]}))

        if not self.edits_applied:
            edits = self._pipeline_edits()
            if edits:
                self.edits_applied = True
                path, content = edits[0]
                self.plan.extend(Action(ActionType.WRITE_FILE, {"path": p, "content": c}) for p, c in edits[1:])
                self.plan.append(Action(ActionType.RUN_SHELL, {"command": self.task["entrypoint"]}))
                self.plan.append(
                    Action(ActionType.SUBMIT, {"path_or_answer": self.task["required_output_path"]})
                )
                return self._set_last_action(Action(ActionType.WRITE_FILE, {"path": path, "content": content}))

        if self.smoke_test_attempts == 0:
            self.smoke_test_attempts += 1
            return self._set_last_action(Action(ActionType.RUN_SHELL, {"command": self.task["entrypoint"]}))
        if not self.submitted:
            self.submitted = True
            return self._set_last_action(
                Action(ActionType.SUBMIT, {"path_or_answer": self.task["required_output_path"]})
            )
        return self._set_last_action(Action(ActionType.SUBMIT, {"path_or_answer": "done"}))

    def _script_repair_edits(self) -> list[tuple[str, str]]:
        assert self.task is not None
        edits: list[tuple[str, str]] = []
        scenario_id = self.task["scenario_id"]
        if scenario_id == "inventory_report":
            analytics = self.file_cache.get("src/repair_target/analytics.py", "")
            report = self.file_cache.get("src/repair_target/report.py", "")
            fixed_analytics = analytics
            fixed_report = report
            fixed_analytics = fixed_analytics.replace(
                "range(len(values) - window)",
                "range(len(values) - window + 1)",
            )
            fixed_analytics = fixed_analytics.replace(
                'if status == "archived":',
                "if status not in summary:",
            )
            fixed_report = fixed_report.replace(
                '"rolling_average": rolling_average(active_counts, 3),',
                '"rolling_average": rolling_average(active_counts, 2),',
            )
            if fixed_analytics != analytics:
                edits.append(("src/repair_target/analytics.py", fixed_analytics))
            if fixed_report != report:
                edits.append(("src/repair_target/report.py", fixed_report))
        elif scenario_id == "path_batch":
            io_helpers = self.file_cache.get("src/repair_target/io_helpers.py", "")
            batch = self.file_cache.get("src/repair_target/batch.py", "")
            fixed_io = io_helpers
            fixed_batch = batch
            if "from pathlib import Path" not in fixed_io:
                fixed_io = fixed_io.replace("from __future__ import annotations\n\n", "from __future__ import annotations\n\nfrom pathlib import Path\n\n")
            fixed_io = fixed_io.replace('data_dir.parent / "measurements.csv"', 'data_dir / "measurements.csv"')
            fixed_batch = fixed_batch.replace('"total": len(values),', '"total": sum(values),')
            fixed_batch = fixed_batch.replace(
                "def compute_batch_summary(base_dir: Path) -> dict[str, int]\n",
                "def compute_batch_summary(base_dir: Path) -> dict[str, int]:\n",
            )
            if fixed_io != io_helpers:
                edits.append(("src/repair_target/io_helpers.py", fixed_io))
            if fixed_batch != batch:
                edits.append(("src/repair_target/batch.py", fixed_batch))
        return edits

    def _pipeline_edits(self) -> list[tuple[str, str]]:
        assert self.task is not None
        edits: list[tuple[str, str]] = []
        config_text = self.file_cache.get("config/pipeline_config.json", "")
        if config_text:
            config = json.loads(config_text)
            config["input_path"] = "data/jobs.json"
            config["output_path"] = self.task["required_output_path"]
            normalized = json.dumps(config, indent=2, sort_keys=True) + "\n"
            if normalized != config_text:
                edits.append(("config/pipeline_config.json", normalized))

        runner = self.file_cache.get("run_pipeline.py", "")
        fixed_runner = runner.replace("normalized = rows", "normalized = normalize_rows(rows)")
        if fixed_runner != runner:
            edits.append(("run_pipeline.py", fixed_runner))

        steps = self.file_cache.get("src/pipeline_app/steps.py", "")
        fixed_steps = steps.replace(
            'summary[team]["total_hours"] = round(float(summary[team]["total_hours"]) + 1, 1)',
            'summary[team]["total_hours"] = round(float(summary[team]["total_hours"]) + float(row["hours"]), 1)',
        )
        if fixed_steps != steps:
            edits.append(("src/pipeline_app/steps.py", fixed_steps))

        io_utils = self.file_cache.get("src/pipeline_app/io_utils.py", "")
        fixed_io = io_utils.replace(
            'Path(path).write_text(str(payload), encoding="utf-8")',
            'Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n", encoding="utf-8")',
        )
        if fixed_io != io_utils:
            edits.append(("src/pipeline_app/io_utils.py", fixed_io))
        return edits
