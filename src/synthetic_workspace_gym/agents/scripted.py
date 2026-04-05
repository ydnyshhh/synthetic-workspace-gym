from __future__ import annotations

from synthetic_workspace_gym.agents.base import BaseAgent, solve_tabular_task
from synthetic_workspace_gym.schemas import Action, ActionType, ToolObservation, ToolState


class ScriptedBaselineAgent(BaseAgent):
    name = "scripted"

    def __init__(self) -> None:
        super().__init__()
        self.plan: list[Action] = []
        self.smoke_test_ran = False
        self.submitted = False

    def reset(self, manifest, initial_observation):
        super().reset(manifest, initial_observation)
        self.plan = [
            Action(ActionType.LIST_DIRECTORY, {"path": "."}),
            Action(ActionType.READ_FILE, {"path": "README.md"}),
            Action(ActionType.READ_FILE, {"path": "task.json"}),
        ]
        self.smoke_test_ran = False
        self.submitted = False

    def act(self, observation: ToolObservation | dict[str, object], tool_state: ToolState) -> Action:
        self.consume_observation(observation)
        if self.plan:
            return self.set_last_action(self.plan.pop(0))

        assert self.manifest is not None
        if self.manifest.family.value == "tabular":
            assert self.task is not None
            missing_inputs = [path for path in self.task["input_files"] if path not in self.file_cache]
            if missing_inputs:
                return self.set_last_action(Action(ActionType.READ_FILE, {"path": missing_inputs[0]}))
            if self.task["output_path"] not in self.file_cache:
                content = solve_tabular_task(self.task, self.file_cache)
                return self.set_last_action(
                    Action(ActionType.WRITE_FILE, {"path": self.task["output_path"], "content": content})
                )
            return self.set_last_action(Action(ActionType.SUBMIT, {"path_or_answer": self.task["output_path"]}))

        if not self.smoke_test_ran and self.task is not None:
            self.smoke_test_ran = True
            return self.set_last_action(Action(ActionType.RUN_SHELL, {"command": self.task["entrypoint"]}))
        if not self.submitted and self.task is not None:
            self.submitted = True
            return self.set_last_action(
                Action(
                    ActionType.SUBMIT,
                    {"path_or_answer": self.task.get("required_output_path", self.task.get("entrypoint", "done"))},
                )
            )
        return self.set_last_action(Action(ActionType.SUBMIT, {"path_or_answer": "done"}))
