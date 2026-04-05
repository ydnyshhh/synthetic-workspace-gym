from __future__ import annotations

from synthetic_workspace_gym.agents.base import BaseAgent
from synthetic_workspace_gym.schemas import Action, ActionType, ToolObservation, ToolState


class HeuristicBaselineAgent(BaseAgent):
    """Privileged reference-solution baseline for infrastructure validation.

    This agent intentionally uses the manifest's stored reference solution
    instead of reasoning over the workspace. It is useful for validating the
    generate -> run -> evaluate loop, but it is not a general-purpose agent.
    """

    name = "heuristic"

    def __init__(self) -> None:
        super().__init__()
        self.plan: list[Action] = []

    def reset(self, manifest, initial_observation):
        super().reset(manifest, initial_observation)
        solution_files = manifest.reference_solution.get("files", {})
        self.plan = [
            Action(ActionType.WRITE_FILE, {"path": path, "content": content})
            for path, content in sorted(solution_files.items())
        ]
        submit_target = self.default_submit_target(solution_files)
        self.plan.append(Action(ActionType.SUBMIT, {"path_or_answer": submit_target}))

    def act(self, observation: ToolObservation | dict[str, object], tool_state: ToolState) -> Action:
        if self.plan:
            return self.set_last_action(self.plan.pop(0))
        return self.set_last_action(Action(ActionType.SUBMIT, {"path_or_answer": "reference-solution"}))

    def default_submit_target(self, solution_files: dict[str, object]) -> str:
        if len(solution_files) == 1:
            return next(iter(solution_files))
        if self.manifest is not None and self.manifest.family.value == "pipeline":
            return "pipeline-reference"
        if self.manifest is not None and self.manifest.family.value == "script_repair":
            return "hidden-tests"
        return "reference-solution"


ReActBaselineAgent = HeuristicBaselineAgent
