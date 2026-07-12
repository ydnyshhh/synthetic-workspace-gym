from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from synthetic_workspace_gym.schemas import ActionType

from .schemas import CounterfactualSnapshot


@dataclass(slots=True)
class Selection:
    snapshot_id: str
    label: str
    priority: float
    reason: str


class BranchPointSelector(Protocol):
    name: str
    def select(self, snapshots: list[CounterfactualSnapshot]) -> list[Selection]: ...


class BeforeFirstWriteSelector:
    name = "before_first_write"
    def select(self, snapshots: list[CounterfactualSnapshot]) -> list[Selection]:
        for item in sorted(snapshots, key=lambda x: (x.step_index, x.metadata.get("phase", ""))):
            if item.metadata.get("phase") == "before" and (item.original_action or {}).get("tool") in {"write_file", "append_file"}:
                return [Selection(item.snapshot_id, self.name, 1.0, "state immediately before the first write")]
        return []


class BeforeSubmitSelector:
    name = "before_submit"
    def select(self, snapshots: list[CounterfactualSnapshot]) -> list[Selection]:
        return [Selection(x.snapshot_id, self.name, 1.0, "state immediately before submit") for x in snapshots
                if x.metadata.get("phase") == "before" and (x.original_action or {}).get("tool") == ActionType.SUBMIT.value]


class AfterFailedCheckSelector:
    name = "after_failed_check"
    def select(self, snapshots: list[CounterfactualSnapshot]) -> list[Selection]:
        return [Selection(x.snapshot_id, self.name, .9, "public check returned a failure") for x in snapshots
                if x.metadata.get("phase") == "after" and (x.original_action or {}).get("tool") in {"run_shell", "run_python"}
                and x.metadata.get("action_success") is False]


class RepeatedActionSelector:
    name = "repeated_action"
    def select(self, snapshots: list[CounterfactualSnapshot]) -> list[Selection]:
        return [Selection(x.snapshot_id, self.name, .8, "action repeats the preceding action") for x in snapshots
                if x.original_action and x.original_action == x.previous_action]


class PostSolutionSelector:
    name = "post_solution"
    def select(self, snapshots: list[CounterfactualSnapshot]) -> list[Selection]:
        for x in snapshots:
            if x.evaluator_score is not None and x.evaluator_score >= 1.0:
                return [Selection(x.snapshot_id, self.name, 1.0, "workspace first reached evaluator-perfect")]
        return []


class ScoreDropSelector:
    name = "score_drop"
    def select(self, snapshots: list[CounterfactualSnapshot]) -> list[Selection]:
        selected, prior = [], None
        for x in sorted(snapshots, key=lambda item: item.step_index):
            if prior is not None and x.evaluator_score is not None and x.evaluator_score < prior:
                selected.append(Selection(x.snapshot_id, self.name, 1.0, "trusted evaluator score decreased"))
            if x.evaluator_score is not None:
                prior = x.evaluator_score
        return selected


SELECTORS = {x.name: x for x in (BeforeFirstWriteSelector(), BeforeSubmitSelector(), AfterFailedCheckSelector(), RepeatedActionSelector(), PostSolutionSelector(), ScoreDropSelector())}
