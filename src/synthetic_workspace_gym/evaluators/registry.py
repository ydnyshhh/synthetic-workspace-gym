from __future__ import annotations

from synthetic_workspace_gym.schemas import EnvironmentFamily

from .pipeline import PipelineEvaluator
from .script_repair import ScriptRepairEvaluator
from .tabular import TabularEvaluator

_EVALUATORS = {
    EnvironmentFamily.TABULAR: TabularEvaluator(),
    EnvironmentFamily.SCRIPT_REPAIR: ScriptRepairEvaluator(),
    EnvironmentFamily.PIPELINE: PipelineEvaluator(),
}


def get_evaluator(family: EnvironmentFamily | str):
    return _EVALUATORS[EnvironmentFamily(family)]


def list_evaluators() -> list[str]:
    return [family.value for family in _EVALUATORS]
