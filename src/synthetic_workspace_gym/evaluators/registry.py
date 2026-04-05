from __future__ import annotations

from importlib import import_module

from synthetic_workspace_gym.schemas import EnvironmentFamily
from synthetic_workspace_gym.evaluators.base import BaseEvaluator

from .pipeline import PipelineEvaluator
from .script_repair import ScriptRepairEvaluator
from .tabular import TabularEvaluator

_EVALUATORS = {
    EnvironmentFamily.TABULAR: TabularEvaluator(),
    EnvironmentFamily.SCRIPT_REPAIR: ScriptRepairEvaluator(),
    EnvironmentFamily.PIPELINE: PipelineEvaluator(),
}


def get_evaluator(family: EnvironmentFamily | str, evaluator_entrypoint: str | None = None):
    if evaluator_entrypoint:
        return _load_evaluator_from_entrypoint(evaluator_entrypoint)
    return _EVALUATORS[EnvironmentFamily(family)]


def list_evaluators() -> list[str]:
    return [family.value for family in _EVALUATORS]


def _load_evaluator_from_entrypoint(entrypoint: str) -> BaseEvaluator:
    module_name, separator, attr_name = entrypoint.partition(":")
    if not separator or not module_name or not attr_name:
        raise ValueError(f"Invalid evaluator entrypoint: {entrypoint}")
    module = import_module(module_name)
    loaded = getattr(module, attr_name)
    if isinstance(loaded, BaseEvaluator):
        return loaded
    instance = loaded()
    if not isinstance(instance, BaseEvaluator):
        raise TypeError(f"Evaluator entrypoint did not resolve to a BaseEvaluator: {entrypoint}")
    return instance
