from __future__ import annotations

from synthetic_workspace_gym.schemas import EnvironmentFamily

from .pipeline_completion import PipelineCompletionGenerator
from .script_repair import ScriptRepairGenerator
from .tabular import TabularTransformationGenerator

_GENERATORS = {
    EnvironmentFamily.TABULAR: TabularTransformationGenerator(),
    EnvironmentFamily.SCRIPT_REPAIR: ScriptRepairGenerator(),
    EnvironmentFamily.PIPELINE: PipelineCompletionGenerator(),
}


def get_generator(family: EnvironmentFamily | str):
    return _GENERATORS[EnvironmentFamily(family)]


def list_generators() -> list[str]:
    return [family.value for family in _GENERATORS]
