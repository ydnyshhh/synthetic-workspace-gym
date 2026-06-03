from __future__ import annotations

from typing import Any

from .compat import vf
from .env import SyntheticWorkspaceVerifiersEnv


SWG_VERIFIERS_ENV_IDS: dict[str, dict[str, Any]] = {
    "swg.tabular.monthly_segment_report": {
        "family": "tabular",
        "scenario": "monthly_segment_report",
    },
    "swg.script_repair.csv_schema_drift": {
        "family": "script_repair",
        "scenario": "csv_schema_drift",
    },
    "swg.pipeline.team_hours_pipeline": {
        "family": "pipeline",
        "scenario": "team_hours_pipeline",
    },
    "swg.retrieval_workspace.service_config_reconciliation": {
        "family": "retrieval_workspace",
        "scenario": "service_config_reconciliation",
    },
}


def list_environments() -> list[str]:
    return sorted(SWG_VERIFIERS_ENV_IDS)


def get_environment_config(env_id: str) -> dict[str, Any]:
    if env_id not in SWG_VERIFIERS_ENV_IDS:
        raise KeyError(f"Unknown SWG verifiers environment id: {env_id}")
    return dict(SWG_VERIFIERS_ENV_IDS[env_id])


def make_environment(env_id: str, **overrides: Any) -> SyntheticWorkspaceVerifiersEnv:
    config = get_environment_config(env_id)
    config.update(overrides)
    return SyntheticWorkspaceVerifiersEnv(**config)


def register_with_verifiers() -> bool:
    if vf is None:
        return False
    for register in _registry_candidates(vf):
        try:
            for env_id, config in SWG_VERIFIERS_ENV_IDS.items():
                register(env_id, lambda env_id=env_id, config=config: SyntheticWorkspaceVerifiersEnv(**config))
            return True
        except TypeError:
            try:
                for env_id, config in SWG_VERIFIERS_ENV_IDS.items():
                    register(id=env_id, entry_point=lambda config=config: SyntheticWorkspaceVerifiersEnv(**config))
                return True
            except TypeError:
                continue
    return False


def _registry_candidates(module: Any) -> list[Any]:
    candidates = []
    direct = getattr(module, "register_environment", None)
    if direct is not None:
        candidates.append(direct)
    registry = getattr(module, "registry", None)
    if registry is not None and getattr(registry, "register", None) is not None:
        candidates.append(registry.register)
    envs = getattr(module, "envs", None)
    if envs is not None and getattr(envs, "register", None) is not None:
        candidates.append(envs.register)
    return candidates
