from __future__ import annotations

from .dataset import SyntheticWorkspacePrimeDataset
from .env import SyntheticWorkspacePrimeEnv
from .tools import SWG_PRIME_TOOL_SCHEMAS, get_tool_schemas
from .verifier import evaluator_result_to_prime_reward, verify_workspace


def make_env(*args: object, **kwargs: object) -> SyntheticWorkspacePrimeEnv:
    return SyntheticWorkspacePrimeEnv(*args, **kwargs)


__all__ = [
    "SWG_PRIME_TOOL_SCHEMAS",
    "SyntheticWorkspacePrimeDataset",
    "SyntheticWorkspacePrimeEnv",
    "evaluator_result_to_prime_reward",
    "get_tool_schemas",
    "make_env",
    "verify_workspace",
]
