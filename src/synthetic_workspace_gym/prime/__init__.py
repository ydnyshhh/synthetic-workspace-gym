from __future__ import annotations

from .dataset import SyntheticWorkspacePrimeDataset
from .env import SyntheticWorkspacePrimeEnv
from .export import (
    build_manifest_row,
    export_existing_environments,
    export_prime_pack,
    generate_and_export_prime_pack,
    write_manifest_jsonl,
    write_metadata_json,
)
from .tools import SWG_PRIME_TOOL_SCHEMAS, get_tool_schemas
from .verifier import evaluator_result_to_prime_reward, verify_workspace


def make_env(*args: object, **kwargs: object) -> SyntheticWorkspacePrimeEnv:
    return SyntheticWorkspacePrimeEnv(*args, **kwargs)


__all__ = [
    "SWG_PRIME_TOOL_SCHEMAS",
    "SyntheticWorkspacePrimeDataset",
    "SyntheticWorkspacePrimeEnv",
    "build_manifest_row",
    "evaluator_result_to_prime_reward",
    "export_existing_environments",
    "export_prime_pack",
    "generate_and_export_prime_pack",
    "get_tool_schemas",
    "make_env",
    "verify_workspace",
    "write_manifest_jsonl",
    "write_metadata_json",
]
