from __future__ import annotations

from .agents import PrimeReActAgent
from .clients import HeuristicReferenceClient, JSONActionClient, ScriptedPrimeClient
from .dataset import SyntheticWorkspacePrimeDataset
from .env import SyntheticWorkspacePrimeEnv
from .export import (
    build_manifest_row,
    export_existing_environments,
    export_prime_pack,
    generate_and_export_prime_pack,
    generate_and_export_task_rows,
    write_manifest_jsonl,
    write_metadata_json,
)
from .rollout import (
    build_batch_summary,
    build_prime_rollout_payload,
    run_prime_branch_rollout,
    run_prime_rollout,
    run_prime_rollout_batch,
    write_prime_rollout_artifacts,
)
from .tools import SWG_PRIME_TOOL_SCHEMAS, get_tool_schemas
from .transcript import make_event, read_transcript_jsonl, write_transcript_jsonl
from .verifier import evaluator_result_to_prime_reward, verify_workspace


def make_env(*args: object, **kwargs: object) -> SyntheticWorkspacePrimeEnv:
    if kwargs.get("branch_manifest_path") is not None:
        from synthetic_workspace_gym.counterfactual.prime import SyntheticWorkspacePrimeBranchEnv
        return SyntheticWorkspacePrimeBranchEnv(*args, **kwargs)
    return SyntheticWorkspacePrimeEnv(*args, **kwargs)


__all__ = [
    "SWG_PRIME_TOOL_SCHEMAS",
    "HeuristicReferenceClient",
    "JSONActionClient",
    "PrimeReActAgent",
    "ScriptedPrimeClient",
    "SyntheticWorkspacePrimeDataset",
    "SyntheticWorkspacePrimeEnv",
    "build_batch_summary",
    "build_manifest_row",
    "build_prime_rollout_payload",
    "evaluator_result_to_prime_reward",
    "export_existing_environments",
    "export_prime_pack",
    "generate_and_export_prime_pack",
    "generate_and_export_task_rows",
    "get_tool_schemas",
    "make_env",
    "make_event",
    "read_transcript_jsonl",
    "run_prime_branch_rollout",
    "run_prime_rollout",
    "run_prime_rollout_batch",
    "verify_workspace",
    "write_manifest_jsonl",
    "write_metadata_json",
    "write_prime_rollout_artifacts",
    "write_transcript_jsonl",
]
