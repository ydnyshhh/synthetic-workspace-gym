from __future__ import annotations

import json
from pathlib import Path

from synthetic_workspace_gym.agents.base import BaseAgent
from synthetic_workspace_gym.utils.io import write_json, write_jsonl

from .analysis import aggregate_outcomes
from .replay import replay_branch
from .schemas import BranchOutcome, BranchTask


def read_branch_manifest(path: Path) -> list[BranchTask]:
    return [BranchTask.from_dict(json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_branches(tasks: list[BranchTask], agent_factory: callable, rollouts_per_branch: int, output_root: Path) -> list[BranchOutcome]:
    outcomes = []
    for task in tasks:
        for index in range(rollouts_per_branch):
            agent: BaseAgent = agent_factory()
            outcomes.append(replay_branch(task, agent, output_root, index).outcome)
    write_jsonl(output_root / "outcomes.jsonl", [item.to_dict() for item in outcomes])
    comparisons = aggregate_outcomes(outcomes)
    write_jsonl(output_root / "comparisons.jsonl", [item.to_dict() for item in comparisons])
    write_json(output_root / "summary.json", {"outcome_count": len(outcomes), "branch_group_count": len(comparisons), "mean_regret": sum(x.decision_regret for x in comparisons) / len(comparisons) if comparisons else 0.0})
    return outcomes
