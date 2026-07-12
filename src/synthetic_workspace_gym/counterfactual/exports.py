from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from synthetic_workspace_gym.utils.io import write_json, write_jsonl

from .schemas import BranchComparison, BranchTask


def export_training_data(comparisons: list[BranchComparison], tasks: dict[str, BranchTask], output: Path, format: str,
                         min_margin: float = .2, min_quality: float = .8, exclude_privileged: bool = False) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    by_candidate = {task.candidate_id: task for task in tasks.values()}
    for comparison in comparisons:
        ranking = comparison.metadata.get("candidate_ranking", list(comparison.candidate_statistics))
        best_id = comparison.best_candidate_id; best_task = by_candidate.get(best_id)
        if best_task is None or (exclude_privileged and best_task.metadata.get("privileged")): continue
        best_stats = comparison.candidate_statistics[best_id]
        if format == "sft" and best_stats["mean"] >= min_quality and best_task.forced_action:
            records.append({"task_id": best_task.task_id, "messages": best_task.prefix_messages, "target_action": best_task.forced_action, "target_return": best_stats["mean"], "source": "counterfactual_branch", "privileged": bool(best_task.metadata.get("privileged"))})
        elif format == "preference":
            for rejected_id in ranking[1:]:
                rejected = by_candidate.get(rejected_id); rejected_stats = comparison.candidate_statistics[rejected_id]
                if rejected and not (exclude_privileged and rejected.metadata.get("privileged")) and best_task.forced_action and rejected.forced_action and best_stats["mean"] - rejected_stats["mean"] >= min_margin:
                    records.append({"task_id": best_task.task_id, "state_id": comparison.snapshot_id, "messages": best_task.prefix_messages, "chosen": best_task.forced_action, "rejected": rejected.forced_action, "chosen_return": best_stats["mean"], "rejected_return": rejected_stats["mean"], "return_margin": best_stats["mean"] - rejected_stats["mean"]})
        elif format == "critic":
            for candidate_id, stats in comparison.candidate_statistics.items():
                task = by_candidate.get(candidate_id)
                if task and task.forced_action and not (exclude_privileged and task.metadata.get("privileged")):
                    records.append({"state_id": comparison.snapshot_id, "messages": task.prefix_messages, "action": task.forced_action, "q_target": stats["mean"], "return_std": stats["std"], "rollout_count": int(stats["count"]), "recoverability_target": comparison.recoverable, "success_probability": stats["success_rate"]})
    write_jsonl(output, records)
    return records


def export_rl_taskset(comparisons: list[BranchComparison], tasks: dict[str, BranchTask], output_root: Path, min_regret: float = .2) -> list[BranchTask]:
    selected = []
    for comparison in comparisons:
        if comparison.decision_regret < min_regret: continue
        source = next((x for x in tasks.values() if x.snapshot_id == comparison.snapshot_id), None)
        if source is None: continue
        target = output_root / "environments" / source.task_id
        if target.exists(): shutil.rmtree(target)
        shutil.copytree(Path(source.environment_path), target)
        task = BranchTask(source.task_id, source.branch_group_id, source.snapshot_id, source.candidate_id, "open", str(target), source.prefix_messages, None, source.remaining_steps, source.time_limit_seconds, source.family, source.scenario_id, source.difficulty, source.seed, {**source.metadata, "training_regret": comparison.decision_regret})
        write_json(target / "branch.json", task.to_dict()); selected.append(task)
    write_jsonl(output_root / "manifest.jsonl", [x.to_dict() for x in selected])
    write_json(output_root / "metadata.json", {"format_version": "1.0", "mode": "open", "task_count": len(selected)})
    return selected


def read_comparisons(path: Path) -> list[BranchComparison]:
    return [BranchComparison.from_dict(json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
