from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from synthetic_workspace_gym.agents.scripted import ScriptedBaselineAgent
from synthetic_workspace_gym.counterfactual.analysis import aggregate_outcomes
from synthetic_workspace_gym.counterfactual.candidates import _candidate
from synthetic_workspace_gym.counterfactual.compiler import compile_pack
from synthetic_workspace_gym.counterfactual.exports import export_rl_taskset, export_training_data
from synthetic_workspace_gym.counterfactual.replay import replay_branch
from synthetic_workspace_gym.counterfactual.runner import read_branch_manifest
from synthetic_workspace_gym.counterfactual.snapshots import NamedSnapshotPolicy, SnapshotCollector, SnapshotContext
from synthetic_workspace_gym.generators.registry import get_generator
from synthetic_workspace_gym.runtime.environment import load_environment
from synthetic_workspace_gym.schemas import Action, ActionType
from synthetic_workspace_gym.utils.io import write_json, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("examples/counterfactual/positive-demo"))
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="swg-cf-positive-") as temp:
        temp_root = Path(temp)
        generator = get_generator("script_repair")
        spec = generator.sample_spec(difficulty=1, seed=23, scenario_id="csv_schema_drift")
        bundle = generator.generate_instance(spec, temp_root / "generated")
        env = load_environment(bundle.root)
        collector = SnapshotCollector(temp_root / "snapshots", NamedSnapshotPolicy("every_step"), max_snapshots=1)
        action = Action(ActionType.SUBMIT, {"path_or_answer": "done"})
        snapshot = collector.maybe_capture(SnapshotContext("positive-demo", None, env.manifest, env.visible_root, env.root, 0, 8, 0.0, action, None, None, [{"role": "user", "content": env.manifest.instruction}], [], "before"))
        assert snapshot is not None
        snapshot_root = Path(snapshot.metadata["snapshot_root"])
        original = _candidate(snapshot, "original", snapshot.original_action)
        patch_path, patch_content = next(iter(env.manifest.reference_solution["files"].items()))
        corrected = _candidate(snapshot, "reference_guided", {"tool": "write_file", "args": {"path": patch_path, "content": patch_content}}, "reference_solution", privileged=True, rationale="deterministic known-good intervention")
        compile_pack([(snapshot, original, snapshot_root), (snapshot, corrected, snapshot_root)], output / "pack", "forced")
    tasks_list = read_branch_manifest(output / "pack" / "manifest.jsonl")
    outcomes = [replay_branch(task, ScriptedBaselineAgent(), output / "run", 0).outcome for task in tasks_list]
    comparisons = aggregate_outcomes(outcomes)
    write_jsonl(output / "run" / "outcomes.jsonl", [row.to_dict() for row in outcomes])
    write_jsonl(output / "run" / "comparisons.jsonl", [row.to_dict() for row in comparisons])
    comparison = comparisons[0]
    write_json(output / "summary.json", {"original_return": comparison.original_mean_return, "best_return": comparison.best_mean_return, "decision_regret": comparison.decision_regret, "recoverable": comparison.recoverable})
    tasks = {task.task_id: task for task in tasks_list}
    export_training_data(comparisons, tasks, output / "preference.jsonl", "preference", min_margin=.2)
    export_rl_taskset(comparisons, tasks, output / "rl-pack", min_regret=.2)


if __name__ == "__main__":
    main()