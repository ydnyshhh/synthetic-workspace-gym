from __future__ import annotations

import argparse
import json
from pathlib import Path

from synthetic_workspace_gym.runtime.environment import load_environment
from synthetic_workspace_gym.runtime.runner import EpisodeRunner
from synthetic_workspace_gym.utils.io import read_json, write_json, write_jsonl

from .analysis import aggregate_outcomes
from .candidates import generate_candidates
from .compiler import compile_pack
from .exports import export_rl_taskset, export_training_data, read_comparisons
from .runner import read_branch_manifest, run_branches
from .schemas import BranchOutcome, CounterfactualSnapshot
from .selectors import SELECTORS
from .snapshots import NamedSnapshotPolicy, SnapshotCollector, load_snapshot


def configure_parser(subparsers: argparse._SubParsersAction) -> None:
    counterfactual = subparsers.add_parser("counterfactual", help="Counterfactual trajectory branching commands")
    cf = counterfactual.add_subparsers(dest="counterfactual_command", required=True)
    collect = cf.add_parser("collect"); collect.add_argument("--environment", type=Path, required=True); collect.add_argument("--agent", choices=["scripted", "heuristic"], default="scripted"); collect.add_argument("--snapshot-policy", choices=["none", "every_step", "writes", "checks", "submits", "writes_checks_submit", "selected"], default="writes_checks_submit"); collect.add_argument("--max-snapshots", type=int, default=3); collect.add_argument("--max-signal-snapshots", type=int, default=2); collect.add_argument("--intermediate-evaluation", action="store_true"); collect.add_argument("--output-dir", type=Path, required=True)
    build = cf.add_parser("build"); build.add_argument("--snapshots", type=Path, required=True); build.add_argument("--selectors", default="before_first_write,before_submit"); build.add_argument("--candidates", default="original,submit,run_public_check,read_relevant_file"); build.add_argument("--mode", choices=["forced", "open"], default="forced"); build.add_argument("--max-branch-points", type=int, default=2, help="Maximum selected states per trajectory"); build.add_argument("--max-branch-points-total", type=int); build.add_argument("--max-candidates", type=int, default=4); build.add_argument("--output-dir", type=Path, required=True)
    run = cf.add_parser("run"); run.add_argument("--manifest", type=Path, required=True); run.add_argument("--client", choices=["scripted", "heuristic"], default="scripted"); run.add_argument("--rollouts-per-branch", type=int, default=1); run.add_argument("--output-dir", type=Path, required=True)
    analyze = cf.add_parser("analyze"); analyze.add_argument("--outcomes", type=Path, required=True); analyze.add_argument("--recoverable-threshold", type=float, default=.95); analyze.add_argument("--optimality-tolerance", type=float, default=.05); analyze.add_argument("--output", type=Path, required=True)
    export = cf.add_parser("export"); export.add_argument("--comparisons", type=Path, required=True); export.add_argument("--branch-manifest", type=Path, required=True); export.add_argument("--format", choices=["sft", "preference", "critic", "rl-taskset"], required=True); export.add_argument("--min-margin", type=float, default=.2); export.add_argument("--min-regret", type=float, default=.2); export.add_argument("--include-privileged", action="store_true", help="Include targets derived from privileged reference data (excluded by default)"); export.add_argument("--output", type=Path, required=True)
    inspect = cf.add_parser("inspect"); inspect.add_argument("--comparisons", type=Path, required=True); inspect.add_argument("--comparison-id", required=True)


def dispatch(args: argparse.Namespace, get_agent) -> int:
    command = args.counterfactual_command
    if command == "collect":
        collector = SnapshotCollector(args.output_dir / "snapshots", NamedSnapshotPolicy(args.snapshot_policy), args.max_snapshots, args.intermediate_evaluation, args.max_signal_snapshots)
        summary = EpisodeRunner(args.output_dir / "episodes", collector).run_episode(load_environment(args.environment), get_agent(args.agent))
        print(json.dumps({"episode": summary.to_dict(), "snapshots": [x.to_dict() for x in collector.snapshots]}, indent=2)); return 0
    if command == "build":
        roots = sorted({path.parent for path in args.snapshots.rglob("snapshot.json")})
        snapshots = [(load_snapshot(root), root) for root in roots]
        selected_set = set(select_branch_snapshot_ids(
            [snapshot for snapshot, _ in snapshots], _csv(args.selectors),
            args.max_branch_points, args.max_branch_points_total,
        ))
        items = []
        for snapshot, root in [(s, r) for s, r in snapshots if s.snapshot_id in selected_set]:
            manifest = load_environment(root).manifest
            for candidate in generate_candidates(snapshot, manifest, root, _csv(args.candidates), args.max_candidates):
                if args.mode == "forced" and candidate.action is None: continue
                items.append((snapshot, candidate, root))
        tasks = compile_pack(items, args.output_dir, args.mode); print(json.dumps({"task_count": len(tasks), "manifest": str(args.output_dir / "manifest.jsonl")}, indent=2)); return 0
    if command == "run":
        outcomes = run_branches(read_branch_manifest(args.manifest), lambda: get_agent(args.client), args.rollouts_per_branch, args.output_dir); print(json.dumps({"outcome_count": len(outcomes)}, indent=2)); return 0
    if command == "analyze":
        outcomes = [BranchOutcome.from_dict(json.loads(line)) for line in args.outcomes.read_text(encoding="utf-8").splitlines() if line.strip()]
        comparisons = aggregate_outcomes(outcomes, args.recoverable_threshold, args.optimality_tolerance); write_jsonl(args.output, [x.to_dict() for x in comparisons]); print(json.dumps({"comparison_count": len(comparisons)}, indent=2)); return 0
    comparisons = read_comparisons(args.comparisons)
    if command == "inspect":
        item = next(x for x in comparisons if x.branch_group_id == args.comparison_id); print(json.dumps(item.to_dict(), indent=2)); return 0
    tasks_list = read_branch_manifest(args.branch_manifest); tasks = {x.task_id: x for x in tasks_list}
    if args.format == "rl-taskset": records = export_rl_taskset(comparisons, tasks, args.output, args.min_regret)
    else: records = export_training_data(comparisons, tasks, args.output, args.format, args.min_margin, exclude_privileged=not args.include_privileged)
    print(json.dumps({"record_count": len(records), "output": str(args.output)}, indent=2)); return 0


def select_branch_snapshot_ids(
    snapshots: list[CounterfactualSnapshot],
    selector_names: list[str],
    max_branch_points: int,
    max_branch_points_total: int | None = None,
) -> list[str]:
    by_trajectory: dict[str, list[CounterfactualSnapshot]] = {}
    for snapshot in snapshots:
        by_trajectory.setdefault(str(snapshot.trajectory_id), []).append(snapshot)
    selected: list[str] = []
    for trajectory_snapshots in by_trajectory.values():
        trajectory_ids: list[str] = []
        for name in selector_names:
            if name not in SELECTORS:
                raise ValueError(f"unknown selector: {name}")
            trajectory_ids.extend(item.snapshot_id for item in SELECTORS[name].select(trajectory_snapshots))
        selected.extend(list(dict.fromkeys(trajectory_ids))[:max_branch_points])
        if max_branch_points_total is not None and len(selected) >= max_branch_points_total:
            return selected[:max_branch_points_total]
    return selected


def _csv(value: str) -> list[str]:
    return [x.strip().replace("-", "_") for x in value.split(",") if x.strip()]
