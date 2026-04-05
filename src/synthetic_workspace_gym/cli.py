from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from synthetic_workspace_gym.agents import HeuristicBaselineAgent, ScriptedBaselineAgent
from synthetic_workspace_gym.evaluators.registry import get_evaluator
from synthetic_workspace_gym.generators.common import normalize_difficulty
from synthetic_workspace_gym.generators.registry import get_generator
from synthetic_workspace_gym.runtime.environment import load_environment
from synthetic_workspace_gym.runtime.runner import EpisodeRunner
from synthetic_workspace_gym.utils.io import write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="swg", description="Synthetic Workspace Gym CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="Generate one or more environments")
    generate.add_argument("--family", required=True, choices=["tabular", "script_repair", "pipeline"])
    generate.add_argument("--count", type=int, default=1)
    generate.add_argument("--difficulty", default="3")
    generate.add_argument("--seed", type=int, default=1)
    generate.add_argument("--output-dir", type=Path, default=Path("generated"))
    generate.add_argument("--skip-validate", action="store_true")

    run = subparsers.add_parser("run", help="Run a single episode with a baseline agent")
    run.add_argument("--environment", type=Path, required=True)
    run.add_argument(
        "--agent",
        choices=["scripted", "heuristic"],
        default="heuristic",
        help="Baseline agent to run: 'scripted' is a weak smoke test, 'heuristic' is the privileged validation baseline.",
    )
    run.add_argument("--output-dir", type=Path, default=Path("episodes"))

    evaluate = subparsers.add_parser("evaluate", help="Evaluate an environment workspace")
    evaluate.add_argument("--environment", type=Path, required=True)
    evaluate.add_argument("--workspace", type=Path)

    benchmark = subparsers.add_parser("benchmark", help="Run a baseline across a directory of environments")
    benchmark.add_argument("--environments", type=Path, required=True)
    benchmark.add_argument(
        "--agent",
        choices=["scripted", "heuristic"],
        default="heuristic",
        help="Baseline agent to run: 'scripted' is a weak smoke test, 'heuristic' is the privileged validation baseline.",
    )
    benchmark.add_argument("--output-dir", type=Path, default=Path("benchmarks"))

    return parser


def get_agent(name: str):
    if name == "scripted":
        return ScriptedBaselineAgent()
    if name in {"heuristic", "react"}:
        return HeuristicBaselineAgent()
    raise ValueError(f"Unsupported agent: {name}")


def command_generate(args: argparse.Namespace) -> int:
    difficulty = normalize_difficulty(args.difficulty)
    generator = get_generator(args.family)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifests = []
    for index in range(args.count):
        seed = args.seed + index
        spec = generator.sample_spec(difficulty=difficulty, seed=seed)
        bundle = generator.generate_instance(spec, args.output_dir, validate=not args.skip_validate)
        manifests.append(bundle.manifest.to_dict())
    print(json.dumps({"generated": manifests}, indent=2, sort_keys=True))
    return 0


def command_run(args: argparse.Namespace) -> int:
    environment = load_environment(args.environment)
    runner = EpisodeRunner(output_root=args.output_dir)
    summary = runner.run_episode(environment, get_agent(args.agent))
    print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
    return 0


def command_evaluate(args: argparse.Namespace) -> int:
    environment = load_environment(args.environment)
    workspace = args.workspace.resolve() if args.workspace else environment.visible_root
    evaluator = get_evaluator(
        environment.manifest.family,
        evaluator_entrypoint=environment.manifest.evaluator_entrypoint,
    )
    result = evaluator.evaluate(workspace, environment.manifest, environment.hidden_root)
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0 if result.success else 1


def command_benchmark(args: argparse.Namespace) -> int:
    env_roots = sorted(path.parent for path in args.environments.rglob("manifest.json"))
    if not env_roots:
        raise SystemExit(f"No environments found under {args.environments}")
    runner = EpisodeRunner(output_root=args.output_dir / "episodes")
    agent_name = get_agent(args.agent).name
    summaries = []
    for env_root in env_roots:
        environment = load_environment(env_root)
        summary = runner.run_episode(environment, get_agent(args.agent))
        summaries.append(summary)
    success_rate = sum(1 for summary in summaries if summary.evaluation.success) / len(summaries)
    mean_score = sum(summary.evaluation.score for summary in summaries) / len(summaries)
    aggregate = {
        "agent": agent_name,
        "environment_count": len(summaries),
        "success_rate": round(success_rate, 4),
        "mean_score": round(mean_score, 4),
        "episodes": [summary.to_dict() for summary in summaries],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / f"benchmark-{agent_name}-{int(time.time())}.json", aggregate)
    print(json.dumps(aggregate, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "generate":
        return command_generate(args)
    if args.command == "run":
        return command_run(args)
    if args.command == "evaluate":
        return command_evaluate(args)
    if args.command == "benchmark":
        return command_benchmark(args)
    raise SystemExit(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
