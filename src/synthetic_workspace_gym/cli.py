from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from synthetic_workspace_gym.agents import HeuristicBaselineAgent, ScriptedBaselineAgent
from synthetic_workspace_gym.analysis.benchmarking import build_benchmark_report, episode_to_row
from synthetic_workspace_gym.evaluators.registry import get_evaluator
from synthetic_workspace_gym.generators.common import normalize_difficulty
from synthetic_workspace_gym.generators.registry import get_generator, list_generators
from synthetic_workspace_gym.prime import get_tool_schemas, verify_workspace
from synthetic_workspace_gym.prime.export import (
    build_manifest_row,
    export_prime_pack,
    write_manifest_jsonl,
)
from synthetic_workspace_gym.runtime.environment import load_environment
from synthetic_workspace_gym.runtime.runner import EpisodeRunner
from synthetic_workspace_gym.utils.io import write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="swg", description="Synthetic Workspace Gym CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="Generate one or more environments")
    generate.add_argument("--family", required=True, choices=list_generators())
    generate.add_argument("--count", type=int, default=1)
    generate.add_argument("--difficulty", default="3")
    generate.add_argument("--seed", type=int, default=1)
    generate.add_argument("--scenario")
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

    prime = subparsers.add_parser("prime", help="Prime/verifiers integration commands")
    prime_subparsers = prime.add_subparsers(dest="prime_command", required=True)

    prime_export = prime_subparsers.add_parser("export", help="Export a Prime-compatible environment pack")
    prime_export.add_argument("--output-dir", type=Path, required=True)
    prime_export.add_argument("--existing-environments", type=Path)
    prime_export.add_argument("--families", default=",".join(list_generators()))
    prime_export.add_argument("--difficulties", default="1,2,3")
    prime_export.add_argument("--seeds", default="0:10")
    prime_export.add_argument("--export-name")
    prime_export.add_argument("--overwrite", action="store_true")

    prime_manifest = prime_subparsers.add_parser("manifest", help="Rebuild manifest.jsonl from exported environments")
    prime_manifest.add_argument("--environments", type=Path, required=True)
    prime_manifest.add_argument("--output", type=Path, required=True)

    prime_verify = prime_subparsers.add_parser("verify", help="Verify a workspace with an exported SWG environment")
    prime_verify.add_argument("--environment", type=Path, required=True)
    prime_verify.add_argument("--workspace", type=Path, required=True)

    prime_smoke = prime_subparsers.add_parser("smoke-test", help="Smoke-test an exported Prime environment")
    prime_smoke.add_argument("--environment", type=Path, required=True)

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
        spec = generator.sample_spec(
            difficulty=difficulty,
            seed=seed,
            scenario_id=getattr(args, "scenario", None),
        )
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
    rows = []
    for env_root in env_roots:
        environment = load_environment(env_root)
        summary = runner.run_episode(environment, get_agent(args.agent))
        rows.append(episode_to_row(summary, environment.manifest))
    aggregate = build_benchmark_report(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / f"benchmark-{aggregate['agent']}-{int(time.time())}.json", aggregate)
    print(json.dumps(aggregate, indent=2, sort_keys=True))
    return 0


def parse_comma_separated(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_int_list_or_range(value: str) -> list[int]:
    value = value.strip()
    if ":" in value:
        start_text, end_text = value.split(":", 1)
        start = int(start_text)
        end = int(end_text)
        step = 1 if end >= start else -1
        return list(range(start, end, step))
    return [int(item) for item in parse_comma_separated(value)]


def parse_difficulty_spec(value: str) -> list[int]:
    value = value.strip()
    if ":" in value:
        start_text, end_text = value.split(":", 1)
        start = int(start_text)
        end = int(end_text)
        step = 1 if end >= start else -1
        difficulties = list(range(start, end + step, step))
    else:
        difficulties = parse_int_list_or_range(value)
    for difficulty in difficulties:
        if not 1 <= difficulty <= 5:
            raise ValueError("difficulty values must be between 1 and 5")
    return difficulties


def command_prime_export(args: argparse.Namespace) -> int:
    summary = export_prime_pack(
        output_dir=args.output_dir,
        existing_environments_dir=args.existing_environments,
        families=parse_comma_separated(args.families),
        difficulties=parse_difficulty_spec(args.difficulties),
        seeds=parse_int_list_or_range(args.seeds),
        export_name=args.export_name,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if not summary.get("errors") else 1


def command_prime_manifest(args: argparse.Namespace) -> int:
    environments_root = args.environments.resolve()
    export_root = environments_root.parent
    environment_paths = sorted(path.parent for path in environments_root.rglob("manifest.json"))
    rows = [build_manifest_row(path, export_root) for path in environment_paths]
    write_manifest_jsonl(args.output, rows)
    print(
        json.dumps(
            {
                "environment_count": len(rows),
                "manifest_path": str(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def command_prime_verify(args: argparse.Namespace) -> int:
    payload = verify_workspace(args.environment, args.workspace)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload.get("success") else 1


def command_prime_smoke_test(args: argparse.Namespace) -> int:
    environment = load_environment(args.environment)
    visible_exists = environment.visible_root.exists() and environment.visible_root.is_dir()
    hidden_exists = environment.hidden_root.exists() and environment.hidden_root.is_dir()
    tool_schemas = get_tool_schemas(environment.manifest.tool_permissions.enabled_tools())
    payload = {
        "environment": str(args.environment),
        "env_id": environment.manifest.env_id,
        "manifest_loads": True,
        "visible_exists": visible_exists,
        "hidden_exists": hidden_exists,
        "tool_schema_count": len(tool_schemas),
        "pass": bool(visible_exists and hidden_exists and tool_schemas),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["pass"] else 1


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
    if args.command == "prime":
        if args.prime_command == "export":
            return command_prime_export(args)
        if args.prime_command == "manifest":
            return command_prime_manifest(args)
        if args.prime_command == "verify":
            return command_prime_verify(args)
        if args.prime_command == "smoke-test":
            return command_prime_smoke_test(args)
    raise SystemExit(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
