from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from synthetic_workspace_gym.agents import HeuristicBaselineAgent, ScriptedBaselineAgent
from synthetic_workspace_gym.analysis.benchmarking import build_benchmark_report, episode_to_row
from synthetic_workspace_gym.evaluators.registry import get_evaluator
from synthetic_workspace_gym.generators.common import normalize_difficulty
from synthetic_workspace_gym.generators.registry import get_generator, list_generators
from synthetic_workspace_gym.prime import get_tool_schemas, verify_workspace
from synthetic_workspace_gym.prime.clients import HeuristicReferenceClient, ScriptedPrimeClient
from synthetic_workspace_gym.prime.export import (
    build_manifest_row,
    export_split_pack,
    export_prime_pack,
    write_manifest_jsonl,
)
from synthetic_workspace_gym.prime.rollout import run_prime_branch_rollout, run_prime_rollout, run_prime_rollout_batch
from synthetic_workspace_gym.runtime.environment import load_environment
from synthetic_workspace_gym.runtime.runner import EpisodeRunner
from synthetic_workspace_gym.sandbox.evaluator import verify_workspace_in_sandbox
from synthetic_workspace_gym.sandbox.runner import build_sandbox_backend, docker_available
from synthetic_workspace_gym.sandbox.schemas import SandboxCommand, SandboxConfig
from synthetic_workspace_gym.splits import (
    build_split_manifest,
    default_split_policy,
    read_split_manifest,
    validate_split_manifest,
    write_split_jsonl,
    write_split_manifest,
)
from synthetic_workspace_gym.utils.io import write_json
from synthetic_workspace_gym.verifiers.compat import is_verifiers_available
from synthetic_workspace_gym.verifiers.registry import (
    SWG_VERIFIERS_ENV_IDS,
    list_environments as list_verifiers_environments,
    make_environment as make_verifiers_environment,
    register_with_verifiers,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="swg", description="Synthetic Workspace Gym CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="Generate one or more environments")
    generate.add_argument("--family", required=True, choices=list_generators())
    generate.add_argument("--count", type=int, default=1)
    generate.add_argument("--difficulty", default="3")
    generate.add_argument("--seed", type=int, default=1)
    generate.add_argument("--scenario")
    generate.add_argument("--split")
    generate.add_argument("--task-id")
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

    prime_export_splits = prime_subparsers.add_parser("export-splits", help="Export a Prime-compatible split pack")
    prime_export_splits.add_argument("--split-manifest", type=Path, required=True)
    prime_export_splits.add_argument("--output-dir", type=Path, required=True)
    prime_export_splits.add_argument("--export-name")
    prime_export_splits.add_argument("--overwrite", action="store_true")

    prime_manifest = prime_subparsers.add_parser("manifest", help="Rebuild manifest.jsonl from exported environments")
    prime_manifest.add_argument("--environments", type=Path, required=True)
    prime_manifest.add_argument("--output", type=Path, required=True)

    prime_verify = prime_subparsers.add_parser("verify", help="Verify a workspace with an exported SWG environment")
    prime_verify.add_argument("--environment", type=Path, required=True)
    prime_verify.add_argument("--workspace", type=Path, required=True)
    add_sandbox_args(prime_verify)

    prime_smoke = prime_subparsers.add_parser("smoke-test", help="Smoke-test an exported Prime environment")
    prime_smoke.add_argument("--environment", type=Path, required=True)
    add_sandbox_args(prime_smoke)

    prime_rollout = prime_subparsers.add_parser("rollout", help="Run one Prime-compatible model/tool rollout")
    prime_rollout.add_argument("--family")
    prime_rollout.add_argument("--scenario")
    prime_rollout.add_argument("--difficulty", type=int, default=3)
    prime_rollout.add_argument("--seed", type=int, default=0)
    prime_rollout.add_argument("--environment", type=Path)
    prime_rollout.add_argument("--client", choices=["scripted", "heuristic-reference"], default="scripted")
    prime_rollout.add_argument("--action-json", action="append", default=[])
    prime_rollout.add_argument("--output-dir", type=Path, default=Path("prime_rollouts"))
    prime_rollout.add_argument("--max-turns", type=int)
    prime_rollout.add_argument("--rollout-id")
    add_sandbox_args(prime_rollout)

    prime_branch = prime_subparsers.add_parser("branch-rollout", help="Run a forced or open counterfactual branch through a Prime model client")
    prime_branch.add_argument("--manifest", type=Path, required=True)
    prime_branch.add_argument("--task-id")
    prime_branch.add_argument("--task-index", type=int, default=0)
    prime_branch.add_argument("--mode", choices=["forced", "open"])
    prime_branch.add_argument("--client", choices=["scripted", "heuristic-reference"], default="scripted")
    prime_branch.add_argument("--action-json", action="append", default=[])
    prime_branch.add_argument("--output-dir", type=Path, default=Path("prime_branch_rollouts"))
    prime_branch.add_argument("--max-turns", type=int)
    prime_branch.add_argument("--rollout-id")
    add_sandbox_args(prime_branch)
    prime_rollout_batch = prime_subparsers.add_parser("rollout-batch", help="Run Prime rollouts from manifest.jsonl")
    prime_rollout_batch.add_argument("--manifest", type=Path, required=True)
    prime_rollout_batch.add_argument("--client", choices=["scripted", "heuristic-reference"], default="scripted")
    prime_rollout_batch.add_argument("--action-json", action="append", default=[])
    prime_rollout_batch.add_argument("--limit", type=int)
    prime_rollout_batch.add_argument("--output-dir", type=Path, default=Path("prime_rollouts"))
    prime_rollout_batch.add_argument("--max-turns", type=int)
    add_sandbox_args(prime_rollout_batch)

    sandbox = subparsers.add_parser("sandbox", help="Sandbox runtime commands")
    sandbox_subparsers = sandbox.add_subparsers(dest="sandbox_command", required=True)

    sandbox_build = sandbox_subparsers.add_parser("build-image", help="Build the SWG Docker runtime image")
    sandbox_build.add_argument("--tag", default="synthetic-workspace-gym-runtime:latest")

    sandbox_check = sandbox_subparsers.add_parser("check", help="Check Docker sandbox availability")
    sandbox_check.add_argument("--image", default="synthetic-workspace-gym-runtime:latest")
    sandbox_check.add_argument("--sandbox-user")

    sandbox_run = sandbox_subparsers.add_parser("run", help="Run a simple command in a sandbox")
    sandbox_run.add_argument("--backend", choices=["local", "docker"], default="local")
    sandbox_run.add_argument("--image", default="synthetic-workspace-gym-runtime:latest")
    sandbox_run.add_argument("--sandbox-user")
    sandbox_run.add_argument("--command", dest="sandbox_command_text", required=True)

    splits = subparsers.add_parser("splits", help="Dataset split manifest commands")
    splits_subparsers = splits.add_subparsers(dest="splits_command", required=True)

    splits_build = splits_subparsers.add_parser("build", help="Build a default split manifest")
    splits_build.add_argument("--output", type=Path, required=True)
    splits_build.add_argument("--assignments-output", type=Path)
    splits_build.add_argument("--families", default=",".join(list_generators()))
    splits_build.add_argument("--max-train", type=int)
    splits_build.add_argument("--max-validation", type=int)
    splits_build.add_argument("--max-test", type=int)
    splits_build.add_argument("--max-heldout", type=int)
    splits_build.add_argument("--shuffle", action="store_true")
    splits_build.add_argument("--shuffle-seed", type=int, default=0)

    splits_validate = splits_subparsers.add_parser("validate", help="Validate a split manifest")
    splits_validate.add_argument("--manifest", type=Path, required=True)

    splits_stats = splits_subparsers.add_parser("stats", help="Print split manifest counts")
    splits_stats.add_argument("--manifest", type=Path, required=True)

    verifiers = subparsers.add_parser("verifiers", help="Native verifiers integration commands")
    verifiers_subparsers = verifiers.add_subparsers(dest="verifiers_command", required=True)

    verifiers_subparsers.add_parser("list", help="List built-in SWG verifiers environment ids")
    verifiers_subparsers.add_parser("check", help="Check optional verifiers integration availability")

    verifiers_smoke = verifiers_subparsers.add_parser("smoke-test", help="Smoke-test a SWG verifiers environment")
    verifiers_smoke.add_argument("--env-id", default="swg.script_repair.csv_schema_drift")
    verifiers_smoke.add_argument("--difficulty", type=int, default=1)
    verifiers_smoke.add_argument("--seed", type=int, default=7)
    add_sandbox_args(verifiers_smoke)

    verifiers_export = verifiers_subparsers.add_parser("export-registry", help="Write SWG verifiers registry metadata")
    verifiers_export.add_argument("--output", type=Path, required=True)

    from synthetic_workspace_gym.counterfactual.cli import configure_parser
    configure_parser(subparsers)
    return parser


def add_sandbox_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--sandbox", choices=["local", "docker"], default="local")
    parser.add_argument("--docker-image")
    parser.add_argument("--sandbox-memory", default="1g")
    parser.add_argument("--sandbox-cpus", type=float, default=1.0)
    parser.add_argument("--sandbox-pids-limit", type=int, default=256)
    parser.add_argument("--sandbox-timeout", type=int, default=30)
    parser.add_argument("--sandbox-network", action="store_true")
    parser.add_argument("--sandbox-user")


def default_sandbox_user() -> str:
    if hasattr(os, "getuid") and hasattr(os, "getgid"):
        return f"{os.getuid()}:{os.getgid()}"
    return "1000:1000"


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
            split=getattr(args, "split", None),
            task_id=getattr(args, "task_id", None),
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


def command_prime_export_splits(args: argparse.Namespace) -> int:
    summary = export_split_pack(
        output_dir=args.output_dir,
        split_manifest_path=args.split_manifest,
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
    config = sandbox_config_from_args(args)
    payload = (
        verify_workspace(args.environment, args.workspace)
        if config.backend == "local"
        else verify_workspace_in_sandbox(args.environment, args.workspace, config)
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload.get("success") else 1


def command_prime_smoke_test(args: argparse.Namespace) -> int:
    config = sandbox_config_from_args(args)
    environment = load_environment(args.environment)
    visible_exists = environment.visible_root.exists() and environment.visible_root.is_dir()
    hidden_exists = environment.hidden_root.exists() and environment.hidden_root.is_dir()
    tool_schemas = get_tool_schemas(environment.manifest.tool_permissions.enabled_tools())
    docker_ok = docker_available() if config.backend == "docker" else None
    sandbox_smoke = None
    if config.backend == "docker" and docker_ok:
        with tempfile.TemporaryDirectory(prefix="swg-sandbox-smoke-") as tmp_dir:
            result = build_sandbox_backend(config).run(
                SandboxCommand(argv=["python", "-c", "print('swg sandbox ok')"], timeout_seconds=config.timeout_seconds),
                Path(tmp_dir),
            )
            sandbox_smoke = result.success
    payload = {
        "environment": str(args.environment),
        "env_id": environment.manifest.env_id,
        "manifest_loads": True,
        "visible_exists": visible_exists,
        "hidden_exists": hidden_exists,
        "tool_schema_count": len(tool_schemas),
        "sandbox": {
            "backend": config.backend,
            "image": config.image,
            "docker_available": docker_ok,
            "sandbox_smoke_test": sandbox_smoke,
        },
        "pass": bool(visible_exists and hidden_exists and tool_schemas and (sandbox_smoke is not False)),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["pass"] else 1


def parse_action_json_rows(values: list[str]) -> list[dict[str, object]]:
    return [json.loads(value) for value in values]


def build_prime_client(name: str, action_json: list[str] | None = None):
    if name == "heuristic-reference":
        return HeuristicReferenceClient()
    if name == "scripted":
        actions = parse_action_json_rows(action_json or [])
        if not actions:
            actions = [
                {"tool": "list_directory", "args": {"path": "."}},
                {"tool": "submit", "args": {"path_or_answer": "done"}},
            ]
        return ScriptedPrimeClient(actions)
    raise ValueError(f"Unsupported Prime client: {name}")


def sandbox_config_from_args(args: argparse.Namespace) -> SandboxConfig:
    return SandboxConfig(
        backend=getattr(args, "sandbox", "local"),
        image=getattr(args, "docker_image", None) or "synthetic-workspace-gym-runtime:latest",
        network_enabled=bool(getattr(args, "sandbox_network", False)),
        memory_limit=getattr(args, "sandbox_memory", "1g"),
        cpus=float(getattr(args, "sandbox_cpus", 1.0)),
        pids_limit=int(getattr(args, "sandbox_pids_limit", 256)),
        timeout_seconds=int(getattr(args, "sandbox_timeout", 30)),
        run_as_user=getattr(args, "sandbox_user", None) or default_sandbox_user(),
    )


def command_prime_rollout(args: argparse.Namespace) -> int:
    config = sandbox_config_from_args(args)
    result = run_prime_rollout(
        family=args.family,
        scenario=args.scenario,
        difficulty=args.difficulty,
        seed=args.seed,
        environment_path=args.environment,
        client=build_prime_client(args.client, args.action_json),
        output_dir=args.output_dir,
        max_turns=args.max_turns,
        rollout_id=args.rollout_id,
        sandbox_backend=config.backend,
        sandbox_config=config,
        docker_image=config.image,
    )
    payload = {
        "rollout_id": result["rollout_id"],
        "env_id": result["env_id"],
        "success": result["success"],
        "final_reward": result["final_reward"],
        "stopped_reason": result["stopped_reason"],
        "artifact_dir": result["artifact_dir"],
        "prime_rollout_path": result["prime_rollout_path"],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0



def command_prime_branch_rollout(args: argparse.Namespace) -> int:
    config = sandbox_config_from_args(args)
    result = run_prime_branch_rollout(
        args.manifest, task_id=args.task_id, task_index=args.task_index, branch_mode=args.mode,
        client=build_prime_client(args.client, args.action_json), output_dir=args.output_dir,
        max_turns=args.max_turns, rollout_id=args.rollout_id, sandbox_backend=config.backend,
        sandbox_config=config, docker_image=config.image,
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0
def command_prime_rollout_batch(args: argparse.Namespace) -> int:
    config = sandbox_config_from_args(args)
    summary = run_prime_rollout_batch(
        args.manifest,
        client_factory=lambda: build_prime_client(args.client, args.action_json),
        output_dir=args.output_dir,
        limit=args.limit,
        max_turns=args.max_turns,
        sandbox_backend=config.backend,
        sandbox_config=config,
        docker_image=config.image,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def command_sandbox_build_image(args: argparse.Namespace) -> int:
    command = ["docker", "build", "-f", "docker/Dockerfile.swg-runtime", "-t", args.tag, "."]
    completed = subprocess.run(command, capture_output=True, text=True)
    payload = {
        "success": completed.returncode == 0,
        "returncode": completed.returncode,
        "tag": args.tag,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if completed.returncode == 0 else 1


def command_sandbox_check(args: argparse.Namespace) -> int:
    available = docker_available()
    image_available = False
    sandbox_smoke_test = False
    if available:
        image_check = subprocess.run(["docker", "image", "inspect", args.image], capture_output=True, text=True)
        image_available = image_check.returncode == 0
        if image_available:
            with tempfile.TemporaryDirectory(prefix="swg-sandbox-check-") as tmp_dir:
                config = SandboxConfig(
                    backend="docker",
                    image=args.image,
                    run_as_user=args.sandbox_user or default_sandbox_user(),
                )
                result = build_sandbox_backend(config).run(
                    SandboxCommand(argv=["python", "-c", "print('swg sandbox ok')"]),
                    Path(tmp_dir),
                )
                sandbox_smoke_test = result.success
    payload = {
        "docker_available": available,
        "image": args.image,
        "image_available": image_available,
        "sandbox_smoke_test": sandbox_smoke_test,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if available and image_available and sandbox_smoke_test else 1


def command_sandbox_run(args: argparse.Namespace) -> int:
    config = SandboxConfig(
        backend=args.backend,
        image=args.image,
        run_as_user=args.sandbox_user or default_sandbox_user(),
    )
    backend = build_sandbox_backend(config)
    with tempfile.TemporaryDirectory(prefix="swg-sandbox-run-") as tmp_dir:
        result = backend.run(SandboxCommand(argv=shlex.split(args.sandbox_command_text)), Path(tmp_dir))
    print(json.dumps(result.to_public_dict(), indent=2, sort_keys=True))
    return 0 if result.success else 1


def command_splits_build(args: argparse.Namespace) -> int:
    families = parse_comma_separated(args.families)
    max_per_split = {
        "train": args.max_train,
        "validation": args.max_validation,
        "test": args.max_test,
        "heldout": args.max_heldout,
    }
    max_per_split = {key: value for key, value in max_per_split.items() if value is not None}
    manifest = build_split_manifest(
        "synthetic-workspace-gym-default-splits",
        default_split_policy(families=families),
        max_per_split=max_per_split,
        shuffle=args.shuffle,
        shuffle_seed=args.shuffle_seed,
        metadata={"families": families, "policy": "default"},
    )
    manifest_path = write_split_manifest(args.output, manifest)
    assignments_path = (
        write_split_jsonl(args.assignments_output, manifest.assignments)
        if args.assignments_output is not None
        else None
    )
    validation = validate_split_manifest(manifest)
    payload = {
        "manifest_path": str(manifest_path),
        "assignments_path": str(assignments_path) if assignments_path is not None else None,
        "assignment_count": len(manifest.assignments),
        "validation": validation,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if validation["valid"] else 1


def command_splits_validate(args: argparse.Namespace) -> int:
    payload = validate_split_manifest(read_split_manifest(args.manifest))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["valid"] else 1


def command_splits_stats(args: argparse.Namespace) -> int:
    manifest = read_split_manifest(args.manifest)
    counts: dict[str, Any] = {
        "by_split": {},
        "by_family": {},
        "by_scenario": {},
        "by_difficulty": {},
    }
    for assignment in manifest.assignments:
        counts["by_split"][assignment.split] = counts["by_split"].get(assignment.split, 0) + 1
        family_key = f"{assignment.split}:{assignment.family}"
        scenario_key = f"{assignment.split}:{assignment.family}:{assignment.scenario or 'default'}"
        difficulty_key = f"{assignment.split}:d{assignment.difficulty}"
        counts["by_family"][family_key] = counts["by_family"].get(family_key, 0) + 1
        counts["by_scenario"][scenario_key] = counts["by_scenario"].get(scenario_key, 0) + 1
        counts["by_difficulty"][difficulty_key] = counts["by_difficulty"].get(difficulty_key, 0) + 1
    print(json.dumps(counts, indent=2, sort_keys=True))
    return 0


def command_verifiers_list(args: argparse.Namespace) -> int:
    print(json.dumps({"environments": list_verifiers_environments()}, indent=2, sort_keys=True))
    return 0


def command_verifiers_check(args: argparse.Namespace) -> int:
    payload = {
        "verifiers_available": is_verifiers_available(),
        "native_adapter_available": True,
        "registered_env_count": len(list_verifiers_environments()),
        "registered_with_verifiers": register_with_verifiers(),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def command_verifiers_export_registry(args: argparse.Namespace) -> int:
    rows = [
        {
            "env_id": env_id,
            **config,
        }
        for env_id, config in sorted(SWG_VERIFIERS_ENV_IDS.items())
    ]
    write_json(args.output, {"environments": rows})
    print(json.dumps({"environment_count": len(rows), "output": str(args.output)}, indent=2, sort_keys=True))
    return 0


def command_verifiers_smoke_test(args: argparse.Namespace) -> int:
    config = sandbox_config_from_args(args)
    env = make_verifiers_environment(
        args.env_id,
        difficulty=args.difficulty,
        seed=args.seed,
        sandbox_backend=config.backend,
        sandbox_config=config,
        docker_image=config.image,
    )
    try:
        reset = env.reset()
        first = env.step({"tool": "list_directory", "args": {"path": "."}})
        second = env.step({"tool": "submit", "args": {"path_or_answer": "done"}})
        reward_payload = env.evaluate()
    finally:
        env.close()
    payload = {
        "env_id": args.env_id,
        "reset": {
            "env_id": reset.get("env_id"),
            "instruction_present": bool(reset.get("instruction")),
            "tool_count": len(reset.get("tools", [])),
        },
        "first_step": {
            "done": first.get("done"),
            "reward": first.get("reward"),
            "success": (first.get("info") or {}).get("success") if isinstance(first.get("info"), dict) else None,
        },
        "submit_step": {
            "done": second.get("done"),
            "reward": second.get("reward"),
        },
        "reward_payload": reward_payload,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
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
    if args.command == "prime":
        if args.prime_command == "export":
            return command_prime_export(args)
        if args.prime_command == "export-splits":
            return command_prime_export_splits(args)
        if args.prime_command == "manifest":
            return command_prime_manifest(args)
        if args.prime_command == "verify":
            return command_prime_verify(args)
        if args.prime_command == "smoke-test":
            return command_prime_smoke_test(args)
        if args.prime_command == "rollout":
            return command_prime_rollout(args)
        if args.prime_command == "branch-rollout":
            return command_prime_branch_rollout(args)
        if args.prime_command == "rollout-batch":
            return command_prime_rollout_batch(args)
    if args.command == "sandbox":
        if args.sandbox_command == "build-image":
            return command_sandbox_build_image(args)
        if args.sandbox_command == "check":
            return command_sandbox_check(args)
        if args.sandbox_command == "run":
            return command_sandbox_run(args)
    if args.command == "splits":
        if args.splits_command == "build":
            return command_splits_build(args)
        if args.splits_command == "validate":
            return command_splits_validate(args)
        if args.splits_command == "stats":
            return command_splits_stats(args)
    if args.command == "verifiers":
        if args.verifiers_command == "list":
            return command_verifiers_list(args)
        if args.verifiers_command == "check":
            return command_verifiers_check(args)
        if args.verifiers_command == "smoke-test":
            return command_verifiers_smoke_test(args)
        if args.verifiers_command == "export-registry":
            return command_verifiers_export_registry(args)
    if args.command == "counterfactual":
        from synthetic_workspace_gym.counterfactual.cli import dispatch
        return dispatch(args, get_agent)
    raise SystemExit(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
