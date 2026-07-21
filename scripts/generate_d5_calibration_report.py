from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from synthetic_workspace_gym.analysis.oracle_calibration import (
    evaluate_ordered_oracle_states,
)
from synthetic_workspace_gym.evaluators.registry import get_evaluator
from synthetic_workspace_gym.generators.registry import get_generator
from synthetic_workspace_gym.generators.retrieval_profile_scenarios import (
    build_profiled_retrieval_scenario,
)
from synthetic_workspace_gym.generators.tabular_program_synthesis import (
    build_account_event_program_scenario,
)
from synthetic_workspace_gym.utils.io import write_json


PROFILE_AWARE_SEEDS = (100, 103, 106, 108, 109)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/d5-calibration-report.json"),
    )
    parser.add_argument(
        "--scratch-root",
        type=Path,
        default=Path(".tmp-tests/d5-calibration"),
    )
    args = parser.parse_args()
    if args.scratch_root.exists():
        raise FileExistsError(
            f"Scratch root already exists; choose a fresh path: {args.scratch_root}"
        )
    args.scratch_root.mkdir(parents=True)

    reports: list[dict[str, object]] = []
    for family in ("pipeline", "retrieval_workspace", "tabular"):
        generator = get_generator(family)
        for seed in PROFILE_AWARE_SEEDS:
            scenario_id = (
                generator.resolve_scenario_id(difficulty=5, seed=seed)
                if family == "retrieval_workspace"
                else {
                    "pipeline": "team_hours_pipeline",
                    "tabular": "account_event_program_synthesis",
                }[family]
            )
            spec = generator.sample_spec(
                difficulty=5,
                seed=seed,
                scenario_id=scenario_id,
                generation_params={"composition_mode": "hard_atomic"},
            )
            instance_root = args.scratch_root / family / f"seed-{seed}"
            bundle = generator.generate_instance(spec, instance_root / "generated")
            scenario = _scenario(generator, family, spec)
            realization = dict(bundle.manifest.metadata["difficulty_realization"])
            report = evaluate_ordered_oracle_states(
                task_id=(
                    f"swg.calibration.{family}.{scenario_id}.d5."
                    f"{realization['profile']}.s{seed}"
                ),
                correct_files=dict(
                    scenario["correct_files"]
                    if "correct_files" in scenario
                    else scenario["files"]
                ),
                selected_bugs=list(scenario["bugs"]),
                visible_template=bundle.visible_root,
                scratch_root=instance_root / "states",
                evaluator=get_evaluator(
                    family,
                    evaluator_entrypoint=bundle.manifest.evaluator_entrypoint,
                ),
                manifest=bundle.manifest,
                hidden_root=bundle.hidden_root,
                capability_count=int(realization.get("capability_count", 0)),
                semantic_dependency_depth=int(
                    realization.get("semantic_dependency_depth", 0)
                ),
            )
            report.update(
                {
                    "family": family,
                    "seed": seed,
                    "profile": realization["profile"],
                }
            )
            reports.append(report)

    payload = {
        "schema_version": "d5-calibration-v1",
        "seeds": list(PROFILE_AWARE_SEEDS),
        "task_count": len(reports),
        "monotonic_task_count": sum(bool(report["monotonic"]) for report in reports),
        "violation_count": sum(len(report["violations"]) for report in reports),
        "tasks": reports,
    }
    write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["violation_count"] == 0 else 2


def _scenario(generator, family: str, spec):
    if family == "pipeline":
        return generator.team_hours_pipeline_scenario(random.Random(spec.seed), spec)
    if family == "retrieval_workspace":
        return build_profiled_retrieval_scenario(
            random.Random(f"{spec.seed}:profiled_retrieval"), spec
        )
    return build_account_event_program_scenario(
        random.Random(f"{spec.seed}:account_event_program_synthesis"), spec
    )


if __name__ == "__main__":
    raise SystemExit(main())
