from __future__ import annotations
import copy
import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "src/synthetic_workspace_gym/frozen_manifests"
V1_OUT = (
    ROOT
    / "environments/tool_use/synthetic_workspace_gym/synthetic_workspace_gym/frozen_manifests"
)
INDEX = ROOT / "configs/releases/swg-0.2.0.dev1-experiment-index.json"
TRAIN_CONFIGS = ROOT / "configs/rl/qwen35-4b-matrix"
EVAL_CONFIGS = ROOT / "configs/evals/qwen35-4b-matrix"
VERSION = "0.2.0.dev1"
CREATED_AT = "2026-07-27T00:00:00+05:30"
ORIGINAL = {
    "tabular": [
        "monthly_segment_report",
        "channel_status_pivot",
        "weekly_refund_rollup",
    ],
    "script_repair": [
        "inventory_report",
        "path_batch",
        "csv_schema_drift",
        "timestamp_normalization",
    ],
    "pipeline": [
        "team_hours_pipeline",
        "sales_csv_pipeline",
        "artifact_stitch_pipeline",
    ],
    "retrieval_workspace": [
        "service_config_reconciliation",
        "migration_plan_bundle",
        "incident_report_bundle",
    ],
}
HELDOUT = {
    "tabular": ["supplier_restock_summary"],
    "script_repair": ["team_roster_export"],
    "pipeline": ["quality_gate_pipeline"],
    "retrieval_workspace": ["client_adapter_sync"],
}
COMPOSITE = "retrieval_guided_pipeline_repair"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def row(split, family, scenario, difficulty, seed, experiment):
    task_id = f"swg.{split}.{family}.{scenario}.d{difficulty}.s{seed}"
    core = {
        "split": split,
        "family": family,
        "scenario": scenario,
        "difficulty": difficulty,
        "seed": seed,
        "task_id": task_id,
    }
    contributors = (
        ["retrieval_workspace", "pipeline"]
        if family == "composite_workspace"
        else [family]
    )
    return {
        **core,
        "env_id": None,
        "metadata": {
            "experiment": experiment,
            "contributing_families": contributors,
            "assignment_fingerprint": digest(core),
        },
    }


def pool(families, split, difficulties, seeds, experiment):
    return [
        row(split, family, scenario, difficulty, seed, experiment)
        for family, scenarios in families.items()
        for scenario in scenarios
        for difficulty in difficulties
        for seed in seeds
    ]


def balanced(items, count, shuffle_seed):
    groups = {}
    for item in items:
        groups.setdefault(
            (item["family"], item["scenario"], item["difficulty"]), []
        ).append(copy.deepcopy(item))
    rng = random.Random(shuffle_seed)
    keys = sorted(groups)
    rng.shuffle(keys)
    for values in groups.values():
        rng.shuffle(values)
    selected = []
    while len(selected) < count:
        progress = False
        for key in keys:
            if groups[key]:
                selected.append(groups[key].pop())
                progress = True
                if len(selected) == count:
                    break
        if not progress:
            raise ValueError("not enough unique tasks")
    return selected


def balanced_by_family(items, count, shuffle_seed):
    families = sorted({item["family"] for item in items})
    random.Random(shuffle_seed).shuffle(families)
    base, remainder = divmod(count, len(families))
    selected = []
    for index, family in enumerate(families):
        family_items = [item for item in items if item["family"] == family]
        selected.extend(
            balanced(family_items, base + int(index < remainder), shuffle_seed + index)
        )
    random.Random(shuffle_seed).shuffle(selected)
    return selected


def freeze(name, assignments, metadata):
    body = {
        "name": name,
        "version": "v1",
        "created_at": CREATED_AT,
        "split_specs": {},
        "assignments": assignments,
    }
    body["metadata"] = {
        "environment_version": VERSION,
        "frozen": True,
        "assignment_count": len(assignments),
        "manifest_fingerprint": digest(body),
        **metadata,
    }
    return body


def build():
    result = {}
    for family, scenarios in ORIGINAL.items():
        name = f"train-specialist-{family}"
        result[name] = freeze(
            name,
            balanced(
                pool({family: scenarios}, "train", [2, 3, 4], range(80), name), 512, 42
            ),
            {"curriculum": "specialist", "shuffle_seed": 42, "composite_fraction": 0.0},
        )
    originals = pool(ORIGINAL, "train", [2, 3, 4], range(80), "all-family")
    composites = pool(
        {"composite_workspace": [COMPOSITE]},
        "train",
        [2, 3, 4],
        range(80),
        "composition",
    )
    for seed in (42, 43):
        name = f"train-all-family-seed-{seed}"
        chosen = balanced_by_family(originals, 512, seed)
        for item in chosen:
            item["metadata"]["experiment"] = name
        result[name] = freeze(
            name,
            chosen,
            {
                "curriculum": "all_family",
                "shuffle_seed": seed,
                "composite_fraction": 0.0,
            },
        )
        name = f"train-composition-20pct-seed-{seed}"
        chosen = balanced_by_family(originals, 410, seed) + balanced(
            composites, 102, seed
        )
        random.Random(seed).shuffle(chosen)
        for item in chosen:
            item["metadata"]["experiment"] = name
        result[name] = freeze(
            name,
            chosen,
            {
                "curriculum": "composition_augmented",
                "shuffle_seed": seed,
                "original_count": 410,
                "composite_count": 102,
                "composite_fraction": 102 / 512,
            },
        )
    name = "eval-id-d3-d5"
    result[name] = freeze(
        name,
        pool(ORIGINAL, "test", [3, 4, 5], range(90, 100), name),
        {"panel": "in_distribution"},
    )
    name = "eval-scenario-heldout"
    result[name] = freeze(
        name,
        pool(HELDOUT, "heldout", [3, 4, 5], range(100, 120), name),
        {"panel": "scenario_heldout"},
    )
    name = "eval-d5-panel-24"
    panel = []
    for family, scenarios in ORIGINAL.items():
        for index in range(6):
            panel.append(
                row(
                    "test",
                    family,
                    scenarios[index % len(scenarios)],
                    5,
                    190 + index,
                    name,
                )
            )
    result[name] = freeze(name, panel, {"panel": "frozen_d5", "rollouts_per_task": 5})

    difficulty_band_panels = {
        "eval-d1-d2-panel-24": [
            row("heldout", family, scenarios[0], difficulty, seed, "eval-d1-d2-panel-24")
            for family, scenarios in HELDOUT.items()
            for difficulty in (1, 2)
            for seed in range(120, 123)
        ],
        "eval-d3-d4-panel-24": [
            row(
                "heldout",
                family,
                scenarios[0],
                difficulty,
                seed,
                "eval-d3-d4-panel-24",
            )
            for family, scenarios in HELDOUT.items()
            for difficulty, seeds in ((3, range(123, 126)), (4, range(220, 223)))
            for seed in seeds
        ],
        "eval-d5-heldout-panel-24": [
            row(
                "heldout",
                family,
                scenarios[0],
                5,
                seed,
                "eval-d5-heldout-panel-24",
            )
            for family, scenarios in HELDOUT.items()
            for seed in range(223, 229)
        ],
    }
    for name, panel in difficulty_band_panels.items():
        result[name] = freeze(
            name,
            panel,
            {
                "panel": "difficulty_band_heldout",
                "scenario_partition": "heldout",
                "rollouts_per_task": 5,
            },
        )

    name = "eval-composite-heldout-24"
    panel = [
        row("heldout", "composite_workspace", COMPOSITE, difficulty, seed, name)
        for difficulty in (3, 4, 5)
        for seed in range(200, 208)
    ]
    result[name] = freeze(
        name, panel, {"panel": "composite_heldout", "document_fixture_split": "heldout"}
    )

    # Training-purpose manifests deliberately use non-overlapping seed bands and
    # explicit split labels.  They are independent of the historical Qwen RL
    # matrix above, which mixed D2-D4 tasks under generic ``train`` manifests.
    name = "sft-easy-v1"
    result[name] = freeze(
        name,
        balanced_by_family(
            pool(ORIGINAL, "sft_train", [1, 2, 3], range(0, 80), name),
            512,
            20260822,
        ),
        {
            "training_purpose": "supervised_midtraining",
            "curriculum": "easy_atomic",
            "difficulty_range": [1, 3],
            "scenario_partition": "development",
            "seed_range": [0, 79],
            "shuffle_seed": 20260822,
        },
    )

    name = "sft-validation-v1"
    result[name] = freeze(
        name,
        pool(HELDOUT, "sft_validation", [1, 2, 3], range(120, 128), name),
        {
            "training_purpose": "supervised_validation",
            "curriculum": "easy_atomic_heldout",
            "difficulty_range": [1, 3],
            "scenario_partition": "heldout",
            "seed_range": [120, 127],
            "disjoint_from": ["sft-easy-v1"],
        },
    )

    name = "rl-hard-v1"
    hard_atomic = pool(ORIGINAL, "rl_train", [4, 5], range(128, 190), name)
    hard_composite = pool(
        {"composite_workspace": [COMPOSITE]},
        "rl_train",
        [4, 5],
        range(128, 190),
        name,
    )
    hard_assignments = balanced_by_family(hard_atomic, 392, 20260823) + balanced(
        hard_composite, 120, 20260823
    )
    random.Random(20260823).shuffle(hard_assignments)
    result[name] = freeze(
        name,
        hard_assignments,
        {
            "training_purpose": "reinforcement_learning",
            "curriculum": "hard_atomic_and_composite",
            "difficulty_range": [4, 5],
            "scenario_partition": "development",
            "seed_range": [128, 189],
            "atomic_count": 392,
            "composite_count": 120,
            "composite_fraction": 120 / 512,
            "shuffle_seed": 20260823,
        },
    )

    name = "rl-eval-v1"
    rl_eval_atomic = pool(
        HELDOUT, "rl_eval", [4, 5], range(220, 230), name
    )
    rl_eval_composite = pool(
        {"composite_workspace": [COMPOSITE]},
        "rl_eval",
        [4, 5],
        range(230, 250),
        name,
    )
    result[name] = freeze(
        name,
        rl_eval_atomic + rl_eval_composite,
        {
            "training_purpose": "reinforcement_learning_evaluation",
            "curriculum": "hard_scenario_seed_and_template_heldout",
            "difficulty_range": [4, 5],
            "scenario_partition": "heldout",
            "atomic_seed_range": [220, 229],
            "composite_seed_range": [230, 249],
            "atomic_count": len(rl_eval_atomic),
            "composite_count": len(rl_eval_composite),
            "disjoint_from": ["rl-hard-v1"],
            "document_fixture_split": "heldout",
        },
    )
    return result


def training_config(manifest_name: str, steps: int = 200) -> str:
    return (
        'model = "Qwen/Qwen3.5-4B"\nloss = "rl"\n'
        f"max_steps = {steps}\nbatch_size = 128\nrollouts_per_example = 8\n\n"
        "[sampling]\nmax_tokens = 1024\ntemperature = 0.7\n\n"
        "[checkpoints]\ninterval = 25\nkeep_cloud = 8\n\n"
        "[adapters]\ninterval = 25\nkeep_last = 8\n\n"
        '[[env]]\nid = "yadnyesh/synthetic-workspace-gym@0.2.0.dev1"\n\n'
        f'[env.args]\nfrozen_manifest = "{manifest_name}"\nsplit = "train"\n'
        "max_examples = 512\nmax_turns = 25\nmax_tool_steps = 64\n"
        'reward_mode = "score"\nsample_strategy = "first"\nshuffle = false\n'
    )


def evaluation_config(manifest_name: str, count: int, rollouts: int, split: str) -> str:
    return (
        'model = "Qwen/Qwen3.5-4B"\n'
        f"num_examples = {count}\nrollouts_per_example = {rollouts}\n"
        "max_tokens = 4096\ntemperature = 0.7\nmax_concurrent = 5\n"
        "max_retries = 0\ntimeout_minutes = 180\n\n"
        '[[eval]]\nid = "yadnyesh/synthetic-workspace-gym"\n'
        f'eval_name = "base-{manifest_name}"\n'
        f'env_args = {{ manifest = "{manifest_name}" }}\n'
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    V1_OUT.mkdir(parents=True, exist_ok=True)
    INDEX.parent.mkdir(parents=True, exist_ok=True)
    TRAIN_CONFIGS.mkdir(parents=True, exist_ok=True)
    EVAL_CONFIGS.mkdir(parents=True, exist_ok=True)
    manifests = build()
    for name, payload in manifests.items():
        serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        for manifest_root in (OUT, V1_OUT):
            (manifest_root / f"{name}.json").write_text(
                serialized,
                encoding="utf-8",
            )
    training = sorted(name for name in manifests if name.startswith("train-"))
    evaluations = sorted(name for name in manifests if name.startswith("eval-"))
    for name in training:
        (TRAIN_CONFIGS / f"{name}.toml").write_text(
            training_config(name), encoding="utf-8"
        )
    (TRAIN_CONFIGS / "pilot-8-step-all-family-seed-42.toml").write_text(
        training_config("train-all-family-seed-42", steps=8), encoding="utf-8"
    )
    for name in evaluations:
        count = len(manifests[name]["assignments"])
        rollouts = 5 if name in {
            "eval-d1-d2-panel-24",
            "eval-d3-d4-panel-24",
            "eval-d5-heldout-panel-24",
            "eval-d5-panel-24",
        } else 1
        split = "heldout" if "heldout" in name else "test"
        (EVAL_CONFIGS / f"base-{name}.toml").write_text(
            evaluation_config(name, count, rollouts, split), encoding="utf-8"
        )
    wheel_path = ROOT / f"dist/synthetic_workspace_gym-{VERSION}-py3-none-any.whl"
    wheel_sha256 = (
        hashlib.sha256(wheel_path.read_bytes()).hexdigest()
        if wheel_path.is_file()
        else None
    )
    index = {
        "environment_version": VERSION,
        "candidate_wheel_sha256": wheel_sha256,
        "created_at": CREATED_AT,
        "training_runs": training,
        "evaluation_panels": evaluations,
        "pilot_manifest": "train-all-family-seed-42",
        "training_purpose_manifests": {
            "sft_train": "sft-easy-v1",
            "sft_validation": "sft-validation-v1",
            "rl_train": "rl-hard-v1",
            "rl_evaluation": "rl-eval-v1",
        },
        "fixed_settings": {
            "model": "Qwen/Qwen3.5-4B",
            "steps": 200,
            "batch_size": 128,
            "rollouts_per_example": 8,
            "temperature": 0.7,
            "max_output_tokens": 1024,
            "max_turns": 25,
            "max_tool_steps": 64,
            "nominal_task_count": 512,
        },
        "external_evaluation": {
            "swe_bench_lite": {
                "instance_count": 25,
                "status": "requires_frozen_external_instance_ids_and_agent_scaffold",
            }
        },
    }
    INDEX.write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "manifests": len(manifests),
                "training": len(training),
                "evaluation": len(evaluations),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
