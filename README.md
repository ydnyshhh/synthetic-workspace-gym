# Synthetic Workspace Gym

I think one of the bottlenecks in agent research is that environments are still too often handcrafted. Building each one manually is tedious, hard to scale, and makes it difficult to study agent behavior systematically across controlled variations in difficulty, structure, and tooling. Prompts alone are not enough; what matters is the executable world around them. Synthetic Workspace Gym is my attempt to address that by treating environments as generated objects rather than fixed artifacts. Instead of writing each workspace by hand, the framework compiles synthetic workspace-style environments with hidden evaluators, structured metadata, and reproducible complexity, giving agents a more scalable substrate for training and evaluation.

## Overview

| Item | Description |
| --- | --- |
| Primary unit | Environment instance, not prompt |
| Focus | Synthetic workspace generation, execution, logging, and trusted evaluation |
| v1 families | `tabular`, `script_repair`, `pipeline` |
| Runtime model | Local scratch workspace + hidden evaluator outside writable scope |
| Evaluator resolution | Dynamic import from manifest `evaluator_entrypoint` |
| Packaging | Python package with `uv` workflow and `swg` CLI |
| Status | v1 research infrastructure |

## What This Is

| This project is | This project is not |
| --- | --- |
| An environment-centric agent research framework | Just a prompt dataset |
| A generator of executable workspaces | Only a static benchmark |
| A place to study trajectories, tool use, and repair behavior | A visible self-evaluation setup |
| A typed substrate for future environment families | A security sandbox product |

## Quick Start

| Task | Command |
| --- | --- |
| Install | `uv sync` |
| Install with local cache | `uv sync --cache-dir .uv-cache` |
| Run tests | `uv run --no-project --cache-dir .uv-cache --python python python -B -m unittest discover -s tests -v` |
| Generate one env | `uv run swg generate --family tabular --scenario monthly_segment_report --count 1 --difficulty 3 --seed 42 --output-dir generated` |
| Run one episode | `uv run swg run --environment generated/<env_id> --agent heuristic --output-dir episodes` |
| Evaluate workspace | `uv run swg evaluate --environment generated/<env_id>` |
| Benchmark baseline | `uv run swg benchmark --environments generated --agent heuristic --output-dir benchmarks` |

## Core Design

### Environment model

Every generated instance is treated as a world with:

| Property | Meaning |
| --- | --- |
| Initial state | Visible files and project state before the agent starts |
| Visible artifacts | Files the agent can inspect and modify |
| Hidden assets | Trusted evaluator code, expected outputs, reference metadata |
| Tool permissions | Allowed tool surface for the episode runtime |
| Success criteria | Evaluator-defined completion logic |
| Complexity metadata | Difficulty plus latent structural factors |

### Core abstractions

| Abstraction | Purpose | Key fields / methods |
| --- | --- | --- |
| `EnvironmentSpec` | Declarative generation request | `env_family`, `difficulty`, `seed`, `max_steps`, `tool_permissions`, `task_params`, `generation_params`, `complexity_profile` |
| `EnvironmentManifest` | Concrete generated instance description | `env_id`, `instruction`, `visible_files`, `hidden_files`, `evaluator_entrypoint`, `metadata`, `reference_solution` |
| `EvaluatorResult` | Structured trusted evaluation result | `success`, `score`, `subscores`, `failure_labels`, `diagnostics`, `runtime_seconds` |
| `TrajectoryEvent` | Step-level execution log | action, args, observation summary, stdout/stderr, exit code, files touched, workspace digest |
| `BaseGenerator` | Generator interface | `sample_spec()`, `generate_instance()`, `validate_instance()` |
| `BaseEvaluator` | Evaluator interface | `evaluate(workspace_path, manifest, hidden_root)` |
| `BaseAgent` | Tool-using agent interface | `reset()`, `act()` |
| `EpisodeRunner` | Runtime orchestrator | environment reset, action loop, logging, final evaluation, artifact export |

### Difficulty model

External difficulty is exposed as `1..5` or `easy/medium/hard`. Internally, generators use richer latent factors:

| Factor | Meaning |
| --- | --- |
| `file_count` | Workspace size |
| `distractor_count` | Irrelevant or misleading visible artifacts |
| `dependency_depth` | Cross-file coupling depth |
| `reasoning_hops` | Number of conceptual steps needed |
| `transformation_count` | Required transformation operations |
| `bug_subtlety` | Repair subtlety or misleadingness |
| `execution_required` | Whether running code/shell is necessary |
| `output_constraint_strength` | Strictness of final artifact contract |

## Implemented Environment Families

| Family | Visible workspace shape | Hidden evaluation | Typical failure modes | Difficulty scaling |
| --- | --- | --- | --- | --- |
| `tabular` | CSV/JSON inputs, task file, README, expected output path | Hidden exact-output comparison | Wrong parsing, dedup, joins, aggregations, sorting | more rows/files, schema mismatch, joins, distractors, stricter output |
| `script_repair` | Small Python project with bugs, smoke-test entrypoint | Hidden unit tests | syntax, off-by-one, condition bugs, path issues, imports, wrong return values | more files, more bugs, cross-file coupling, subtler faults |
| `pipeline` | Multi-file mini-project with config + code + artifacts | Hidden execution + artifact validation | wrong config path, broken pipeline step, output format mismatch, aggregation error | more config coupling, broken assumptions, missing steps, stricter artifact requirements |

### Current base scenario pools

| Family | Current scenario ids | Coverage intent |
| --- | --- | --- |
| `tabular` | `monthly_segment_report`, `channel_status_pivot`, `weekly_refund_rollup`, `supplier_restock_summary` | grouped monthly reporting, pivot-style aggregation, ISO-week time bucketing, alias normalization plus restock joins |
| `script_repair` | `inventory_report`, `path_batch`, `csv_schema_drift`, `timestamp_normalization`, `team_roster_export` | aggregation repair, file/path handling, schema drift, datetime normalization, serialization and cross-file contract repair |
| `pipeline` | `team_hours_pipeline`, `sales_csv_pipeline`, `artifact_stitch_pipeline`, `quality_gate_pipeline` | JSON summary generation, CSV normalization, artifact stitching, multi-stage quality/filter/aggregate pipelines |

## Runtime Model

### Tool API

| Tool | Purpose |
| --- | --- |
| `read_file(path)` | Read visible workspace file content |
| `write_file(path, content)` | Replace file content |
| `append_file(path, content)` | Append to file |
| `list_directory(path)` | Inspect workspace tree |
| `run_shell(command)` | Run shell command in workspace |
| `run_python(command_or_script)` | Run Python code or script in workspace |
| `submit(path_or_answer)` | Signal completion |

### Runtime guarantees

| Guarantee | v1 behavior |
| --- | --- |
| Isolation | Each episode runs in a fresh scratch copy of `visible/` |
| Hidden evaluator boundary | Hidden assets stay outside the agent writable workspace |
| Logging | Every tool step is recorded as a structured trajectory event |
| Limits | `max_steps` and wall-clock `time_limit_seconds` enforced |
| Tool execution guard | Shell and Python execution are restricted to local workspace-oriented usage with parent-traversal, absolute-path, inline-env, network, and inline-Python guardrails |
| Evaluation trigger | Runs after episode termination or `submit` |
| Artifacts | Manifest copy, trajectory, evaluator result, summary, final diff, final workspace |

### Runtime Security Model

The runtime policy blocks common workspace-escape patterns such as parent traversal (`../`), absolute filesystem paths, inline environment-variable assignment in shell commands, inline/module Python execution through `run_shell`, and common network utilities like `curl` and `wget`. This improves evaluator integrity for local research workflows, but it is still a denylist policy wrapped around normal local subprocesses.

v1 is not a hardened OS sandbox. A model-backed agent should be treated as running in a best-effort local integrity envelope rather than in a security boundary suitable for hostile code. "Hidden evaluator" in this project means hidden from the normal visible workspace layout and file-tool surface, not cryptographically or kernel-level isolated from a determined adversary.

## Trusted Evaluation

| Family | Evaluator strategy |
| --- | --- |
| `tabular` | Compare generated output to hidden expected JSON |
| `script_repair` | Execute hidden tests against the repaired workspace |
| `pipeline` | Execute repaired project and compare final artifact to hidden expected output |

Generation-time validation is built in: every environment is checked by applying the stored reference solution to a scratch copy of the workspace, verifying that the hidden evaluator returns success, and asserting that the stored solution actually changes visible artifacts. Evaluators expose partial credit through `score` and `subscores` instead of only binary pass/fail outputs, and subprocess-backed evaluators return structured timeout failures rather than raising raw exceptions.

## Artifact Layout

### Generated environment

| Path | Purpose |
| --- | --- |
| `generated/<env_id>/manifest.json` | Concrete environment manifest |
| `generated/<env_id>/visible/` | Agent-visible workspace |
| `generated/<env_id>/hidden/` | Hidden evaluator assets and solution metadata |

### Episode rollout

| Path | Purpose |
| --- | --- |
| `episodes/<episode_id>/manifest.json` | Manifest snapshot used for the run |
| `episodes/<episode_id>/trajectory.jsonl` | Step-by-step action/observation log |
| `episodes/<episode_id>/evaluator_result.json` | Trusted final evaluation |
| `episodes/<episode_id>/summary.json` | Rollout summary metrics |
| `episodes/<episode_id>/final_diff.txt` | Unified diff from initial to final workspace |
| `episodes/<episode_id>/final_workspace/` | Final workspace snapshot |

### Benchmark report

`swg benchmark` now builds a normalized analysis row per episode and then emits grouped summaries instead of only a flat global mean.

| Section | Purpose |
| --- | --- |
| `rows` | Per-episode normalized analysis rows merging runtime outcome and manifest metadata |
| `overall` | Global aggregate metrics across the run set |
| `by_family` | Aggregate by `tabular`, `script_repair`, `pipeline` |
| `by_difficulty` | Aggregate by difficulty level |
| `by_scenario_id` | Aggregate by concrete scenario id |
| `by_family_and_difficulty` | Aggregate by combined family/difficulty buckets |
| `by_bug_scope`, `by_failure_mode`, `by_repair_surface`, `by_smoke_test_quality` | Structure-aware slices for scenario analysis when metadata is available |

Each bucket reports `count`, `success_rate`, `mean_score`, `median_score`, `perfect_rate`, `mean_step_count`, `mean_duration_seconds`, `failure_label_counts`, and `mean_subscores`.

## Repository Layout

| Path | Responsibility |
| --- | --- |
| `src/synthetic_workspace_gym/schemas/` | Typed schemas and serialization helpers |
| `src/synthetic_workspace_gym/generators/` | Family generators, difficulty mapping, registry |
| `src/synthetic_workspace_gym/evaluators/` | Trusted evaluators and registry |
| `src/synthetic_workspace_gym/runtime/` | Environment loader, tool executor, episode runner |
| `src/synthetic_workspace_gym/agents/` | Baseline agents |
| `src/synthetic_workspace_gym/analysis/` | Artifact export, snapshotting, diff utilities, benchmark analysis |
| `src/synthetic_workspace_gym/cli.py` | CLI entrypoint |
| `tests/` | Schema, generator, evaluator, runtime, and end-to-end tests |

## CLI

| Command | Description |
| --- | --- |
| `swg generate` | Generate one or more environments from a family, difficulty, and optional explicit `--scenario` id |
| `swg run` | Run a baseline agent on one environment |
| `swg evaluate` | Evaluate a workspace against the hidden evaluator |
| `swg benchmark` | Run a baseline across a directory of generated environments |

### Examples

```bash
uv run swg generate --family script_repair --scenario csv_schema_drift --count 10 --difficulty 4 --seed 100 --output-dir generated
uv run swg run --environment generated/script_repair-d4-s100-XXXXXXXX --agent heuristic --output-dir episodes
uv run swg evaluate --environment generated/script_repair-d4-s100-XXXXXXXX
uv run swg benchmark --environments generated --agent heuristic --output-dir benchmarks
```

For comparable benchmarks, prefer explicit scenario addressing: `--scenario <scenario_id> --seed <seed>`. Seed-only routing remains available as a convenience, but by design it follows the current scenario pool order and can change when new scenarios are added to a family.

## Baseline Agents

| Agent | Role |
| --- | --- |
| `scripted` | Minimal heuristic smoke-test baseline; intentionally weak |
| `heuristic` | Privileged validation baseline that applies `manifest.reference_solution["files"]` directly and submits |

The baseline layer is intentionally modular so stronger model-backed agents can plug into the same runtime without changing the environment format. The `heuristic` baseline should not be treated as a language-model benchmark; it is a privileged infrastructure check that replays the stored reference solution rather than reasoning about the task. A `ReActBaselineAgent` compatibility alias still exists in Python for older code, but the CLI now exposes only `scripted` and `heuristic`.

## Development Notes

| Topic | Details |
| --- | --- |
| Python | `>=3.11` |
| Package entrypoint | `swg = synthetic_workspace_gym.cli:main` |
| Build backend | `hatchling` |
| Test framework | `unittest` |
| Serialization style | dataclass-based typed schemas with explicit `to_dict()` / `from_dict()` |
| Dependency policy | minimal v1 surface; standard library first |

## Adding a New Environment Family

| Step | What to implement |
| --- | --- |
| 1 | Add a new `EnvironmentFamily` entry |
| 2 | Implement a generator in `src/synthetic_workspace_gym/generators/` |
| 3 | Implement a trusted evaluator in `src/synthetic_workspace_gym/evaluators/` |
| 4 | Register both in the generator/evaluator registries |
| 5 | Emit visible files, hidden assets, manifest metadata, and reference solution metadata |
| 6 | Add tests for structural validity, unsolved failure, reference solution success, and episode execution |

## Current v1 Boundaries

| Included | Deferred |
| --- | --- |
| local runtime, hidden evaluators, trajectory logging, typed manifests, three families, explicit scenario selection, baseline agents, CLI, tests, dynamic evaluator loading, partial-credit scoring | integrity-aware compilation, reward-hacking variants, decoy leakage files, editable evaluators, provenance monitoring, multi-stage / lifelong environments |
