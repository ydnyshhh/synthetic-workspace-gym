# Synthetic Workspace Gym

Synthetic Workspace Gym is a research-oriented framework for generating and evaluating small, executable workspace environments for tool-using agents.

The unit of work is an environment instance, not a prompt. Each instance is treated as a self-contained world with:

- an initial visible workspace
- hidden evaluator assets outside the writable agent workspace
- explicit tool permissions
- a structured manifest
- trusted success criteria
- controllable difficulty and latent complexity metadata
- full episode trajectory logging

This makes the project useful as infrastructure for agent training and evaluation, rather than as a flat prompt dataset or benchmark-only task list.

## Why This Exists

Most agent benchmarks stop at instructions and final answers. Real coding and workspace agents operate in richer environments:

- files exist before the agent starts
- tools mutate state over time
- hidden evaluators determine success
- trajectory quality matters, not just the final answer
- difficulty comes from workspace structure, coupling, and repair dynamics

Synthetic Workspace Gym focuses on those environment-level concerns directly.

## v1 Scope

The current version implements three environment families:

1. `tabular`
   Messy CSV/JSON transformation tasks with deterministic hidden reference outputs.
2. `script_repair`
   Small Python repair tasks with hidden tests.
3. `pipeline`
   Multi-file mini-project completion tasks with broken config/code/output assumptions.

Each generated environment includes:

- `visible/` workspace files for the agent
- `hidden/` evaluator assets and solution metadata
- `manifest.json` describing the concrete instance

Each episode run exports:

- manifest copy
- trajectory JSONL
- evaluator result JSON
- summary JSON
- final workspace snapshot
- final unified diff

## Project Layout

```text
src/synthetic_workspace_gym/
  schemas/        Typed core schemas and event models
  generators/     Environment-family generators and difficulty logic
  evaluators/     Trusted hidden evaluators
  runtime/        Tool execution, environment loading, episode runner
  agents/         Baseline agents
  analysis/       Artifact export and diff helpers
  cli.py          Command-line entrypoint
tests/            Generator, evaluator, runtime, schema, and e2e tests
```

## Core Abstractions

### `EnvironmentSpec`

Declarative generation input with:

- `env_family`
- `difficulty`
- `seed`
- `max_steps`
- `time_limit_seconds`
- `tool_permissions`
- `observability`
- `task_params`
- `evaluator_params`
- `generation_params`
- `complexity_profile`

### `EnvironmentManifest`

Concrete generated-instance manifest with:

- `env_id`
- `family`
- `difficulty`
- `seed`
- `instruction`
- `workspace_root`
- `visible_files`
- `hidden_root`
- `hidden_files`
- `tool_permissions`
- `max_steps`
- `time_limit_seconds`
- `metadata`
- `evaluator_entrypoint`
- `reference_solution`

### `EvaluatorResult`

Structured trusted evaluation output with:

- `success`
- `score`
- `subscores`
- `failure_labels`
- `diagnostics`
- `runtime_seconds`

### Trajectory Schema

Every tool step records:

- step index
- timestamp
- action type
- action arguments
- observation summary
- stdout/stderr
- exit code
- files touched
- workspace digest
- success flag

### Extension Interfaces

- `BaseGenerator`
- `BaseEvaluator`
- `BaseAgent`
- `EpisodeRunner`

## Install

### Recommended

```bash
uv sync
```

If your environment restricts access to the global `uv` cache, use a repo-local cache:

```bash
uv sync --cache-dir .uv-cache
```

## Run Tests

```bash
uv run --no-project --cache-dir .uv-cache --python python python -B -m unittest discover -s tests -v
```

In a normal unrestricted environment, `uv run python -m unittest discover -s tests -v` is also fine after `uv sync`.

## CLI

The package exposes the `swg` CLI.

### Generate environments

```bash
uv run swg generate --family tabular --count 5 --difficulty 3 --seed 100 --output-dir generated
```

### Run one episode

```bash
uv run swg run --environment generated/tabular-d3-s100-XXXXXXXX --agent react --output-dir episodes
```

### Evaluate a workspace

```bash
uv run swg evaluate --environment generated/tabular-d3-s100-XXXXXXXX
```

You can also evaluate a modified workspace explicitly:

```bash
uv run swg evaluate --environment generated/tabular-d3-s100-XXXXXXXX --workspace /path/to/workspace
```

### Benchmark a baseline

```bash
uv run swg benchmark --environments generated --agent react --output-dir benchmarks
```

## Runtime and Tool Model

Agents interact through a structured tool API:

- `read_file(path)`
- `write_file(path, content)`
- `append_file(path, content)`
- `list_directory(path)`
- `run_shell(command)`
- `run_python(command_or_script)`
- `submit(path_or_answer)`

The runtime is intentionally local and simple in v1:

- each episode runs in an isolated scratch workspace under the configured output root
- visible workspace files are copied into that scratch directory
- hidden evaluator assets stay outside the agent workspace
- max steps and wall-clock time are enforced
- every action and observation is logged
- evaluation happens only at the end of the episode or after `submit`

This is a cooperative research sandbox, not a security-hardened OS isolation layer.

## Environment Families

### 1. Tabular / Data Transformation

Visible workspace contents typically include:

- `README.md`
- `task.json`
- `data/orders.csv`
- optional lookup/adjustment files

Hidden assets include:

- `expected_output.json`
- `evaluator_config.json`
- `reference_solution.json`

Difficulty scales with factors such as:

- more rows and input files
- deduplication requirements
- joins
- extra transformation steps
- distractor files
- stricter output constraints

### 2. Script / Code Repair

Visible workspace contents typically include:

- `README.md`
- `task.json`
- `src/repair_target/*.py`
- a visible smoke-test entrypoint
- scenario data files

Hidden assets include:

- a hidden test runner
- evaluator config
- solution file metadata

Difficulty scales with factors such as:

- more files
- more simultaneous bugs
- subtler bug types
- misleading visible hints
- cross-file repair requirements

### 3. Pipeline / Config Completion

Visible workspace contents typically include:

- `README.md`
- `task.json`
- `config/*.json`
- `src/pipeline_app/*.py`
- executable `run_pipeline.py`
- input data files

Hidden assets include:

- `expected_output.json`
- `evaluator_config.json`
- solution file metadata

Difficulty scales with factors such as:

- config/code coupling
- missing transformation steps
- broken output assumptions
- distractor notes/configs
- stricter output contracts

## Artifact Layout

### Generated environment

```text
generated/<env_id>/
  manifest.json
  visible/
    ...
  hidden/
    ...
```

### Episode rollout

```text
episodes/<episode_id>/
  manifest.json
  trajectory.jsonl
  evaluator_result.json
  summary.json
  final_diff.txt
  final_workspace/
    ...
```

## Baseline Agents

Two baseline agents are included:

1. `scripted`
   A minimal heuristic smoke-test agent. It solves the tabular family and performs lightweight visible checks for the repair-style families.
2. `react`
   A simple iterative tool-using baseline that reads instructions, inspects task files, runs smoke tests, applies family-specific repairs, and submits.

These are deliberately modular placeholders so stronger model-backed agents can be dropped in later behind the same interfaces.

## How Evaluation Works

Evaluation is trusted and hidden from the agent workspace.

- Tabular tasks compare the final produced artifact against a hidden expected output.
- Script-repair tasks execute hidden tests outside the agent’s writable scope.
- Pipeline tasks execute the repaired project and compare the resulting artifact to a hidden expected output.

During generation, every environment is validated by applying the reference solution metadata to a scratch copy of the workspace and ensuring the evaluator returns success.

## Difficulty and Complexity

External difficulty is exposed as `1..5` or `easy/medium/hard`, while internal complexity is represented with richer latent factors:

- `file_count`
- `distractor_count`
- `dependency_depth`
- `reasoning_hops`
- `transformation_count`
- `bug_subtlety`
- `execution_required`
- `output_constraint_strength`

These factors are stored in environment metadata so downstream research can analyze success and failure modes beyond a single flat difficulty number.

## Adding a New Environment Family

1. Add a new `EnvironmentFamily` enum entry.
2. Implement a new generator in `src/synthetic_workspace_gym/generators/`.
3. Implement a new evaluator in `src/synthetic_workspace_gym/evaluators/`.
4. Register both in the generator/evaluator registries.
5. Emit:
   - visible workspace files
   - hidden evaluator assets
   - manifest metadata
   - reference solution metadata
6. Add family-specific tests covering:
   - generation validity
   - unsolved workspace failure
   - reference solution success
   - episode execution if applicable

## Notes for Future Versions

The current abstractions are intended to extend toward:

- integrity-aware compilation
- reward-hacking variants
- decoy leakage files
- editable vs sealed evaluators
- provenance monitoring
- multi-stage / lifelong environments

v1 keeps the implementation local, typed, modular, and research-friendly without overbuilding distributed infrastructure.
