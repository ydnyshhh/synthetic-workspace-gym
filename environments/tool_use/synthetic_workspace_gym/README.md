# Synthetic Workspace Gym V1

Synthetic Workspace Gym (SWG) is a native Verifiers V1 taskset for training terminal agents on generated, executable workspaces. Tasks cover tabular transformations, script repair, data pipelines, local-document retrieval, and composite workflows. Each task has a deterministic assignment, an agent-visible workspace, and a trusted evaluator with partial-credit scoring.

## Design

`SyntheticWorkspaceTaskset` loads a versioned frozen assignment manifest and materializes tasks lazily. Each assignment becomes a typed `SyntheticWorkspaceTask` with typed `SyntheticWorkspaceData` and `SyntheticWorkspaceState`.

The task declares `NEEDS_CONTAINER = True` and a working directory of `/workspace`. Verifiers and the selected standard harness own model interaction, terminal access, runtime provisioning, timeouts, and trace capture. SWG does not wrap a harness or implement a rollout loop.

During task setup, SWG writes only the generated `visible/` tree into the task runtime. Hidden evaluator files remain in task state, which is excluded from serialized traces. After the standard harness exits, finalization injects the hidden assets and a minimal trusted grader into a randomized `/tmp` directory in the same isolated runtime, runs grading there, and retrieves only the structured result. Model-edited code is never executed on the Verifiers host. The weighted `workspace_score` reward is the evaluator's normalized score in `[0, 1]`; metrics report binary success, changed/final file counts, and all evaluator subscores.

The task requests framework-only network access. A fresh runtime is used for every rollout, and generated paths are validated before being written or extracted.

## Install

From the `prime-envs` repository root:

```bash
uv pip install -e environments/tool_use/synthetic_workspace_gym
```

## Smoke evaluation

Run three tasks with the standard Codex harness and a Prime runtime:

```bash
uv run eval synthetic-workspace-gym \
  -n 3 -r 1 --rich false -v --no-push \
  --env.agent.timeout.rollout 180 \
  --env.agent.harness.id codex \
  --env.agent.runtime.type prime \
  --env.agent.runtime.vm true
```

Use the default `train-all-family-seed-42` manifest or select a packaged manifest and filters:

```bash
uv run eval synthetic-workspace-gym \
  --env.taskset.manifest eval-scenario-heldout \
  --env.taskset.families '["script_repair","pipeline"]' \
  --env.taskset.difficulties '[4,5]' \
  -n 20 -r 3 --rich false --no-push
```

Available manifests include balanced and specialist training assignments plus in-distribution, scenario-heldout, composite-heldout, and difficulty-5 evaluation panels. Verifiers' normal `-n`, `-r`, `--shuffle`, harness, model, sampling, and runtime options apply without environment-specific rollout arguments.

## Configuration

| Field | Default | Purpose |
| --- | --- | --- |
| `manifest` | `train-all-family-seed-42` | Packaged frozen assignment manifest |
| `families` | all in manifest | Optional family filter |
| `difficulties` | all in manifest | Optional difficulty filter (`1` through `5`) |
| `tasks` | all in manifest | Optional exact task-ID filter |
| `validate_generation` | `false` | Re-run the gold-solution generation gate while loading |
| `image` | `python:3.12-slim` | Base image for the standard harness runtime |
| `task.max_result_bytes` | `2097152` | Maximum structured grading result returned from the runtime |

Task generation is deterministic over family, scenario, difficulty, seed, and the packaged manifest assignment. Changing task or reward semantics requires a package version bump.
