# SWG RL Exp 1: Small Model Script Repair Improvement

## Motivation

Synthetic Workspace Gym should demonstrate trainability, not only evaluation
coverage. The first clean experiment asks whether a small open model can improve
through direct reinforcement learning interaction with SWG hidden-evaluator
rewards.

A mentor concern motivating this run is that SWG should not become only a
teacher-imitation benchmark. This experiment therefore excludes GPT, Claude,
GLM, privileged heuristic, and other teacher trajectories from training.

## Hypothesis

`Qwen/Qwen3.5-0.8B` should improve on validation/test SWG `script_repair`
tasks after RL rollouts on the train split using `reward_mode = "score"`.
Heldout scenario transfer is measured separately as a stronger generalization
probe.

## Setup

- Model: `Qwen/Qwen3.5-0.8B`
- Environment: `yadnyesh/synthetic-workspace-gym`
- Family: `script_repair`
- Reward mode: `score`
- Training steps: `50`
- Batch size: `128`
- Rollouts per example: `8`
- Sampling max tokens: `1024`
- Training examples: `128`
- Eval examples: about `40` per split
- Max turns: `8` for this first reproducible run, matching the hosted config and
  scripts. A later sensitivity run can raise this to `16`.

No core environment logic, evaluator semantics, hidden reward calculation, or
generated task correctness is changed for this experiment.

## Split Design

The split script uses SWG's existing deterministic split support:

- `train`: `script_repair`, difficulties `1,2,3`, default train seeds, RL rollouts only
- `validation`: `script_repair`, difficulties `2,3,4`, unseen seeds, reward/harness/config debugging only
- `test`: `script_repair`, difficulties `3,4,5`, unseen seeds, final same-family generalization
- `heldout`: `script_repair`, difficulties `3,4,5`, scenario-heldout where available plus unseen seeds

Use `validation` for debugging the harness and training configuration. Do not
tune on `test` or `heldout`.

## Commands

Create the branch:

```bash
git checkout -b exp/swg-small-model-rl-only
```

Build, validate, inspect, and export deterministic splits:

```bash
bash scripts/experiments/build_swg_rl_exp1_splits.sh
```

If refreshing existing generated split artifacts:

```bash
bash scripts/experiments/build_swg_rl_exp1_splits.sh --overwrite
```

Run the baseline hosted evals:

```bash
bash scripts/experiments/baseline_eval_swg_rl_exp1.sh
```

Launch hosted RL:

```bash
prime train run configs/rl/swg-qwen-0.8b-script-repair.toml
```

Monitor:

```bash
prime train logs <run-id> -f
```

Run post-RL evals:

```bash
bash scripts/experiments/post_rl_eval_swg_rl_exp1.sh <trained-checkpoint-id>
```

## Metrics To Report

- mean score
- median score
- perfect solve rate
- zero score rate
- submitted count
- max-turn count
- error count
- average turns to submit
- failure label counts
- by-scenario score
- by-difficulty score
- train vs validation vs test vs heldout gap

All improvement claims must compare the trained checkpoint against the original
`Qwen/Qwen3.5-0.8B` baseline under the same eval settings.

## Expected Failure Modes

- The small model may fail to use tools reliably enough for RL to get useful signal.
- The model may overfit train scenarios and show little or no heldout improvement.
- Score gains may come from submitting more often while still failing hidden checks.
- Max-turn failures may remain high if exploration is too weak.
- Hosted RL may need environment packaging or quota fixes before training runs cleanly.

## Positive Result

A positive result is validation and test improvement after RL training on the
train split, compared with the original Qwen-0.8B baseline. Strong supporting
signals include lower max-turn failure rate, higher perfect solve rate, and
better by-difficulty scores. Heldout improvement may be smaller than test
improvement and still be useful.

## Invalid Claims

Do not claim GPT trajectory distillation, because this run performs RL on the
environment and does not train on GPT, Claude, GLM, privileged heuristic, or
reference-agent trajectories.

Do not claim general SWE-bench improvement or broad agentic reasoning
improvement from this small `script_repair`-only experiment.

Do not tune on `test` or `heldout`. Do not expose hidden evaluator files to the
model-facing workspace. Do not use the privileged heuristic/reference agent as a
model benchmark.
