# Counterfactual trajectory branching

SWG normally assigns trusted reward to a final workspace. That reward cannot distinguish a harmful action from a useful action followed by a later mistake. The counterfactual subsystem snapshots an intermediate filesystem and its normalized conversation prefix, applies matched alternative interventions, evaluates each continuation with the standard evaluator, and estimates action value under a specified continuation policy.

## Modes

In **forced** mode, SWG executes the candidate tool action before the continuation agent acts. The resulting observation is added to context and tagged `forced`; it must not be treated as a model-selected training action. This estimates `Q(state, forced_action)`.

In **open** mode, the copied branch workspace and prefix become the initial state and the policy chooses the next action. Open branches are intended for evaluation and RL tasksets. Budgets, model settings, evaluator, snapshot, and prefix should be matched within a comparison group.

## Quick start

```bash
uv run swg counterfactual collect \
  --environment generated/example \
  --agent scripted \
  --snapshot-policy writes_checks_submit \
  --max-snapshots 3 \
  --output-dir counterfactual-artifacts/root

uv run swg counterfactual build \
  --snapshots counterfactual-artifacts/root/snapshots \
  --selectors before-first-write,before-submit \
  --candidates original,submit,run-public-check,read-relevant-file \
  --mode forced \
  --output-dir branch-packs/demo

uv run swg counterfactual run \
  --manifest branch-packs/demo/manifest.jsonl \
  --client scripted --rollouts-per-branch 4 \
  --output-dir counterfactual-runs/demo

uv run swg counterfactual analyze \
  --outcomes counterfactual-runs/demo/outcomes.jsonl \
  --output counterfactual-runs/demo/comparisons.jsonl

uv run swg counterfactual export \
  --comparisons counterfactual-runs/demo/comparisons.jsonl \
  --branch-manifest branch-packs/demo/manifest.jsonl \
  --format preference --min-margin 0.20 \
  --output datasets/swg-cf-preference.jsonl
```

Use `--format sft`, `critic`, or `rl-taskset` for the other outputs. RL export writes open-action environments immediately before states whose regret passes `--min-regret`.

## Artifacts

Snapshots contain `snapshot.json`, `branch_state.json`, a rewritten `manifest.json`, `visible/`, trusted-only `hidden/`, and `trajectory_prefix.jsonl`. Branch packs contain `metadata.json`, `manifest.jsonl`, and one independently loadable environment per task. Runs contain `outcomes.jsonl`, `comparisons.jsonl`, `summary.json`, and isolated rollout directories with trajectory, final workspace, evaluator result, diff, and outcome.

For candidate `a` at branch group `b`, SWG reports `Q_hat(b,a)` as mean final reward. Counterfactual delta is `Q_hat(b,a) - Q_hat(b,original)` and decision regret is `max_a Q_hat(b,a) - Q_hat(b,original)`. Recoverability means at least one tested candidate reaches the configured threshold. These are intervention estimates under the chosen continuation distribution, not policy-independent causal facts.

## Trust boundary and cautions

Hidden evaluator assets are copied only for trusted execution and never placed under `visible/` or included in model messages. Forced actions pass normal permission checks; file candidates reject absolute paths and parent traversal; every rollout starts in a fresh copy.

One continuation does not establish causality. Stochastic policies require repeated continuations, matched budgets, and variance-aware interpretation. Privileged candidate sources must remain labeled and should normally be excluded from training exports. Intermediate evaluator results are selection metadata only and must never be model-visible. Results depend on the continuation policy and candidate coverage.

## Prime and Verifiers direction

Each branch environment uses the existing SWG `manifest.json` layout and filesystem loader, so it can be mounted as an environment path today. Hosted execution still needs packaging or Hub resolution for `branch_manifest_path`, plus thin Prime/Verifiers adapters that load `prefix_messages`, execute forced actions before the first sampled turn, and preserve forced-action metadata. No Prime RL changes are required: the `rl-taskset` export is an open-action manifest suitable for an environment loader argument.

Recommended first real experiment: 40 roots, two branch points, four candidates, and four continuations (`1,280` branch continuations). Primary reporting should include the share of states with regret above `0.20`, recoverability, original-action optimality, candidate-type value, and family/difficulty breakdowns.
