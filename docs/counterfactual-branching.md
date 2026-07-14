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

## Hosted package generation

Generate one immutable Environment Hub package per counterfactual experiment:

```bash
uv run swg counterfactual package-hosted \
  --branch-pack artifacts/pilot-pack \
  --output-dir dist/swg-counterfactual-pilot \
  --package-name swg-counterfactual-pilot \
  --pack-id swg-cf-pilot-2026-07-13 \
  --swg-ref df0e0462de3c2c006ba4a4db69785e60ec8cccc4
```

The command validates every manifest row, rejects paths outside the pack and missing hidden evaluator assets, copies the unchanged pack under the generated Python package, computes a deterministic SHA-256 over every pack path and file, runs a local native-Verifiers smoke test, builds the wheel, and verifies that every pack file is present in it. The generated wrapper injects `pack_id` and `pack_sha256` into hosted branch metadata so rollout state remains tied to the immutable input pack.

The generated directory contains `environment.py`, `pyproject.toml`, `README.md`, and `src/<package_module>/branch_pack/`. Use `--force` only when intentionally replacing an existing generated directory. Real packs contain trusted assets under `hidden/`; publish them privately and use a new package version and pack ID for corrections instead of updating a collected experiment in place.

## Artifacts

Snapshots contain `snapshot.json`, `branch_state.json`, a rewritten `manifest.json`, `visible/`, trusted-only `hidden/`, and `trajectory_prefix.jsonl`. Action-value selectors compile only before-action states. Signals observed after an action (such as a failed check or score drop) are carried to the next before-action snapshot, whose `original_action` is the policy's actual next choice. Branch packs contain `metadata.json`, `manifest.jsonl`, and one independently loadable environment per task. Manifest environment paths are POSIX-relative to the pack and resolved by the loader, so packs can be moved across working directories and operating systems. Runs contain `outcomes.jsonl`, `comparisons.jsonl`, `summary.json`, and isolated rollout directories with trajectory, final workspace, evaluator result, diff, and outcome.

For candidate `a` at branch group `b`, SWG reports `Q_hat(b,a)` as mean final reward. Counterfactual delta is `Q_hat(b,a) - Q_hat(b,original)` and decision regret is `max_a Q_hat(b,a) - Q_hat(b,original)`. Recoverability means at least one tested candidate reaches the configured threshold. These are intervention estimates under the chosen continuation distribution, not policy-independent causal facts.

## Trust boundary and cautions

Hidden evaluator assets are copied only for trusted execution and never placed under `visible/` or included in model messages. Forced actions pass normal permission checks; file candidates reject absolute paths and parent traversal; every rollout starts in a fresh copy.

One continuation does not establish causality. Stochastic policies require repeated continuations, matched budgets, and variance-aware interpretation. Privileged candidate sources must remain labeled and should normally be excluded from training exports. Intermediate evaluator results are selection metadata only and must never be model-visible. Results depend on the continuation policy and candidate coverage.

## Prime and Verifiers direction

Local deterministic agents rehydrate their caches from exact stored tool messages without replaying tool effects. Prime model clients and Verifiers policies use the same exact prefix directly. `swg prime branch-rollout` loads a packaged task, injects a forced action before the first sampled model turn (or leaves the first action open), appends its observation, and records forced-action metadata in the transcript and model metadata.

```bash
uv run swg prime branch-rollout \
  --manifest examples/counterfactual/demo-pack/manifest.jsonl \
  --task-index 2 --mode forced --client scripted \
  --output-dir prime-branch-runs/demo
```

`SyntheticWorkspacePrimeBranchEnv`, `run_prime_branch_rollout`, and `SyntheticWorkspaceVerifiersEnv(branch_manifest_path=...)` provide JSON-friendly programmatic entry points. Hosted execution requires the branch pack to be bundled in the published environment or resolved from a Hub artifact; no additional prefix or forced-action adapter is required. No Prime RL changes are required: `rl-taskset` produces an open-action manifest suitable for `branch_manifest_path`.

The evaluator-backed positive demo is reproducible with:

```bash
uv run python examples/counterfactual/positive_demo.py
```

It intentionally uses a privileged known-good patch to validate the analysis pipeline: original return `0.0`, corrected return `1.0`, regret `1.0`, and `recoverable=true`. It is a pipeline demonstration, not a claim about model capability.

Recommended first real experiment: 40 roots, two branch points, four candidates, and four continuations (`1,280` branch continuations). Primary reporting should include the share of states with regret above `0.20`, recoverability, original-action optimality, candidate-type value, and family/difficulty breakdowns.


## Hosted isolation and provenance boundary

Hosted counterfactual packages fail closed unless sandbox_backend is set to docker. The generated loader rejects local and unknown backends because the host process must retain the branch manifest, evaluator entrypoint, and installed hidden assets while untrusted tool processes receive only the active visible workspace. Evaluator processes separately receive the hidden directory as a read-only mount.

[Prime-hosted evaluations](https://docs.primeintellect.ai/tutorials-environments/hosted-evaluations) run the environment itself in a Prime-managed sandbox, and [Prime Sandboxes](https://docs.primeintellect.ai/sandboxes/overview) are remote disposable Docker environments. Prime's public documentation does not guarantee Docker-in-Docker inside a hosted environment, so SWG does not silently assume it. A hosted deployment must provide either the supported SWG Docker backend or a future explicitly integrated Prime-native sandbox backend.

Every hosted load must also provide wheel_sha256, the exact SHA-256 reported by package-result.json. The generated runtime manifest attaches that value alongside pack_id, pack_sha256, source_swg_commit, and hosted_package_version to each branch row and rollout state. The wheel hash is a detached attestation because a wheel cannot contain its own final cryptographic hash.

The packaging pipeline builds the wheel first, creates a clean temporary virtual environment, installs the wheel and its exact Git-pinned SWG dependency, imports the installed package, verifies local execution is refused, and loads representative forced non-terminal, forced terminal, and open branches when present.
