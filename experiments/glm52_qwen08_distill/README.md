# GLM-5.2 to Qwen3.5-0.8B Trajectory Distillation

This experiment tests whether Qwen3.5-0.8B can learn the SWG closed-loop tool-control policy from GLM-5.2 expert trajectories. The first stage builds a perfect-only supervised dataset from GLM trajectories. Later stages will compare raw multi-tool training, sequentialized single-tool training, scenario-balanced partial-trace training, and recovery-state training from Qwen failure traces.

## Motivation

The target behavior is not full trace memorization. The dataset turns successful hosted-eval trajectories into state to next tool-action examples that teach the control loop:

```text
inspect -> read relevant files -> write/edit -> run public check -> read artifact -> submit
```

The first experiment extracts only perfect-reward GLM-5.2 traces and converts assistant tool-call turns into SFT-ready action windows.

## Teacher Run

- Teacher model: `GLM-5.2`
- Evaluation ID: `kxhqr8w6kxeficm93rp7s5k6`
- Total examples: 390
- Mean reward: about 0.9167
- Perfect examples: 304
- Max-turn episodes: 0
- Successful submissions: 390 / 390

## Hypothesis

Sequentialized single-tool action windows should be easier for `Qwen/Qwen3.5-0.8B` to learn than raw multi-tool windows because they reduce target complexity and match the SWG instruction to return exactly one tool call per turn.

## Inputs

Place the hosted-eval page exports locally under:

```text
data/raw_traces/glm52/
```

Expected files are JSON page exports such as:

```text
data/raw_traces/glm52/samples-page-01.json
...
data/raw_traces/glm52/samples-page-20.json
```

The loader reads `*.json` files with `utf-8-sig` encoding and skips JSON files that do not contain a `samples` list.

This branch also includes a single-file raw trace bundle:

```text
data/raw_traces/glm52/glm52_raw_traces_pages.json
```

The builder can use either the original page-export directory or this bundled raw trace JSON file as `--input-dir`.

## Build Command

```bash
python experiments/glm52_qwen08_distill/build_dataset.py \
  --input-dir data/raw_traces/glm52/glm52_raw_traces_pages.json \
  --output-dir data/processed_traces/glm52_qwen08 \
  --report-dir data/reports/glm52_qwen08 \
  --teacher glm-5.2 \
  --student Qwen/Qwen3.5-0.8B \
  --eval-id kxhqr8w6kxeficm93rp7s5k6 \
  --reward-filter perfect \
  --write-raw \
  --write-sequential
```

Use `--dry-run` to inspect stats without writing datasets or reports. Use `inspect_traces.py` for the same dry-run path:

```bash
python experiments/glm52_qwen08_distill/inspect_traces.py \
  --input-dir data/raw_traces/glm52 \
  --output-dir data/processed_traces/glm52_qwen08 \
  --report-dir data/reports/glm52_qwen08 \
  --teacher glm-5.2 \
  --student Qwen/Qwen3.5-0.8B \
  --eval-id kxhqr8w6kxeficm93rp7s5k6
```

## Outputs

The full build writes:

```text
data/processed_traces/glm52_qwen08/glm52_perfect_raw_actions.jsonl
data/processed_traces/glm52_qwen08/glm52_perfect_sequential_actions.jsonl
data/reports/glm52_qwen08/perfect_dataset_report.json
data/reports/glm52_qwen08/perfect_dataset_report.md
```

Variant A keeps raw assistant action turns, including multi-tool targets. Variant B splits multi-tool assistant turns into single-tool targets and appends aligned tool observations to later split-window histories when `tool_call_id` matching is available.

## Safety

Do not commit raw hosted-eval traces, generated JSONL datasets, model checkpoints, adapters, logs, WandB outputs, or report outputs under `data/`.

Only commit scripts, configs, README files, small synthetic fixtures, report templates, and tests.

## Next Steps

1. Build and inspect the perfect-only raw and sequentialized datasets.
2. Review scenario coverage and data-quality counters in the report.
3. Prefer the sequentialized variant if raw targets contain frequent multi-tool calls.
4. Build a scenario-balanced partial-trace variant if perfect-only coverage is sparse.
5. Later, add recovery-state examples from Qwen failure traces before launching training.
