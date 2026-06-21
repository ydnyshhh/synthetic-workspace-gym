# GLM-5.2 to Qwen3.5-0.8B Trajectory Distillation

This experiment tests whether Qwen3.5-0.8B can learn the SWG closed-loop tool-control policy from GLM-5.2 expert trajectories. The first stage builds a perfect-only supervised dataset from GLM trajectories. Later stages will compare raw multi-tool training, sequentialized single-tool training, scenario-balanced partial-trace training, and recovery-state training from Qwen failure traces.

## Motivation

The target behavior is not full trace memorization. The dataset turns successful hosted-eval trajectories into state to next tool-action examples that teach the control loop:

```text
inspect -> read relevant files -> write/edit -> run public check -> read artifact -> submit
```

The first experiment extracts only perfect-reward GLM-5.2 traces and converts assistant tool-call turns into intermediate action windows. These examples are not yet assumed to be Prime trainer-ready.

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

The builder can also read a single JSON file containing either a `samples` list or a `pages` list, but raw trace bundles should remain local and ignored.

## Build Command

```bash
python experiments/glm52_qwen08_distill/build_dataset.py \
  --input-dir data/raw_traces/glm52 \
  --output-dir data/processed_traces/glm52_qwen08 \
  --report-dir data/reports/glm52_qwen08 \
  --teacher glm-5.2 \
  --student Qwen/Qwen3.5-0.8B \
  --eval-id kxhqr8w6kxeficm93rp7s5k6 \
  --reward-filter perfect \
  --write-raw \
  --write-sequential
```

The default quality gate writes a report but refuses JSONL output when critical issues are found. Use `--allow-quality-warnings` only for analysis-only dataset output after reviewing the audit section.

The report separates observed quality issues from issues that remain in the written dataset. `ready_for_sft` is based on the written dataset after invalid target windows are excluded, while the default gate still requires review for any observed critical issue. Target validation checks tool names, required tool arguments such as `submit.path_or_answer`, absolute path attempts, and invalid `run_python` script paths. Invalid assistant actions are kept in later histories with their tool corrections, but they are not emitted as supervised targets.

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

The JSONL shape is:

```json
{"messages": [...], "target": {...}, "metadata": {...}}
```

That shape is intentionally inspectable and intermediate.

This branch includes two generic exporters. Prime-RL's SFT docs say prompt/completion rows mask the prompt and compute loss only over the completion, while `messages` rows train over all assistant turns and take precedence if both formats are present. For action-window distillation, prefer the prompt/completion export unless the target trainer explicitly supports final-assistant-only masking for `messages`.

Use the prompt/completion exporter when you need unambiguous final-target-only supervision:

```bash
python experiments/glm52_qwen08_distill/export_prompt_completion_sft.py \
  --input-jsonl data/processed_traces/glm52_qwen08/glm52_perfect_sequential_train.jsonl \
  --output-jsonl data/processed_traces/glm52_qwen08/glm52_perfect_sequential_train_prompt_completion.jsonl
```

It writes records shaped as:

```json
{"prompt": [...], "completion": {"role": "assistant", "content": "", "tool_calls": [...]}, "metadata": {...}}
```

Prime tool-use SFT also needs tool definitions on each row. After exporting prompt/completion split files, add SWG tool definitions and drop metadata for trainer-facing files:

```powershell
python experiments/glm52_qwen08_distill/scripts/add_swg_tool_defs.py `
  --input-jsonl data/processed_traces/glm52_qwen08/glm52_perfect_sequential_train_prompt_completion.jsonl `
  --output-jsonl data/processed_traces/glm52_qwen08/glm52_perfect_sequential_train_pc_tooldefs.jsonl `
  --drop-metadata

python experiments/glm52_qwen08_distill/scripts/add_swg_tool_defs.py `
  --input-jsonl data/processed_traces/glm52_qwen08/glm52_perfect_sequential_dev_prompt_completion.jsonl `
  --output-jsonl data/processed_traces/glm52_qwen08/glm52_perfect_sequential_dev_pc_tooldefs.jsonl `
  --drop-metadata
```

These rows keep only `prompt`, `completion`, and `tool_defs`. The `tool_defs` value is a list of SWG tool schemas for `read_file`, `write_file`, `append_file`, `list_directory`, `run_shell`, `run_python`, and `submit`.

The messages-format exporter is also available as a schema candidate:

```bash
python experiments/glm52_qwen08_distill/export_prime_sft.py \
  --input-jsonl data/processed_traces/glm52_qwen08/glm52_perfect_sequential_actions.jsonl \
  --output-jsonl data/processed_traces/glm52_qwen08/glm52_perfect_sequential_prime_sft.jsonl
```

Before training, split by trace rather than by action window to avoid leakage:

```bash
python experiments/glm52_qwen08_distill/split_dataset.py \
  --input-jsonl data/processed_traces/glm52_qwen08/glm52_perfect_sequential_actions.jsonl \
  --output-dir data/processed_traces/glm52_qwen08 \
  --prefix glm52_perfect_sequential \
  --dev-ratio 0.1 \
  --trace-test-ratio 0.1 \
  --seed 42
```

This writes:

```text
glm52_perfect_sequential_train.jsonl
glm52_perfect_sequential_dev.jsonl
glm52_perfect_sequential_trace_test.jsonl
```

## Safety

Do not commit raw hosted-eval traces, generated JSONL datasets, model checkpoints, adapters, logs, WandB outputs, or report outputs under `data/`.

Only commit scripts, configs, README files, small synthetic fixtures, report templates, and tests.

## Tests

Run the synthetic fixture tests with:

```bash
python -m unittest experiments.glm52_qwen08_distill.tests.test_dataset_builder -v
```

The tests use `experiments/glm52_qwen08_distill/tests/fixtures/tiny_trace.json` and cover quality-gate behavior, invalid `run_python` detection, malformed submit argument detection, absolute-path auditing, sequential single-tool targets, tool-call ID preservation in history, the messages-format exporter, the prompt/completion exporter, SWG tool-definition injection, and trace-group split behavior.

## Next Steps

1. Build and inspect the perfect-only raw and sequentialized datasets.
2. Review scenario coverage and data-quality counters in the report.
3. Prefer the sequentialized variant if raw targets contain frequent multi-tool calls.
4. Build a scenario-balanced partial-trace variant if perfect-only coverage is sparse.
5. Later, add recovery-state examples from Qwen failure traces before launching training.
6. Verify or adapt the generic messages-format exporter once the intermediate dataset has passed quality review.
