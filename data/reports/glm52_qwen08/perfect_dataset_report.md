# Perfect Dataset Report

## Run-Level Stats

- Evaluation ID: `kxhqr8w6kxeficm93rp7s5k6`
- Total files loaded: 20
- Total samples: 390
- Unique example IDs: 390
- Perfect examples: 304
- Non-perfect examples: 86
- Reward min / mean / median / max: 0.333333 / 0.9166604692307693 / 1.0 / 1.0

## Dataset Stats

| variant | SFT examples | traces used | avg examples/trace | avg history messages | avg target tool calls | max target tool calls | submit | write_file | run_shell | run_python |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw | 2084 | 304 | 6.86 | 12.62 | 2.04 | 8 | 305 | 454 | 217 | 72 |
| sequential | 4244 | 304 | 13.96 | 12.27 | 1.00 | 1 | 305 | 454 | 217 | 72 |

## Data Quality

- malformed_tool_calls: 0
- unknown_tools: 0
- absolute_path_attempts: 3
- invalid_run_python_calls: 0
- assistant_prose_only_turns_skipped: 0
- reasoning_content_fields_stripped: 825
- samples_with_missing_reward: 0
- samples_with_missing_task_metadata: 0
- duplicate_example_ids: 0
- duplicate_trace_ids: 0
- sequentialization_warnings: 0

## Scenario Coverage

| scenario | total traces | perfect traces | action examples raw | action examples sequential |
| --- | ---: | ---: | ---: | ---: |
| artifact_stitch_pipeline | 30 | 30 | 260 | 579 |
| channel_status_pivot | 30 | 10 | 58 | 93 |
| csv_schema_drift | 30 | 30 | 195 | 373 |
| incident_report_bundle | 30 | 28 | 175 | 470 |
| inventory_report | 30 | 29 | 222 | 393 |
| **migration_plan_bundle** | 30 | 0 | 0 | 0 |
| monthly_segment_report | 30 | 30 | 175 | 281 |
| path_batch | 30 | 25 | 186 | 355 |
| sales_csv_pipeline | 30 | 28 | 207 | 466 |
| service_config_reconciliation | 30 | 20 | 122 | 331 |
| team_hours_pipeline | 30 | 30 | 211 | 464 |
| timestamp_normalization | 30 | 14 | 104 | 175 |
| weekly_refund_rollup | 30 | 30 | 169 | 264 |

## Coverage Warnings

- Zero perfect coverage: migration_plan_bundle
- Low perfect coverage: migration_plan_bundle

## Recommendation

- Ready for SFT: False
- Perfect-only coverage too sparse: True
- Sequentialized variant preferable: True
- Next dataset variant: Build a scenario-balanced partial-trace dataset that adds high-reward non-perfect traces for sparse scenarios.
