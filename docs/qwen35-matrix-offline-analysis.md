# Qwen3.5-4B training-matrix offline trajectory analysis

Export timestamp: `2026-07-29T14:26:24.094167+00:00`

## Archive coverage

- Normalized trajectories: **5,234**
- Training milestone trajectories: **4,480**
- Hosted-evaluation trajectories: **754**
- Recorded training tokens: **8,245,925,134**
- Recorded training cost: **$679.87**
- Exporting and analysis launched no inference and consumed no training credits.

Training coverage is a fixed milestone panel, not every rollout from every step. Hosted-evaluation exports include every sample the platform retained.

## Final sampled training behavior

| Run | n | Mean reward | Perfect | Submit | Tools | Explore flag |
|---|---|---|---|---|---|---|
| all_family_seed_42 | 64 | 0.970 | 90.6% | 96.9% | 14.9 | 50.0% |
| all_family_seed_43 | 64 | 0.993 | 98.4% | 65.6% | 22.1 | 81.2% |
| checkpoint_pipeline_step_100 | 64 | 0.689 | 50.0% | 78.1% | 12.8 | 45.3% |
| checkpoint_script_repair_step_150 | 64 | 0.819 | 51.6% | 100.0% | 12.9 | 56.2% |
| composition_seed_42 | 64 | 0.998 | 98.4% | 100.0% | 18.1 | 70.3% |
| composition_seed_43 | 64 | 0.995 | 98.4% | 90.6% | 17.0 | 73.4% |
| specialist_pipeline | 64 | 1.000 | 100.0% | 100.0% | 16.6 | 95.3% |
| specialist_retrieval | 64 | 0.971 | 87.5% | 96.9% | 16.3 | 100.0% |
| specialist_script_repair | 64 | 1.000 | 100.0% | 100.0% | 12.1 | 0.0% |
| specialist_tabular | 64 | 1.000 | 100.0% | 100.0% | 10.3 | 0.0% |

These are training-distribution samples. Near-perfect values indicate saturation and must not be interpreted as held-out generalization.

## Existing hosted-evaluation results

| Evaluation | n | Mean reward | Perfect | Partial | Zero |
|---|---|---|---|---|---|
| base_composite_heldout | 24 | 0.723 | 16.7% | 83.3% | 0.0% |
| base_d5_panel | 100 | 0.557 | 22.0% | 76.0% | 2.0% |
| base_in_distribution | 390 | 0.701 | 48.2% | 50.0% | 1.8% |
| base_scenario_heldout | 240 | 0.758 | 57.1% | 41.2% | 1.7% |

## Behavioral findings

1. **Specialist saturation is real.** Pipeline, script repair, and tabular converge to almost entirely perfect training batches, leaving little or no usable policy gradient.
2. **Submission is not enforced by reward.** Several perfect trajectories never call `submit`, especially in the second all-family seed. This is an evaluator/reward mismatch.
3. **Retrieval and composite work remain exploration-heavy.** Agents repeatedly list and read broad document trees before committing to authoritative evidence.
4. **Tool errors are usually recoverable.** Missing files, rejected inline Python, and premature execution usually lead to a correction; failures persist when the correction loop reaches the horizon.
5. **Verification remains uneven.** Some code trajectories submit after source edits without a successful post-edit public check. Retrieval writes are often not reread.
6. **No strong context-loss signature dominates.** Most low rewards are better explained by evidence selection, weak action ordering, missing verification, or horizon exhaustion.

## Family-level metrics

| Source | Run | Family | n | Reward | Perfect | Submit |
|---|---|---|---|---|---|---|
| evaluation | base_composite_heldout | composite_workspace | 24 | 0.723 | 16.7% | 95.8% |
| evaluation | base_d5_panel | pipeline | 30 | 0.655 | 30.0% | 63.3% |
| evaluation | base_d5_panel | retrieval_workspace | 10 | 0.778 | 50.0% | 100.0% |
| evaluation | base_d5_panel | script_repair | 30 | 0.396 | 6.7% | 96.7% |
| evaluation | base_d5_panel | tabular | 30 | 0.546 | 20.0% | 73.3% |
| evaluation | base_in_distribution | pipeline | 90 | 0.658 | 47.8% | 80.0% |
| evaluation | base_in_distribution | retrieval_workspace | 90 | 0.812 | 57.8% | 100.0% |
| evaluation | base_in_distribution | script_repair | 120 | 0.657 | 45.8% | 98.3% |
| evaluation | base_in_distribution | tabular | 90 | 0.693 | 42.2% | 92.2% |
| evaluation | base_scenario_heldout | pipeline | 60 | 0.527 | 30.0% | 96.7% |
| evaluation | base_scenario_heldout | retrieval_workspace | 60 | 0.925 | 56.7% | 100.0% |
| evaluation | base_scenario_heldout | script_repair | 60 | 0.727 | 65.0% | 95.0% |
| evaluation | base_scenario_heldout | tabular | 60 | 0.852 | 76.7% | 95.0% |
| training | all_family_seed_42 | pipeline | 148 | 0.910 | 83.1% | 83.1% |
| training | all_family_seed_42 | retrieval_workspace | 150 | 0.893 | 66.7% | 98.7% |
| training | all_family_seed_42 | script_repair | 139 | 0.854 | 74.1% | 89.2% |
| training | all_family_seed_42 | tabular | 139 | 0.887 | 83.5% | 89.2% |
| training | all_family_seed_43 | pipeline | 116 | 0.921 | 87.1% | 86.2% |
| training | all_family_seed_43 | retrieval_workspace | 201 | 0.897 | 62.7% | 98.5% |
| training | all_family_seed_43 | script_repair | 104 | 0.915 | 87.5% | 68.3% |
| training | all_family_seed_43 | tabular | 155 | 0.872 | 83.2% | 76.8% |
| training | checkpoint_pipeline_step_100 | pipeline | 5 | 1.000 | 100.0% | 80.0% |
| training | checkpoint_pipeline_step_100 | retrieval_workspace | 28 | 0.704 | 28.6% | 100.0% |
| training | checkpoint_pipeline_step_100 | script_repair | 18 | 1.000 | 100.0% | 100.0% |
| training | checkpoint_pipeline_step_100 | tabular | 13 | 0.108 | 7.7% | 0.0% |
| training | checkpoint_script_repair_step_150 | retrieval_workspace | 41 | 0.717 | 24.4% | 100.0% |
| training | checkpoint_script_repair_step_150 | script_repair | 23 | 1.000 | 100.0% | 100.0% |
| training | composition_seed_42 | composite_workspace | 88 | 0.991 | 93.2% | 86.4% |
| training | composition_seed_42 | pipeline | 104 | 0.889 | 81.7% | 81.7% |
| training | composition_seed_42 | retrieval_workspace | 146 | 0.902 | 61.0% | 97.3% |
| training | composition_seed_42 | script_repair | 121 | 0.934 | 86.8% | 91.7% |
| training | composition_seed_42 | tabular | 117 | 0.848 | 79.5% | 63.2% |
| training | composition_seed_43 | composite_workspace | 89 | 0.948 | 83.1% | 59.6% |
| training | composition_seed_43 | pipeline | 129 | 0.881 | 81.4% | 87.6% |
| training | composition_seed_43 | retrieval_workspace | 101 | 0.880 | 57.4% | 100.0% |
| training | composition_seed_43 | script_repair | 89 | 0.884 | 79.8% | 84.3% |
| training | composition_seed_43 | tabular | 168 | 0.806 | 75.6% | 57.7% |
| training | specialist_pipeline | pipeline | 384 | 0.922 | 88.5% | 90.1% |
| training | specialist_retrieval | retrieval_workspace | 576 | 0.935 | 72.4% | 90.8% |
| training | specialist_script_repair | script_repair | 512 | 0.914 | 88.1% | 85.4% |
| training | specialist_tabular | tabular | 576 | 0.859 | 80.6% | 75.7% |

## Interpretation constraints

- Milestone batches are not matched task panels, so longitudinal reward changes are descriptive.
- Behavioral flags are deterministic heuristics. They identify review candidates rather than ground truth labels.
- `wrong_evidence` is only assigned to sub-0.9 retrieval/composite repairs and should be manually confirmed.
- Final all-family and composition checkpoints still require the frozen held-out suite before transfer claims.

## Representative review queue

### wrong_evidence: `checkpoint_script_repair_step_150` / `swg.train.retrieval_workspace.migration_plan_bundle.d2.s45`

- Reward: `0.200000`; family: `retrieval_workspace`; difficulty: `2`; submitted: `True`
- Tool sequence: `list_directory -> read_file -> read_file -> list_directory -> list_directory -> list_directory -> list_directory -> list_directory -> read_file -> read_file -> read_file -> read_file -> read_file -> write_file -> submit`
- Error: Directory not found: src error: directory_not_found
- Final reasoning: Done. Submit the artifact.

### wrong_evidence: `checkpoint_script_repair_step_150` / `swg.train.retrieval_workspace.migration_plan_bundle.d3.s36`

- Reward: `0.200000`; family: `retrieval_workspace`; difficulty: `3`; submitted: `True`
- Tool sequence: `list_directory -> read_file -> read_file -> list_directory -> list_directory -> list_directory -> list_directory -> read_file -> read_file -> read_file -> read_file -> read_file -> read_file -> write_file -> submit`
- Error: Directory not found: documents error: directory_not_found
- Final reasoning: Now submit the final artifact.

### wrong_evidence: `checkpoint_pipeline_step_100` / `swg.train.retrieval_workspace.migration_plan_bundle.d2.s45`

- Reward: `0.200000`; family: `retrieval_workspace`; difficulty: `2`; submitted: `True`
- Tool sequence: `list_directory -> read_file -> read_file -> list_directory -> read_file -> list_directory -> list_directory -> read_file -> read_file -> write_file -> submit`
- Error: none
- Final reasoning: documents: - Schema v5 changes from specs/schema_v5.md (field renames and drops) - Backfill rules from notes/backfill_rules.md - Ignored docs/retention_policy.md as it doesn't affect the target artifact The guidance says the artifact reflects visible evidence, so I should submit.

### weak_planning: `checkpoint_pipeline_step_100` / `swg.train.tabular.monthly_segment_report.d3.s37`

- Reward: `0.000000`; family: `tabular`; difficulty: `3`; submitted: `False`
- Tool sequence: `list_directory -> read_file -> read_file -> list_directory -> read_file -> read_file -> write_file -> run_python -> write_file`
- Error: Ran python command: process_report.py stderr: Traceback (most recent call last): File "/tmp/swg-prime-jqdg_8v0/active-workspaces/tabular-d3-s37-6f89c9bf/process_report.py", line 19, in <module> parsed = datetime.strptime(date_order, '%Y-%m-
- Final reasoning: The date parsing logic needs to be fixed. The issue is that the if/else chain is checking formats in the wrong order - it's trying to parse '06-Mar-2024' with '%Y-%m-%d' first. I need to fix the order of checks.

### weak_planning: `checkpoint_pipeline_step_100` / `swg.train.tabular.monthly_segment_report.d3.s37`

- Reward: `0.000000`; family: `tabular`; difficulty: `3`; submitted: `False`
- Tool sequence: `list_directory -> read_file -> read_file -> list_directory -> read_file -> read_file -> write_file -> run_python -> write_file`
- Error: Ran python command: process_report.py stderr: Traceback (most recent call last): File "/tmp/swg-prime-b9z9digc/active-workspaces/tabular-d3-s37-6f89c9bf/process_report.py", line 15, in <module> parsed_date = datetime.strptime(row['order_dat
- Final reasoning: The date parsing logic has a bug - it's trying to parse ISO format dates with the wrong format. I need to fix the date parsing to handle multiple formats properly.

### weak_planning: `checkpoint_pipeline_step_100` / `swg.train.tabular.monthly_segment_report.d3.s37`

- Reward: `0.000000`; family: `tabular`; difficulty: `3`; submitted: `False`
- Tool sequence: `read_file -> read_file -> list_directory -> read_file -> read_file -> list_directory -> write_file -> run_python -> write_file`
- Error: Ran python command: process_report.py stderr: Traceback (most recent call last): File "/tmp/swg-prime-w_ij142q/active-workspaces/tabular-d3-s37-6f89c9bf/process_report.py", line 20, in <module> order_date = datetime.strptime(f"{year}-{month
- Final reasoning: There's an issue with parsing dates. The date format `11-2024-03` seems unusual - it looks like `L-655` from legacy_code was being parsed instead of the order_date. Let me fix the script to handle date parsing more carefully.

### excessive_exploration: `checkpoint_script_repair_step_150` / `swg.train.retrieval_workspace.migration_plan_bundle.d2.s45`

- Reward: `0.200000`; family: `retrieval_workspace`; difficulty: `2`; submitted: `True`
- Tool sequence: `list_directory -> read_file -> read_file -> list_directory -> list_directory -> list_directory -> list_directory -> list_directory -> read_file -> read_file -> read_file -> read_file -> read_file -> write_file -> submit`
- Error: Directory not found: src error: directory_not_found
- Final reasoning: Done. Submit the artifact.

### excessive_exploration: `checkpoint_script_repair_step_150` / `swg.train.retrieval_workspace.migration_plan_bundle.d3.s36`

- Reward: `0.200000`; family: `retrieval_workspace`; difficulty: `3`; submitted: `True`
- Tool sequence: `list_directory -> read_file -> read_file -> list_directory -> list_directory -> list_directory -> list_directory -> read_file -> read_file -> read_file -> read_file -> read_file -> read_file -> write_file -> submit`
- Error: Directory not found: documents error: directory_not_found
- Final reasoning: Now submit the final artifact.

### excessive_exploration: `checkpoint_pipeline_step_100` / `swg.train.retrieval_workspace.incident_report_bundle.d3.s3`

- Reward: `0.400000`; family: `retrieval_workspace`; difficulty: `3`; submitted: `True`
- Tool sequence: `list_directory -> read_file -> read_file -> list_directory -> list_directory -> list_directory -> read_file -> read_file -> read_file -> read_file -> write_file -> submit`
- Error: none
- Final reasoning: 3 mapping, owner pipeline-ops) - Ordered timeline preserving all event=action detail=... lines from the log - All timestamps and markers preserved This reflects the visible evidence from docs/action_capture.md, logs/inc-512.log, and notes/severity_policy.md. The task is complete.

### tool_error: `checkpoint_pipeline_step_100` / `swg.train.tabular.monthly_segment_report.d3.s37`

- Reward: `0.000000`; family: `tabular`; difficulty: `3`; submitted: `False`
- Tool sequence: `list_directory -> read_file -> read_file -> list_directory -> read_file -> read_file -> write_file -> run_python -> write_file`
- Error: Ran python command: process_report.py stderr: Traceback (most recent call last): File "/tmp/swg-prime-jqdg_8v0/active-workspaces/tabular-d3-s37-6f89c9bf/process_report.py", line 19, in <module> parsed = datetime.strptime(date_order, '%Y-%m-
- Final reasoning: The date parsing logic needs to be fixed. The issue is that the if/else chain is checking formats in the wrong order - it's trying to parse '06-Mar-2024' with '%Y-%m-%d' first. I need to fix the order of checks.

### tool_error: `checkpoint_pipeline_step_100` / `swg.train.tabular.monthly_segment_report.d3.s37`

- Reward: `0.000000`; family: `tabular`; difficulty: `3`; submitted: `False`
- Tool sequence: `list_directory -> read_file -> read_file -> list_directory -> read_file -> read_file -> write_file -> run_python -> write_file`
- Error: Ran python command: process_report.py stderr: Traceback (most recent call last): File "/tmp/swg-prime-b9z9digc/active-workspaces/tabular-d3-s37-6f89c9bf/process_report.py", line 15, in <module> parsed_date = datetime.strptime(row['order_dat
- Final reasoning: The date parsing logic has a bug - it's trying to parse ISO format dates with the wrong format. I need to fix the date parsing to handle multiple formats properly.

### tool_error: `checkpoint_pipeline_step_100` / `swg.train.tabular.monthly_segment_report.d3.s37`

- Reward: `0.000000`; family: `tabular`; difficulty: `3`; submitted: `False`
- Tool sequence: `read_file -> read_file -> list_directory -> read_file -> read_file -> list_directory -> write_file -> run_python -> write_file`
- Error: Ran python command: process_report.py stderr: Traceback (most recent call last): File "/tmp/swg-prime-w_ij142q/active-workspaces/tabular-d3-s37-6f89c9bf/process_report.py", line 20, in <module> order_date = datetime.strptime(f"{year}-{month
- Final reasoning: There's an issue with parsing dates. The date format `11-2024-03` seems unusual - it looks like `L-655` from legacy_code was being parsed instead of the order_date. Let me fix the script to handle date parsing more carefully.

### context_loss: `all_family_seed_42` / `swg.train.pipeline.artifact_stitch_pipeline.d4.s17`

- Reward: `0.566666`; family: `pipeline`; difficulty: `4`; submitted: `False`
- Tool sequence: `list_directory -> read_file -> list_directory -> read_file -> read_file -> list_directory -> list_directory -> list_directory -> read_file -> read_file -> list_directory -> list_directory -> read_file -> read_file -> read_file -> read_file -> write_file -> run_shell -> write_file -> run_shell -> read_file -> list_directory -> list_directory -> read_file -> read_file`
- Error: Directory not found: data/fragment error: directory_not_found
- Final reasoning: The config looks correct. Let me re-read the artifacts/merged_report.json file again.

### context_loss: `composition_seed_42` / `swg.train.composite_workspace.retrieval_guided_pipeline_repair.d3.s13`

- Reward: `1.000000`; family: `composite_workspace`; difficulty: `3`; submitted: `True`
- Tool sequence: `list_directory -> read_file -> read_file -> read_file -> read_file -> read_file -> read_file -> read_file -> list_directory -> list_directory -> list_directory -> list_directory -> list_directory -> list_directory -> read_file -> read_file -> read_file -> read_file -> read_file -> read_file -> read_file -> read_file -> read_file -> read_file -> read_file -> read_file -> read_file -> read_file -> read_file -> write_file -> write_file -> write_file -> run_python -> run_python -> read_file -> read_file -> read_file -> submit`
- Error: File not found: release/pointer.txt error: file_not_found
- Final reasoning: The pipeline has been fixed. Let me submit the answer.

### context_loss: `composition_seed_42` / `swg.train.composite_workspace.retrieval_guided_pipeline_repair.d2.s71`

- Reward: `1.000000`; family: `composite_workspace`; difficulty: `2`; submitted: `True`
- Tool sequence: `list_directory -> read_file -> read_file -> read_file -> read_file -> read_file -> read_file -> read_file -> read_file -> list_directory -> list_directory -> read_file -> read_file -> list_directory -> read_file -> read_file -> read_file -> read_file -> read_file -> read_file -> read_file -> run_python -> run_python -> read_file -> read_file -> read_file -> write_file -> write_file -> run_python -> read_file -> read_file -> read_file -> run_python -> read_file -> read_file -> read_file -> submit`
- Error: File not found: release/pointer.txt error: file_not_found
- Final reasoning: The pipeline is working correctly. The public check passed. Let me submit.

### missing_verification: `checkpoint_pipeline_step_100` / `swg.train.tabular.monthly_segment_report.d3.s37`

- Reward: `0.000000`; family: `tabular`; difficulty: `3`; submitted: `False`
- Tool sequence: `list_directory -> read_file -> read_file -> list_directory -> read_file -> read_file -> write_file -> run_python -> write_file`
- Error: Ran python command: process_report.py stderr: Traceback (most recent call last): File "/tmp/swg-prime-jqdg_8v0/active-workspaces/tabular-d3-s37-6f89c9bf/process_report.py", line 19, in <module> parsed = datetime.strptime(date_order, '%Y-%m-
- Final reasoning: The date parsing logic needs to be fixed. The issue is that the if/else chain is checking formats in the wrong order - it's trying to parse '06-Mar-2024' with '%Y-%m-%d' first. I need to fix the order of checks.

### missing_verification: `checkpoint_pipeline_step_100` / `swg.train.tabular.monthly_segment_report.d3.s37`

- Reward: `0.000000`; family: `tabular`; difficulty: `3`; submitted: `False`
- Tool sequence: `list_directory -> read_file -> read_file -> list_directory -> read_file -> read_file -> write_file -> run_python -> write_file`
- Error: Ran python command: process_report.py stderr: Traceback (most recent call last): File "/tmp/swg-prime-b9z9digc/active-workspaces/tabular-d3-s37-6f89c9bf/process_report.py", line 15, in <module> parsed_date = datetime.strptime(row['order_dat
- Final reasoning: The date parsing logic has a bug - it's trying to parse ISO format dates with the wrong format. I need to fix the date parsing to handle multiple formats properly.

### missing_verification: `checkpoint_pipeline_step_100` / `swg.train.tabular.monthly_segment_report.d3.s37`

- Reward: `0.000000`; family: `tabular`; difficulty: `3`; submitted: `False`
- Tool sequence: `read_file -> read_file -> list_directory -> read_file -> read_file -> list_directory -> write_file -> run_python -> write_file`
- Error: Ran python command: process_report.py stderr: Traceback (most recent call last): File "/tmp/swg-prime-w_ij142q/active-workspaces/tabular-d3-s37-6f89c9bf/process_report.py", line 20, in <module> order_date = datetime.strptime(f"{year}-{month
- Final reasoning: There's an issue with parsing dates. The date format `11-2024-03` seems unusual - it looks like `L-655` from legacy_code was being parsed instead of the order_date. Let me fix the script to handle date parsing more carefully.

### failed_recovery: `checkpoint_pipeline_step_100` / `swg.train.tabular.monthly_segment_report.d3.s37`

- Reward: `0.000000`; family: `tabular`; difficulty: `3`; submitted: `False`
- Tool sequence: `list_directory -> read_file -> read_file -> list_directory -> read_file -> read_file -> write_file -> run_python -> write_file`
- Error: Ran python command: process_report.py stderr: Traceback (most recent call last): File "/tmp/swg-prime-jqdg_8v0/active-workspaces/tabular-d3-s37-6f89c9bf/process_report.py", line 19, in <module> parsed = datetime.strptime(date_order, '%Y-%m-
- Final reasoning: The date parsing logic needs to be fixed. The issue is that the if/else chain is checking formats in the wrong order - it's trying to parse '06-Mar-2024' with '%Y-%m-%d' first. I need to fix the order of checks.

### failed_recovery: `checkpoint_pipeline_step_100` / `swg.train.tabular.monthly_segment_report.d3.s37`

- Reward: `0.000000`; family: `tabular`; difficulty: `3`; submitted: `False`
- Tool sequence: `list_directory -> read_file -> read_file -> list_directory -> read_file -> read_file -> write_file -> run_python -> write_file`
- Error: Ran python command: process_report.py stderr: Traceback (most recent call last): File "/tmp/swg-prime-b9z9digc/active-workspaces/tabular-d3-s37-6f89c9bf/process_report.py", line 15, in <module> parsed_date = datetime.strptime(row['order_dat
- Final reasoning: The date parsing logic has a bug - it's trying to parse ISO format dates with the wrong format. I need to fix the date parsing to handle multiple formats properly.
