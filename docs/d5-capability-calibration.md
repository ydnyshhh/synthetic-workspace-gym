# D5 capability calibration

D5 uses deterministic internal profiles without changing the external difficulty
number. The seed's final decimal digit selects the profile:

- `d5_a`: digits 0–2 (30%)
- `d5_b`: digits 3–7 (50%)
- `d5_c`: digits 8–9 (20%)

The profile is stored in `difficulty_realization.profile`, benchmark rows, and D5
task IDs. D1–D4 do not receive a profile.

## Capability weights

Pipeline uses execution 0.05, top-level shape 0.05, row schema 0.10,
normalization 0.15, deduplication 0.15, filtering 0.10, aggregation 0.25,
ordering 0.05, and determinism 0.10. Visible and hidden semantic scores are
combined. Output keys and semantics remain fully visible.

Retrieval uses collection 0.10, quantity 0.10, cursor 0.10, request ID 0.05,
record count 0.10, total quantity 0.15, warehouse default 0.10, warehouse
sorting/deduplication 0.10, empty records 0.05, and alternate-payload
generalization 0.15. Valid partial adapters are not capped.

Tabular uses script existence 0.03, execution 0.05, valid JSON 0.04, schema
0.05, active coercion 0.08, fractional aggregation 0.08, canonical identity
0.12, deduplication 0.13, timestamp normalization 0.10, temporal join 0.12,
hidden end-to-end behavior 0.15, and determinism 0.05. Focused fixtures isolate
the semantic capabilities. Determinism does not depend on correctness.

Hard failures are reserved for states where semantic evaluation is impossible,
such as a missing required target, import/entrypoint failure, timeout, missing
output, or invalid JSON.

## Offline calibration

Generate the five-seed, three-family oracle report with:

```powershell
$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe scripts\generate_d5_calibration_report.py `
  --scratch-root .tmp-tests\d5-calibration-fresh `
  --output analysis\d5-calibration-report.json
```

The scratch path must not already exist. The report records untouched,
one-capability, two-capability, all-but-one, and full reference states and exits
nonzero when a monotonicity violation is found.

Hosted Qwen3.5-4B configurations are prepared under `configs/evals/` for the
proposed 0.1.25 release. They must not be launched until that version is
explicitly approved and published.
