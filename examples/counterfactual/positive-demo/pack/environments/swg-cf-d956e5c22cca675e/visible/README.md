# CSV Schema Drift Repair

One or more Python files in this workspace are buggy. Repair the code so the hidden tests pass.

## What to preserve
- Keep the public function names stable.
- Prefer targeted fixes over rewrites.
- Use the visible smoke test command to sanity-check your changes.

## Smoke test
- `python run_example.py`

## Likely target files
- `src/repair_target/parser.py`

## Expected behavior
- Load every row keyed by the visible `account_id` column; do not look for `customer_id`.
- Normalize `region` with `strip().lower()` before reporting.
- Exclude rows whose normalized status is `cancelled` in the report step.
- Return report rows sorted lexicographically by lowercase `region`.

## Hints
- The CSV headers and the parser must agree on the account identifier column.
- Cancelled rows should be excluded from the final report, not dropped before parsing.
- The final report is sorted by region, not by aggregate size.
