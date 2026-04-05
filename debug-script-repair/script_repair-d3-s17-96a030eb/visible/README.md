# Batch Summary Repair

One or more Python files in this workspace are buggy. Repair the code so the hidden tests pass.

## What to preserve
- Keep the public function names stable.
- Prefer targeted fixes over rewrites.
- Use the visible smoke test command to sanity-check your changes.

## Smoke test
- `python run_example.py`

## Likely target files
- `src/repair_target/batch.py`
- `src/repair_target/io_helpers.py`

## Hints
- The measurement loader should read from the visible `data` directory.
- The batch summary should aggregate every row in the CSV.
- If the module fails to import, inspect recent edits around function signatures and imports.
