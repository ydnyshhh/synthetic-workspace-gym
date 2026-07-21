# Retrieval D5 structural hardening

Retrieval D5 uses a family-specific deterministic profile distribution: 20% D5-A, 50% D5-B, and 30% D5-C. Tabular remains frozen at its previous profile distribution and capability weights. Pipeline changes only from 30/50/20 to 20/50/30, shifting ten percent of tasks from A to C.

## Profiles

- D5-A (`client_adapter_sync`) is the recoverable single-adapter task. API, pagination, migration, and warehouse policy evidence is distributed instead of presenting direct rename instructions.
- D5-B (`client_adapter_policy_sync`) requires consistent changes to `config/client_runtime.json`, `src/client_parser.py`, and `src/client_summary.py`. Hidden evaluation varies configuration.
- D5-C (`versioned_client_migration`) requires resolving `release/current_manifest.json` against v2/v3 and cutover evidence before updating `config/client.json`, `src/adapter.py`, and `src/serializer.py`.

B and C independently test authority resolution, schema mapping, numeric-string quantity parsing, pagination, missing-value policy, regional aliases, SKU deduplication, timezone-aware latest-record selection, the output contract, and hidden configuration/payload generalization.

## Capability weights

| Capability | Weight |
|---|---:|
| authority resolution | 0.10 |
| schema mapping | 0.15 |
| quantity parsing | 0.10 |
| pagination | 0.10 |
| missing-value policy | 0.10 |
| regional override | 0.10 |
| deduplication | 0.10 |
| timestamp resolution | 0.10 |
| output contract | 0.05 |
| hidden generalization | 0.10 |

Authority credit is based on resulting configuration and behavior, never document-read telemetry.

## Offline oracle gate

Run:

```powershell
$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe scripts\generate_retrieval_d5_oracle_report.py --scratch-root .tmp-tests\retrieval-d5-oracle-fresh --output analysis\retrieval-d5-oracle-report.json
```

For B and C the required state bands are:

| State | Reward band |
|---|---:|
| untouched | at most 0.15 |
| version/config only | 0.15–0.30 |
| parser only | 0.25–0.45 |
| parser + config | 0.40–0.65 |
| all except edge/output behavior | 0.65–0.85 |
| full | 1.00 |

## Prepared hosted calibration

`configs/evals/swg-0.1.26-qwen35-4b-retrieval-d5-profile-balanced-15-hosted.toml` contains fifteen distinct seeds: five A, five B, and five C. It is prepared only and must not be launched until SWG 0.1.26 is reviewed and published.

Acceptance targets are combined mean 0.40–0.65, at least six distinct rewards, no bucket over 40%, no more than five perfect results, and partial reward on at least half of trajectories. Profile means should be A 0.60–0.80, B 0.35–0.60, and C 0.15–0.45.