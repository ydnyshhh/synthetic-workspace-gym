from __future__ import annotations

import json
import random
from textwrap import dedent

from synthetic_workspace_gym.schemas import EnvironmentSpec

DOCUMENT_PREFIXES = ("docs/", "notes/", "specs/", "logs/", "changelog/")


def render_json(payload: object) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def difficulty_settings(difficulty: int) -> dict[str, object]:
    return {
        "retrieval_hops": {1: 1, 2: 2, 3: 3, 4: 4, 5: 5}[difficulty],
        "distractor_count": {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}[difficulty],
        "evidence_distribution": (
            "single_source"
            if difficulty == 1
            else "primary_plus_supporting"
            if difficulty == 2
            else "multi_source"
        ),
        "staleness_pattern": "none" if difficulty <= 3 else "stale_note" if difficulty == 4 else "superseded_changelog",
    }


def count_documents(files: dict[str, str]) -> int:
    return sum(1 for path in files if path.startswith(DOCUMENT_PREFIXES))


def document_roots(files: dict[str, str]) -> list[str]:
    roots = {path.split("/", 1)[0] for path in files if path.startswith(DOCUMENT_PREFIXES)}
    return sorted(roots)


def base_profile(
    *,
    files: dict[str, str],
    difficulty: int,
    task_type: str,
    output_style: str,
    repair_surface: str,
    failure_mode: str,
    smoke_test_quality: str,
) -> dict[str, object]:
    settings = difficulty_settings(difficulty)
    return {
        "task_type": task_type,
        "document_count": count_documents(files),
        "retrieval_hops": settings["retrieval_hops"],
        "evidence_distribution": settings["evidence_distribution"],
        "distractor_count": settings["distractor_count"],
        "staleness_pattern": settings["staleness_pattern"],
        "output_style": output_style,
        "repair_surface": repair_surface,
        "bug_scope": "cross_document",
        "failure_mode": failure_mode,
        "smoke_test_quality": smoke_test_quality,
    }


def add_distractor_documents(files: dict[str, str], rng: random.Random, count: int) -> None:
    distractors = [
        (
            "docs/retention_policy.md",
            "# Retention Policy\n\nAnalytics snapshots are retained for 30 days. This note does not affect the target artifact.\n",
        ),
        (
            "docs/auth_rotation.md",
            "# Auth Rotation\n\nService-to-service tokens rotate every 14 days. No target workspace file needs this value.\n",
        ),
        (
            "notes/archive_window.md",
            "# Archive Window\n\nCold storage compaction runs on Sundays at 02:00 UTC.\n",
        ),
        (
            "logs/heartbeat.log",
            "2026-03-20T10:00:00Z INFO worker-heartbeat ok\n2026-03-20T10:05:00Z INFO worker-heartbeat ok\n",
        ),
        (
            "docs/team_calendar.md",
            "# Team Calendar\n\nQuarter planning is scheduled for April 14.\n",
        ),
        (
            "notes/metrics_backfill.md",
            "# Metrics Backfill\n\nDashboards replay from warehouse snapshots every six hours.\n",
        ),
    ]
    rng.shuffle(distractors)
    added = 0
    for path, content in distractors:
        if path in files:
            continue
        files[path] = content
        added += 1
        if added >= count:
            return


def build_service_config_reconciliation_scenario(rng: random.Random, spec: EnvironmentSpec) -> dict[str, object]:
    difficulty = spec.difficulty
    expected_config = {
        "base_url": "https://svc.internal/v2",
        "cohort_limit": 250,
        "enable_shadow_mode": True,
        "region": "ap-south-1",
        "retry_attempts": 3,
        "service_name": "ledger-sync",
        "timeout_seconds": 45,
    }
    visible_files: dict[str, str] = {
        "config/service_config.json": render_json(
            {
                "base_url": "https://svc.internal/v1",
                "cohort_limit": 50,
                "enable_shadow_mode": False,
                "region": "us-east-1",
                "retry_attempts": 1,
                "service_name": "ledger-sync",
                "timeout_seconds": 30,
            }
        )
    }
    if difficulty == 1:
        visible_files["specs/runtime_contract.md"] = dedent(
            """
            # Runtime Contract

            The `config/service_config.json` file must contain:

            - `service_name`: `ledger-sync`
            - `base_url`: `https://svc.internal/v2`
            - `timeout_seconds`: `45`
            - `retry_attempts`: `3`
            - `region`: `ap-south-1`
            - `enable_shadow_mode`: `true`
            - `cohort_limit`: `250`
            """
        ).strip() + "\n"
    else:
        visible_files["specs/runtime_contract.md"] = dedent(
            """
            # Runtime Contract

            The service contract now uses:

            - `service_name`: `ledger-sync`
            - `base_url`: `https://svc.internal/v2`
            - `timeout_seconds`: `45`
            - `retry_attempts`: `3`
            """
        ).strip() + "\n"
        visible_files["notes/rollout_plan.md"] = dedent(
            """
            # Rollout Plan

            Shadow mode is enabled for the beta cohort.

            - `enable_shadow_mode`: `true`
            - `cohort_limit`: `250`
            """
        ).strip() + "\n"
        if difficulty >= 3:
            visible_files["notes/region_override.md"] = dedent(
                """
                # Region Override

                The current deployment stays pinned to `ap-south-1` until the next failover rehearsal.
                """
            ).strip() + "\n"
    if difficulty == 4:
        visible_files["notes/legacy_service_config.md"] = dedent(
            """
            # Legacy Service Config

            Old rollout draft:

            - `base_url`: `https://svc.internal/v1`
            - `timeout_seconds`: `30`
            - `region`: `us-east-1`
            """
        ).strip() + "\n"
    if difficulty == 5:
        visible_files["changelog/2026-01-runtime-rollout.md"] = dedent(
            """
            # 2026-01 Runtime Rollout

            Superseded by the current runtime contract and rollout plan.

            Historical values:

            - `base_url`: `https://svc.internal/v1`
            - `enable_shadow_mode`: `false`
            - `cohort_limit`: `50`
            """
        ).strip() + "\n"
    add_distractor_documents(visible_files, rng, int(difficulty_settings(difficulty)["distractor_count"]))
    output_contract = [
        "Update `config/service_config.json` in place.",
        "Keep `service_name` unchanged.",
        "The final file must be valid JSON with the required values from the document set.",
    ]
    return {
        "scenario_id": "service_config_reconciliation",
        "title": "Service Config Reconciliation",
        "description": "The runtime config has drifted from the local document set. Reconcile the config using only the visible documents in the workspace.",
        "target_path": "config/service_config.json",
        "task_type": "config_alignment",
        "output_style": "config_patch",
        "entrypoint": None,
        "files": visible_files,
        "expected_output": expected_config,
        "reference_solution_files": {
            "config/service_config.json": render_json(expected_config),
        },
        "evaluator_config": {
            "mode": "exact_json",
            "output_path": "config/service_config.json",
            "expected_path": "expected_output.json",
        },
        "hints": [
            "The runtime contract is authoritative for transport settings.",
            "Rollout notes may define feature flags separately from the base config.",
            "Ignore clearly stale legacy notes when newer documents supersede them.",
        ],
        "output_contract": output_contract,
        "structure": base_profile(
            files=visible_files,
            difficulty=difficulty,
            task_type="config_alignment",
            output_style="config_patch",
            repair_surface="config_alignment",
            failure_mode="grounding_and_semantic",
            smoke_test_quality="none",
        ),
        "document_roots": document_roots(visible_files),
    }


def build_migration_plan_bundle_scenario(rng: random.Random, spec: EnvironmentSpec) -> dict[str, object]:
    difficulty = spec.difficulty
    expected_output = {
        "backfill_rules": [
            {"source": "profiles.country_code", "target": "country"},
            {"source": "events.last_active_at", "target": "activity.last_seen_at"},
        ],
        "drop_fields": ["legacy_score"],
        "ordered_steps": [
            "Create v3 columns account_id, subscription_tier, country, and activity.last_seen_at.",
            "Backfill country from profiles.country_code and activity.last_seen_at from events.last_active_at.",
            "Update downstream readers to consume account_id and subscription_tier before cutover.",
            "Drop legacy_score after backfill validation passes.",
        ],
        "rename_fields": [
            {"from": "customer_id", "to": "account_id"},
            {"from": "plan_code", "to": "subscription_tier"},
        ],
        "schema_version": "v3",
        "validation_checks": [
            "Row counts remain unchanged after migration.",
            "No null account_id values remain after cutover.",
            "Country values match profiles.country_code for migrated rows.",
        ],
    }
    visible_files: dict[str, str] = {
        "artifacts/migration_plan.json": render_json(
            {
                "backfill_rules": [],
                "drop_fields": ["legacy_score"],
                "ordered_steps": ["Create new columns."],
                "rename_fields": [{"from": "customer_id", "to": "customer_id"}],
                "schema_version": "v2",
                "validation_checks": [],
            }
        )
    }
    if difficulty == 1:
        visible_files["specs/schema_v3.md"] = dedent(
            """
            # Schema v3

            Produce a migration plan with:

            - schema version `v3`
            - rename `customer_id -> account_id`
            - rename `plan_code -> subscription_tier`
            - drop `legacy_score`
            - backfill `country` from `profiles.country_code`
            - backfill `activity.last_seen_at` from `events.last_active_at`
            - validation checks: row counts unchanged, no null account_id values, country values preserved
            """
        ).strip() + "\n"
    else:
        visible_files["specs/schema_v3.md"] = dedent(
            """
            # Schema v3

            Required schema changes:

            - rename `customer_id -> account_id`
            - rename `plan_code -> subscription_tier`
            - drop `legacy_score`
            - schema version `v3`
            """
        ).strip() + "\n"
        visible_files["notes/backfill_rules.md"] = dedent(
            """
            # Backfill Rules

            - `profiles.country_code -> country`
            - `events.last_active_at -> activity.last_seen_at`
            """
        ).strip() + "\n"
        if difficulty >= 3:
            visible_files["docs/client_cutover.md"] = dedent(
                """
                # Client Cutover

                Recommended cutover sequence:

                1. Create v3 columns account_id, subscription_tier, country, and activity.last_seen_at.
                2. Backfill country and activity.last_seen_at.
                3. Update downstream readers before final cutover.
                4. Drop legacy_score after validation passes.

                Validation checks:

                - row counts remain unchanged
                - no null account_id values remain
                - country values match profiles.country_code
                """
            ).strip() + "\n"
    if difficulty == 4:
        visible_files["notes/v2_cutover.md"] = dedent(
            """
            # v2 Cutover Draft

            Old plan draft:

            - keep `customer_id`
            - keep `plan_code`
            - ignore `country`
            """
        ).strip() + "\n"
    if difficulty == 5:
        visible_files["changelog/2026-02-migration.md"] = dedent(
            """
            # 2026-02 Migration

            Superseded by the schema v3 documents.

            Historical draft:

            - rename `customer_id -> member_id`
            - keep `plan_code`
            """
        ).strip() + "\n"
    add_distractor_documents(visible_files, rng, int(difficulty_settings(difficulty)["distractor_count"]))
    return {
        "scenario_id": "migration_plan_bundle",
        "title": "Migration Plan Bundle",
        "description": "The workspace includes migration notes, compatibility guidance, and schema docs. Synthesize them into the required migration plan artifact.",
        "target_path": "artifacts/migration_plan.json",
        "task_type": "migration_planning",
        "output_style": "migration_plan",
        "entrypoint": None,
        "files": visible_files,
        "expected_output": expected_output,
        "reference_solution_files": {
            "artifacts/migration_plan.json": render_json(expected_output),
        },
        "evaluator_config": {
            "mode": "exact_json",
            "output_path": "artifacts/migration_plan.json",
            "expected_path": "expected_output.json",
        },
        "hints": [
            "The schema spec defines field renames and removals.",
            "Backfill rules may be split away from the cutover order.",
            "Older migration drafts can be wrong once the v3 schema is introduced.",
        ],
        "output_contract": [
            "Write the final plan to `artifacts/migration_plan.json`.",
            "Keep the JSON keys and list ordering stable.",
            "The plan must reflect the current v3 schema, not older migration drafts.",
        ],
        "structure": base_profile(
            files=visible_files,
            difficulty=difficulty,
            task_type="migration_planning",
            output_style="migration_plan",
            repair_surface="migration_spec_synthesis",
            failure_mode="grounding_and_semantic",
            smoke_test_quality="none",
        ),
        "document_roots": document_roots(visible_files),
    }


def build_incident_report_bundle_scenario(rng: random.Random, spec: EnvironmentSpec) -> dict[str, object]:
    difficulty = spec.difficulty
    expected_output = {
        "actions": ["increase export worker quota", "replay delayed jobs"],
        "customer_impact": "delayed_exports",
        "incident_id": "INC-204",
        "owner": "ops-reliability",
        "primary_cause": "quota_guard_trip",
        "resolution_utc": "2026-03-21T08:05:00Z",
        "service": "workspace-sync",
        "severity": "sev2",
        "start_utc": "2026-03-21T07:10:00Z",
    }
    visible_files: dict[str, str] = {
        "artifacts/oncall_report.json": render_json(
            {
                "actions": [],
                "customer_impact": "unknown",
                "incident_id": "INC-204",
                "owner": "platform",
                "primary_cause": "unknown",
                "resolution_utc": "",
                "service": "workspace-sync",
                "severity": "sev3",
                "start_utc": "",
            }
        ),
        "logs/incident_204.log": dedent(
            """
            2026-03-21T07:10:00Z incident=INC-204 service=workspace-sync event=start impact=delayed_exports priority=P2
            2026-03-21T07:16:00Z incident=INC-204 event=diagnosis marker=quota_guard_trip
            2026-03-21T07:52:00Z incident=INC-204 event=action detail=increase export worker quota
            2026-03-21T08:01:00Z incident=INC-204 event=action detail=replay delayed jobs
            2026-03-21T08:05:00Z incident=INC-204 event=resolved owner_alias=ops-reliability
            """
        ).strip() + "\n",
        "specs/report_contract.md": dedent(
            """
            # Report Contract

            The final artifact must be JSON with these keys:

            - `incident_id`
            - `service`
            - `severity`
            - `owner`
            - `customer_impact`
            - `start_utc`
            - `resolution_utc`
            - `primary_cause`
            - `actions`
            """
        ).strip() + "\n",
    }
    if difficulty >= 2:
        visible_files["notes/severity_policy.md"] = dedent(
            """
            # Severity Policy

            - `P1 -> sev1`
            - `P2 -> sev2`
            - `P3 -> sev3`
            """
        ).strip() + "\n"
    if difficulty >= 3:
        visible_files["docs/action_capture.md"] = dedent(
            """
            # Action Capture

            Every `event=action detail=...` line in the incident log must be preserved in order in the final report.
            The `owner_alias` on the resolved line is the report owner.
            """
        ).strip() + "\n"
    if difficulty == 4:
        visible_files["notes/postmortem_draft.md"] = dedent(
            """
            # Postmortem Draft

            Draft notes from another incident:

            - severity: sev3
            - owner: platform
            """
        ).strip() + "\n"
    if difficulty == 5:
        visible_files["changelog/2026-02-reporting.md"] = dedent(
            """
            # 2026-02 Reporting

            Superseded severity mapping:

            - `P2 -> sev3`
            """
        ).strip() + "\n"
    add_distractor_documents(visible_files, rng, int(difficulty_settings(difficulty)["distractor_count"]))
    structure = base_profile(
        files=visible_files,
        difficulty=difficulty,
        task_type="incident_reporting",
        output_style="structured_json",
        repair_surface="incident_report_grounding",
        failure_mode="grounding_and_semantic",
        smoke_test_quality="none",
    )
    if difficulty == 1:
        structure["evidence_distribution"] = "primary_plus_supporting"
        structure["retrieval_hops"] = 2
    return {
        "scenario_id": "incident_report_bundle",
        "title": "Incident Report Bundle",
        "description": "Use the local logs, contract notes, and operational guidance to build the required on-call incident report artifact.",
        "target_path": "artifacts/oncall_report.json",
        "task_type": "incident_reporting",
        "output_style": "structured_json",
        "entrypoint": None,
        "files": visible_files,
        "expected_output": expected_output,
        "reference_solution_files": {
            "artifacts/oncall_report.json": render_json(expected_output),
        },
        "evaluator_config": {
            "mode": "exact_json",
            "output_path": "artifacts/oncall_report.json",
            "expected_path": "expected_output.json",
        },
        "hints": [
            "The log contains the authoritative timestamps, root-cause marker, and ordered actions.",
            "Severity translation may live outside the report schema.",
            "Ignore stale drafts that describe a different severity or owner.",
        ],
        "output_contract": [
            "Write the final artifact to `artifacts/oncall_report.json`.",
            "Preserve ordered actions from the log.",
            "Use the visible document set to ground severity and ownership.",
        ],
        "structure": structure,
        "document_roots": document_roots(visible_files),
    }


def build_client_adapter_sync_scenario(rng: random.Random, spec: EnvironmentSpec) -> dict[str, object]:
    difficulty = spec.difficulty
    sample_response = {
        "next_cursor": "cursor-2",
        "records": [
            {"quantity": 4, "sku": "A-1", "warehouse": "east"},
            {"quantity": 2, "sku": "B-3"},
            {"quantity": 5, "sku": "C-7", "warehouse": "west"},
        ],
        "request_id": "req-204",
    }
    correct_adapter = dedent(
        """
        from __future__ import annotations


        def build_summary(response: dict[str, object]) -> dict[str, object]:
            records = list(response.get("records", []))
            total_quantity = 0
            warehouses: set[str] = set()
            for record in records:
                total_quantity += int(record["quantity"])
                warehouses.add(str(record.get("warehouse", "unknown")))
            return {
                "request_id": str(response["request_id"]),
                "next_cursor": response.get("next_cursor"),
                "record_count": len(records),
                "total_quantity": total_quantity,
                "warehouses": sorted(warehouses),
            }
        """
    ).strip() + "\n"
    buggy_adapter = dedent(
        """
        from __future__ import annotations


        def build_summary(response: dict[str, object]) -> dict[str, object]:
            items = list(response.get("items", []))
            total_quantity = 0
            warehouses: set[str] = set()
            for item in items:
                total_quantity += int(item["count"])
                warehouses.add(str(item["warehouse"]))
            return {
                "request_id": str(response["request_id"]),
                "next_cursor": response.get("cursor"),
                "record_count": len(items),
                "total_quantity": total_quantity,
                "warehouses": sorted(warehouses),
            }
        """
    ).strip() + "\n"
    visible_files: dict[str, str] = {
        "src/client_adapter.py": buggy_adapter,
        "samples/response.json": render_json(sample_response),
        "run_example.py": dedent(
            """
            from __future__ import annotations

            import json
            import sys
            from pathlib import Path

            workspace = Path(__file__).resolve().parent
            sys.path.insert(0, str(workspace / "src"))

            from client_adapter import build_summary


            def main() -> None:
                payload = json.loads((workspace / "samples" / "response.json").read_text(encoding="utf-8"))
                print(json.dumps(build_summary(payload), indent=2, sort_keys=True))


            if __name__ == "__main__":
                main()
            """
        ).strip() + "\n",
        "docs/api_reference.md": dedent(
            """
            # API Reference

            Latest response payload:

            - `request_id`: string
            - `records`: list of objects
            - each record includes `sku`, `quantity`, and optional `warehouse`
            - pagination field: `next_cursor`
            """
        ).strip() + "\n",
    }
    if difficulty >= 2:
        visible_files["docs/output_contract.md"] = dedent(
            """
            # Output Contract

            `build_summary()` must return:

            - `request_id`
            - `next_cursor`
            - `record_count`
            - `total_quantity`
            - `warehouses` sorted ascending
            """
        ).strip() + "\n"
    if difficulty >= 3:
        visible_files["notes/api_changelog.md"] = dedent(
            """
            # API Changelog

            Renames in the current release:

            - `items -> records`
            - `count -> quantity`
            - `cursor -> next_cursor`

            Missing `warehouse` values should default to `unknown`.
            """
        ).strip() + "\n"
    if difficulty == 4:
        visible_files["notes/legacy_adapter.md"] = dedent(
            """
            # Legacy Adapter

            Old adapter behavior:

            - read `items`
            - read `count`
            - fail if `warehouse` is missing
            """
        ).strip() + "\n"
    if difficulty == 5:
        visible_files["changelog/2026-01-client.md"] = dedent(
            """
            # 2026-01 Client Notes

            Superseded client response shape:

            - pagination field `cursor`
            - collection field `items`
            """
        ).strip() + "\n"
    add_distractor_documents(visible_files, rng, int(difficulty_settings(difficulty)["distractor_count"]))
    hidden_runner = dedent(
        """
        from __future__ import annotations

        import json
        import sys
        import unittest
        from pathlib import Path


        def build_suite(workspace: Path) -> unittest.TestSuite:
            sys.path.insert(0, str(workspace / "src"))
            from client_adapter import build_summary

            sample = json.loads((workspace / "samples" / "response.json").read_text(encoding="utf-8"))

            class HiddenTests(unittest.TestCase):
                def test_sample_payload(self) -> None:
                    self.assertEqual(
                        build_summary(sample),
                        {
                            "next_cursor": "cursor-2",
                            "record_count": 3,
                            "request_id": "req-204",
                            "total_quantity": 11,
                            "warehouses": ["east", "unknown", "west"],
                        },
                    )

                def test_missing_warehouse_defaults_to_unknown(self) -> None:
                    payload = {
                        "request_id": "req-9",
                        "next_cursor": None,
                        "records": [{"sku": "X", "quantity": 1}],
                    }
                    self.assertEqual(
                        build_summary(payload),
                        {
                            "next_cursor": None,
                            "record_count": 1,
                            "request_id": "req-9",
                            "total_quantity": 1,
                            "warehouses": ["unknown"],
                        },
                    )

            return unittest.defaultTestLoader.loadTestsFromTestCase(HiddenTests)


        def main() -> None:
            workspace = Path(sys.argv[1]).resolve()
            suite = build_suite(workspace)
            result = unittest.TextTestRunner(verbosity=2).run(suite)
            payload = {
                "success": result.wasSuccessful(),
                "score": 1.0 if result.wasSuccessful() else 0.0,
                "subscores": {
                    "tests_passed": result.testsRun - len(result.failures) - len(result.errors),
                    "tests_total": result.testsRun,
                },
                "failure_labels": ["hidden_tests_failed"] if not result.wasSuccessful() else [],
                "diagnostics": {
                    "tests_run": result.testsRun,
                    "failures": [case[0].id() for case in result.failures],
                    "errors": [case[0].id() for case in result.errors],
                },
            }
            print(json.dumps(payload, sort_keys=True))
            sys.exit(0 if result.wasSuccessful() else 1)


        if __name__ == "__main__":
            main()
        """
    ).strip() + "\n"
    return {
        "scenario_id": "client_adapter_sync",
        "title": "Client Adapter Sync",
        "description": "Repair the adapter implementation so it aligns with the local API documents and examples.",
        "target_path": "src/client_adapter.py",
        "task_type": "spec_to_code_alignment",
        "output_style": "code_patch",
        "entrypoint": "python run_example.py",
        "files": visible_files,
        "expected_output": None,
        "reference_solution_files": {
            "src/client_adapter.py": correct_adapter,
        },
        "evaluator_config": {
            "mode": "hidden_tests",
            "runner": "run_hidden_tests.py",
        },
        "hidden_text_assets": {
            "run_hidden_tests.py": hidden_runner,
        },
        "hints": [
            "The API reference names the canonical response fields.",
            "The changelog may describe field renames and defaulting behavior.",
            "Use the local sample payload to sanity-check the adapter after patching it.",
        ],
        "output_contract": [
            "Update `src/client_adapter.py` so it matches the visible API docs.",
            "Preserve the public `build_summary()` function signature.",
            "The hidden evaluator checks behavior, not just file text.",
        ],
        "structure": base_profile(
            files=visible_files,
            difficulty=difficulty,
            task_type="spec_to_code_alignment",
            output_style="code_patch",
            repair_surface="doc_assisted_code_repair",
            failure_mode="interface_and_semantic",
            smoke_test_quality="informative",
        ),
        "document_roots": document_roots(visible_files),
    }
