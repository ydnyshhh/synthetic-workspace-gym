from __future__ import annotations

import json
from textwrap import dedent


def build_csv_schema_drift_scenario(generator) -> dict[str, object]:
    orders_csv = (
        "account_id,region,status,amount\n"
        "A-100,North,complete,120.00\n"
        "A-101,West,complete,95.50\n"
        "A-102,East,cancelled,20.00\n"
        "A-103,North,complete,90.00\n"
        "A-104,East,complete,85.00\n"
    )
    expected_report = [
        {"region": "east", "row_count": 1, "total_amount": 85.0},
        {"region": "north", "row_count": 2, "total_amount": 210.0},
        {"region": "west", "row_count": 1, "total_amount": 95.5},
    ]
    parser = dedent(
        """\
        from __future__ import annotations

        import csv
        from pathlib import Path


        def load_orders(path: Path) -> list[dict[str, object]]:
            with Path(path).open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                rows = []
                for row in reader:
                    if not row.get("account_id"):
                        continue
                    rows.append(
                        {
                            "account_id": row["account_id"],
                            "region": str(row["region"]).strip().lower(),
                            "status": str(row["status"]).strip().lower(),
                            "amount": round(float(row["amount"]), 2),
                        }
                    )
            return rows
        """
    )
    report = dedent(
        """\
        from __future__ import annotations


        def build_region_report(rows: list[dict[str, object]]) -> list[dict[str, object]]:
            summary: dict[str, dict[str, object]] = {}
            for row in rows:
                if row["status"] == "cancelled":
                    continue
                region = str(row["region"])
                if region not in summary:
                    summary[region] = {"region": region, "row_count": 0, "total_amount": 0.0}
                summary[region]["row_count"] = int(summary[region]["row_count"]) + 1
                summary[region]["total_amount"] = round(
                    float(summary[region]["total_amount"]) + float(row["amount"]),
                    2,
                )
            return sorted(summary.values(), key=lambda item: str(item["region"]))
        """
    )
    return {
        "scenario_id": "csv_schema_drift",
        "title": "CSV Schema Drift Repair",
        "debug_note": "The smoke test may still run after a partial fix, but the hidden tests check that rows are not silently dropped and the final ordering contract is preserved.\n",
        "hints": [
            "The CSV headers and the parser must agree on the account identifier column.",
            "Cancelled rows should be excluded from the final report, not dropped before parsing.",
            "The final report is sorted by region, not by aggregate size.",
        ],
        "structure": {
            "repair_surface": "parser_interface",
            "bug_scope": "cross_file",
            "failure_mode": "semantic_and_formatting",
            "smoke_test_quality": "partially_informative",
        },
        "files": {
            "src/repair_target/__init__.py": "",
            "src/repair_target/parser.py": parser,
            "src/repair_target/report.py": report,
            "run_example.py": generator._json_runner(
                import_block="""
                from repair_target.parser import load_orders
                from repair_target.report import build_region_report
                """,
                expression='build_region_report(load_orders(workspace / "data" / "orders.csv"))',
            ),
            "data/orders.csv": orders_csv,
        },
        "hidden_json_assets": {
            "expected_region_report.json": expected_report,
        },
        "bugs": [
            {
                "label": "schema_mapping_bug",
                "target_path": "src/repair_target/parser.py",
                "apply": generator._replace_once(
                    '"account_id": row["account_id"],',
                    '"account_id": row["customer_id"],',
                    label="schema_mapping_bug",
                    target_path="src/repair_target/parser.py",
                ),
            },
            {
                "label": "missing_rows_bug",
                "target_path": "src/repair_target/parser.py",
                "apply": generator._replace_once(
                    'if not row.get("account_id"):',
                    'if not row.get("customer_id"):',
                    label="missing_rows_bug",
                    target_path="src/repair_target/parser.py",
                ),
            },
            {
                "label": "wrong_sort_key",
                "target_path": "src/repair_target/report.py",
                "apply": generator._replace_once(
                    'return sorted(summary.values(), key=lambda item: str(item["region"]))',
                    'return sorted(summary.values(), key=lambda item: str(item["row_count"]))',
                    label="wrong_sort_key",
                    target_path="src/repair_target/report.py",
                ),
            },
        ],
        "test_runner": generator._hidden_runner(
            asset_setup='expected = json.loads((hidden_root / "expected_region_report.json").read_text(encoding="utf-8"))',
            import_block="""
            from repair_target.parser import load_orders
            from repair_target.report import build_region_report
            """,
            test_methods="""
            def test_rows_are_loaded(self) -> None:
                rows = load_orders(workspace / "data" / "orders.csv")
                self.assertEqual(len(rows), 5)
                self.assertEqual(rows[0]["account_id"], "A-100")

            def test_report_contract(self) -> None:
                rows = load_orders(workspace / "data" / "orders.csv")
                self.assertEqual(build_region_report(rows), expected)

            def test_sorted_regions(self) -> None:
                rows = load_orders(workspace / "data" / "orders.csv")
                actual = build_region_report(rows)
                self.assertEqual([row["region"] for row in actual], ["east", "north", "west"])
            """,
        ),
    }


def build_timestamp_normalization_scenario(generator) -> dict[str, object]:
    events = [
        {"amount": 2.0, "event_id": "evt-1", "timestamp": "2024-02-01 09:30"},
        {"amount": 1.5, "event_id": "evt-2", "timestamp": "01/31/2024 23:00"},
        {"amount": 3.0, "event_id": "evt-3", "timestamp": "2024/02/01 08:15"},
        {"amount": 4.0, "event_id": "evt-4", "timestamp": "02-Feb-2024 10:00"},
    ]
    expected_report = {
        "daily_totals": [
            {"day": "2024-01-31", "total_amount": 1.5},
            {"day": "2024-02-01", "total_amount": 5.0},
            {"day": "2024-02-02", "total_amount": 4.0},
        ],
        "ordered_ids": ["evt-2", "evt-3", "evt-1", "evt-4"],
    }
    time_utils = dedent(
        """\
        from __future__ import annotations

        from datetime import datetime

        _FORMATS = ("%Y-%m-%d %H:%M", "%m/%d/%Y %H:%M", "%Y/%m/%d %H:%M", "%d-%b-%Y %H:%M")


        def parse_timestamp(value: str) -> datetime:
            for fmt in _FORMATS:
                try:
                    return datetime.strptime(value, fmt)
                except ValueError:
                    continue
            raise ValueError(f"Unsupported timestamp format: {value}")


        def sort_events(rows: list[dict[str, object]]) -> list[dict[str, object]]:
            return sorted(rows, key=lambda row: parse_timestamp(str(row["timestamp"])))
        """
    )
    report = dedent(
        """\
        from __future__ import annotations

        from repair_target.time_utils import parse_timestamp, sort_events


        def build_report(rows: list[dict[str, object]]) -> dict[str, object]:
            ordered = sort_events(rows)
            daily_totals: dict[str, float] = {}
            ordered_ids: list[str] = []
            for row in ordered:
                stamp = parse_timestamp(str(row["timestamp"]))
                ordered_ids.append(str(row["event_id"]))
                day = stamp.strftime("%Y-%m-%d")
                daily_totals[day] = round(daily_totals.get(day, 0.0) + float(row["amount"]), 1)
            return {
                "ordered_ids": ordered_ids,
                "daily_totals": [
                    {"day": day, "total_amount": daily_totals[day]}
                    for day in sorted(daily_totals)
                ],
            }
        """
    )
    return {
        "scenario_id": "timestamp_normalization",
        "title": "Timestamp Normalization Repair",
        "debug_note": "The visible smoke test can look plausible even when timestamps are sorted lexicographically instead of chronologically.\n",
        "hints": [
            "This workspace mixes several timestamp formats.",
            "Chronological ordering should use parsed datetimes, not raw strings.",
            "The final daily summary uses ISO day strings.",
        ],
        "structure": {
            "repair_surface": "normalization_and_sorting",
            "bug_scope": "cross_file",
            "failure_mode": "semantic",
            "smoke_test_quality": "misleading",
        },
        "files": {
            "src/repair_target/__init__.py": "",
            "src/repair_target/time_utils.py": time_utils,
            "src/repair_target/report.py": report,
            "run_example.py": generator._json_runner(
                import_block="from repair_target.report import build_report",
                expression='build_report(json.loads((workspace / "data" / "events.json").read_text(encoding="utf-8")))',
            ),
            "data/events.json": json.dumps(events, indent=2, sort_keys=True) + "\n",
        },
        "hidden_json_assets": {
            "expected_timeline_report.json": expected_report,
        },
        "bugs": [
            {
                "label": "missing_format",
                "target_path": "src/repair_target/time_utils.py",
                "apply": generator._replace_once(
                    '"%Y/%m/%d %H:%M"',
                    '"%Y-%d-%m %H:%M"',
                    label="missing_format",
                    target_path="src/repair_target/time_utils.py",
                ),
            },
            {
                "label": "wrong_sort_key",
                "target_path": "src/repair_target/time_utils.py",
                "apply": generator._replace_once(
                    'return sorted(rows, key=lambda row: parse_timestamp(str(row["timestamp"])))',
                    'return sorted(rows, key=lambda row: str(row["timestamp"]))',
                    label="wrong_sort_key",
                    target_path="src/repair_target/time_utils.py",
                ),
            },
            {
                "label": "wrong_output_schema",
                "target_path": "src/repair_target/report.py",
                "apply": generator._replace_once(
                    'day = stamp.strftime("%Y-%m-%d")',
                    'day = stamp.strftime("%m/%d/%Y")',
                    label="wrong_output_schema",
                    target_path="src/repair_target/report.py",
                ),
            },
        ],
        "test_runner": generator._hidden_runner(
            asset_setup='expected = json.loads((hidden_root / "expected_timeline_report.json").read_text(encoding="utf-8"))',
            import_block="""
            from repair_target.report import build_report
            from repair_target.time_utils import parse_timestamp, sort_events
            """,
            test_methods="""
            def test_timestamp_formats(self) -> None:
                self.assertEqual(parse_timestamp("2024/02/01 08:15").strftime("%Y-%m-%d %H:%M"), "2024-02-01 08:15")

            def test_chronological_order(self) -> None:
                rows = json.loads((workspace / "data" / "events.json").read_text(encoding="utf-8"))
                ordered = sort_events(rows)
                self.assertEqual([row["event_id"] for row in ordered], expected["ordered_ids"])

            def test_report(self) -> None:
                rows = json.loads((workspace / "data" / "events.json").read_text(encoding="utf-8"))
                self.assertEqual(build_report(rows), expected)
            """,
        ),
    }


def build_team_roster_export_scenario(generator) -> dict[str, object]:
    players = [
        {"active": True, "name": "Ada", "score": 8, "team": "alpha"},
        {"active": True, "name": "Ben", "score": 5, "team": "beta"},
        {"active": False, "name": "Cai", "score": 7, "team": "alpha"},
        {"active": True, "name": "Dee", "score": 4, "team": "alpha"},
        {"active": True, "name": "Eli", "score": 6, "team": "beta"},
    ]
    expected_rows = [
        {"member_count": 2, "team": "alpha", "total_score": 12},
        {"member_count": 2, "team": "beta", "total_score": 11},
    ]
    contracts = dedent(
        """\
        from __future__ import annotations


        def build_rows(players: list[dict[str, object]]) -> list[dict[str, object]]:
            summary: dict[str, dict[str, object]] = {}
            for row in players:
                if not bool(row["active"]):
                    continue
                team = str(row["team"]).lower()
                if team not in summary:
                    summary[team] = {"team": team, "member_count": 0, "total_score": 0}
                summary[team]["member_count"] = int(summary[team]["member_count"]) + 1
                summary[team]["total_score"] = int(summary[team]["total_score"]) + int(row["score"])
            return sorted(summary.values(), key=lambda item: str(item["team"]))
        """
    )
    writer = dedent(
        """\
        from __future__ import annotations

        import json
        from pathlib import Path


        def write_report(path: Path, payload: object) -> None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
        """
    )
    runner = dedent(
        """\
        from __future__ import annotations

        import json
        import sys
        from pathlib import Path

        workspace = Path(__file__).resolve().parent
        sys.path.insert(0, str(workspace / "src"))

        from repair_target.contracts import build_rows
        from repair_target.writer import write_report


        def main() -> None:
            players = json.loads((workspace / "data" / "players.json").read_text(encoding="utf-8"))
            output_path = workspace / "artifacts" / "report.json"
            write_report(output_path, build_rows(players))
            print(output_path.read_text(encoding="utf-8"))


        if __name__ == "__main__":
            main()
        """
    )
    return {
        "scenario_id": "team_roster_export",
        "title": "Team Roster Export Repair",
        "debug_note": "The visible smoke test only proves that something was written to disk. Hidden tests still care about valid JSON and the contract between the transformer and writer.\n",
        "hints": [
            "The exported artifact should be valid JSON, not a Python repr.",
            "Downstream code expects the team field to keep its original public name.",
            "The final rows are sorted by team.",
        ],
        "structure": {
            "repair_surface": "serialization_and_interface",
            "bug_scope": "cross_file",
            "failure_mode": "formatting_and_contract",
            "smoke_test_quality": "weak",
        },
        "files": {
            "src/repair_target/__init__.py": "",
            "src/repair_target/contracts.py": contracts,
            "src/repair_target/writer.py": writer,
            "run_example.py": runner,
            "data/players.json": json.dumps(players, indent=2, sort_keys=True) + "\n",
        },
        "hidden_json_assets": {
            "expected_roster_report.json": expected_rows,
        },
        "bugs": [
            {
                "label": "cross_file_contract_bug",
                "target_path": "src/repair_target/contracts.py",
                "apply": generator._replace_once(
                    '{"team": team, "member_count": 0, "total_score": 0}',
                    '{"team_name": team, "member_count": 0, "total_score": 0}',
                    label="cross_file_contract_bug",
                    target_path="src/repair_target/contracts.py",
                ),
            },
            {
                "label": "wrong_sort_key",
                "target_path": "src/repair_target/contracts.py",
                "apply": generator._replace_once(
                    'return sorted(summary.values(), key=lambda item: str(item["team"]))',
                    'return sorted(summary.values(), key=lambda item: int(item["member_count"]))',
                    label="wrong_sort_key",
                    target_path="src/repair_target/contracts.py",
                ),
            },
            {
                "label": "serialization_bug",
                "target_path": "src/repair_target/writer.py",
                "apply": generator._replace_once(
                    'Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n", encoding="utf-8")',
                    'Path(path).write_text(str(payload), encoding="utf-8")',
                    label="serialization_bug",
                    target_path="src/repair_target/writer.py",
                ),
            },
        ],
        "test_runner": generator._hidden_runner(
            asset_setup='expected = json.loads((hidden_root / "expected_roster_report.json").read_text(encoding="utf-8"))',
            import_block="""
            from repair_target.contracts import build_rows
            from repair_target.writer import write_report
            """,
            test_methods="""
            def test_contract_shape(self) -> None:
                players = json.loads((workspace / "data" / "players.json").read_text(encoding="utf-8"))
                rows = build_rows(players)
                self.assertEqual(rows, expected)
                self.assertEqual(sorted(rows[0].keys()), ["member_count", "team", "total_score"])

            def test_artifact_is_json(self) -> None:
                players = json.loads((workspace / "data" / "players.json").read_text(encoding="utf-8"))
                output_path = workspace / "artifacts" / "report.json"
                write_report(output_path, build_rows(players))
                actual = json.loads(output_path.read_text(encoding="utf-8"))
                self.assertEqual(actual, expected)
            """,
        ),
    }
