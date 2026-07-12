from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


def build_suite(workspace: Path) -> unittest.TestSuite:
    hidden_root = Path(__file__).resolve().parent
    expected = json.loads((hidden_root / "expected_region_report.json").read_text(encoding="utf-8"))
    sys.path.insert(0, str(workspace / "src"))
    from repair_target.parser import load_orders
    from repair_target.report import build_region_report

    class HiddenTests(unittest.TestCase):
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
