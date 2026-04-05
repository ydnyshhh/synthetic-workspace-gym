from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


def build_suite(workspace: Path) -> unittest.TestSuite:
    hidden_root = Path(__file__).resolve().parent
    expected = json.loads((hidden_root / "expected_batch_summary.json").read_text(encoding="utf-8"))
    sys.path.insert(0, str(workspace / "src"))
    from repair_target.batch import compute_batch_summary
    from repair_target.io_helpers import load_measurements

    class HiddenTests(unittest.TestCase):
        def test_loader(self) -> None:
            values = load_measurements(workspace / "data")
            self.assertEqual(values, [5, 8, 3, 9])

        def test_summary(self) -> None:
            self.assertEqual(compute_batch_summary(workspace), expected)

    return unittest.defaultTestLoader.loadTestsFromTestCase(HiddenTests)


def main() -> None:
    workspace = Path(sys.argv[1]).resolve()
    suite = build_suite(workspace)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    payload = {{
        "success": result.wasSuccessful(),
        "score": 1.0 if result.wasSuccessful() else 0.0,
        "subscores": {{
            "tests_passed": result.testsRun - len(result.failures) - len(result.errors),
            "tests_total": result.testsRun,
        }},
        "failure_labels": ["hidden_tests_failed"] if not result.wasSuccessful() else [],
        "diagnostics": {{
            "tests_run": result.testsRun,
            "failures": [case[0].id() for case in result.failures],
            "errors": [case[0].id() for case in result.errors],
        }},
    }}
    print(json.dumps(payload, sort_keys=True))
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
