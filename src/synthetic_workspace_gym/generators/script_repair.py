from __future__ import annotations

import json
import random
from pathlib import Path

from synthetic_workspace_gym.generators.base import BaseGenerator, GeneratedPayload
from synthetic_workspace_gym.schemas import EnvironmentFamily, EnvironmentSpec
from synthetic_workspace_gym.utils.io import write_json, write_text


class ScriptRepairGenerator(BaseGenerator):
    family = EnvironmentFamily.SCRIPT_REPAIR

    def _replace_once(self, old: str, new: str, *, label: str, target_path: str):
        def apply(content: str) -> str:
            updated = content.replace(old, new, 1)
            if updated == content:
                raise ValueError(
                    f"Bug application '{label}' did not modify {target_path!r}; canonical source drifted."
                )
            return updated

        return apply

    def _build_environment(self, spec: EnvironmentSpec, *, root: Path, visible_root: Path, hidden_root: Path) -> GeneratedPayload:
        rng = random.Random(spec.seed)
        scenario = rng.choice([self._inventory_report_scenario(), self._path_batch_scenario()])
        bug_budget = {1: 1, 2: 1, 3: 2, 4: 3, 5: 3}[spec.difficulty]
        candidates = list(scenario["bugs"])
        if spec.difficulty < 5:
            candidates = [bug for bug in candidates if bug["label"] != "syntax_error"]
        selected_bugs = candidates[:bug_budget]
        if len(candidates) > bug_budget:
            selected_bugs = rng.sample(candidates, k=bug_budget)

        correct_files = dict(scenario["files"])
        buggy_files = dict(correct_files)
        applied_bug_labels: list[str] = []
        touched_files: set[str] = set()
        for bug in selected_bugs:
            target_path = bug["target_path"]
            buggy_files[target_path] = bug["apply"](buggy_files[target_path])
            touched_files.add(target_path)
            applied_bug_labels.append(bug["label"])

        for relative_path, content in buggy_files.items():
            write_text(visible_root / relative_path, content)

        if spec.difficulty >= 4:
            write_text(
                visible_root / "notes" / "incident_log.md",
                "Recent debugging note: the visible smoke test and the hidden evaluator may exercise different code paths.\n",
            )

        task_descriptor = {
            "family": "script_repair",
            "scenario_id": scenario["scenario_id"],
            "entrypoint": "python run_example.py",
            "target_files": sorted(touched_files),
            "hints": scenario["hints"],
        }
        write_text(visible_root / "README.md", self._build_readme(scenario, task_descriptor))
        write_json(visible_root / "task.json", task_descriptor)

        hidden_runner = scenario["test_runner"](task_descriptor)
        write_text(hidden_root / "run_hidden_tests.py", hidden_runner)
        for relative_path, payload in scenario.get("hidden_json_assets", {}).items():
            write_json(hidden_root / relative_path, payload)
        reference_solution = {
            "files": {path: correct_files[path] for path in sorted(touched_files)},
            "scenario_id": scenario["scenario_id"],
            "bug_labels": applied_bug_labels,
        }
        write_json(hidden_root / "solution_files.json", reference_solution)
        write_json(
            hidden_root / "evaluator_config.json",
            {
                "runner": "run_hidden_tests.py",
                "scenario_id": scenario["scenario_id"],
            },
        )

        metadata = {
            "task_descriptor": task_descriptor,
            "complexity_profile": spec.complexity_profile.to_dict() if spec.complexity_profile else {},
            "bug_labels": applied_bug_labels,
            "scenario_id": scenario["scenario_id"],
        }
        return GeneratedPayload(
            instruction="Repair the provided Python workspace so that the hidden tests pass.",
            metadata=metadata,
            reference_solution=reference_solution,
            evaluator_entrypoint="synthetic_workspace_gym.evaluators.script_repair:ScriptRepairEvaluator",
        )

    def _build_readme(self, scenario: dict[str, object], task_descriptor: dict[str, object]) -> str:
        hints = "\n".join(f"- {hint}" for hint in task_descriptor["hints"])
        targets = "\n".join(f"- `{item}`" for item in task_descriptor["target_files"])
        return (
            f"# {scenario['title']}\n\n"
            "One or more Python files in this workspace are buggy. Repair the code so the hidden tests pass.\n\n"
            "## What to preserve\n"
            "- Keep the public function names stable.\n"
            "- Prefer targeted fixes over rewrites.\n"
            "- Use the visible smoke test command to sanity-check your changes.\n\n"
            "## Smoke test\n"
            f"- `{task_descriptor['entrypoint']}`\n\n"
            "## Likely target files\n"
            f"{targets}\n\n"
            "## Hints\n"
            f"{hints}\n"
        )

    def _inventory_report_scenario(self) -> dict[str, object]:
        items = [
            {"name": "alpha", "status": "active", "count": 3},
            {"name": "beta", "status": "archived", "count": 2},
            {"name": "gamma", "status": "active", "count": 4},
            {"name": "delta", "status": "active", "count": 6},
            {"name": "epsilon", "status": "archived", "count": 3},
        ]
        expected_report = {
            "summary": {"active": 13, "archived": 5},
            "rolling_average": [3.5, 5.0],
        }
        analytics = """from __future__ import annotations


def rolling_average(values: list[int], window: int) -> list[float]:
    if window <= 0:
        raise ValueError("window must be positive")
    if len(values) < window:
        return []
    averages = []
    for index in range(len(values) - window + 1):
        chunk = values[index : index + window]
        averages.append(round(sum(chunk) / window, 2))
    return averages


def summarize_items(rows: list[dict[str, object]]) -> dict[str, int]:
    summary = {"active": 0, "archived": 0}
    for row in rows:
        status = str(row["status"]).lower()
        if status not in summary:
            continue
        summary[status] += int(row["count"])
    return summary
"""
        report = """from __future__ import annotations

import json
from pathlib import Path

from repair_target.analytics import rolling_average, summarize_items


def build_report(data_path: Path) -> dict[str, object]:
    rows = json.loads(Path(data_path).read_text(encoding="utf-8"))
    active_counts = [int(row["count"]) for row in rows if str(row["status"]).lower() == "active"]
    return {
        "summary": summarize_items(rows),
        "rolling_average": rolling_average(active_counts, 2),
    }
"""
        runner = """from __future__ import annotations

import json
import sys
from pathlib import Path

workspace = Path(__file__).resolve().parent
sys.path.insert(0, str(workspace / "src"))

from repair_target.report import build_report


def main() -> None:
    data_path = workspace / "data" / "items.json"
    print(json.dumps(build_report(data_path), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
"""
        test_runner = """from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


def build_suite(workspace: Path) -> unittest.TestSuite:
    hidden_root = Path(__file__).resolve().parent
    expected = json.loads((hidden_root / "expected_inventory_report.json").read_text(encoding="utf-8"))
    sys.path.insert(0, str(workspace / "src"))
    from repair_target.analytics import rolling_average, summarize_items
    from repair_target.report import build_report

    class HiddenTests(unittest.TestCase):
        def test_rolling_average(self) -> None:
            self.assertEqual(rolling_average([3, 4, 6], 2), [3.5, 5.0])

        def test_summary(self) -> None:
            rows = json.loads((workspace / "data" / "items.json").read_text(encoding="utf-8"))
            self.assertEqual(summarize_items(rows), expected["summary"])

        def test_report(self) -> None:
            actual = build_report(workspace / "data" / "items.json")
            self.assertEqual(actual, expected)

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
        return {
            "scenario_id": "inventory_report",
            "title": "Inventory Report Repair",
            "hints": [
                "The rolling average should include every complete 2-item window.",
                "Archived rows still contribute to the summary counts.",
                "The smoke test should print a stable JSON report.",
            ],
            "files": {
                "src/repair_target/__init__.py": "",
                "src/repair_target/analytics.py": analytics,
                "src/repair_target/report.py": report,
                "run_example.py": runner,
                "data/items.json": json.dumps(items, indent=2, sort_keys=True) + "\n",
            },
            "hidden_json_assets": {
                "expected_inventory_report.json": expected_report,
            },
            "bugs": [
                {
                    "label": "off_by_one",
                    "target_path": "src/repair_target/analytics.py",
                    "apply": self._replace_once(
                        "range(len(values) - window + 1)",
                        "range(len(values) - window)",
                        label="off_by_one",
                        target_path="src/repair_target/analytics.py",
                    ),
                },
                {
                    "label": "wrong_condition",
                    "target_path": "src/repair_target/analytics.py",
                    "apply": self._replace_once(
                        "if status not in summary:",
                        'if status == "archived":',
                        label="wrong_condition",
                        target_path="src/repair_target/analytics.py",
                    ),
                },
                {
                    "label": "wrong_return_value",
                    "target_path": "src/repair_target/report.py",
                    "apply": self._replace_once(
                        '"rolling_average": rolling_average(active_counts, 2),',
                        '"rolling_average": rolling_average(active_counts, 3),',
                        label="wrong_return_value",
                        target_path="src/repair_target/report.py",
                    ),
                },
            ],
            "test_runner": lambda _task: test_runner,
        }

    def _path_batch_scenario(self) -> dict[str, object]:
        measurements = "day,value\nmon,5\ntue,8\nwed,3\nthu,9\n"
        expected_summary = {"count": 4, "maximum": 9, "minimum": 3, "total": 25}
        io_helpers = """from __future__ import annotations

from pathlib import Path


def load_measurements(data_dir: Path) -> list[int]:
    path = data_dir / "measurements.csv"
    rows = Path(path).read_text(encoding="utf-8").strip().splitlines()[1:]
    return [int(row.split(",")[1]) for row in rows]
"""
        batch = """from __future__ import annotations

from pathlib import Path

from repair_target.io_helpers import load_measurements


def compute_batch_summary(base_dir: Path) -> dict[str, int]:
    data_dir = Path(base_dir) / "data"
    values = load_measurements(data_dir)
    return {
        "count": len(values),
        "minimum": min(values),
        "maximum": max(values),
        "total": sum(values),
    }
"""
        runner = """from __future__ import annotations

import json
import sys
from pathlib import Path

workspace = Path(__file__).resolve().parent
sys.path.insert(0, str(workspace / "src"))

from repair_target.batch import compute_batch_summary


def main() -> None:
    print(json.dumps(compute_batch_summary(workspace), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
"""
        test_runner = """from __future__ import annotations

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
        return {
            "scenario_id": "path_batch",
            "title": "Batch Summary Repair",
            "hints": [
                "The measurement loader should read from the visible `data` directory.",
                "The batch summary should aggregate every row in the CSV.",
                "If the module fails to import, inspect recent edits around function signatures and imports.",
            ],
            "files": {
                "src/repair_target/__init__.py": "",
                "src/repair_target/io_helpers.py": io_helpers,
                "src/repair_target/batch.py": batch,
                "run_example.py": runner,
                "data/measurements.csv": measurements,
            },
            "hidden_json_assets": {
                "expected_batch_summary.json": expected_summary,
            },
            "bugs": [
                {
                    "label": "missing_import",
                    "target_path": "src/repair_target/io_helpers.py",
                    "apply": self._replace_once(
                        "from pathlib import Path\n\n",
                        "",
                        label="missing_import",
                        target_path="src/repair_target/io_helpers.py",
                    ),
                },
                {
                    "label": "file_path_issue",
                    "target_path": "src/repair_target/io_helpers.py",
                    "apply": self._replace_once(
                        'data_dir / "measurements.csv"',
                        'data_dir.parent / "measurements.csv"',
                        label="file_path_issue",
                        target_path="src/repair_target/io_helpers.py",
                    ),
                },
                {
                    "label": "aggregation_bug",
                    "target_path": "src/repair_target/batch.py",
                    "apply": self._replace_once(
                        '"total": sum(values),',
                        '"total": len(values),',
                        label="aggregation_bug",
                        target_path="src/repair_target/batch.py",
                    ),
                },
                {
                    "label": "syntax_error",
                    "target_path": "src/repair_target/batch.py",
                    "apply": self._replace_once(
                        "def compute_batch_summary(base_dir: Path) -> dict[str, int]:",
                        "def compute_batch_summary(base_dir: Path) -> dict[str, int]",
                        label="syntax_error",
                        target_path="src/repair_target/batch.py",
                    ),
                },
            ],
            "test_runner": lambda _task: test_runner,
        }
