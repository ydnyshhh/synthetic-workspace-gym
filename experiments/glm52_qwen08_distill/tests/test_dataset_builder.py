from __future__ import annotations

import json
import subprocess
import sys
import unittest
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / "experiments" / "glm52_qwen08_distill"
FIXTURE = EXPERIMENT / "tests" / "fixtures" / "tiny_trace.json"
BUILD_DATASET = EXPERIMENT / "build_dataset.py"
EXPORT_PRIME_SFT = EXPERIMENT / "export_prime_sft.py"
TMP_ROOT = EXPERIMENT / "tests" / "tmp"


class DatasetBuilderTest(unittest.TestCase):
    def run_builder(
        self,
        output_dir: Path,
        report_dir: Path,
        *extra_args: str,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        command = [
            sys.executable,
            str(BUILD_DATASET),
            "--input-dir",
            str(FIXTURE),
            "--output-dir",
            str(output_dir),
            "--report-dir",
            str(report_dir),
            "--eval-id",
            "tiny-eval",
            "--allow-non-390",
            "--write-raw",
            "--write-sequential",
            *extra_args,
        ]
        return subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=check,
        )

    def test_quality_gate_reports_but_refuses_jsonl_by_default(self) -> None:
        root = make_case_root()
        output_dir = root / "processed"
        report_dir = root / "reports"
        result = self.run_builder(output_dir, report_dir, check=False)

        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("Quality gate failed", result.stdout)
        self.assertFalse((output_dir / "glm52_perfect_raw_actions.jsonl").exists())
        self.assertTrue((report_dir / "perfect_dataset_report.json").exists())

        report = json.loads((report_dir / "perfect_dataset_report.json").read_text(encoding="utf-8"))
        quality = report["data_quality_stats"]
        self.assertEqual(quality["invalid_run_python_calls"], 1)
        self.assertEqual(quality["absolute_path_attempts"], 1)
        self.assertEqual(quality["invalid_target_windows_excluded"], 2)
        self.assertEqual(
            report["absolute_path_examples"],
            [
                {
                    "argument_path": "command",
                    "example_id": 1,
                    "scenario": "tiny_scenario",
                    "target_index": 5,
                    "tool": "run_shell",
                    "trace_id": "tiny-trace-1",
                    "value": "cd /tmp && python check.py",
                }
            ],
        )

    def test_action_windows_and_exporter(self) -> None:
        root = make_case_root()
        output_dir = root / "processed"
        report_dir = root / "reports"
        self.run_builder(output_dir, report_dir, "--allow-quality-warnings")

        raw_path = output_dir / "glm52_perfect_raw_actions.jsonl"
        sequential_path = output_dir / "glm52_perfect_sequential_actions.jsonl"
        raw_examples = read_jsonl(raw_path)
        sequential_examples = read_jsonl(sequential_path)

        self.assertEqual(len(raw_examples), 2)
        self.assertEqual(len(sequential_examples), 3)
        self.assertTrue(all("reasoning_content" not in item["target"] for item in raw_examples))
        self.assertLessEqual(
            max(len(item["target"]["tool_calls"]) for item in sequential_examples),
            1,
        )
        self.assertTrue(
            all("id" not in call for item in sequential_examples for call in item["target"]["tool_calls"])
        )

        history_pairs = [
            (messages[index], messages[index + 1])
            for item in sequential_examples
            for messages in [item["messages"]]
            for index in range(len(messages) - 1)
            if messages[index].get("role") == "assistant"
            and messages[index].get("tool_calls")
            and messages[index + 1].get("role") == "tool"
        ]
        self.assertTrue(history_pairs)
        self.assertTrue(
            any(
                pair[0]["tool_calls"][0].get("id") == pair[1].get("tool_call_id")
                for pair in history_pairs
            )
        )

        sft_path = root / "prime_sft.jsonl"
        subprocess.run(
            [
                sys.executable,
                str(EXPORT_PRIME_SFT),
                "--input-jsonl",
                str(sequential_path),
                "--output-jsonl",
                str(sft_path),
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
        sft_examples = read_jsonl(sft_path)
        self.assertEqual(len(sft_examples), len(sequential_examples))
        self.assertEqual(sft_examples[0]["messages"][-1], sequential_examples[0]["target"])


def make_case_root() -> Path:
    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    root = TMP_ROOT / uuid.uuid4().hex
    root.mkdir(parents=True, exist_ok=False)
    return root


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


if __name__ == "__main__":
    unittest.main()
