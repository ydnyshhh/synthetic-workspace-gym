from __future__ import annotations

import argparse
import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from test_support import workspace_tempdir

from synthetic_workspace_gym.cli import command_benchmark, command_generate


class CliTests(unittest.TestCase):
    def test_benchmark_command_writes_summary(self) -> None:
        with workspace_tempdir() as tmp_dir:
            root = Path(tmp_dir)
            generated_root = root / "generated"
            benchmark_root = root / "benchmarks"
            with redirect_stdout(io.StringIO()):
                command_generate(
                    argparse.Namespace(
                        family="tabular",
                        count=2,
                        difficulty="2",
                        seed=70,
                        output_dir=generated_root,
                        skip_validate=False,
                    )
                )
                exit_code = command_benchmark(
                    argparse.Namespace(
                        environments=generated_root,
                        agent="heuristic",
                        output_dir=benchmark_root,
                    )
                )
            self.assertEqual(exit_code, 0)
            result_files = list(benchmark_root.glob("benchmark-heuristic-*.json"))
            self.assertEqual(len(result_files), 1)
            payload = json.loads(result_files[0].read_text(encoding="utf-8"))
            self.assertEqual(payload["agent"], "heuristic")
            self.assertEqual(payload["environment_count"], 2)


if __name__ == "__main__":
    unittest.main()
