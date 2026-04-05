from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from synthetic_workspace_gym.evaluators.base import BaseEvaluator
from synthetic_workspace_gym.schemas import EnvironmentManifest, EvaluatorResult
from synthetic_workspace_gym.utils.io import read_json


class ScriptRepairEvaluator(BaseEvaluator):
    def evaluate(self, workspace_path: Path, manifest: EnvironmentManifest, hidden_root: Path) -> EvaluatorResult:
        started = time.perf_counter()
        config = read_json(hidden_root / "evaluator_config.json")
        runner_path = (hidden_root / config["runner"]).resolve()
        try:
            completed = subprocess.run(
                [sys.executable, str(runner_path), str(workspace_path)],
                cwd=str(hidden_root.resolve()),
                capture_output=True,
                text=True,
                timeout=manifest.time_limit_seconds,
                env={**dict(os.environ), "PYTHONDONTWRITEBYTECODE": "1"},
            )
        except subprocess.TimeoutExpired as exc:
            return EvaluatorResult(
                success=False,
                score=0.0,
                subscores={"tests_passed": 0.0, "tests_total": 0.0, "tests_passed_ratio": 0.0},
                failure_labels=["timeout"],
                diagnostics={
                    "stdout": exc.stdout or "",
                    "stderr": exc.stderr or "",
                    "runner": str(runner_path),
                },
                runtime_seconds=time.perf_counter() - started,
            )
        payload = self.extract_payload(completed.stdout)
        if payload is None:
            return EvaluatorResult(
                success=False,
                score=0.0,
                subscores={"tests_passed": 0.0, "tests_total": 0.0, "tests_passed_ratio": 0.0},
                failure_labels=["hidden_tests_failed"],
                diagnostics={
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                    "returncode": completed.returncode,
                },
                runtime_seconds=time.perf_counter() - started,
            )

        tests_passed = float(payload["subscores"].get("tests_passed", 0))
        tests_total = float(payload["subscores"].get("tests_total", 0))
        tests_passed_ratio = (tests_passed / tests_total) if tests_total else 0.0
        return EvaluatorResult(
            success=bool(payload["success"]),
            score=1.0 if bool(payload["success"]) else round(tests_passed_ratio, 6),
            subscores={
                "tests_passed": tests_passed,
                "tests_total": tests_total,
                "tests_passed_ratio": round(tests_passed_ratio, 6),
            },
            failure_labels=list(payload.get("failure_labels", [])),
            diagnostics={
                **payload.get("diagnostics", {}),
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "returncode": completed.returncode,
            },
            runtime_seconds=time.perf_counter() - started,
        )

    def extract_payload(self, stdout: str) -> dict[str, object] | None:
        for line in reversed([item.strip() for item in stdout.splitlines() if item.strip()]):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
        return None
