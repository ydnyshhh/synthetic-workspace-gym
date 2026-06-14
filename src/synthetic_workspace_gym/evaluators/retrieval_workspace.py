from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from synthetic_workspace_gym.evaluators.base import BaseEvaluator
from synthetic_workspace_gym.evaluators.metrics import flatten_json, json_field_diff_diagnostics, row_overlap_metrics, weighted_match_score
from synthetic_workspace_gym.schemas import EnvironmentManifest, EvaluatorResult
from synthetic_workspace_gym.utils.io import read_json


class RetrievalWorkspaceEvaluator(BaseEvaluator):
    def evaluate(self, workspace_path: Path, manifest: EnvironmentManifest, hidden_root: Path) -> EvaluatorResult:
        config = read_json(hidden_root / "evaluator_config.json")
        mode = str(config.get("mode", "exact_json"))
        if mode == "exact_json":
            return self.evaluate_exact_json(workspace_path, manifest, hidden_root, config)
        if mode == "hidden_tests":
            return self.evaluate_hidden_tests(workspace_path, manifest, hidden_root, config)
        raise ValueError(f"Unsupported retrieval workspace evaluation mode: {mode}")

    def evaluate_exact_json(
        self,
        workspace_path: Path,
        manifest: EnvironmentManifest,
        hidden_root: Path,
        config: dict[str, object],
    ) -> EvaluatorResult:
        started = time.perf_counter()
        output_path = workspace_path / str(config["output_path"])
        if not output_path.exists():
            return EvaluatorResult(
                success=False,
                score=0.0,
                subscores={
                    "output_exists": 0.0,
                    "valid_json": 0.0,
                    "field_precision": 0.0,
                    "field_recall": 0.0,
                    "field_f1": 0.0,
                    "exact_match": 0.0,
                },
                failure_labels=["output_missing"],
                diagnostics={"required_output_path": str(config["output_path"])},
                runtime_seconds=time.perf_counter() - started,
            )
        try:
            actual = json.loads(output_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            return EvaluatorResult(
                success=False,
                score=0.2,
                subscores={
                    "output_exists": 1.0,
                    "valid_json": 0.0,
                    "field_precision": 0.0,
                    "field_recall": 0.0,
                    "field_f1": 0.0,
                    "exact_match": 0.0,
                },
                failure_labels=["invalid_json"],
                diagnostics={"error": str(exc), "required_output_path": str(config["output_path"])},
                runtime_seconds=time.perf_counter() - started,
            )

        expected = read_json(hidden_root / str(config.get("expected_path", "expected_output.json")))
        metrics = self.field_overlap_metrics(expected, actual)
        score = weighted_match_score(
            output_exists=1.0,
            valid_structure=1.0,
            metrics={
                "row_precision": metrics["field_precision"],
                "row_recall": metrics["field_recall"],
                "row_f1": metrics["field_f1"],
                "exact_match": metrics["exact_match"],
            },
        )
        success = actual == expected
        diagnostics = {
            "required_output_path": str(config["output_path"]),
            "expected_preview": self.preview(expected),
            "actual_preview": self.preview(actual),
        }
        if not success:
            diagnostics.update(json_field_diff_diagnostics(expected, actual))
        return EvaluatorResult(
            success=success,
            score=score,
            subscores={"output_exists": 1.0, "valid_json": 1.0, **metrics},
            failure_labels=[] if success else ["output_mismatch"],
            diagnostics=diagnostics,
            runtime_seconds=time.perf_counter() - started,
        )

    def evaluate_hidden_tests(
        self,
        workspace_path: Path,
        manifest: EnvironmentManifest,
        hidden_root: Path,
        config: dict[str, object],
    ) -> EvaluatorResult:
        started = time.perf_counter()
        runner_path = (hidden_root / str(config["runner"])).resolve()
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

        tests_passed = float(dict(payload.get("subscores", {})).get("tests_passed", 0))
        tests_total = float(dict(payload.get("subscores", {})).get("tests_total", 0))
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
                **dict(payload.get("diagnostics", {})),
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "returncode": completed.returncode,
            },
            runtime_seconds=time.perf_counter() - started,
        )

    def field_overlap_metrics(self, expected: Any, actual: Any) -> dict[str, float]:
        row_metrics = row_overlap_metrics(self.flatten_json(expected), self.flatten_json(actual))
        return {
            "field_precision": row_metrics["row_precision"],
            "field_recall": row_metrics["row_recall"],
            "field_f1": row_metrics["row_f1"],
            "exact_match": row_metrics["exact_match"],
        }

    def flatten_json(self, value: Any, *, prefix: str = "$") -> list[dict[str, object]]:
        return flatten_json(value, prefix=prefix)

    def preview(self, value: Any) -> Any:
        if isinstance(value, list):
            return value[:2]
        if isinstance(value, dict):
            keys = sorted(value)[:4]
            return {key: value[key] for key in keys}
        return value

    def extract_payload(self, stdout: str) -> dict[str, object] | None:
        for line in reversed([item.strip() for item in stdout.splitlines() if item.strip()]):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
        return None
