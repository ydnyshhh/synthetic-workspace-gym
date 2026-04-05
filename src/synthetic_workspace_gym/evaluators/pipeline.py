from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from synthetic_workspace_gym.evaluators.base import BaseEvaluator
from synthetic_workspace_gym.evaluators.metrics import row_overlap_metrics, weighted_match_score
from synthetic_workspace_gym.schemas import EnvironmentManifest, EvaluatorResult
from synthetic_workspace_gym.utils.io import read_json


class PipelineEvaluator(BaseEvaluator):
    def evaluate(self, workspace_path: Path, manifest: EnvironmentManifest, hidden_root: Path) -> EvaluatorResult:
        started = time.perf_counter()
        config = read_json(hidden_root / "evaluator_config.json")
        try:
            completed = subprocess.run(
                [sys.executable, str(workspace_path / config["entrypoint"])],
                cwd=str(workspace_path),
                capture_output=True,
                text=True,
                timeout=manifest.time_limit_seconds,
                env={**dict(os.environ), "PYTHONDONTWRITEBYTECODE": "1"},
            )
        except subprocess.TimeoutExpired as exc:
            return EvaluatorResult(
                success=False,
                score=0.0,
                subscores={
                    "execution": 0.0,
                    "valid_json": 0.0,
                    "row_precision": 0.0,
                    "row_recall": 0.0,
                    "row_f1": 0.0,
                    "exact_match": 0.0,
                },
                failure_labels=["timeout"],
                diagnostics={
                    "stdout": exc.stdout or "",
                    "stderr": exc.stderr or "",
                    "entrypoint": config["entrypoint"],
                },
                runtime_seconds=time.perf_counter() - started,
            )
        if completed.returncode != 0:
            return EvaluatorResult(
                success=False,
                score=0.0,
                subscores={
                    "execution": 0.0,
                    "valid_json": 0.0,
                    "row_precision": 0.0,
                    "row_recall": 0.0,
                    "row_f1": 0.0,
                    "exact_match": 0.0,
                },
                failure_labels=["execution_failed"],
                diagnostics={
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                    "returncode": completed.returncode,
                },
                runtime_seconds=time.perf_counter() - started,
            )

        output_path = workspace_path / config["required_output_path"]
        if not output_path.exists():
            return EvaluatorResult(
                success=False,
                score=0.2,
                subscores={
                    "execution": 1.0,
                    "valid_json": 0.0,
                    "row_precision": 0.0,
                    "row_recall": 0.0,
                    "row_f1": 0.0,
                    "exact_match": 0.0,
                },
                failure_labels=["output_missing"],
                diagnostics={"required_output_path": config["required_output_path"]},
                runtime_seconds=time.perf_counter() - started,
            )

        try:
            actual = json.loads(output_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            return EvaluatorResult(
                success=False,
                score=0.4,
                subscores={
                    "execution": 1.0,
                    "valid_json": 0.0,
                    "row_precision": 0.0,
                    "row_recall": 0.0,
                    "row_f1": 0.0,
                    "exact_match": 0.0,
                },
                failure_labels=["invalid_json"],
                diagnostics={"error": str(exc)},
                runtime_seconds=time.perf_counter() - started,
            )

        expected = read_json(hidden_root / "expected_output.json")
        success = actual == expected
        metrics = row_overlap_metrics(expected, actual)
        score = weighted_match_score(output_exists=1.0, valid_structure=1.0, metrics=metrics)
        diagnostics = {
            "required_output_path": config["required_output_path"],
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "returncode": completed.returncode,
        }
        if not success:
            diagnostics["expected_preview"] = expected[:2] if isinstance(expected, list) else expected
            diagnostics["actual_preview"] = actual[:2] if isinstance(actual, list) else actual
        return EvaluatorResult(
            success=success,
            score=score,
            subscores={"execution": 1.0, "valid_json": 1.0, **metrics},
            failure_labels=[] if success else ["output_mismatch"],
            diagnostics=diagnostics,
            runtime_seconds=time.perf_counter() - started,
        )
