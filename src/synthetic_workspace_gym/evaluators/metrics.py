from __future__ import annotations

import json
from collections import Counter
from typing import Any


def row_overlap_metrics(expected: Any, actual: Any) -> dict[str, float]:
    if not isinstance(expected, list) or not isinstance(actual, list):
        return {
            "row_precision": 0.0,
            "row_recall": 0.0,
            "row_f1": 0.0,
            "exact_match": 1.0 if actual == expected else 0.0,
        }
    expected_rows = Counter(normalize_row(row) for row in expected)
    actual_rows = Counter(normalize_row(row) for row in actual)
    matches = sum((expected_rows & actual_rows).values())
    precision = matches / len(actual) if actual else (1.0 if not expected else 0.0)
    recall = matches / len(expected) if expected else (1.0 if not actual else 0.0)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = (2 * precision * recall) / (precision + recall)
    return {
        "row_precision": round(precision, 6),
        "row_recall": round(recall, 6),
        "row_f1": round(f1, 6),
        "exact_match": 1.0 if actual == expected else 0.0,
    }


def weighted_match_score(*, output_exists: float, valid_structure: float, metrics: dict[str, float]) -> float:
    if metrics["exact_match"] == 1.0 and output_exists == 1.0 and valid_structure == 1.0:
        return 1.0
    score = (
        0.2 * output_exists
        + 0.2 * valid_structure
        + 0.3 * metrics["row_precision"]
        + 0.2 * metrics["row_recall"]
        + 0.1 * metrics["exact_match"]
    )
    return round(min(1.0, max(0.0, score)), 6)


def normalize_row(row: Any) -> str:
    return json.dumps(row, sort_keys=True, separators=(",", ":"))
