from __future__ import annotations

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
