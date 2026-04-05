from __future__ import annotations

from pathlib import Path


def load_measurements(data_dir: Path) -> list[int]:
    path = data_dir.parent / "measurements.csv"
    rows = Path(path).read_text(encoding="utf-8").strip().splitlines()[1:]
    return [int(row.split(",")[1]) for row in rows]
