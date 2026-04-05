from __future__ import annotations

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
