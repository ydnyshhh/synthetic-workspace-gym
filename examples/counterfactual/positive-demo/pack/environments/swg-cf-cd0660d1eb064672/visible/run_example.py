from __future__ import annotations

import json
import sys
from pathlib import Path

workspace = Path(__file__).resolve().parent
sys.path.insert(0, str(workspace / "src"))

from repair_target.parser import load_orders
from repair_target.report import build_region_report


def main() -> None:
    print(json.dumps(build_region_report(load_orders(workspace / "data" / "orders.csv")), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
