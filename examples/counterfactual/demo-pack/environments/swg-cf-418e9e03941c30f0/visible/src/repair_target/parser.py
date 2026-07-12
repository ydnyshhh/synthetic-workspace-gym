from __future__ import annotations

import csv
from pathlib import Path


def load_orders(path: Path) -> list[dict[str, object]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            if not row.get("customer_id"):
                continue
            rows.append(
                {
                    "account_id": row["account_id"],
                    "region": str(row["region"]).strip().lower(),
                    "status": str(row["status"]).strip().lower(),
                    "amount": round(float(row["amount"]), 2),
                }
            )
    return rows
