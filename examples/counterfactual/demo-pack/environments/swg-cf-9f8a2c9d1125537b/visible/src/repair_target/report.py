from __future__ import annotations


def build_region_report(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summary: dict[str, dict[str, object]] = {}
    for row in rows:
        if row["status"] == "cancelled":
            continue
        region = str(row["region"])
        if region not in summary:
            summary[region] = {"region": region, "row_count": 0, "total_amount": 0.0}
        summary[region]["row_count"] = int(summary[region]["row_count"]) + 1
        summary[region]["total_amount"] = round(
            float(summary[region]["total_amount"]) + float(row["amount"]),
            2,
        )
    return sorted(summary.values(), key=lambda item: str(item["region"]))
