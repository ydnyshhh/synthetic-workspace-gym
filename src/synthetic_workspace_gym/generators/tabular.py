from __future__ import annotations

import csv
import json
import random
from datetime import date, datetime, timedelta
from pathlib import Path

from synthetic_workspace_gym.generators.base import BaseGenerator, GeneratedPayload
from synthetic_workspace_gym.schemas import EnvironmentFamily, EnvironmentSpec
from synthetic_workspace_gym.utils.io import write_json, write_text


class TabularTransformationGenerator(BaseGenerator):
    family = EnvironmentFamily.TABULAR

    date_formats = ("%Y-%m-%d", "%m/%d/%Y", "%d-%b-%Y")
    segments = ("enterprise", "midmarket", "smb")
    regions = ("north", "south", "east", "west")
    statuses = ("completed", "processing", "cancelled")

    def build_environment(self, spec: EnvironmentSpec, *, root: Path, visible_root: Path, hidden_root: Path) -> GeneratedPayload:
        rng = random.Random(spec.seed)
        include_customers = spec.difficulty >= 3
        include_adjustments = spec.difficulty >= 4
        include_duplicates = spec.difficulty >= 2
        include_distractor = spec.difficulty >= 4

        customers = self.build_customers(rng, count=4 + spec.difficulty)
        orders = self.build_orders(rng, customers, row_count=10 + (spec.difficulty * 5), include_duplicates=include_duplicates)
        adjustments = self.build_adjustments(rng, orders) if include_adjustments else []

        output_path = "outputs/report.json"
        task_descriptor = {
            "family": "tabular",
            "scenario_id": "monthly_segment_report",
            "input_files": ["data/orders.csv"],
            "output_path": output_path,
            "operations": [
                "parse_dates",
                "filter_cancelled",
                *(["deduplicate_latest"] if include_duplicates else []),
            ],
            "group_by": ["month"] + (["segment"] if include_customers else []),
            "sort_by": ["month"] + (["segment"] if include_customers else []),
        }
        if include_customers:
            task_descriptor["input_files"].append("data/customers.json")
            task_descriptor["operations"].append("join_customers")
        if include_adjustments:
            task_descriptor["input_files"].append("data/adjustments.csv")
            task_descriptor["operations"].append("apply_adjustments")

        expected_output = self.compute_expected_output(
            orders=orders,
            customers=customers,
            adjustments=adjustments,
            include_customers=include_customers,
            include_duplicates=include_duplicates,
            include_adjustments=include_adjustments,
        )

        self.write_orders_csv(visible_root / "data" / "orders.csv", orders)
        if include_customers:
            write_json(visible_root / "data" / "customers.json", customers)
        if include_adjustments:
            self.write_adjustments_csv(visible_root / "data" / "adjustments.csv", adjustments)
        if include_distractor:
            write_text(
                visible_root / "notes" / "legacy_columns.md",
                "The `sales_owner` and `legacy_code` fields are historical artifacts and do not belong in the final report.\n",
            )

        write_text(visible_root / "README.md", self.build_readme(task_descriptor))
        write_json(visible_root / "task.json", task_descriptor)

        write_json(hidden_root / "expected_output.json", expected_output)
        write_json(
            hidden_root / "evaluator_config.json",
            {
                "output_path": output_path,
                "comparison_mode": "exact_json",
            },
        )
        reference_solution = {
            "files": {
                output_path: json.dumps(expected_output, indent=2, sort_keys=True) + "\n",
            },
            "seed": spec.seed,
            "task_descriptor": task_descriptor,
        }
        write_json(hidden_root / "reference_solution.json", reference_solution)

        metadata = {
            "task_descriptor": task_descriptor,
            "complexity_profile": spec.complexity_profile.to_dict() if spec.complexity_profile else {},
            "input_row_count": len(orders),
            "customer_count": len(customers) if include_customers else 0,
            "adjustment_count": len(adjustments),
            "visible_artifact_layout": {
                "workspace_root": "visible",
                "output_path": output_path,
            },
        }
        return GeneratedPayload(
            instruction=self.build_instruction(task_descriptor),
            metadata=metadata,
            reference_solution=reference_solution,
            evaluator_entrypoint="synthetic_workspace_gym.evaluators.tabular:TabularEvaluator",
        )

    def build_customers(self, rng: random.Random, *, count: int) -> list[dict[str, str]]:
        customers: list[dict[str, str]] = []
        for index in range(count):
            customers.append(
                {
                    "customer_id": f"C{index + 1:03d}",
                    "segment": self.segments[index % len(self.segments)],
                    "region": self.regions[(index + 1) % len(self.regions)],
                }
            )
        rng.shuffle(customers)
        return customers

    def build_orders(
        self,
        rng: random.Random,
        customers: list[dict[str, str]],
        *,
        row_count: int,
        include_duplicates: bool,
    ) -> list[dict[str, str]]:
        base_date = date(2024, 1, 5)
        orders: list[dict[str, str]] = []
        for index in range(row_count):
            customer = rng.choice(customers)
            order_date = base_date + timedelta(days=rng.randint(0, 75))
            updated_at = datetime(2024, 3, 1, 8, 0, 0) + timedelta(hours=index)
            amount = round(rng.uniform(75, 900), 2)
            row = {
                "order_id": f"O{index + 1:04d}",
                "customer_id": customer["customer_id"],
                "status": rng.choices(self.statuses, weights=(0.65, 0.2, 0.15), k=1)[0],
                "order_date": order_date.strftime(rng.choice(self.date_formats)),
                "updated_at": updated_at.isoformat(),
                "amount": f"{amount:.2f}",
                "sales_owner": f"rep-{rng.randint(1, 8)}",
                "legacy_code": f"L-{rng.randint(100, 999)}",
            }
            orders.append(row)

        if include_duplicates:
            duplicate_indices = rng.sample(range(len(orders)), k=max(1, row_count // 5))
            for duplicate_index in duplicate_indices:
                source = orders[duplicate_index]
                updated_at = datetime.fromisoformat(source["updated_at"]) + timedelta(hours=24)
                revised_amount = round(float(source["amount"]) + rng.uniform(-20, 60), 2)
                orders.append(
                    {
                        **source,
                        "updated_at": updated_at.isoformat(),
                        "amount": f"{revised_amount:.2f}",
                    }
                )
        rng.shuffle(orders)
        return orders

    def build_adjustments(self, rng: random.Random, orders: list[dict[str, str]]) -> list[dict[str, str]]:
        unique_order_ids = sorted({row["order_id"] for row in orders})
        selected = rng.sample(unique_order_ids, k=max(2, len(unique_order_ids) // 4))
        adjustments = []
        for order_id in selected:
            adjustments.append(
                {
                    "order_id": order_id,
                    "adjustment": f"{round(rng.uniform(-15, 25), 2):.2f}",
                }
            )
        return adjustments

    def compute_expected_output(
        self,
        *,
        orders: list[dict[str, str]],
        customers: list[dict[str, str]],
        adjustments: list[dict[str, str]],
        include_customers: bool,
        include_duplicates: bool,
        include_adjustments: bool,
    ) -> list[dict[str, str | int | float]]:
        customer_lookup = {row["customer_id"]: row for row in customers}
        adjustment_lookup = {row["order_id"]: float(row["adjustment"]) for row in adjustments}
        deduped: dict[str, dict[str, str]] = {}
        iterable = orders
        if include_duplicates:
            for row in iterable:
                current = deduped.get(row["order_id"])
                if current is None or current["updated_at"] < row["updated_at"]:
                    deduped[row["order_id"]] = row
            iterable = list(deduped.values())

        grouped: dict[tuple[str, ...], dict[str, str | int | float]] = {}
        for row in iterable:
            if row["status"] == "cancelled":
                continue
            parsed_date = self.parse_date(row["order_date"])
            month = parsed_date.strftime("%Y-%m")
            amount = float(row["amount"])
            if include_adjustments:
                amount += adjustment_lookup.get(row["order_id"], 0.0)
            key_parts = [month]
            if include_customers:
                key_parts.append(customer_lookup[row["customer_id"]]["segment"])
            key = tuple(key_parts)
            if key not in grouped:
                entry: dict[str, str | int | float] = {"month": month, "order_count": 0, "total_amount": 0.0}
                if include_customers:
                    entry["segment"] = key_parts[1]
                grouped[key] = entry
            grouped[key]["order_count"] = int(grouped[key]["order_count"]) + 1
            grouped[key]["total_amount"] = float(grouped[key]["total_amount"]) + amount

        rows = list(grouped.values())
        for row in rows:
            row["total_amount"] = round(float(row["total_amount"]), 2)
        sort_keys = ["month"] + (["segment"] if include_customers else [])
        return sorted(rows, key=lambda item: tuple(str(item[key]) for key in sort_keys))

    def parse_date(self, value: str) -> datetime:
        for fmt in self.date_formats:
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                continue
        raise ValueError(f"Unsupported date format: {value}")

    def write_orders_csv(self, path: Path, rows: list[dict[str, str]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def write_adjustments_csv(self, path: Path, rows: list[dict[str, str]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["order_id", "adjustment"])
            writer.writeheader()
            writer.writerows(rows)

    def build_instruction(self, task_descriptor: dict[str, object]) -> str:
        parts = [
            "Create the cleaned monthly report described in README.md.",
            f"Write the final artifact to {task_descriptor['output_path']}.",
            "Use the provided visible workspace files only; the evaluator is hidden.",
        ]
        return " ".join(parts)

    def build_readme(self, task_descriptor: dict[str, object]) -> str:
        operations = "\n".join(
            f"{index}. `{operation}`"
            for index, operation in enumerate(task_descriptor["operations"], start=1)
        )
        inputs = "\n".join(f"- `{item}`" for item in task_descriptor["input_files"])
        group_by = ", ".join(f"`{field}`" for field in task_descriptor["group_by"])
        return (
            "# Monthly Segment Report\n\n"
            "You are working in a synthetic data-cleaning workspace. Build the final report from the provided tabular files.\n\n"
            "## Inputs\n"
            f"{inputs}\n\n"
            "## Required operations\n"
            f"{operations}\n\n"
            "## Output contract\n"
            f"- Write a JSON array to `{task_descriptor['output_path']}`.\n"
            f"- Group by {group_by}.\n"
            "- Include `order_count` and `total_amount` for every group.\n"
            "- Sort rows lexicographically by the grouping keys.\n"
            "- Round `total_amount` to 2 decimal places.\n"
        )
