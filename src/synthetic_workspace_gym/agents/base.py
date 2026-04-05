from __future__ import annotations

import csv
import io
import json
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any

from synthetic_workspace_gym.schemas import Action, ActionType, EnvironmentManifest, ToolObservation, ToolState


class BaseAgent(ABC):
    name = "base"

    def __init__(self) -> None:
        self.manifest: EnvironmentManifest | None = None
        self.initial_observation: dict[str, object] = {}
        self.last_action: Action | None = None
        self.file_cache: dict[str, str] = {}
        self.directory_cache: dict[str, list[str]] = {}
        self.task: dict[str, Any] | None = None

    def reset(self, manifest: EnvironmentManifest, initial_observation: dict[str, object]) -> None:
        self.manifest = manifest
        self.initial_observation = initial_observation
        self.last_action = None
        self.file_cache = {}
        self.directory_cache = {}
        self.task = None

    @abstractmethod
    def act(self, observation: ToolObservation | dict[str, object], tool_state: ToolState) -> Action:
        raise NotImplementedError

    def _consume_observation(self, observation: ToolObservation | dict[str, object]) -> None:
        if self.last_action is None or not isinstance(observation, ToolObservation):
            return
        action = self.last_action
        path = str(action.arguments.get("path", ""))
        if action.action_type == ActionType.READ_FILE and observation.success and path:
            self.file_cache[path] = observation.content or ""
            if path == "task.json":
                self.task = json.loads(observation.content or "{}")
        elif action.action_type == ActionType.WRITE_FILE and observation.success and path:
            self.file_cache[path] = str(action.arguments.get("content", ""))
        elif action.action_type == ActionType.APPEND_FILE and observation.success and path:
            self.file_cache[path] = self.file_cache.get(path, "") + str(action.arguments.get("content", ""))
        elif action.action_type == ActionType.LIST_DIRECTORY and observation.success:
            listed_path = str(action.arguments.get("path", "."))
            self.directory_cache[listed_path] = observation.listing

    def _set_last_action(self, action: Action) -> Action:
        self.last_action = action
        return action


def parse_csv_rows(content: str) -> list[dict[str, str]]:
    reader = csv.DictReader(io.StringIO(content))
    return [dict(row) for row in reader]


def parse_json_content(content: str) -> Any:
    return json.loads(content)


def solve_tabular_task(task: dict[str, Any], file_cache: dict[str, str]) -> str:
    customers = {
        row["customer_id"]: row
        for row in parse_json_content(file_cache.get("data/customers.json", "[]"))
    }
    adjustments = {
        row["order_id"]: float(row["adjustment"])
        for row in parse_csv_rows(file_cache.get("data/adjustments.csv", "order_id,adjustment\n"))
    }
    orders = parse_csv_rows(file_cache["data/orders.csv"])
    if "deduplicate_latest" in task.get("operations", []):
        deduped: dict[str, dict[str, str]] = {}
        for row in orders:
            existing = deduped.get(row["order_id"])
            if existing is None or existing["updated_at"] < row["updated_at"]:
                deduped[row["order_id"]] = row
        orders = list(deduped.values())

    grouped: dict[tuple[str, ...], dict[str, object]] = {}
    for row in orders:
        if row["status"] == "cancelled":
            continue
        order_date = parse_date(row["order_date"])
        key_values: list[str] = []
        for key in task.get("group_by", []):
            if key == "month":
                key_values.append(order_date.strftime("%Y-%m"))
            elif key == "segment":
                key_values.append(customers[row["customer_id"]]["segment"])
            else:
                key_values.append(str(row[key]))
        key = tuple(key_values)
        if key not in grouped:
            entry = {name: value for name, value in zip(task.get("group_by", []), key_values)}
            entry["order_count"] = 0
            entry["total_amount"] = 0.0
            grouped[key] = entry
        amount = float(row["amount"])
        if "apply_adjustments" in task.get("operations", []):
            amount += adjustments.get(row["order_id"], 0.0)
        grouped[key]["order_count"] = int(grouped[key]["order_count"]) + 1
        grouped[key]["total_amount"] = round(float(grouped[key]["total_amount"]) + amount, 2)

    sort_keys = task.get("sort_by", task.get("group_by", []))
    rows = sorted(grouped.values(), key=lambda item: tuple(str(item[key]) for key in sort_keys))
    return json.dumps(rows, indent=2, sort_keys=True) + "\n"


def parse_date(value: str) -> datetime:
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d-%b-%Y"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unsupported date format: {value}")
