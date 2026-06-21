from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split intermediate action-window JSONL by trace/example group, stratified by scenario."
    )
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", default=None)
    parser.add_argument("--dev-ratio", type=float, default=0.1)
    parser.add_argument("--trace-test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            records.append(record)
    return records


def group_key(record: dict[str, Any]) -> str:
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Every record must include metadata for trace-level splitting")
    key = metadata.get("trace_id")
    if key in (None, ""):
        key = metadata.get("example_id")
    if key in (None, ""):
        raise ValueError("Every record must include metadata.trace_id or metadata.example_id")
    return str(key)


def scenario_key(records: list[dict[str, Any]]) -> str:
    scenarios = {
        str(record.get("metadata", {}).get("scenario"))
        for record in records
        if isinstance(record.get("metadata"), dict) and record.get("metadata", {}).get("scenario") is not None
    }
    return sorted(scenarios)[0] if scenarios else "unknown"


def allocation_counts(total: int, dev_ratio: float, test_ratio: float) -> tuple[int, int]:
    if total <= 1:
        return 0, 0
    test_count = int(round(total * test_ratio))
    dev_count = int(round(total * dev_ratio))
    if test_ratio > 0 and total >= 3:
        test_count = max(1, test_count)
    if dev_ratio > 0 and total >= 3:
        dev_count = max(1, dev_count)
    while dev_count + test_count >= total:
        if dev_count >= test_count and dev_count > 0:
            dev_count -= 1
        elif test_count > 0:
            test_count -= 1
        else:
            break
    return dev_count, test_count


def split_records(
    records: list[dict[str, Any]],
    dev_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[group_key(record)].append(record)

    by_scenario: dict[str, list[tuple[str, list[dict[str, Any]]]]] = defaultdict(list)
    for key, group_records in grouped.items():
        by_scenario[scenario_key(group_records)].append((key, group_records))

    rng = random.Random(seed)
    splits = {"train": [], "dev": [], "trace_test": []}
    for scenario in sorted(by_scenario):
        groups = by_scenario[scenario]
        rng.shuffle(groups)
        dev_count, test_count = allocation_counts(len(groups), dev_ratio, test_ratio)
        test_groups = groups[:test_count]
        dev_groups = groups[test_count : test_count + dev_count]
        train_groups = groups[test_count + dev_count :]
        for split_name, split_groups in (
            ("train", train_groups),
            ("dev", dev_groups),
            ("trace_test", test_groups),
        ):
            for _, group_records in split_groups:
                splits[split_name].extend(group_records)

    return splits


def output_prefix(input_jsonl: Path, prefix: str | None) -> str:
    if prefix:
        return prefix
    stem = input_jsonl.stem
    return stem.removesuffix("_actions")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> int:
    args = parse_args()
    if not (0 <= args.dev_ratio < 1) or not (0 <= args.trace_test_ratio < 1):
        raise ValueError("Split ratios must be in [0, 1)")
    if args.dev_ratio + args.trace_test_ratio >= 1:
        raise ValueError("dev-ratio + trace-test-ratio must be less than 1")

    records = read_jsonl(args.input_jsonl)
    splits = split_records(records, args.dev_ratio, args.trace_test_ratio, args.seed)
    prefix = output_prefix(args.input_jsonl, args.prefix)
    paths = {
        "train": args.output_dir / f"{prefix}_train.jsonl",
        "dev": args.output_dir / f"{prefix}_dev.jsonl",
        "trace_test": args.output_dir / f"{prefix}_trace_test.jsonl",
    }
    for split_name, path in paths.items():
        write_jsonl(path, splits[split_name])
    print(
        "Wrote splits: "
        + ", ".join(f"{name}={len(splits[name])} -> {paths[name]}" for name in ("train", "dev", "trace_test"))
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
