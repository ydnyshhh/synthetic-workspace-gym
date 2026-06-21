from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert intermediate action-window JSONL into prompt/completion SFT JSONL."
    )
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument(
        "--drop-metadata",
        action="store_true",
        help="Write only {'prompt': ..., 'completion': ...} records instead of preserving metadata.",
    )
    return parser.parse_args()


def convert_record(record: dict[str, Any], drop_metadata: bool) -> dict[str, Any]:
    prompt = copy.deepcopy(record.get("messages"))
    completion = copy.deepcopy(record.get("target"))
    if not isinstance(prompt, list) or not isinstance(completion, dict):
        raise ValueError("Input record must contain list 'messages' and object 'target'")
    if completion.get("role") != "assistant":
        raise ValueError("Input target must be an assistant message")
    if completion.get("tool_calls") and not isinstance(completion["tool_calls"], list):
        raise ValueError("Input target tool_calls must be a list")

    output = {
        "prompt": prompt,
        "completion": completion,
    }
    if not drop_metadata and isinstance(record.get("metadata"), dict):
        output["metadata"] = copy.deepcopy(record["metadata"])
    return output


def main() -> int:
    args = parse_args()
    count = 0
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.input_jsonl.open("r", encoding="utf-8") as source:
        with args.output_jsonl.open("w", encoding="utf-8") as target:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    converted = convert_record(record, args.drop_metadata)
                except Exception as exc:
                    raise ValueError(f"{args.input_jsonl}:{line_number}: {exc}") from exc
                target.write(json.dumps(converted, ensure_ascii=False, sort_keys=True) + "\n")
                count += 1
    print(f"Wrote {count} prompt/completion SFT records to {args.output_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
