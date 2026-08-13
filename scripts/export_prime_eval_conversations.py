from __future__ import annotations

import argparse
import collections
import json
import re
import subprocess
from pathlib import Path
from typing import Any


def _run_prime(prime: str, args: list[str]) -> dict[str, Any]:
    completed = subprocess.run(
        [prime, *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8-sig",
    )
    return json.loads(completed.stdout)


def _message_text(message: dict[str, Any]) -> str:
    parts: list[str] = []
    content = message.get("content")
    if content:
        parts.append(str(content))
    reasoning = message.get("reasoning_content")
    if reasoning:
        parts.append(f"[reasoning]\n{reasoning}")
    calls = message.get("tool_calls") or []
    for call in calls:
        parts.append(f"[tool_call]\n{call}")
    return "\n\n".join(parts).strip()


def _write_conversation(path: Path, sample: dict[str, Any]) -> None:
    lines = [
        f"# example {sample.get('example_id')} rollout {sample.get('rollout_number')}",
        "",
        f"- trace_id: {sample.get('trace_id')}",
        f"- reward: {sample.get('reward') or sample.get('swg_reward')}",
        f"- num_steps: {sample.get('num_steps')}",
        f"- total_time: {sample.get('total_time')}",
        "",
    ]
    messages = []
    messages.extend(sample.get("prompt") or [])
    messages.extend(sample.get("completion") or [])
    for idx, message in enumerate(messages):
        role = message.get("role", "unknown")
        lines.append(f"## {idx:03d} {role}")
        text = _message_text(message)
        lines.append(text if text else "(empty)")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _extract_metadata(sample: dict[str, Any]) -> dict[str, str]:
    prompt = sample.get("prompt") or []
    text = ""
    if len(prompt) > 1:
        text = str(prompt[1].get("content") or "")
    metadata: dict[str, str] = {}
    for key in ("task_id", "split", "family", "scenario", "difficulty", "seed"):
        match = re.search(rf"- {key}: ([^\n]+)", text)
        if match:
            metadata[key] = match.group(1).strip()
    if not metadata and "Repair the provided Python workspace" in text:
        metadata["family"] = "legacy_or_unannotated"
    return metadata


def _tool_names(sample: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for message in sample.get("completion") or []:
        for raw in message.get("tool_calls") or []:
            try:
                call = json.loads(raw)
            except Exception:
                continue
            name = call.get("name")
            if name:
                names.append(str(name))
    return names


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("eval_id")
    parser.add_argument("--prime", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--page-size", type=int, default=100)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    pages_dir = args.out / "pages"
    conversations_dir = args.out / "conversations"
    pages_dir.mkdir(exist_ok=True)
    conversations_dir.mkdir(exist_ok=True)

    metadata = _run_prime(args.prime, ["eval", "get", args.eval_id, "--output", "json", "--plain"])
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    all_samples: list[dict[str, Any]] = []
    page = 1
    total = None
    while total is None or len(all_samples) < total:
        payload = _run_prime(
            args.prime,
            [
                "eval",
                "samples",
                args.eval_id,
                "--page",
                str(page),
                "--num",
                str(args.page_size),
                "--output",
                "json",
                "--plain",
            ],
        )
        (pages_dir / f"page-{page:03d}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        total = int(payload.get("total") or 0)
        samples = payload.get("samples") or []
        if not samples:
            break
        all_samples.extend(samples)
        page += 1

    (args.out / "samples.json").write_text(json.dumps({"samples": all_samples}, indent=2), encoding="utf-8")

    unique: dict[tuple[Any, Any], dict[str, Any]] = {}
    duplicate_counts: collections.Counter[tuple[Any, Any]] = collections.Counter()
    for sample in all_samples:
        key = (sample.get("example_id"), sample.get("rollout_number"))
        duplicate_counts[key] += 1
        unique.setdefault(key, sample)

    rows: list[dict[str, Any]] = []
    for key, sample in sorted(unique.items(), key=lambda item: (item[0][0] or -1, item[0][1] or -1)):
        example_id, rollout_number = key
        metadata_fields = _extract_metadata(sample)
        tools = _tool_names(sample)
        reward = sample.get("reward")
        if reward is None:
            reward = sample.get("swg_reward")
        rows.append(
            {
                "example_id": example_id,
                "rollout_number": rollout_number,
                "duplicate_count": duplicate_counts[key],
                "reward": reward,
                "num_steps": sample.get("num_steps"),
                "family": metadata_fields.get("family"),
                "scenario": metadata_fields.get("scenario"),
                "difficulty": metadata_fields.get("difficulty"),
                "task_id": metadata_fields.get("task_id"),
                "tools": collections.Counter(tools),
                "submit_count": tools.count("submit"),
                "tool_count": len(tools),
            }
        )
        _write_conversation(
            conversations_dir / f"example-{int(example_id):04d}-rollout-{int(rollout_number):02d}.md",
            sample,
        )

    summary = {
        "eval_id": args.eval_id,
        "status": metadata.get("status"),
        "declared_total_samples": metadata.get("total_samples"),
        "fetched_samples": len(all_samples),
        "unique_samples": len(unique),
        "duplicate_count_distribution": collections.Counter(duplicate_counts.values()),
        "families": collections.Counter(row.get("family") for row in rows),
        "reward_distribution": collections.Counter(str(row.get("reward")) for row in rows),
        "average_reward": (
            sum(float(row["reward"]) for row in rows if isinstance(row.get("reward"), (int, float))) / len(rows)
            if rows
            else None
        ),
    }
    (args.out / "export-summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (args.out / "summary.json").write_text(json.dumps(rows, indent=2, default=dict), encoding="utf-8")

    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
