from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


TRAINING_RUNS: dict[str, dict[str, Any]] = {
    "specialist_pipeline": {
        "id": "l5glxbgjdntswv2l3bv9wvyg",
        "steps": [0, 25, 50, 75, 100, 122],
    },
    "specialist_retrieval": {
        "id": "lvr4kva16s9y8cla8gyih3rw",
        "steps": [0, 25, 50, 75, 100, 125, 150, 175, 199],
    },
    "specialist_script_repair": {
        "id": "fchv5udvzonemg8tek4veiw7",
        "steps": [0, 25, 50, 75, 100, 125, 150, 152],
    },
    "specialist_tabular": {
        "id": "a3sxjn4taxdjp1ueg32xa95f",
        "steps": [0, 25, 50, 75, 100, 125, 150, 175, 199],
    },
    "all_family_seed_42": {
        "id": "zj4tkmu8wcojd3vzmec5wroy",
        "steps": [0, 25, 50, 75, 100, 125, 150, 175, 199],
    },
    "all_family_seed_43": {
        "id": "bko57pm43ro6lt18ni2ul0gm",
        "steps": [0, 25, 50, 75, 100, 125, 150, 175, 199],
    },
    "composition_seed_42": {
        "id": "sjye7uiee2x9kjf6o0x1ygan",
        "steps": [0, 25, 50, 75, 100, 125, 150, 175, 199],
    },
    "composition_seed_43": {
        "id": "lidiv8ngaqb5py3dexm2uyqo",
        "steps": [0, 25, 50, 75, 100, 125, 150, 175, 199],
    },
    "checkpoint_pipeline_step_100": {
        "id": "h9bl28st7r0lb5ktfvbgs1my",
        "steps": [100],
    },
    "checkpoint_script_repair_step_150": {
        "id": "g4x25u3unmknj64ezff5yew1",
        "steps": [150],
    },
}

EVALUATIONS: dict[str, str] = {
    "base_in_distribution": "hv70l4ygf93v0s8woj0g4o0r",
    "base_scenario_heldout": "pwdn7pnbu1fwzpbav8oyxzng",
    "base_d5_panel": "ipz03q0il1rj0z5cwzdtltrq",
    "base_composite_heldout": "pi59auqw69mqrtq5iofawovk",
}


def run_json(prime: str, args: list[str], retries: int = 3) -> dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PRIME_DISABLE_VERSION_CHECK"] = "1"
    last_error = ""
    for attempt in range(1, retries + 1):
        completed = subprocess.run(
            [prime, *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8-sig",
            env=env,
            check=False,
        )
        if completed.returncode == 0:
            try:
                return json.loads(completed.stdout)
            except json.JSONDecodeError as exc:
                last_error = f"invalid JSON: {exc}; stdout={completed.stdout[:500]!r}"
        else:
            last_error = completed.stderr.strip() or completed.stdout.strip()
        if attempt < retries:
            time.sleep(2**attempt)
    raise RuntimeError(
        f"prime {' '.join(args)} failed after {retries} attempts: {last_error}"
    )


def write_gzip_json(path: Path, payload: Any) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    with gzip.open(path, "wb", compresslevel=6) as handle:
        handle.write(encoded)
    compressed = path.read_bytes()
    return {
        "path": path.as_posix(),
        "uncompressed_bytes": len(encoded),
        "compressed_bytes": len(compressed),
        "sha256": hashlib.sha256(compressed).hexdigest(),
    }


def load_gzip_json(path: Path) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def export_training(
    prime: str,
    root: Path,
    page_size: int,
    force: bool,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for label, spec in TRAINING_RUNS.items():
        run_id = str(spec["id"])
        run_dir = root / "training" / label
        run_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = run_dir / "metadata.json.gz"
        if force or not metadata_path.exists():
            metadata = {
                "run": run_json(
                    prime, ["train", "get", run_id, "--output", "json", "--plain"]
                ),
                "usage": run_json(
                    prime, ["train", "usage", run_id, "--output", "json", "--plain"]
                ),
                "progress": run_json(prime, ["train", "progress", run_id, "--plain"]),
                "metrics": run_json(prime, ["train", "metrics", run_id, "--plain"]),
                "checkpoints": run_json(
                    prime,
                    ["train", "checkpoints", run_id, "--output", "json", "--plain"],
                ),
            }
            meta_record = write_gzip_json(metadata_path, metadata)
        else:
            metadata = load_gzip_json(metadata_path)
            meta_record = {
                "path": metadata_path.as_posix(),
                "compressed_bytes": metadata_path.stat().st_size,
                "sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
            }
        records.append(
            {
                "kind": "training_metadata",
                "label": label,
                "run_id": run_id,
                **meta_record,
            }
        )
        available = set(
            (metadata.get("progress") or {}).get("steps_with_samples") or []
        )
        for step in spec["steps"]:
            if step not in available:
                records.append(
                    {
                        "kind": "training_rollouts",
                        "label": label,
                        "run_id": run_id,
                        "step": step,
                        "status": "unavailable",
                    }
                )
                continue
            path = run_dir / f"step-{step:03d}.json.gz"
            if force or not path.exists():
                payload = run_json(
                    prime,
                    [
                        "train",
                        "rollouts",
                        run_id,
                        "--step",
                        str(step),
                        "--num",
                        str(page_size),
                        "--plain",
                    ],
                )
                file_record = write_gzip_json(path, payload)
            else:
                payload = load_gzip_json(path)
                file_record = {
                    "path": path.as_posix(),
                    "compressed_bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            records.append(
                {
                    "kind": "training_rollouts",
                    "label": label,
                    "run_id": run_id,
                    "step": step,
                    "status": "exported",
                    "samples": len(payload.get("samples") or []),
                    "reported_total": payload.get("total"),
                    **file_record,
                }
            )
            print(
                f"training {label} step {step}: {len(payload.get('samples') or [])} samples"
            )
    return records


def export_evaluations(
    prime: str,
    root: Path,
    page_size: int,
    force: bool,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for label, eval_id in EVALUATIONS.items():
        eval_dir = root / "evaluations" / label
        eval_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = eval_dir / "metadata.json.gz"
        if force or not metadata_path.exists():
            metadata = run_json(
                prime, ["eval", "get", eval_id, "--output", "json", "--plain"]
            )
            meta_record = write_gzip_json(metadata_path, metadata)
        else:
            metadata = load_gzip_json(metadata_path)
            meta_record = {
                "path": metadata_path.as_posix(),
                "compressed_bytes": metadata_path.stat().st_size,
                "sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
            }
        records.append(
            {
                "kind": "evaluation_metadata",
                "label": label,
                "evaluation_id": eval_id,
                "status": metadata.get("status"),
                **meta_record,
            }
        )
        total = int(metadata.get("total_samples") or 0)
        fetched = 0
        page = 1
        while fetched < total:
            path = eval_dir / f"page-{page:03d}.json.gz"
            if force or not path.exists():
                payload = run_json(
                    prime,
                    [
                        "eval",
                        "samples",
                        eval_id,
                        "--page",
                        str(page),
                        "--num",
                        str(page_size),
                        "--output",
                        "json",
                        "--plain",
                    ],
                )
                file_record = write_gzip_json(path, payload)
            else:
                payload = load_gzip_json(path)
                file_record = {
                    "path": path.as_posix(),
                    "compressed_bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            samples = payload.get("samples") or []
            if not samples:
                break
            fetched += len(samples)
            records.append(
                {
                    "kind": "evaluation_samples",
                    "label": label,
                    "evaluation_id": eval_id,
                    "page": page,
                    "samples": len(samples),
                    **file_record,
                }
            )
            print(f"evaluation {label} page {page}: {len(samples)} samples")
            page += 1
    return records


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export frozen Qwen3.5-4B matrix trajectories without launching inference."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("analysis/qwen35-4b-offline/exports"),
    )
    parser.add_argument("--prime", default=shutil.which("prime") or "prime")
    parser.add_argument("--train-page-size", type=int, default=128)
    parser.add_argument("--eval-page-size", type=int, default=100)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-evaluations", action="store_true")
    args = parser.parse_args()

    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    if not args.skip_training:
        records.extend(
            export_training(args.prime, root, args.train_page_size, args.force)
        )
    if not args.skip_evaluations:
        records.extend(
            export_evaluations(args.prime, root, args.eval_page_size, args.force)
        )

    manifest = {
        "schema_version": 1,
        "exported_at_utc": datetime.now(UTC).isoformat(),
        "prime_cli": args.prime,
        "policy": {
            "training_steps": "fixed milestones plus terminal step",
            "training_page_size": args.train_page_size,
            "evaluation_pages": "all available samples",
            "paid_inference_launched": False,
        },
        "records": records,
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
