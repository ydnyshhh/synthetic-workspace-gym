from __future__ import annotations

import argparse
import json
from pathlib import Path

from synthetic_workspace_gym.evaluators.registry import get_evaluator
from synthetic_workspace_gym.prime.verifier import evaluator_result_to_prime_reward
from synthetic_workspace_gym.schemas import EnvironmentManifest
from synthetic_workspace_gym.utils.io import read_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--hidden", type=Path, required=True)
    args = parser.parse_args()

    manifest = EnvironmentManifest.from_dict(read_json(args.manifest))
    evaluator = get_evaluator(manifest.family, evaluator_entrypoint=manifest.evaluator_entrypoint)
    result = evaluator.evaluate(args.workspace, manifest, args.hidden)
    print(json.dumps(evaluator_result_to_prime_reward(result), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
