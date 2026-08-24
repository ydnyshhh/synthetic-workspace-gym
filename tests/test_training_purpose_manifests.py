from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
MANIFEST_ROOTS = (
    ROOT / "src/synthetic_workspace_gym/frozen_manifests",
    ROOT
    / "environments/tool_use/synthetic_workspace_gym/synthetic_workspace_gym/frozen_manifests",
)


def _load(name: str, root: Path = MANIFEST_ROOTS[0]) -> dict[str, object]:
    return json.loads((root / f"{name}.json").read_text(encoding="utf-8"))


def _assignments(name: str) -> list[dict[str, object]]:
    return list(_load(name)["assignments"])


def test_training_purpose_manifests_are_mirrored_byte_for_byte() -> None:
    for name in ("sft-easy-v1", "sft-validation-v1", "rl-hard-v1", "rl-eval-v1"):
        left = MANIFEST_ROOTS[0] / f"{name}.json"
        right = MANIFEST_ROOTS[1] / f"{name}.json"
        assert left.read_bytes() == right.read_bytes()


def test_sft_curriculum_is_easy_and_validation_is_disjoint() -> None:
    train = _assignments("sft-easy-v1")
    validation = _assignments("sft-validation-v1")

    assert len(train) == 512
    assert {int(row["difficulty"]) for row in train} == {1, 2, 3}
    assert {str(row["family"]) for row in train} == {
        "pipeline",
        "retrieval_workspace",
        "script_repair",
        "tabular",
    }
    assert {str(row["split"]) for row in train} == {"sft_train"}
    assert {str(row["split"]) for row in validation} == {"sft_validation"}
    assert {int(row["difficulty"]) for row in validation} == {1, 2, 3}
    assert {int(row["seed"]) for row in train}.isdisjoint(
        {int(row["seed"]) for row in validation}
    )
    train_scenarios = {
        (str(row["family"]), str(row["scenario"])) for row in train
    }
    validation_scenarios = {
        (str(row["family"]), str(row["scenario"])) for row in validation
    }
    assert train_scenarios.isdisjoint(validation_scenarios)


def test_rl_curriculum_is_hard_and_evaluation_is_disjoint() -> None:
    train = _assignments("rl-hard-v1")
    evaluation = _assignments("rl-eval-v1")

    assert len(train) == 512
    assert len(evaluation) == 120
    assert {int(row["difficulty"]) for row in train} == {4, 5}
    assert {int(row["difficulty"]) for row in evaluation} == {4, 5}
    assert sum(row["family"] == "composite_workspace" for row in train) == 120
    assert sum(row["family"] == "composite_workspace" for row in evaluation) == 40
    assert {int(row["seed"]) for row in train}.isdisjoint(
        {int(row["seed"]) for row in evaluation}
    )
    assert {str(row["split"]) for row in train} == {"rl_train"}
    assert {str(row["split"]) for row in evaluation} == {"rl_eval"}


def test_manifest_metadata_declares_training_purpose_and_is_frozen() -> None:
    expected = {
        "sft-easy-v1": "supervised_midtraining",
        "sft-validation-v1": "supervised_validation",
        "rl-hard-v1": "reinforcement_learning",
        "rl-eval-v1": "reinforcement_learning_evaluation",
    }
    for name, purpose in expected.items():
        payload = _load(name)
        metadata = payload["metadata"]
        assert payload["version"] == "v1"
        assert metadata["frozen"] is True
        assert metadata["training_purpose"] == purpose
        assert metadata["manifest_fingerprint"]
