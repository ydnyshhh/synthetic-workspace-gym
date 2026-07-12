from __future__ import annotations

import json
from pathlib import Path

import pytest

from synthetic_workspace_gym.counterfactual.analysis import aggregate_outcomes
from synthetic_workspace_gym.counterfactual.candidates import generate_candidates
from synthetic_workspace_gym.counterfactual.compiler import compile_pack
from synthetic_workspace_gym.counterfactual.exports import export_training_data
from synthetic_workspace_gym.counterfactual.schemas import BranchOutcome, BranchTask, CandidateAction, CounterfactualSnapshot, stable_id
from synthetic_workspace_gym.counterfactual.selectors import BeforeFirstWriteSelector, BeforeSubmitSelector, RepeatedActionSelector, ScoreDropSelector
from synthetic_workspace_gym.counterfactual.snapshots import NamedSnapshotPolicy, SnapshotCollector, SnapshotContext, load_snapshot
from synthetic_workspace_gym.generators.registry import get_generator
from synthetic_workspace_gym.runtime.environment import load_environment
from synthetic_workspace_gym.schemas import Action, ActionType


def snapshot(**overrides) -> CounterfactualSnapshot:
    data = dict(snapshot_id="snapshot-1", trajectory_id="trajectory-1", episode_id=None, env_id="env-1", family="script_repair", scenario_id="csv_schema_drift", difficulty=3, seed=1, step_index=2, remaining_steps=5, elapsed_seconds=1., workspace_path="visible", manifest_path="manifest.json", branch_state_path="branch_state.json", original_action={"tool": "write_file", "args": {"path": "x.py", "content": "x"}}, metadata={"phase": "before"})
    data.update(overrides); return CounterfactualSnapshot(**data)


def outcome(candidate: str, reward: float, index: int = 0, kind: str = "other") -> BranchOutcome:
    return BranchOutcome(f"r-{candidate}-{index}", f"t-{candidate}", "g", candidate, "s", "model", index, reward, reward, reward >= .95, step_count=2, metadata={"candidate_type": kind})


def test_schema_roundtrip_and_invariants() -> None:
    item = snapshot(); assert CounterfactualSnapshot.from_dict(item.to_dict()) == item
    assert stable_id("x", 1, {"a": 2}) == stable_id("x", 1, {"a": 2})
    with pytest.raises(ValueError): BranchTask("t", "g", "s", "c", "bad", "e", [], None, 1, 1, "f", None, 1, 1)
    with pytest.raises(ValueError): CandidateAction("c", "g", "s", "original", {}, "test")


def test_selectors() -> None:
    first = snapshot(); submit = snapshot(snapshot_id="s2", step_index=4, original_action={"tool": "submit", "args": {}}, previous_action={"tool": "submit", "args": {}}, evaluator_score=.7)
    drop = snapshot(snapshot_id="s3", step_index=5, evaluator_score=.4, metadata={"phase": "after"})
    assert BeforeFirstWriteSelector().select([first, submit])[0].snapshot_id == first.snapshot_id
    assert BeforeSubmitSelector().select([first, submit])[0].snapshot_id == submit.snapshot_id
    assert RepeatedActionSelector().select([submit])
    assert ScoreDropSelector().select([submit, drop])


def test_analysis_and_preference_export(tmp_path: Path) -> None:
    rows = [outcome("original", .2, kind="original"), outcome("better", .9), outcome("better", 1., 1)]
    comparison = aggregate_outcomes(rows, recoverable_threshold=.9)[0]
    assert comparison.decision_regret == pytest.approx(.75); assert comparison.recoverable
    common = dict(branch_group_id="g", snapshot_id="s", mode="forced", environment_path="env", prefix_messages=[], remaining_steps=2, time_limit_seconds=10, family="f", scenario_id=None, difficulty=1, seed=1)
    tasks = {"a": BranchTask("a", candidate_id="original", forced_action={"tool": "submit", "args": {}}, metadata={"candidate_type": "original"}, **common), "b": BranchTask("b", candidate_id="better", forced_action={"tool": "read_file", "args": {"path": "README.md"}}, metadata={"candidate_type": "read_relevant_file"}, **common)}
    records = export_training_data([comparison], tasks, tmp_path / "preference.jsonl", "preference", min_margin=.2)
    assert len(records) == 1 and records[0]["state_id"] == "s"


def test_snapshot_candidate_compile_pipeline(tmp_path: Path) -> None:
    generator = get_generator("script_repair")
    spec = generator.sample_spec(difficulty=1, seed=7, scenario_id="csv_schema_drift")
    bundle = generator.generate_instance(spec, tmp_path / "generated")
    env = load_environment(bundle.root)
    collector = SnapshotCollector(tmp_path / "snapshots", NamedSnapshotPolicy("every_step"), max_snapshots=2)
    action = Action(ActionType.SUBMIT, {"path_or_answer": "done"})
    context = SnapshotContext("trajectory", "episode", env.manifest, env.visible_root, env.root, 1, 4, .5, action, None, "ready", [{"role": "user", "content": env.manifest.instruction}], [], "before")
    saved = collector.maybe_capture(context); assert saved is not None
    root = Path(saved.metadata["snapshot_root"]); loaded = load_snapshot(root)
    assert (root / "visible").exists() and (root / "hidden").exists()
    candidates = generate_candidates(loaded, load_environment(root).manifest, root, ["original", "submit", "run_public_check"], 3)
    tasks = compile_pack([(loaded, candidate, root) for candidate in candidates], tmp_path / "pack", "forced")
    assert tasks and (tmp_path / "pack" / "manifest.jsonl").exists()
    assert not (root / "visible" / "hidden").exists()


def test_disabled_policy_writes_nothing(tmp_path: Path) -> None:
    assert not NamedSnapshotPolicy("none").should_snapshot(None if False else type("C", (), {"action": None})()).selected
