from __future__ import annotations

import json
from pathlib import Path

import pytest

from synthetic_workspace_gym.counterfactual.analysis import aggregate_outcomes
from synthetic_workspace_gym.counterfactual.candidates import generate_candidates
from synthetic_workspace_gym.counterfactual.compiler import compile_pack
from synthetic_workspace_gym.counterfactual.exports import export_training_data
from synthetic_workspace_gym.counterfactual.replay import replay_branch
from synthetic_workspace_gym.counterfactual.runner import read_branch_manifest
from synthetic_workspace_gym.counterfactual.schemas import BranchOutcome, BranchTask, CandidateAction, CounterfactualSnapshot, stable_id
from synthetic_workspace_gym.counterfactual.selectors import AfterFailedCheckSelector, BeforeFirstWriteSelector, BeforeSubmitSelector, RepeatedActionSelector, ScoreDropSelector
from synthetic_workspace_gym.counterfactual.snapshots import NamedSnapshotPolicy, SnapshotCollector, SnapshotContext, load_snapshot
from synthetic_workspace_gym.generators.registry import get_generator
from synthetic_workspace_gym.agents.base import BaseAgent
from synthetic_workspace_gym.agents.scripted import ScriptedBaselineAgent
from synthetic_workspace_gym.runtime.runner import EpisodeRunner
from synthetic_workspace_gym.schemas import ToolState
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
    with pytest.raises(ValueError): BranchTask("t", "g", "s", "c", "forced", "e", [], None, 1, 1, "f", None, 1, 1)


def test_selectors() -> None:
    first = snapshot(); submit = snapshot(snapshot_id="s2", step_index=4, original_action={"tool": "submit", "args": {}}, previous_action={"tool": "submit", "args": {}}, evaluator_score=.7)
    drop = snapshot(snapshot_id="s3", step_index=5, evaluator_score=.4, metadata={"phase": "before", "previous_event_type": "score_drop"})
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


def test_public_check_is_shell_and_pack_paths_are_portable(tmp_path: Path) -> None:
    generator = get_generator("script_repair")
    spec = generator.sample_spec(difficulty=1, seed=11, scenario_id="csv_schema_drift")
    bundle = generator.generate_instance(spec, tmp_path / "generated")
    env = load_environment(bundle.root)
    collector = SnapshotCollector(tmp_path / "snapshots", NamedSnapshotPolicy("every_step"), max_snapshots=1)
    action = Action(ActionType.SUBMIT, {"path_or_answer": "done"})
    context = SnapshotContext("trajectory", None, env.manifest, env.visible_root, env.root, 1, 4, .1, action, None, None, [{"role": "user", "content": env.manifest.instruction}], [], "before")
    saved = collector.maybe_capture(context)
    assert saved is not None
    snapshot_root = Path(saved.metadata["snapshot_root"])
    candidate = generate_candidates(saved, env.manifest, snapshot_root, ["run_public_check"], 1)[0]
    assert candidate.action == {"tool": "run_shell", "args": {"command": "python run_example.py"}}
    compile_pack([(saved, candidate, snapshot_root)], tmp_path / "pack", "forced")
    raw = json.loads((tmp_path / "pack" / "manifest.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert raw["environment_path"].startswith("environments/")
    assert "\\" not in raw["environment_path"]
    loaded = read_branch_manifest(tmp_path / "pack" / "manifest.jsonl")[0]
    assert Path(loaded.environment_path).is_absolute()


def test_script_repair_branch_evaluator_uses_absolute_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    generator = get_generator("script_repair")
    spec = generator.sample_spec(difficulty=1, seed=13, scenario_id="csv_schema_drift")
    bundle = generator.generate_instance(spec, Path("generated"))
    env = load_environment(bundle.root)
    collector = SnapshotCollector(Path("snapshots"), NamedSnapshotPolicy("every_step"), max_snapshots=1)
    action = Action(ActionType.SUBMIT, {"path_or_answer": "done"})
    context = SnapshotContext("trajectory", None, env.manifest, env.visible_root, env.root, 1, 4, .1, action, None, None, [{"role": "user", "content": env.manifest.instruction}], [], "before")
    saved = collector.maybe_capture(context)
    assert saved is not None
    snapshot_root = Path(saved.metadata["snapshot_root"])
    candidate = generate_candidates(saved, env.manifest, snapshot_root, ["submit"], 1)[0]
    compile_pack([(saved, candidate, snapshot_root)], Path("pack"), "forced")
    task = read_branch_manifest(Path("pack/manifest.jsonl"))[0]
    outcome = replay_branch(task, ScriptedBaselineAgent(), Path("run")).outcome
    assert outcome.diagnostics.get("returncode") is not None
    assert "ModuleNotFoundError" not in outcome.diagnostics.get("stderr", "")
    assert outcome.subscores["tests_total"] > 0

class _FailedCheckThenReadAgent(BaseAgent):
    name = "failed-check-then-read"

    def __init__(self) -> None:
        super().__init__()
        self.actions = []

    def reset(self, manifest, initial_observation):
        super().reset(manifest, initial_observation)
        self.actions = [
            Action(ActionType.RUN_SHELL, {"command": "python missing-public-check.py"}),
            Action(ActionType.READ_FILE, {"path": "task.json"}),
            Action(ActionType.SUBMIT, {"path_or_answer": "done"}),
        ]

    def act(self, observation, tool_state: ToolState) -> Action:
        return self.actions.pop(0)


def test_failed_check_selects_actual_next_before_action_even_at_snapshot_limit(tmp_path: Path) -> None:
    generator = get_generator("script_repair")
    spec = generator.sample_spec(difficulty=1, seed=19, scenario_id="csv_schema_drift")
    bundle = generator.generate_instance(spec, tmp_path / "generated")
    collector = SnapshotCollector(tmp_path / "snapshots", NamedSnapshotPolicy("writes_checks_submit"), max_snapshots=1)
    EpisodeRunner(tmp_path / "episodes", collector).run_episode(load_environment(bundle.root), _FailedCheckThenReadAgent())
    assert len(collector.snapshots) == 2
    selected = AfterFailedCheckSelector().select(collector.snapshots)
    assert len(selected) == 1
    branch = next(item for item in collector.snapshots if item.snapshot_id == selected[0].snapshot_id)
    assert branch.metadata["phase"] == "before"
    assert branch.metadata["previous_event_type"] == "failed_public_check"
    assert branch.original_action == {"tool": "read_file", "args": {"path": "task.json"}}