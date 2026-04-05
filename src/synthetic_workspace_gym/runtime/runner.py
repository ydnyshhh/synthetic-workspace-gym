from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from shutil import copytree

from synthetic_workspace_gym.agents.base import BaseAgent
from synthetic_workspace_gym.analysis.artifacts import (
    build_unified_diff,
    export_episode_artifacts,
    snapshot_texts,
)
from synthetic_workspace_gym.evaluators.registry import get_evaluator
from synthetic_workspace_gym.runtime.environment import LoadedEnvironment
from synthetic_workspace_gym.runtime.tools import WorkspaceToolExecutor
from synthetic_workspace_gym.schemas import ActionType, EpisodeSummary, ToolObservation, ToolState, TrajectoryEvent, utc_timestamp
from synthetic_workspace_gym.utils.scratch import scratch_directory


@dataclass(slots=True)
class EpisodeRunner:
    output_root: Path

    def run_episode(self, environment: LoadedEnvironment, agent: BaseAgent) -> EpisodeSummary:
        evaluator = get_evaluator(
            environment.manifest.family,
            evaluator_entrypoint=environment.manifest.evaluator_entrypoint,
        )
        episode_id = f"{environment.manifest.env_id}-{agent.name}-{int(time.time() * 1000)}"
        artifact_root = self.output_root / episode_id
        started = time.perf_counter()

        scratch_root = self.output_root / ".tmp"
        scratch_root.mkdir(parents=True, exist_ok=True)
        with scratch_directory(scratch_root, "swg-episode-") as tmp_dir:
            workspace = tmp_dir / "workspace"
            copytree(environment.visible_root, workspace)
            executor = WorkspaceToolExecutor(workspace, environment.manifest.tool_permissions)
            trajectory: list[TrajectoryEvent] = []
            initial_snapshot = snapshot_texts(workspace)
            initial_observation = {
                "instruction": environment.manifest.instruction,
                "top_level_files": sorted(item.name for item in workspace.iterdir()),
            }
            agent.reset(environment.manifest, initial_observation)
            observation: ToolObservation | dict[str, object] = initial_observation
            submitted = False
            recent_files: list[str] = []
            touched_files: set[str] = set()
            last_exit_code: int | None = None

            for step_index in range(environment.manifest.max_steps):
                elapsed = time.perf_counter() - started
                remaining_time_seconds = environment.manifest.time_limit_seconds - elapsed
                if remaining_time_seconds <= 0:
                    break
                state = ToolState(
                    step_index=step_index,
                    remaining_steps=environment.manifest.max_steps - step_index,
                    available_tools=environment.manifest.tool_permissions.enabled_tools(),
                    recent_files=recent_files,
                    last_exit_code=last_exit_code,
                    submitted=submitted,
                )
                action = agent.act(observation, state)
                observation = executor.execute(action, remaining_time_seconds=remaining_time_seconds)
                recent_files = observation.touched_files
                touched_files.update(observation.touched_files)
                last_exit_code = observation.exit_code
                trajectory.append(
                    TrajectoryEvent(
                        step_index=step_index,
                        timestamp=utc_timestamp(),
                        action_type=action.action_type,
                        action_arguments=action.arguments,
                        observation_summary=observation.message,
                        stdout=observation.stdout,
                        stderr=observation.stderr,
                        exit_code=observation.exit_code,
                        files_touched=observation.touched_files,
                        workspace_digest=observation.workspace_digest or executor.workspace_digest,
                        success=observation.success,
                    )
                )
                if action.action_type == ActionType.SUBMIT:
                    submitted = True
                    break

            evaluator_result = evaluator.evaluate(workspace, environment.manifest, environment.hidden_root)
            duration = time.perf_counter() - started
            final_snapshot = snapshot_texts(workspace)
            final_diff = build_unified_diff(initial_snapshot, final_snapshot)
            summary = EpisodeSummary(
                episode_id=episode_id,
                env_id=environment.manifest.env_id,
                agent_name=agent.name,
                step_count=len(trajectory),
                submitted=submitted,
                duration_seconds=duration,
                files_touched=sorted(touched_files),
                evaluation=evaluator_result,
                artifact_root=str(artifact_root),
            )
            export_episode_artifacts(
                artifact_root,
                manifest=environment.manifest,
                trajectory=trajectory,
                evaluator_result=evaluator_result,
                summary=summary,
                final_workspace=workspace,
                final_diff=final_diff,
            )
            return summary
