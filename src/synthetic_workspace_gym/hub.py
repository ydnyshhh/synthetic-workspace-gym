from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from synthetic_workspace_gym.prime.env import SyntheticWorkspacePrimeEnv
from synthetic_workspace_gym.prime.tools import get_tool_schemas
from synthetic_workspace_gym.sandbox.schemas import SandboxConfig
from synthetic_workspace_gym.splits.schemas import VALID_SPLITS, normalize_split_name
from synthetic_workspace_gym.verifiers.compat import vf
from synthetic_workspace_gym.verifiers.dataset import SWGVerifiersDataset
from synthetic_workspace_gym.verifiers.env import SYSTEM_PROMPT, SyntheticWorkspaceVerifiersEnv
from synthetic_workspace_gym.verifiers.parser import SWGToolCallParser
from synthetic_workspace_gym.verifiers.rewards import compute_reward, normalize_reward_payload, to_verifiers_info


DEFAULT_ENV_ID = "synthetic-workspace-gym"
HUB_SYSTEM_PROMPT = (
    f"{SYSTEM_PROMPT}\n\n"
    "SWG tasks are budgeted workspace repair tasks. Solve efficiently: inspect task.json, README.md, "
    "and only the files needed to understand the requested behavior. Use relative paths only. Do not "
    "look for or ask about hidden tests, evaluator files, or absolute testbed paths; hidden tests are "
    "represented by the public task description and the visible workspace contract. Prefer small, "
    "targeted edits over broad rewrites. After an edit, run one focused public check when useful. "
    "If the check passes or the fix is clearly complete, call submit immediately instead of continuing "
    "to search for more evidence. Keep reasoning brief and spend turns on tool actions.\n\n"
    "If native tool calling is unavailable, respond with exactly one JSON object and no extra text: "
    '{"tool":"list_directory","args":{"path":"."}}. '
    "Available tools are read_file, write_file, append_file, list_directory, run_shell, run_python, and submit."
)


def load_environment(
    split: str | None = "train",
    family: str | None = None,
    scenario: str | None = None,
    difficulty: int | None = None,
    seed: int | None = None,
    families: str | list[str] | tuple[str, ...] | None = None,
    difficulties: str | list[int] | tuple[int, ...] | None = None,
    seeds: str | list[int] | tuple[int, ...] | None = None,
    split_manifest_path: str | None = None,
    include_splits: str | list[str] | tuple[str, ...] | None = None,
    exclude_splits: str | list[str] | tuple[str, ...] | None = None,
    task_id: str | None = None,
    max_examples: int = -1,
    max_turns: int = 12,
    max_tool_steps: int | None = None,
    sandbox_backend: str = "local",
    docker_image: str | None = None,
    reward_mode: str = "score",
    output_dir: str | None = None,
    env_id: str = DEFAULT_ENV_ID,
) -> object:
    """Load SWG as a Prime Intellect / Verifiers Environment Hub package.

    Prime's hosted evals and hosted training pass JSON ``env_args`` directly to
    this function. The arguments intentionally stay JSON-friendly so runs can
    select SWG splits, task families, scenarios, and small smoke subsets from
    the Prime dashboard or CLI.
    """

    rows = _build_rows(
        split=split,
        family=family,
        scenario=scenario,
        difficulty=difficulty,
        seed=seed,
        families=families,
        difficulties=difficulties,
        seeds=seeds,
        split_manifest_path=split_manifest_path,
        include_splits=include_splits,
        exclude_splits=exclude_splits,
        task_id=task_id,
        max_examples=max_examples,
    )
    env_args = {
        "split": split,
        "family": family,
        "scenario": scenario,
        "difficulty": difficulty,
        "seed": seed,
        "families": families,
        "difficulties": difficulties,
        "seeds": seeds,
        "split_manifest_path": split_manifest_path,
        "include_splits": include_splits,
        "exclude_splits": exclude_splits,
        "task_id": task_id,
        "max_examples": max_examples,
        "max_turns": max_turns,
        "max_tool_steps": max_tool_steps,
        "sandbox_backend": sandbox_backend,
        "docker_image": docker_image,
        "reward_mode": reward_mode,
        "output_dir": output_dir,
    }

    if _native_hub_available():
        return SyntheticWorkspaceHubEnv(
            rows=rows,
            env_id=env_id,
            env_args=env_args,
            max_turns=max_turns,
            max_tool_steps=max_tool_steps,
            sandbox_backend=sandbox_backend,
            docker_image=docker_image,
            reward_mode=reward_mode,
            output_dir=output_dir,
        )

    first = rows[0] if rows else {}
    return SyntheticWorkspaceVerifiersEnv(
        family=str(first.get("family") or family or "script_repair"),
        scenario=first.get("scenario") or scenario,
        difficulty=int(first.get("difficulty") or difficulty or 3),
        seed=int(first.get("seed") or seed or 0),
        environment_path=first.get("environment_path"),
        sandbox_backend=sandbox_backend,
        docker_image=docker_image,
        reward_mode=reward_mode,
        max_turns=max_turns,
        output_dir=output_dir,
    )


class SyntheticWorkspaceHubEnv:  # pragma: no cover - class is exercised when verifiers is installed.
    pass


def _native_hub_available() -> bool:
    if vf is None:
        return False
    try:
        from verifiers.envs.multiturn_env import MultiTurnEnv  # noqa: F401
        from verifiers.rubrics.rubric import Rubric  # noqa: F401
        from verifiers.types import ToolMessage  # noqa: F401
    except Exception:
        return False
    return True


if _native_hub_available():
    from verifiers.envs.multiturn_env import MultiTurnEnv
    from verifiers.rubrics.rubric import Rubric
    from verifiers.types import ToolMessage
    from verifiers.utils.message_utils import normalize_messages

    class SyntheticWorkspaceHubEnv(MultiTurnEnv):  # type: ignore[no-redef]
        """Native multi-turn Verifiers environment for hosted SWG runs."""

        def __init__(
            self,
            *,
            rows: list[dict[str, Any]],
            env_id: str,
            env_args: dict[str, Any],
            max_turns: int,
            max_tool_steps: int | None,
            sandbox_backend: str,
            docker_image: str | None,
            reward_mode: str,
            output_dir: str | None,
        ) -> None:
            self.rows = [dict(row) for row in rows]
            self.sandbox_backend = sandbox_backend
            self.docker_image = docker_image
            self.reward_mode = reward_mode
            self.output_dir = Path(output_dir).resolve() if output_dir else None
            self.max_tool_steps = _resolve_max_tool_steps(max_turns, max_tool_steps)
            super().__init__(
                dataset=_to_dataset(self.rows),
                system_prompt=HUB_SYSTEM_PROMPT,
                rubric=Rubric(funcs=[swg_reward]),
                tool_defs=get_tool_schemas(),
                max_turns=max_turns,
                env_id=env_id,
                env_args=env_args,
            )

        async def setup_state(self, state: Any) -> None:
            row = dict(state.get("input", {}) or {})
            sandbox_config = SandboxConfig(backend=self.sandbox_backend)
            if self.docker_image is not None:
                sandbox_config.image = self.docker_image
            output_dir = None
            if self.output_dir is not None:
                output_dir = self.output_dir / str(state.get("trajectory_id", "rollout"))
            env = SyntheticWorkspacePrimeEnv(
                family=str(row.get("family") or "script_repair"),
                scenario=row.get("scenario"),
                difficulty=int(row.get("difficulty") or 3),
                seed=int(row.get("seed") or 0),
                max_steps=self.max_tool_steps,
                workspace_root=row.get("environment_path"),
                output_dir=output_dir,
                sandbox_backend=self.sandbox_backend,
                sandbox_config=sandbox_config,
                docker_image=self.docker_image,
            )
            observation = env.reset()
            state["swg_env"] = env
            state["swg_reset"] = observation
            state["swg_reward_payload"] = None
            state["swg_reward_mode"] = self.reward_mode
            state["swg_has_written"] = False
            state["swg_successful_check_after_write"] = False
            state["prompt"] = normalize_messages(
                [
                    {"role": "system", "content": HUB_SYSTEM_PROMPT},
                    {"role": "user", "content": str(observation.get("instruction", ""))},
                ],
                field_name="swg.prompt",
            )
            tool_schemas = observation.get("tool_schemas")
            if isinstance(tool_schemas, list) and tool_schemas:
                state["tool_defs"] = self._normalize_tool_defs(tool_schemas) or []

        async def env_response(self, messages: Any, state: Any, **kwargs: Any) -> Any:
            last_msg = messages[-1]
            tool_calls = getattr(last_msg, "tool_calls", None)
            if not tool_calls:
                action = _text_action_or_none(getattr(last_msg, "content", None))
                if action is None:
                    return normalize_messages(
                        [
                            {
                                "role": "user",
                                "content": (
                                    "No tool call was provided. Respond with exactly one JSON tool action, "
                                    'for example {"tool":"list_directory","args":{"path":"."}}. '
                                    "Do not answer in prose; use tools to inspect, edit, check, and submit."
                                ),
                            }
                        ],
                        field_name="swg.format_correction",
                    )
                content, done = _execute_swg_action(state, action)
                response = normalize_messages(
                    [{"role": "user", "content": f"Observation:\n{content}"}],
                    field_name="swg.text_tool_response",
                )
                if done:
                    state["final_env_response"] = response
                return response

            tool_messages = []
            for tool_call in tool_calls:
                content, done = _execute_swg_action(state, _tool_call_to_action(tool_call))
                tool_messages.append(
                    ToolMessage(
                        role="tool",
                        content=content,
                        tool_call_id=str(getattr(tool_call, "id", "")),
                    )
                )
                if done:
                    state["final_env_response"] = tool_messages
                    break
            return tool_messages

        async def cleanup(self, state: Any, task: object | None = None, resources: object | None = None) -> None:
            await super().cleanup(state, task=task, resources=resources)
            env = state.get("swg_env")
            if env is None:
                return
            try:
                if state.get("swg_reward_payload") is None:
                    state["swg_reward_payload"] = env.evaluate()
                payload = normalize_reward_payload(state["swg_reward_payload"])
                state["swg_reward_payload"] = payload
                state["swg_verifiers_info"] = to_verifiers_info(payload)
            finally:
                env.close()


async def swg_reward(state: Any, **kwargs: Any) -> float:
    payload = state.get("swg_reward_payload")
    if payload is None:
        return 0.0
    normalized = normalize_reward_payload(payload)
    return compute_reward(normalized, mode=str(state.get("swg_reward_mode") or "score"))


def _build_rows(
    *,
    split: str | None,
    family: str | None,
    scenario: str | None,
    difficulty: int | None,
    seed: int | None,
    families: str | list[str] | tuple[str, ...] | None,
    difficulties: str | list[int] | tuple[int, ...] | None,
    seeds: str | list[int] | tuple[int, ...] | None,
    split_manifest_path: str | None,
    include_splits: str | list[str] | tuple[str, ...] | None,
    exclude_splits: str | list[str] | tuple[str, ...] | None,
    task_id: str | None,
    max_examples: int,
) -> list[dict[str, Any]]:
    family_values = _coerce_str_list(families) or ([family] if family else None)
    difficulty_values = _coerce_int_list(difficulties) or ([difficulty] if difficulty is not None else None)
    seed_values = _coerce_int_list(seeds) or ([seed] if seed is not None else None)
    scenarios = {family: [scenario]} if family and scenario else None
    dataset = SWGVerifiersDataset(
        families=tuple(family_values or ("tabular", "script_repair", "pipeline", "retrieval_workspace")),
        scenarios=scenarios,
        difficulties=tuple(difficulty_values or (1, 2, 3, 4, 5)),
        seeds=tuple(seed_values or range(100)),
        split=_official_split(split),
        split_manifest_path=split_manifest_path,
        include_splits=_coerce_str_list(include_splits),
        exclude_splits=_coerce_str_list(exclude_splits),
    )
    rows = dataset.to_list()
    if task_id is not None:
        rows = [row for row in rows if row.get("task_id") == task_id]
    if max_examples > 0:
        rows = rows[:max_examples]
    if not rows:
        raise ValueError("SWG load_environment produced no task rows for the requested arguments.")
    return [_with_prompt(row) for row in rows]


def _with_prompt(row: dict[str, Any]) -> dict[str, Any]:
    payload = dict(row)
    payload["prompt"] = [{"role": "user", "content": str(payload.get("question") or payload.get("task_id"))}]
    return payload


def _to_dataset(rows: list[dict[str, Any]]) -> object:
    from datasets import Dataset  # type: ignore[import-not-found]

    return Dataset.from_list(rows)


def _tool_call_to_action(tool_call: object) -> dict[str, object]:
    name = str(getattr(tool_call, "name", ""))
    raw_args = getattr(tool_call, "arguments", "{}")
    try:
        args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
    except Exception:
        args = {}
    if not isinstance(args, dict):
        args = {"value": args}
    return {"tool": name, "args": args}


def _text_action_or_none(content: object) -> dict[str, object] | None:
    if content is None:
        return None
    text = str(content).strip()
    if not text:
        return None
    if not (text.startswith("{") or "```" in text):
        return None
    action = SWGToolCallParser().parse(text)
    if action.get("parse_error"):
        return None
    return action


def _execute_swg_action(state: Any, action: dict[str, object]) -> tuple[str, bool]:
    env = state["swg_env"]
    result = env.step(action)
    content = str(result.get("observation", ""))
    info = dict(result.get("info", {}) or {})
    content = _with_efficiency_guidance(state, action, content, info)
    if info.get("reward_payload") is not None:
        state["swg_reward_payload"] = info["reward_payload"]
        state["swg_verifiers_info"] = to_verifiers_info(normalize_reward_payload(info["reward_payload"]))
    return content, bool(result.get("done", False))


def _with_efficiency_guidance(
    state: Any,
    action: dict[str, object],
    content: str,
    info: dict[str, object],
) -> str:
    tool_name = str(action.get("tool", ""))
    success = bool(info.get("success"))
    guidance: str | None = None

    if tool_name in {"write_file", "append_file"}:
        if success:
            state["swg_has_written"] = True
            guidance = (
                "Guidance: You changed the workspace. Run one focused public check if needed; "
                "if the fix is verified or clearly complete, call submit next. Do not search for hidden tests."
            )
        else:
            guidance = (
                "Guidance: The edit did not succeed. Use the error to make the smallest corrective action, "
                "then continue with the repair."
            )
    elif tool_name in {"run_shell", "run_python"} and state.get("swg_has_written"):
        if success:
            state["swg_successful_check_after_write"] = True
            guidance = (
                "Guidance: A check ran after your edit. If this check supports the fix, call submit next "
                "instead of continuing broad verification. Do not look for hidden tests."
            )
        else:
            guidance = (
                "Guidance: Use this check output to make a targeted follow-up edit, then rerun only the "
                "focused check needed to confirm the fix."
            )

    if guidance is None:
        return content
    return f"{content}\n\n{guidance}" if content else guidance


def _resolve_max_tool_steps(max_turns: int, max_tool_steps: int | None) -> int:
    if max_tool_steps is not None:
        return max(1, int(max_tool_steps))
    if max_turns > 0:
        return max(24, int(max_turns) * 4)
    return 24


def _coerce_str_list(value: str | list[str] | tuple[str, ...] | None) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item) for item in value]


def _coerce_int_list(value: str | list[int] | tuple[int, ...] | None) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [int(item.strip()) for item in value.split(",") if item.strip()]
    return [int(item) for item in value]


def _official_split(split: str | None) -> str | None:
    if split is None:
        return None
    normalized = normalize_split_name(split)
    return normalized if normalized in VALID_SPLITS else split
