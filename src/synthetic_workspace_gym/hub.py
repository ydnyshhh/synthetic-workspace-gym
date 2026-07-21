from __future__ import annotations

import asyncio
import json
import random
import threading
from pathlib import Path
from typing import Any

from synthetic_workspace_gym.prime.env import SyntheticWorkspacePrimeEnv
from synthetic_workspace_gym.prime.tools import get_tool_schemas
from synthetic_workspace_gym.sandbox.schemas import SandboxConfig
from synthetic_workspace_gym.splits.schemas import VALID_SPLITS, normalize_split_name
from synthetic_workspace_gym.verifiers.compat import vf
from synthetic_workspace_gym.verifiers.dataset import SWGVerifiersDataset
from synthetic_workspace_gym.verifiers.env import (
    SYSTEM_PROMPT,
    SyntheticWorkspaceVerifiersEnv,
)
from synthetic_workspace_gym.verifiers.messages import (
    to_verifiers_branch_messages as _to_verifiers_branch_messages,
    to_verifiers_tool_exchange as _to_verifiers_tool_exchange,
)
from synthetic_workspace_gym.verifiers.parser import SWGToolCallParser
from synthetic_workspace_gym.verifiers.rewards import (
    compute_reward,
    normalize_reward_payload,
    to_verifiers_info,
)


DEFAULT_ENV_ID = "synthetic-workspace-gym"
HUB_SYSTEM_PROMPT = (
    f"{SYSTEM_PROMPT}\n\n"
    "SWG tasks are budgeted workspace repair tasks. Solve efficiently: inspect task.json, README.md, "
    "and only the files needed to understand the requested behavior. Use relative paths only. Do not "
    "look for or ask about hidden tests, evaluator files, or absolute testbed paths; hidden tests are "
    "represented by the public task description and the visible workspace contract. Prefer small, "
    "targeted edits over broad rewrites. After an edit, run one focused public check when useful. "
    "If the check passes or the fix is clearly complete, call submit immediately instead of continuing "
    "to search for more evidence. Never stop after only writing a file; a repair is complete only after "
    "the required artifact exists or the relevant public check has run, followed by submit. Keep reasoning "
    "brief and spend turns on tool actions.\n\n"
    "Family protocols:\n"
    "- Tabular: read README.md, task.json, and the listed input files; write or edit a processing script only "
    "if useful; run it with run_python; read the required output JSON; submit the required output path.\n"
    "- Pipeline: inspect task.json, README.md, config, entrypoint, and relevant src files; make targeted edits; "
    "run the public entrypoint with run_shell; read the required artifact; submit the required artifact path.\n"
    "- Script repair: inspect task.json and target files; make the smallest source edit; run the public "
    "entrypoint when useful; submit the changed target file.\n"
    "- Retrieval workspace: inspect the named document roots; update the target artifact from visible evidence; "
    "submit the target artifact path.\n\n"
    "Tool-use constraints: run_python accepts only a workspace-relative Python script path such as "
    "`scripts/check.py` or `process_report.py`; do not pass inline Python, `python -c`, `python -m`, "
    "or `python script.py` to run_python. If you need Python logic, first create a small script with "
    "write_file, include any needed output-directory creation inside that script, then run it with "
    "run_python using only the script path. Use only the Python standard library unless a dependency is "
    "already visible in the workspace. Shell commands must use relative paths only. Native tool calling "
    "may include one or more calls in a model turn; SWG executes those calls sequentially, and every call "
    "consumes the separate tool-step budget. `max_turns` limits model responses, while `max_tool_steps` "
    "limits executed workspace actions.\n\n"
    "If native tool calling is unavailable, respond with exactly one JSON object and no extra text: "
    '{"tool":"list_directory","args":{"path":"."}}. '
    "Available tools are read_file, write_file, append_file, list_directory, run_shell, run_python, and submit."
)

_TRANSIENT_SANDBOX_ERROR_NAMES = {
    "APIConnectionError",
    "ConnectError",
    "ConnectTimeout",
    "DownloadTimeoutError",
    "NetworkError",
    "PoolTimeout",
    "ReadError",
    "ReadTimeout",
    "RemoteProtocolError",
    "UploadTimeoutError",
    "WriteError",
    "WriteTimeout",
}


def _is_transient_sandbox_failure(exc: BaseException) -> bool:
    """Recognize remote-sandbox transport failures through wrapped cause chains."""
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if current.__class__.__name__ in _TRANSIENT_SANDBOX_ERROR_NAMES:
            return True
        if any(name in str(current) for name in _TRANSIENT_SANDBOX_ERROR_NAMES):
            return True
        current = current.__cause__ or current.__context__
    return False


async def _run_blocking_swg_operation(operation: Any, *args: Any) -> Any:
    """Keep synchronous sandbox I/O off-loop and expose transient failures to retries."""
    try:
        return await asyncio.to_thread(operation, *args)
    except Exception as exc:
        if _is_transient_sandbox_failure(exc) and vf is not None:
            raise vf.InfraError(
                f"Transient SWG sandbox infrastructure failure: {exc}"
            ) from exc
        raise


def load_environment(
    split: str | None = "train",
    family: str | None = None,
    scenario: str | None = None,
    difficulty: int | None = None,
    seed: int | None = None,
    composition_mode: str | None = None,
    families: str | list[str] | tuple[str, ...] | None = None,
    difficulties: str | list[int] | tuple[int, ...] | None = None,
    seeds: str | list[int] | tuple[int, ...] | None = None,
    split_manifest_path: str | None = None,
    include_splits: str | list[str] | tuple[str, ...] | None = None,
    exclude_splits: str | list[str] | tuple[str, ...] | None = None,
    task_id: str | None = None,
    max_examples: int = -1,
    sample_strategy: str = "first",
    shuffle: bool | str = False,
    shuffle_seed: int = 0,
    max_turns: int = 12,
    max_tool_steps: int | None = None,
    time_limit_seconds: int | None = None,
    max_observation_chars: int | None = 20000,
    sandbox_backend: str = "local",
    docker_image: str | None = None,
    reward_mode: str = "score",
    output_dir: str | None = None,
    env_id: str = DEFAULT_ENV_ID,
    branch_manifest_path: str | None = None,
    branch_task_id: str | None = None,
    branch_mode: str | None = None,
) -> object:
    """Load SWG as a Prime Intellect / Verifiers Environment Hub package.

    Prime's hosted evals and hosted training pass JSON ``env_args`` directly to
    this function. The arguments intentionally stay JSON-friendly so runs can
    select SWG splits, task families, scenarios, and small smoke subsets from
    the Prime dashboard or CLI.
    """

    if branch_manifest_path is not None:
        rows = _build_branch_rows(
            branch_manifest_path=branch_manifest_path,
            branch_task_id=branch_task_id,
            branch_mode=branch_mode,
            max_examples=max_examples,
            sample_strategy=sample_strategy,
            shuffle=shuffle,
            shuffle_seed=shuffle_seed,
        )
    else:
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
            sample_strategy=sample_strategy,
            shuffle=shuffle,
            shuffle_seed=shuffle_seed,
        )
    env_args = {
        "split": split,
        "family": family,
        "scenario": scenario,
        "difficulty": difficulty,
        "seed": seed,
        "composition_mode": composition_mode,
        "families": families,
        "difficulties": difficulties,
        "seeds": seeds,
        "split_manifest_path": split_manifest_path,
        "include_splits": include_splits,
        "exclude_splits": exclude_splits,
        "task_id": task_id,
        "max_examples": max_examples,
        "sample_strategy": sample_strategy,
        "shuffle": shuffle,
        "shuffle_seed": shuffle_seed,
        "max_turns": max_turns,
        "max_tool_steps": max_tool_steps,
        "time_limit_seconds": time_limit_seconds,
        "max_observation_chars": max_observation_chars,
        "sandbox_backend": sandbox_backend,
        "docker_image": docker_image,
        "reward_mode": reward_mode,
        "output_dir": output_dir,
        "branch_manifest_path": branch_manifest_path,
        "branch_task_id": branch_task_id,
        "branch_mode": branch_mode,
    }

    if _native_hub_available():
        return SyntheticWorkspaceHubEnv(
            rows=rows,
            env_id=env_id,
            env_args=env_args,
            max_turns=max_turns,
            max_tool_steps=max_tool_steps,
            time_limit_seconds=time_limit_seconds,
            max_observation_chars=max_observation_chars,
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
        composition_mode=composition_mode,
        environment_path=first.get("environment_path"),
        sandbox_backend=sandbox_backend,
        docker_image=docker_image,
        reward_mode=reward_mode,
        max_turns=int(first.get("remaining_steps") or max_turns),
        time_limit_seconds=int(first.get("time_limit_seconds") or time_limit_seconds)
        if (first.get("time_limit_seconds") or time_limit_seconds) is not None
        else None,
        output_dir=output_dir,
        branch_manifest_path=branch_manifest_path,
        branch_task_id=branch_task_id,
        branch_mode=branch_mode,
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
            time_limit_seconds: int | None,
            max_observation_chars: int | None,
            sandbox_backend: str,
            docker_image: str | None,
            reward_mode: str,
            output_dir: str | None,
        ) -> None:
            self.rows = [dict(row) for row in rows]
            self.composition_mode = env_args.get("composition_mode")
            self.sandbox_backend = sandbox_backend
            self.docker_image = docker_image
            self.reward_mode = reward_mode
            self.output_dir = Path(output_dir).resolve() if output_dir else None
            self.max_tool_steps = _resolve_max_tool_steps(max_turns, max_tool_steps)
            self.time_limit_seconds = _resolve_time_limit_seconds(
                max_turns, time_limit_seconds
            )
            self.max_observation_chars = _resolve_max_observation_chars(
                max_observation_chars
            )
            self._row_cursor = 0
            self._row_lock = threading.Lock()
            dataset = _to_dataset(self.rows)
            super().__init__(
                dataset=dataset,
                eval_dataset=dataset,
                system_prompt=HUB_SYSTEM_PROMPT,
                rubric=Rubric(funcs=[swg_reward]),
                tool_defs=get_tool_schemas(),
                max_turns=max_turns,
                env_id=env_id,
                env_args=env_args,
            )

        async def setup_state(self, state: Any) -> None:
            row = _state_row(state)
            if not _has_task_identity(row):
                row = self._fallback_row(state)
            sandbox_config = SandboxConfig(backend=self.sandbox_backend)
            if self.docker_image is not None:
                sandbox_config.image = self.docker_image
            output_dir = None
            if self.output_dir is not None:
                output_dir = self.output_dir / str(
                    state.get("trajectory_id", "rollout")
                )
            env = SyntheticWorkspacePrimeEnv(
                family=str(row.get("family") or "script_repair"),
                scenario=row.get("scenario"),
                difficulty=int(row.get("difficulty") or 3),
                seed=int(row.get("seed") or 0),
                composition_mode=self.composition_mode,
                max_steps=int(row.get("remaining_steps") or self.max_tool_steps),
                time_limit_seconds=int(
                    row.get("time_limit_seconds") or self.time_limit_seconds
                ),
                workspace_root=row.get("environment_path"),
                output_dir=output_dir,
                sandbox_backend=self.sandbox_backend,
                sandbox_config=sandbox_config,
                docker_image=self.docker_image,
            )
            # Prime sandbox setup performs synchronous network and archive I/O.
            # Running it on the Verifiers event-loop thread prevents worker
            # heartbeats and causes the router to kill otherwise healthy tasks.
            observation = await _run_blocking_swg_operation(env.reset)
            state["swg_env"] = env
            state["swg_reset"] = observation
            state["swg_task"] = _task_metadata(row)
            state["swg_task_row"] = row
            state["swg_reward_payload"] = None
            state["swg_reward_mode"] = self.reward_mode
            state["swg_max_observation_chars"] = self.max_observation_chars
            state["swg_has_written"] = False
            state["swg_successful_check_after_write"] = False
            state["swg_required_output_path"] = _required_output_path(observation)
            state["swg_entrypoint"] = _entrypoint(observation)
            state["swg_target_files"] = _target_files(observation)
            prefix_messages = row.get("prefix_messages")
            if isinstance(prefix_messages, list):
                prompt = _to_verifiers_branch_messages(
                    [dict(message) for message in prefix_messages]
                )
                if not prompt or prompt[0].get("role") != "system":
                    prompt.insert(0, {"role": "system", "content": HUB_SYSTEM_PROMPT})
            else:
                prompt = [
                    {"role": "system", "content": HUB_SYSTEM_PROMPT},
                    {"role": "user", "content": _task_user_prompt(row, observation)},
                ]
            state["swg_branch"] = _branch_metadata(row)
            forced_action = (
                row.get("forced_action") if row.get("branch_mode") == "forced" else None
            )
            state["swg_forced_action"] = forced_action
            if isinstance(forced_action, dict):
                (
                    forced_content,
                    forced_done,
                    forced_info,
                ) = await _run_blocking_swg_operation(
                    _execute_forced_swg_action, state, forced_action
                )
                forced_call_id = "counterfactual-forced-action"
                forced_messages = _to_verifiers_tool_exchange(
                    forced_action,
                    forced_content,
                    forced_call_id,
                    metadata={"forced": True},
                )
                prompt.extend(forced_messages)
                state["swg_forced_prefix_length"] = len(forced_messages)
                state["swg_policy_start_message_index"] = len(prompt)
                state["swg_loss_mask_metadata"] = {
                    "exclude_restored_messages": True,
                    "exclude_forced_messages": True,
                    "forced_tool_call_id": forced_call_id,
                }
                state["swg_forced_action_result"] = {
                    "observation": forced_content,
                    "done": forced_done,
                    "info": forced_info,
                    "success": bool(
                        forced_info.get(
                            "success", forced_info.get("reward_payload") is not None
                        )
                    ),
                }
                if forced_done:
                    state["final_env_response"] = normalize_messages(
                        [forced_messages[1]],
                        field_name="swg.forced_response",
                    )
            state.setdefault("swg_forced_prefix_length", 0)
            state.setdefault("swg_policy_start_message_index", len(prompt))
            state.setdefault(
                "swg_loss_mask_metadata",
                {
                    "exclude_restored_messages": True,
                    "exclude_forced_messages": False,
                    "forced_tool_call_id": None,
                },
            )
            state["prompt"] = normalize_messages(prompt, field_name="swg.prompt")
            tool_schemas = observation.get("tool_schemas")
            if isinstance(tool_schemas, list) and tool_schemas:
                state["tool_defs"] = self._normalize_tool_defs(tool_schemas) or []

        def _fallback_row(self, state: Any) -> dict[str, Any]:
            index = _state_example_index(state)
            if index is None:
                with self._row_lock:
                    index = self._row_cursor
                    self._row_cursor += 1
            if not self.rows:
                return {}
            return dict(self.rows[int(index) % len(self.rows)])

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
                content, done = await _run_blocking_swg_operation(
                    _execute_swg_action, state, action
                )
                response = normalize_messages(
                    [{"role": "user", "content": f"Observation:\n{content}"}],
                    field_name="swg.text_tool_response",
                )
                if done:
                    state["final_env_response"] = response
                return response

            tool_messages = []
            for tool_call in tool_calls:
                content, done = await _run_blocking_swg_operation(
                    _execute_swg_action, state, _tool_call_to_action(tool_call)
                )
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

        async def cleanup(
            self,
            state: Any,
            task: object | None = None,
            resources: object | None = None,
        ) -> None:
            await super().cleanup(state, task=task, resources=resources)
            env = state.get("swg_env")
            if env is None:
                return
            try:
                if state.get("swg_reward_payload") is None:
                    state["swg_reward_payload"] = await _run_blocking_swg_operation(
                        env.evaluate
                    )
                payload = normalize_reward_payload(state["swg_reward_payload"])
                state["swg_reward_payload"] = payload
                state["swg_verifiers_info"] = to_verifiers_info(payload)
            finally:
                await _run_blocking_swg_operation(env.close)


async def swg_reward(state: Any, **kwargs: Any) -> float:
    payload = state.get("swg_reward_payload")
    if payload is None:
        return 0.0
    normalized = normalize_reward_payload(payload)
    return compute_reward(normalized, mode=str(state.get("swg_reward_mode") or "score"))


def _build_branch_rows(
    *,
    branch_manifest_path: str,
    branch_task_id: str | None,
    branch_mode: str | None,
    max_examples: int,
    sample_strategy: str,
    shuffle: bool | str,
    shuffle_seed: int,
) -> list[dict[str, Any]]:
    from synthetic_workspace_gym.counterfactual.runner import read_branch_manifest

    tasks = read_branch_manifest(Path(branch_manifest_path))
    if branch_task_id is not None:
        tasks = [task for task in tasks if task.task_id == branch_task_id]
    rows = []
    for task in tasks:
        mode = branch_mode or task.mode
        if mode not in {"forced", "open"}:
            raise ValueError("branch_mode must be 'forced' or 'open'")
        if mode == "forced" and task.forced_action is None:
            raise ValueError(f"branch task {task.task_id!r} has no forced action")
        rows.append(
            _with_prompt(
                {
                    "task_id": task.task_id,
                    "environment_path": task.environment_path,
                    "prefix_messages": task.prefix_messages,
                    "forced_action": task.forced_action if mode == "forced" else None,
                    "branch_mode": mode,
                    "remaining_steps": task.remaining_steps,
                    "time_limit_seconds": task.time_limit_seconds,
                    "branch_group_id": task.branch_group_id,
                    "snapshot_id": task.snapshot_id,
                    "candidate_id": task.candidate_id,
                    "family": task.family,
                    "scenario": task.scenario_id,
                    "difficulty": task.difficulty,
                    "seed": task.seed,
                    "metadata": dict(task.metadata),
                    "question": "Continue the counterfactual branch from the restored state.",
                }
            )
        )
    rows = _sample_rows(
        rows,
        max_examples=max_examples,
        sample_strategy=sample_strategy,
        shuffle=_coerce_bool(shuffle),
        shuffle_seed=shuffle_seed,
    )
    if not rows:
        raise ValueError(
            f"SWG branch manifest produced no task rows for branch_task_id={branch_task_id!r}."
        )
    return rows


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
    sample_strategy: str,
    shuffle: bool | str,
    shuffle_seed: int,
) -> list[dict[str, Any]]:
    family_values = _coerce_str_list(families) or ([family] if family else None)
    difficulty_values = _coerce_int_list(difficulties) or (
        [difficulty] if difficulty is not None else None
    )
    seed_values = _coerce_int_list(seeds) or ([seed] if seed is not None else None)
    scenarios = {family: [scenario]} if family and scenario else None
    dataset = SWGVerifiersDataset(
        families=tuple(
            family_values
            or ("tabular", "script_repair", "pipeline", "retrieval_workspace")
        ),
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
    rows = _sample_rows(
        rows,
        max_examples=max_examples,
        sample_strategy=sample_strategy,
        shuffle=_coerce_bool(shuffle),
        shuffle_seed=shuffle_seed,
    )
    if not rows:
        raise ValueError(
            "SWG load_environment produced no task rows for the requested arguments."
        )
    return [_with_prompt(row) for row in rows]


def _sample_rows(
    rows: list[dict[str, Any]],
    *,
    max_examples: int,
    sample_strategy: str,
    shuffle: bool,
    shuffle_seed: int,
) -> list[dict[str, Any]]:
    strategy = str(sample_strategy or "first").strip().lower()
    if strategy in {"balanced", "stratified"}:
        return _balanced_sample_rows(
            rows, max_examples=max_examples, shuffle=shuffle, shuffle_seed=shuffle_seed
        )
    if strategy not in {"first", "head"}:
        raise ValueError(f"Unsupported SWG sample_strategy: {sample_strategy}")
    sampled = [dict(row) for row in rows]
    if shuffle:
        random.Random(int(shuffle_seed)).shuffle(sampled)
    if max_examples > 0:
        sampled = sampled[:max_examples]
    return sampled


def _balanced_sample_rows(
    rows: list[dict[str, Any]],
    *,
    max_examples: int,
    shuffle: bool,
    shuffle_seed: int,
) -> list[dict[str, Any]]:
    if max_examples <= 0:
        sampled = [dict(row) for row in rows]
        if shuffle:
            random.Random(int(shuffle_seed)).shuffle(sampled)
        return sampled

    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row.get("family") or ""),
            str(row.get("scenario") or ""),
            int(row.get("difficulty") or 0),
        )
        grouped.setdefault(key, []).append(dict(row))

    rng = random.Random(int(shuffle_seed))
    keys = sorted(grouped)
    if shuffle:
        rng.shuffle(keys)
        for key in keys:
            rng.shuffle(grouped[key])

    sampled: list[dict[str, Any]] = []
    while len(sampled) < max_examples and keys:
        next_keys: list[tuple[str, str, int]] = []
        for key in keys:
            bucket = grouped[key]
            if not bucket:
                continue
            sampled.append(bucket.pop(0))
            if len(sampled) >= max_examples:
                break
            if bucket:
                next_keys.append(key)
        else:
            keys = next_keys
            continue
        break
    return sampled


def _with_prompt(row: dict[str, Any]) -> dict[str, Any]:
    payload = dict(row)
    payload["prompt"] = [
        {
            "role": "user",
            "content": str(payload.get("question") or payload.get("task_id")),
        }
    ]
    return payload


def _to_dataset(rows: list[dict[str, Any]]) -> object:
    from datasets import Dataset  # type: ignore[import-not-found]

    return Dataset.from_list(rows)


def _state_row(state: Any) -> dict[str, Any]:
    for key in ("input", "task"):
        value = state.get(key)
        if isinstance(value, dict) and value:
            return dict(value)
    return {}


def _has_task_identity(row: dict[str, Any]) -> bool:
    return any(
        row.get(key) is not None
        for key in ("task_id", "family", "scenario", "environment_path")
    )


def _state_example_index(state: Any) -> int | None:
    for key in ("example_id", "example_index", "sample_index", "row_index"):
        value = state.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _task_metadata(row: dict[str, Any]) -> dict[str, object]:
    return {
        "task_id": row.get("task_id"),
        "env_id": row.get("env_id"),
        "split": row.get("split"),
        "family": row.get("family"),
        "scenario": row.get("scenario"),
        "difficulty": row.get("difficulty"),
        "seed": row.get("seed"),
        "environment_path": row.get("environment_path"),
        **_branch_metadata(row),
    }


def _branch_metadata(row: dict[str, Any]) -> dict[str, object]:
    if row.get("branch_group_id") is None:
        return {}
    return {
        **dict(row.get("metadata") or {}),
        "counterfactual": True,
        "branch_group_id": row.get("branch_group_id"),
        "snapshot_id": row.get("snapshot_id"),
        "candidate_id": row.get("candidate_id"),
        "branch_mode": row.get("branch_mode"),
        "remaining_steps": row.get("remaining_steps"),
    }


def _task_user_prompt(row: dict[str, Any], observation: dict[str, object]) -> str:
    metadata = _task_metadata(row)
    lines = ["Task metadata:"]
    for key in ("task_id", "split", "family", "scenario", "difficulty", "seed"):
        value = metadata.get(key)
        if value is not None:
            lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("Instruction:")
    lines.append(
        str(
            observation.get("instruction")
            or row.get("question")
            or "Solve the SWG workspace task."
        )
    )
    required_output = _required_output_path(observation)
    entrypoint = _entrypoint(observation)
    if required_output:
        lines.append("")
        lines.append(f"Required final artifact: {required_output}")
    if entrypoint:
        lines.append(f"Public check/entrypoint: {entrypoint}")
    return "\n".join(lines)


def _task_descriptor(observation: dict[str, object]) -> dict[str, object]:
    metadata = observation.get("metadata")
    if not isinstance(metadata, dict):
        return {}
    descriptor = metadata.get("task_descriptor")
    return dict(descriptor) if isinstance(descriptor, dict) else {}


def _required_output_path(observation: dict[str, object]) -> str | None:
    descriptor = _task_descriptor(observation)
    for key in ("required_output_path", "output_path", "target_path"):
        value = descriptor.get(key)
        if value:
            return str(value)
    metadata = observation.get("metadata")
    if isinstance(metadata, dict):
        layout = metadata.get("visible_artifact_layout")
        if isinstance(layout, dict):
            for key in ("required_output_path", "output_path", "target_path"):
                value = layout.get(key)
                if value:
                    return str(value)
    return None


def _entrypoint(observation: dict[str, object]) -> str | None:
    value = _task_descriptor(observation).get("entrypoint")
    return str(value) if value else None


def _target_files(observation: dict[str, object]) -> list[str]:
    value = _task_descriptor(observation).get("target_files")
    if isinstance(value, list):
        return [str(item) for item in value]
    return []


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


def _execute_forced_swg_action(
    state: Any,
    action: dict[str, object],
) -> tuple[str, bool, dict[str, object]]:
    result = state["swg_env"].step(action)
    content = _truncate_observation(
        str(result.get("observation", "")),
        int(state.get("swg_max_observation_chars") or 0),
    )
    info = dict(result.get("info", {}) or {})
    if info.get("reward_payload") is not None:
        state["swg_reward_payload"] = info["reward_payload"]
        state["swg_verifiers_info"] = to_verifiers_info(
            normalize_reward_payload(info["reward_payload"])
        )
    return content, bool(result.get("done", False)), info


def _execute_swg_action(state: Any, action: dict[str, object]) -> tuple[str, bool]:
    submit_correction = _submit_correction_or_none(state, action)
    if submit_correction is not None:
        return submit_correction, False

    env = state["swg_env"]
    result = env.step(action)
    content = str(result.get("observation", ""))
    content = _truncate_observation(
        content, int(state.get("swg_max_observation_chars") or 0)
    )
    info = dict(result.get("info", {}) or {})
    content = _with_efficiency_guidance(state, action, content, info)
    if info.get("reward_payload") is not None:
        state["swg_reward_payload"] = info["reward_payload"]
        state["swg_verifiers_info"] = to_verifiers_info(
            normalize_reward_payload(info["reward_payload"])
        )
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
    family = str((state.get("swg_task") or {}).get("family") or "")
    required_output_path = str(state.get("swg_required_output_path") or "")
    entrypoint = str(state.get("swg_entrypoint") or "")
    target_files = [str(item) for item in state.get("swg_target_files", []) or []]
    action_path = _action_path(action)

    if tool_name in {"write_file", "append_file"}:
        if success:
            state["swg_has_written"] = True
            state["swg_last_written_path"] = action_path
            guidance = _post_write_guidance(
                family=family,
                written_path=action_path,
                required_output_path=required_output_path,
                entrypoint=entrypoint,
                target_files=target_files,
            )
        else:
            guidance = (
                "Guidance: The edit did not succeed. Use the error to make the smallest corrective action, "
                "then continue with the repair."
            )
    elif tool_name in {"run_shell", "run_python"} and state.get("swg_has_written"):
        if success:
            state["swg_successful_check_after_write"] = True
            guidance = _post_check_guidance(
                state,
                family=family,
                required_output_path=required_output_path,
                target_files=target_files,
            )
        else:
            guidance = (
                "Guidance: Use this check output to make a targeted follow-up edit, then rerun only the "
                "focused check needed to confirm the fix."
            )

    if guidance is None:
        return content
    return f"{content}\n\n{guidance}" if content else guidance


def _post_write_guidance(
    *,
    family: str,
    written_path: str,
    required_output_path: str,
    entrypoint: str,
    target_files: list[str],
) -> str:
    if family == "tabular":
        if written_path.endswith(".py") and written_path != required_output_path:
            return (
                "Guidance: You wrote a processing script, but the required artifact is not complete until "
                f"`{required_output_path}` exists. Next run `run_python` on `{written_path}`, read "
                f"`{required_output_path}`, then call `submit` with `{required_output_path}`. "
                "Do not stop after writing the script."
            )
        if written_path == required_output_path:
            return (
                f"Guidance: You wrote the required artifact `{required_output_path}`. Read it if you need "
                f"one quick validation, then call `submit` with `{required_output_path}`."
            )
        return (
            f"Guidance: You changed the workspace. For tabular tasks, the required final artifact is "
            f"`{required_output_path}`; run the focused script/check that creates it, read it, then submit it."
        )
    if family == "pipeline":
        run_hint = f"`{entrypoint}`" if entrypoint else "the public entrypoint"
        return (
            "Guidance: You changed the pipeline. Next run the public entrypoint "
            f"{run_hint} with `run_shell`, read `{required_output_path}`, then call `submit` with "
            f"`{required_output_path}`. Do not stop after editing config or source."
        )
    if family == "retrieval_workspace":
        target = required_output_path or written_path
        return (
            f"Guidance: You changed the retrieval artifact. If `{target}` reflects the visible evidence, "
            f"call `submit` with `{target}` now; otherwise make one targeted correction."
        )
    if family == "script_repair":
        target = target_files[0] if target_files else written_path
        check = (
            f" Run `{entrypoint}` with `run_shell` if you need one focused check."
            if entrypoint
            else ""
        )
        return (
            f"Guidance: You changed the repair target.{check} If the fix is complete, call `submit` "
            f"with `{target}` instead of continuing broad verification."
        )
    return (
        "Guidance: You changed the workspace. Run one focused public check if needed; "
        "if the fix is verified or clearly complete, call submit next. Do not search for hidden tests."
    )


def _post_check_guidance(
    state: Any,
    *,
    family: str,
    required_output_path: str,
    target_files: list[str],
) -> str:
    if family in {"tabular", "pipeline"} and required_output_path:
        if _workspace_path_exists(state, required_output_path):
            return (
                f"Guidance: A check ran and `{required_output_path}` exists. Read that artifact if needed, "
                f"then call `submit` with `{required_output_path}`. Do not continue broad verification."
            )
        return (
            f"Guidance: A check ran, but the required artifact `{required_output_path}` is not present yet. "
            "Make the smallest targeted fix so the next check creates it."
        )
    if family == "script_repair":
        target = (
            target_files[0]
            if target_files
            else str(state.get("swg_last_written_path") or "")
        )
        return (
            f"Guidance: A check ran after your edit. If this supports the fix, call `submit` with `{target}` "
            "instead of continuing broad verification. Do not look for hidden tests."
        )
    return (
        "Guidance: A check ran after your edit. If this check supports the fix, call submit next "
        "instead of continuing broad verification. Do not look for hidden tests."
    )


def _submit_correction_or_none(state: Any, action: dict[str, object]) -> str | None:
    if str(action.get("tool", "")) != "submit":
        return None
    family = str((state.get("swg_task") or {}).get("family") or "")
    required_output_path = str(state.get("swg_required_output_path") or "")
    if (
        family not in {"tabular", "pipeline", "retrieval_workspace"}
        or not required_output_path
    ):
        return None
    submitted_path = (
        str((action.get("args") or {}).get("path_or_answer", ""))
        .strip()
        .replace("\\", "/")
    )
    required = required_output_path.replace("\\", "/")
    if submitted_path != required:
        return (
            f"Submit correction: this task must submit the required artifact path `{required}`. "
            f"You tried to submit `{submitted_path or '<empty>'}`. If the artifact is ready, call "
            f"`submit` with `{required}`; otherwise create or fix it first."
        )
    if not _workspace_path_exists(state, required):
        return (
            f"Submit correction: `{required}` does not exist yet. Create it by running the focused "
            "script or public entrypoint, read it if needed, then submit that path."
        )
    return None


def _workspace_path_exists(state: Any, relative_path: str) -> bool:
    env = state.get("swg_env")
    if env is None or not relative_path:
        return False
    try:
        return (env.active_workspace / relative_path).exists()
    except Exception:
        return False


def _action_path(action: dict[str, object]) -> str:
    args = action.get("args")
    if not isinstance(args, dict):
        return ""
    for key in ("path", "command_or_script", "path_or_answer"):
        if key in args:
            return str(args.get(key) or "").replace("\\", "/")
    return ""


def _resolve_max_tool_steps(max_turns: int, max_tool_steps: int | None) -> int:
    if max_tool_steps is not None:
        return max(1, int(max_tool_steps))
    if max_turns > 0:
        return max(24, int(max_turns) * 4)
    return 24


def _resolve_time_limit_seconds(max_turns: int, time_limit_seconds: int | None) -> int:
    if time_limit_seconds is not None:
        return max(1, int(time_limit_seconds))
    if max_turns > 0:
        return max(180, int(max_turns) * 12)
    return 180


def _resolve_max_observation_chars(max_observation_chars: int | None) -> int:
    if max_observation_chars is None:
        return 0
    return max(0, int(max_observation_chars))


def _truncate_observation(content: str, limit: int) -> str:
    if limit <= 0 or len(content) <= limit:
        return content
    omitted = len(content) - limit
    return (
        content[:limit]
        + f"\n\n[Observation truncated by SWG after {limit} characters; {omitted} characters omitted. "
        "Use narrower file reads or focused checks when possible.]"
    )


def _coerce_str_list(
    value: str | list[str] | tuple[str, ...] | None,
) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item) for item in value]


def _coerce_int_list(
    value: str | list[int] | tuple[int, ...] | None,
) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [int(item.strip()) for item in value.split(",") if item.strip()]
    return [int(item) for item in value]


def _coerce_bool(value: bool | str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off", ""}:
        return False
    raise ValueError(f"Expected boolean value, got: {value}")


def _official_split(split: str | None) -> str | None:
    if split is None:
        return None
    normalized = normalize_split_name(split)
    return normalized if normalized in VALID_SPLITS else split
