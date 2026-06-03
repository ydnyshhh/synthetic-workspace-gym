from __future__ import annotations

import json
from typing import Any

from synthetic_workspace_gym.prime.clients import HeuristicReferenceClient, PrimeModelClient
from synthetic_workspace_gym.prime.env import SyntheticWorkspacePrimeEnv
from synthetic_workspace_gym.prime.transcript import make_event


DEFAULT_SYSTEM_PROMPT = (
    "You are operating inside a local synthetic workspace. Use only the provided tools. "
    "Return exactly one JSON object per turn. To call a tool, return:\n"
    '{"tool": "<tool_name>", "args": {...}}\n'
    "When finished, call:\n"
    '{"tool": "submit", "args": {"path_or_answer": "<path or answer>"}}\n'
    "Do not invent files. Inspect the workspace before editing."
)


class PrimeReActAgent:
    def __init__(
        self,
        client: PrimeModelClient,
        system_prompt: str | None = None,
        max_turns: int | None = None,
    ) -> None:
        self.client = client
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.max_turns = max_turns

    def run(self, env: SyntheticWorkspacePrimeEnv) -> dict[str, Any]:
        initial_observation = env.reset()
        tools = list(initial_observation["tool_schemas"])
        limit = self.max_turns or int(initial_observation.get("max_steps", 12))
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": str(initial_observation["instruction"])},
        ]
        events = [
            make_event("system", {"content": self.system_prompt}),
            make_event(
                "user",
                {
                    "content": str(initial_observation["instruction"]),
                    "env_id": initial_observation.get("env_id"),
                },
            ),
        ]
        model_metadata = self._model_metadata(initial_observation, env)
        final_step: dict[str, Any] | None = None
        final_reward_payload: dict[str, Any] | None = None
        tool_calls: list[dict[str, Any]] = []
        observations: list[dict[str, Any]] = []

        for step_index in range(limit):
            try:
                response = self.client.complete(messages, tools, metadata=model_metadata)
            except Exception as exc:
                events.append(
                    make_event(
                        "error",
                        {"message": str(exc), "exception_type": type(exc).__name__},
                        step_index=step_index,
                    )
                )
                break

            if response.get("type") == "tool_call":
                action = {"tool": response.get("tool"), "args": dict(response.get("args", {}) or {})}
                assistant_content = json.dumps(action, sort_keys=True)
            else:
                content = str(response.get("content", ""))
                action = {"tool": "submit", "args": {"path_or_answer": content}}
                assistant_content = content

            messages.append({"role": "assistant", "content": assistant_content})
            events.append(
                make_event(
                    "assistant",
                    {"content": assistant_content, "raw": response.get("raw", response)},
                    step_index=step_index,
                )
            )
            tool_calls.append({"step_index": step_index, **action})
            events.append(make_event("tool_call", action, step_index=step_index))

            step_result = env.step(action)
            observations.append({"step_index": step_index, **step_result})
            events.append(make_event("tool_observation", step_result, step_index=step_index))
            messages.append({"role": "tool", "content": str(step_result["observation"])})
            final_step = step_result
            if step_result.get("done"):
                reward_payload = dict(step_result.get("info", {}).get("reward_payload", {}) or {})
                final_reward_payload = reward_payload
                events.append(make_event("evaluation", reward_payload, step_index=step_index))
                break

        if final_reward_payload is None:
            try:
                final_reward_payload = env.evaluate()
                events.append(make_event("evaluation", final_reward_payload, step_index=len(tool_calls)))
            except Exception as exc:
                final_reward_payload = {
                    "reward": 0.0,
                    "success": False,
                    "score": 0.0,
                    "subscores": {},
                    "failure_labels": ["rollout_evaluation_error"],
                    "diagnostics": {"error": str(exc), "exception_type": type(exc).__name__},
                    "runtime_seconds": None,
                }
                events.append(make_event("error", final_reward_payload, step_index=len(tool_calls)))

        return {
            "initial_observation": initial_observation,
            "messages": messages,
            "transcript_events": events,
            "tool_calls": tool_calls,
            "observations": observations,
            "final_step": final_step,
            "reward_payload": final_reward_payload,
            "turn_count": len(tool_calls),
        }

    def _model_metadata(self, initial_observation: dict[str, Any], env: SyntheticWorkspacePrimeEnv) -> dict[str, Any]:
        metadata = dict(initial_observation.get("metadata", {}) or {})
        if isinstance(self.client, HeuristicReferenceClient):
            metadata["reference_solution"] = dict(env.manifest.reference_solution)
        return metadata
