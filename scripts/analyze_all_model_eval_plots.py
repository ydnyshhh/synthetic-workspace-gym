from __future__ import annotations

import argparse
import json
import math
import re
import textwrap
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


TASK_RE = re.compile(r"- (task_id|split|family|scenario|difficulty|seed):\s*(.*)")
REASONING_RE = re.compile(r'["\']reasoning_effort["\']\s*:\s*["\']([^"\']+)', re.I)
MODEL_ORDER = [
    "Qwen3.5-0.8B",
    "Qwen3.5-4B",
    "Qwen3-235B-Instruct",
    "Qwen3-235B-Thinking",
    "Nemotron-120B",
    "GPT-5.3-Codex (high)",
    "GPT-5.5 (low)",
    "GPT-5.5 (medium)",
    "GPT-5.5 (high)",
    "GLM-5.2",
    "Kimi-K2.7-Code",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build comprehensive seaborn model evaluation analyses."
    )
    parser.add_argument("--exports-dir", default="prime-eval-exports")
    parser.add_argument("--output-dir", default="analysis/model_comparison_plots")
    args = parser.parse_args()

    exports_dir = Path(args.exports_dir)
    output_dir = Path(args.output_dir)
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    samples, runs = load_all_exports(exports_dir)
    if samples.empty:
        raise SystemExit(f"No samples found under {exports_dir}")
    samples = enrich_samples(samples)
    runs = enrich_runs(runs, samples)
    core_runs = choose_core_runs(runs)
    core = samples[samples["run_id"].isin(core_runs["run_id"])].copy()
    if core.empty:
        raise SystemExit("No benchmark-comparable runs were found")

    samples.to_csv(tables_dir / "all_samples_normalized.csv", index=False)
    runs.to_csv(tables_dir / "run_inventory.csv", index=False)
    scorecard = aggregate(core, ["model_variant", "model_family", "run_id"])
    scorecard = scorecard.merge(
        core_runs[["run_id", "reasoning_effort", "max_turns", "time_limit_seconds"]],
        on="run_id",
        how="left",
    )
    scorecard = scorecard.sort_values("mean_reward", ascending=False)
    scorecard.to_csv(tables_dir / "core_model_scorecard.csv", index=False)
    family = aggregate(core, ["model_variant", "family"])
    family.to_csv(tables_dir / "family_metrics.csv", index=False)
    scenario = aggregate(core, ["model_variant", "family", "scenario"])
    scenario.to_csv(tables_dir / "scenario_metrics.csv", index=False)

    pairwise = pairwise_task_advantage(core)
    pairwise.to_csv(tables_dir / "pairwise_task_advantage.csv")
    stop = stop_condition_table(core)
    stop.to_csv(tables_dir / "stop_condition_rates.csv")

    configure_theme()
    palette = model_palette(samples["model_variant"].unique())
    plot_run_inventory(runs, figures_dir, palette)
    plot_core_scorecard(scorecard, figures_dir, palette)
    plot_reward_distributions(core, figures_dir, palette)
    plot_family_heatmap(core, figures_dir)
    plot_scenario_heatmap(core, figures_dir)
    plot_difficulty_lines(core, figures_dir, palette)
    plot_behavior_heatmap(core, figures_dir)
    plot_tool_signature(core, figures_dir)
    plot_efficiency_frontier(scorecard, figures_dir, palette)
    plot_reasoning_effort(core, figures_dir)
    plot_pairwise_advantage(pairwise, figures_dir)
    plot_relative_specialization(core, figures_dir)
    plot_stop_conditions(stop, figures_dir)
    plot_tool_count_response(core, figures_dir, palette)
    plot_consistency(core, figures_dir, palette)
    plot_task_win_share(core, figures_dir, palette)

    insights = build_insights(core, scorecard, family, scenario, pairwise)
    pd.DataFrame({"insight": insights}).to_csv(
        tables_dir / "key_insights.csv", index=False
    )
    write_report(
        output_dir, samples, runs, core_runs, scorecard, family, scenario, insights
    )
    print(
        f"Loaded {len(samples):,} deduplicated samples across {runs['run_id'].nunique()} runs"
    )
    print(
        f"Selected {len(core_runs)} benchmark-comparable model variants ({len(core):,} samples)"
    )
    print(f"Wrote {len(list(figures_dir.glob('*.png')))} figures to {figures_dir}")
    print(f"Wrote tables and analysis report to {output_dir}")


def load_all_exports(exports_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    for run_dir in sorted(
        (path for path in exports_dir.iterdir() if path.is_dir()), key=lambda p: p.name
    ):
        metadata = (
            load_json(run_dir / "metadata.json")
            if (run_dir / "metadata.json").exists()
            else {}
        )
        raw, source = load_run_samples(run_dir)
        if not raw:
            continue
        raw_count = len(raw)
        raw = deduplicate_samples(raw)
        run_id = str(metadata.get("evaluation_id") or run_dir.name)
        run = parse_run_metadata(run_id, metadata, raw, source, raw_count)
        run_rows.append(run)
        sample_rows.extend(parse_sample(run, sample) for sample in raw)
    return pd.DataFrame(sample_rows), pd.DataFrame(run_rows)


def load_run_samples(run_dir: Path) -> tuple[list[dict[str, Any]], str]:
    samples_path = run_dir / "samples.json"
    if samples_path.exists():
        payload = load_json(samples_path)
        samples = extract_samples(payload)
        if samples:
            return samples, "samples.json"
    page_paths = sorted(run_dir.glob("samples-page-*.json"), key=numeric_page_key)
    if not page_paths and (run_dir / "pages").is_dir():
        page_paths = sorted(
            (run_dir / "pages").glob("page-*.json"), key=numeric_page_key
        )
    rows: list[dict[str, Any]] = []
    for path in page_paths:
        rows.extend(extract_samples(load_json(path)))
    return rows, "paged export" if rows else "none"


def extract_samples(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        for key in ("samples", "data", "results"):
            value = payload.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
    return []


def deduplicate_samples(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for sample in samples:
        key = str(sample.get("trace_id") or "")
        if not key:
            key = "|".join(
                str(sample.get(field) or "")
                for field in ("example_id", "rollout_number", "created_at", "reward")
            )
        if key in seen:
            continue
        seen.add(key)
        result.append(sample)
    return result


def numeric_page_key(path: Path) -> tuple[int, str]:
    match = re.search(r"(\d+)(?=\.json$)", path.name)
    return (int(match.group(1)) if match else math.inf, path.name)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def parse_run_metadata(
    run_id: str,
    metadata: dict[str, Any],
    samples: list[dict[str, Any]],
    source: str,
    raw_count: int,
) -> dict[str, Any]:
    cfg = metadata.get("eval_config") or {}
    env = cfg.get("env_args") or {}
    model = str(
        metadata.get("model_name")
        or metadata.get("inference_model")
        or cfg.get("model")
        or "unknown"
    )
    command = str(cfg.get("eval_command") or metadata.get("eval_command") or "")
    sampling = cfg.get("sampling_args") or metadata.get("sampling_args") or {}
    effort = (
        str(sampling.get("reasoning_effort") or "")
        if isinstance(sampling, dict)
        else ""
    )
    if not effort:
        match = REASONING_RE.search(command)
        effort = match.group(1).lower() if match else "default"
    family = short_model_name(model)
    variant = variant_name(family, effort)
    return {
        "run_id": run_id,
        "model": model,
        "model_family": family,
        "model_variant": variant,
        "reasoning_effort": effort,
        "created_at": str(metadata.get("created_at") or ""),
        "sample_strategy": str(env.get("sample_strategy") or "head/default"),
        "configured_examples": int(
            metadata.get("total_samples") or cfg.get("num_examples") or len(samples)
        ),
        "raw_rows": raw_count,
        "loaded_rows": len(samples),
        "source": source,
        "max_turns": to_float(env.get("max_turns")),
        "max_tool_steps": to_float(env.get("max_tool_steps")),
        "time_limit_seconds": to_float(env.get("time_limit_seconds")),
    }


def parse_sample(run: dict[str, Any], sample: dict[str, Any]) -> dict[str, Any]:
    prompt = sample.get("prompt") or []
    user = first_user_message(prompt)
    task = sample.get("task") or {}
    metadata = parse_task_metadata(user)
    if isinstance(task, dict):
        for key in ("task_id", "family", "scenario", "difficulty", "seed"):
            if task.get(key) is not None and key not in metadata:
                metadata[key] = str(task[key])
    completion = sample.get("completion") or []
    tools, paths = parse_tool_calls(completion)
    info = sample.get("info") or {}
    timing = info.get("timing") or {}
    usage = info.get("token_usage") or {}
    metrics = info.get("metrics") or {}
    text = completion_text_blob(completion)
    reward = first_not_none(
        sample.get("reward"), sample.get("swg_reward"), metrics.get("swg_reward"), 0.0
    )
    family = metadata.get("family", "unknown")
    scenario = metadata.get("scenario", "unknown")
    return {
        **{
            key: run[key]
            for key in (
                "run_id",
                "model",
                "model_family",
                "model_variant",
                "reasoning_effort",
                "created_at",
                "sample_strategy",
                "max_turns",
                "time_limit_seconds",
            )
        },
        "trace_id": str(sample.get("trace_id") or ""),
        "example_id": sample.get("example_id"),
        "task_id": metadata.get("task_id", ""),
        "family": family,
        "scenario": scenario,
        "scenario_label": f"{family}/{scenario}",
        "difficulty": int(to_float(metadata.get("difficulty"))),
        "seed": int(to_float(metadata.get("seed"))),
        "reward": float(reward or 0.0),
        "num_turns": to_float(
            first_not_none(metrics.get("num_turns"), sample.get("num_steps"), 0.0)
        ),
        "time_s": to_float(
            first_not_none(timing.get("total"), sample.get("total_time"), 0.0)
        ),
        "input_tokens": to_float(usage.get("input_tokens")),
        "output_tokens": to_float(usage.get("output_tokens")),
        "stop_condition": str(info.get("stop_condition") or "unknown"),
        "is_truncated": bool(info.get("is_truncated") or False),
        "tool_count": len(tools),
        "read_count": tools.count("read_file"),
        "list_count": tools.count("list_directory"),
        "write_count": tools.count("write_file") + tools.count("append_file"),
        "run_count": sum(
            tools.count(name) for name in ("run_shell", "run_python", "shell_command")
        ),
        "submit_count": tools.count("submit"),
        "tool_error_count": count_tool_errors(completion),
        "guidance_count": count_text(completion, "Guidance:"),
        "submit_correction_count": count_text(completion, "Submit correction"),
        "uses_pandas": bool(re.search(r"\b(import pandas|pd\.|pandas)\b", text, re.I)),
        "has_key_error": "KeyError" in text,
        "has_file_not_found": "FileNotFoundError" in text,
        "has_syntax_error": bool(re.search(r"(SyntaxError|IndentationError)", text)),
        "has_model_error": "ModelError" in str(info.get("error") or ""),
        "paths_tail": " | ".join(paths[-6:]),
    }


def first_user_message(prompt: Any) -> str:
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        for message in prompt:
            if isinstance(message, dict) and message.get("role") == "user":
                return str(message.get("content") or "")
    return ""


def parse_task_metadata(user_message: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in user_message.splitlines():
        match = TASK_RE.match(line.strip())
        if match:
            result[match.group(1)] = match.group(2).strip()
    if not result.get("task_id"):
        match = re.search(r"\b(swg\.[\w.-]+\.d\d+\.s\d+)\b", user_message)
        if match:
            result["task_id"] = match.group(1)
    if result.get("task_id"):
        parts = result["task_id"].split(".")
        if len(parts) >= 5:
            result.setdefault("family", parts[2])
            result.setdefault("scenario", parts[3])
            dmatch = re.match(r"d(\d+)", parts[4])
            if dmatch:
                result.setdefault("difficulty", dmatch.group(1))
            if len(parts) > 5:
                smatch = re.match(r"s(\d+)", parts[5])
                if smatch:
                    result.setdefault("seed", smatch.group(1))
    return result


def parse_tool_calls(messages: Any) -> tuple[list[str], list[str]]:
    tools: list[str] = []
    paths: list[str] = []
    if not isinstance(messages, list):
        return tools, paths
    for message in messages:
        if not isinstance(message, dict):
            continue
        calls = message.get("tool_calls") or []
        for raw in calls:
            call = parse_jsonish(raw)
            function = (
                call.get("function") if isinstance(call.get("function"), dict) else call
            )
            name = str(function.get("name") or call.get("name") or "")
            args = parse_jsonish(
                function.get("arguments") or call.get("arguments") or {}
            )
            path = first_not_none(
                args.get("path"),
                args.get("path_or_answer"),
                args.get("command_or_script"),
                args.get("command"),
                "",
            )
            if name:
                tools.append(name)
                paths.append(str(path))
    return tools, paths


def parse_jsonish(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except (json.JSONDecodeError, TypeError):
            return {}
    return {}


def completion_text_blob(messages: Any) -> str:
    if not isinstance(messages, list):
        return str(messages or "")
    chunks: list[str] = []
    for message in messages:
        if isinstance(message, dict):
            chunks.append(str(message.get("content") or ""))
            chunks.append(str(message.get("reasoning_content") or ""))
            chunks.extend(str(call) for call in message.get("tool_calls") or [])
    return "\n".join(chunks)


def count_tool_errors(messages: Any) -> int:
    if not isinstance(messages, list):
        return 0
    markers = ("error:", "failed", "rejected", "traceback", "filenotfounderror")
    return sum(
        any(marker in str(message.get("content") or "").lower() for marker in markers)
        for message in messages
        if isinstance(message, dict) and message.get("role") == "tool"
    )


def count_text(messages: Any, needle: str) -> int:
    if not isinstance(messages, list):
        return 0
    return sum(
        needle in str(message.get("content") or "")
        for message in messages
        if isinstance(message, dict) and message.get("role") == "tool"
    )


def short_model_name(model: str) -> str:
    lowered = model.lower()
    if "qwen3.5-0.8b" in lowered or "qwen3-5-0-8b" in lowered:
        return "Qwen3.5-0.8B"
    if "qwen3.5-4b" in lowered or "qwen3-5-4b" in lowered:
        return "Qwen3.5-4B"
    if "235b" in lowered and "thinking" in lowered:
        return "Qwen3-235B-Thinking"
    if "235b" in lowered and "instruct" in lowered:
        return "Qwen3-235B-Instruct"
    if "nemotron" in lowered:
        return "Nemotron-120B"
    if "gpt-5.3-codex" in lowered:
        return "GPT-5.3-Codex"
    if "gpt-5.5" in lowered:
        return "GPT-5.5"
    if "glm-5.2" in lowered or "glm52" in lowered:
        return "GLM-5.2"
    if "kimi-k2.7-code" in lowered:
        return "Kimi-K2.7-Code"
    return model.split("/")[-1]


def variant_name(model_family: str, effort: str) -> str:
    if model_family in {"GPT-5.3-Codex", "GPT-5.5"}:
        return f"{model_family} ({effort})"
    return model_family


def enrich_samples(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_perfect"] = np.isclose(df["reward"], 1.0)
    df["is_zero"] = np.isclose(df["reward"], 0.0)
    df["is_strong"] = df["reward"].ge(0.8)
    df["no_submit"] = df["submit_count"].eq(0)
    df["has_tool_error"] = df["tool_error_count"].gt(0)
    df["has_any_runtime_error"] = df[
        ["has_key_error", "has_file_not_found", "has_syntax_error", "has_model_error"]
    ].any(axis=1)
    df["hit_limit"] = df["stop_condition"].str.contains(
        "max_|time", case=False, na=False
    )
    df["reward_per_1k_tokens"] = df["reward"] / (
        df["output_tokens"].replace(0, np.nan) / 1000
    )
    df["reward_per_minute"] = df["reward"] / (df["time_s"].replace(0, np.nan) / 60)
    return df


def enrich_runs(runs: pd.DataFrame, samples: pd.DataFrame) -> pd.DataFrame:
    metrics = aggregate(samples, ["run_id"])
    known = (
        samples.groupby("run_id")["family"]
        .apply(lambda x: x.ne("unknown").mean())
        .rename("known_task_rate")
    )
    unique_tasks = samples.groupby("run_id")["task_id"].nunique().rename("unique_tasks")
    df = (
        runs.merge(metrics, on="run_id", how="left")
        .merge(known, on="run_id")
        .merge(unique_tasks, on="run_id")
    )
    df["configured_coverage"] = df["loaded_rows"] / df["configured_examples"].replace(
        0, np.nan
    )
    df["comparison_eligible"] = (
        df["sample_strategy"].eq("balanced")
        & df["n"].ge(350)
        & df["unique_tasks"].ge(350)
        & df["known_task_rate"].ge(0.95)
    )
    return df.sort_values(["created_at", "run_id"]).reset_index(drop=True)


def choose_core_runs(runs: pd.DataFrame) -> pd.DataFrame:
    eligible = runs[runs["comparison_eligible"]].copy()
    if eligible.empty:
        return eligible
    # Repeated exports/config retries exist. Use the latest complete run per distinct model/effort variant.
    return (
        eligible.sort_values(["created_at", "run_id"])
        .groupby("model_variant", as_index=False, sort=False)
        .tail(1)
        .sort_values("model_variant", key=lambda s: s.map(order_index))
    )


def aggregate(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    result = (
        df.groupby(keys, dropna=False)
        .agg(
            n=("reward", "size"),
            mean_reward=("reward", "mean"),
            median_reward=("reward", "median"),
            std_reward=("reward", "std"),
            perfect_rate=("is_perfect", "mean"),
            strong_rate=("is_strong", "mean"),
            zero_rate=("is_zero", "mean"),
            mean_turns=("num_turns", "mean"),
            mean_time_s=("time_s", "mean"),
            median_time_s=("time_s", "median"),
            mean_output_tokens=("output_tokens", "mean"),
            no_submit_rate=("no_submit", "mean"),
            tool_error_rate=("has_tool_error", "mean"),
            runtime_error_rate=("has_any_runtime_error", "mean"),
            limit_rate=("hit_limit", "mean"),
            mean_tools=("tool_count", "mean"),
            mean_reads=("read_count", "mean"),
            mean_lists=("list_count", "mean"),
            mean_writes=("write_count", "mean"),
            mean_runs=("run_count", "mean"),
        )
        .reset_index()
    )
    result["reward_ci95"] = (
        1.96 * result["std_reward"].fillna(0) / np.sqrt(result["n"].clip(lower=1))
    )
    result["reward_per_minute"] = result["mean_reward"] / (
        result["mean_time_s"].replace(0, np.nan) / 60
    )
    result["reward_per_1k_tokens"] = result["mean_reward"] / (
        result["mean_output_tokens"].replace(0, np.nan) / 1000
    )
    return result


def pairwise_task_advantage(core: pd.DataFrame) -> pd.DataFrame:
    pivot = core.pivot_table(
        index="task_id", columns="model_variant", values="reward", aggfunc="mean"
    )
    variants = sorted(pivot.columns, key=order_index)
    matrix = pd.DataFrame(index=variants, columns=variants, dtype=float)
    for left in variants:
        for right in variants:
            matched = pivot[[left, right]].dropna()
            matrix.loc[left, right] = (
                0.0
                if left == right
                else (matched[left] > matched[right]).mean()
                - (matched[left] < matched[right]).mean()
            )
    return matrix


def stop_condition_table(core: pd.DataFrame) -> pd.DataFrame:
    data = core.copy()
    data["stop_group"] = data["stop_condition"].map(normalize_stop)
    table = data.groupby(["model_variant", "stop_group"]).size().unstack(fill_value=0)
    table = table.div(table.sum(axis=1), axis=0)
    return table.reindex(sorted(table.index, key=order_index))


def normalize_stop(value: str) -> str:
    lowered = str(value).lower()
    if "final" in lowered or "submit" in lowered:
        return "submitted/final"
    if "max_turn" in lowered:
        return "max turns"
    if "time" in lowered:
        return "time limit"
    if "error" in lowered:
        return "error"
    return "other/unknown"


def configure_theme() -> None:
    sns.set_theme(
        context="talk",
        style="whitegrid",
        font_scale=0.82,
        rc={
            "figure.dpi": 130,
            "savefig.dpi": 150,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        },
    )


def model_palette(models: Iterable[str]) -> dict[str, Any]:
    ordered = sorted(set(models), key=order_index)
    colors = sns.color_palette("colorblind", n_colors=max(3, len(ordered)))
    return dict(zip(ordered, colors))


def short_plot_name(model: str) -> str:
    names = {
        "Qwen3.5-0.8B": "Qwen 0.8B",
        "Qwen3.5-4B": "Qwen 4B",
        "Qwen3-235B-Thinking": "Qwen 235B",
        "GPT-5.3-Codex (high)": "Codex 5.3",
        "GPT-5.5 (low)": "GPT-5.5 low",
        "GPT-5.5 (medium)": "GPT-5.5 med",
        "GPT-5.5 (high)": "GPT-5.5 high",
        "GLM-5.2": "GLM-5.2",
        "Kimi-K2.7-Code": "Kimi K2.7",
    }
    return names.get(model, model)


def scenario_plot_name(value: str) -> str:
    scenario = value.split("/", 1)[-1].replace("_", " ")
    return textwrap.fill(scenario, width=24)


def plot_run_inventory(runs: pd.DataFrame, out: Path, palette: dict[str, Any]) -> None:
    data = runs.sort_values("mean_reward", ascending=True).copy()
    data["run_label"] = data.apply(
        lambda r: (
            f"{r.model_variant} · {r.run_id[:6]} · n={int(r.n)}"
            + (" · core" if r.comparison_eligible else "")
        ),
        axis=1,
    )
    fig, ax = plt.subplots(figsize=(14, max(8, 0.42 * len(data))))
    colors = [palette.get(model, (0.5, 0.5, 0.5)) for model in data["model_variant"]]
    ax.barh(data["run_label"], data["mean_reward"], color=colors, alpha=0.9)
    ax.errorbar(
        data["mean_reward"],
        np.arange(len(data)),
        xerr=data["reward_ci95"],
        fmt="none",
        ecolor="0.2",
        capsize=2,
    )
    for y, value in enumerate(data["mean_reward"]):
        ax.text(min(value + 0.012, 1.01), y, f"{value:.3f}", va="center", fontsize=9)
    ax.set_xlim(0, 1.06)
    ax.set_xlabel("mean reward (95% normal CI)")
    ax.set_ylabel("")
    ax.set_title("All Evaluation Runs: Performance and Coverage Inventory")
    savefig(fig, out / "01_all_run_inventory.png")


def plot_core_scorecard(
    scorecard: pd.DataFrame, out: Path, palette: dict[str, Any]
) -> None:
    data = scorecard.sort_values("mean_reward", ascending=True).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(11.5, max(6.5, 0.62 * len(data))))
    colors = [palette.get(model, (0.5, 0.5, 0.5)) for model in data["model_variant"]]
    y = np.arange(len(data))
    ax.barh(y, data["mean_reward"], color=colors, height=0.72)
    ax.errorbar(
        data["mean_reward"],
        y,
        xerr=data["reward_ci95"],
        fmt="none",
        ecolor="0.15",
        capsize=3,
        linewidth=1.4,
    )
    for position, row in enumerate(data.itertuples()):
        label_x = row.mean_reward + row.reward_ci95 + 0.014
        ax.text(
            label_x,
            position,
            f"{row.mean_reward:.3f}  |  perfect {row.perfect_rate:.0%}",
            va="center",
            fontsize=9,
        )
    ax.set_yticks(y, data["model_variant"])
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("mean reward (error bars: 95% CI)")
    ax.set_ylabel("")
    ax.set_title("Benchmark-Comparable Model Performance", pad=12)
    savefig(fig, out / "02_core_model_scorecard.png")


def plot_reward_distributions(
    core: pd.DataFrame, out: Path, palette: dict[str, Any]
) -> None:
    order = ordered_variants(core)
    fig, ax = plt.subplots(figsize=(15, 8))
    sns.violinplot(
        data=core,
        x="model_variant",
        y="reward",
        order=order,
        hue="model_variant",
        palette=palette,
        inner="quart",
        cut=0,
        legend=False,
        ax=ax,
    )
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("")
    ax.set_ylabel("reward")
    ax.set_title("Reward Distribution: Reliability Beyond the Mean")
    ax.tick_params(axis="x", rotation=32)
    savefig(fig, out / "03_reward_distribution_violin.png")


def plot_family_heatmap(core: pd.DataFrame, out: Path) -> None:
    matrix = core.pivot_table(
        index="model_variant", columns="family", values="reward", aggfunc="mean"
    )
    matrix = matrix.reindex(sorted(matrix.index, key=order_index))
    perfect = core.pivot_table(
        index="model_variant", columns="family", values="is_perfect", aggfunc="mean"
    ).reindex_like(matrix)
    annot = matrix.copy().astype(object)
    for row in matrix.index:
        for column in matrix.columns:
            annot.loc[row, column] = (
                f"{matrix.loc[row, column]:.2f}\n({perfect.loc[row, column]:.0%} perfect)"
            )
    fig, ax = plt.subplots(figsize=(12, max(6, 0.68 * len(matrix))))
    sns.heatmap(
        matrix,
        cmap="crest",
        vmin=0.4,
        vmax=1,
        annot=annot,
        fmt="",
        linewidths=0.6,
        ax=ax,
    )
    ax.set_xlabel("task family")
    ax.set_ylabel("")
    ax.set_title("Family Robustness: Mean Reward and Perfect-Completion Rate")
    savefig(fig, out / "04_family_performance_heatmap.png")


def plot_scenario_heatmap(core: pd.DataFrame, out: Path) -> None:
    matrix = core.pivot_table(
        index="scenario_label", columns="model_variant", values="reward", aggfunc="mean"
    )
    matrix = matrix.reindex(columns=sorted(matrix.columns, key=order_index))
    matrix = matrix.loc[matrix.mean(axis=1).sort_values().index]
    matrix.index = [scenario_plot_name(value) for value in matrix.index]
    matrix.columns = [short_plot_name(value) for value in matrix.columns]
    fig, ax = plt.subplots(figsize=(14, max(8.5, 0.58 * len(matrix))))
    sns.heatmap(
        matrix,
        cmap="RdYlGn",
        vmin=0.35,
        vmax=1,
        annot=True,
        fmt=".2f",
        linewidths=0.45,
        cbar_kws={"label": "mean reward", "shrink": 0.86},
        ax=ax,
    )
    ax.set_xlabel("model")
    ax.set_ylabel("scenario (hardest at top)")
    ax.set_title("Scenario Performance Heatmap", pad=12)
    ax.tick_params(axis="x", rotation=30)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    ax.tick_params(axis="y", rotation=0)
    savefig(fig, out / "05_scenario_performance_heatmap.png")


def plot_difficulty_lines(
    core: pd.DataFrame, out: Path, palette: dict[str, Any]
) -> None:
    order = ordered_variants(core)
    g = sns.relplot(
        data=core,
        x="difficulty",
        y="reward",
        hue="model_variant",
        hue_order=order,
        palette=palette,
        col="family",
        col_wrap=2,
        kind="line",
        estimator="mean",
        errorbar=("ci", 95),
        marker="o",
        height=4.4,
        aspect=1.35,
    )
    g.set(ylim=(0.25, 1.02), xticks=sorted(core["difficulty"].unique()))
    g.set_axis_labels("difficulty", "mean reward")
    g.set_titles("{col_name}")
    g.fig.suptitle("Difficulty Scaling by Task Family", y=1.02, fontweight="bold")
    g.fig.savefig(out / "06_difficulty_scaling_lines.png", bbox_inches="tight", dpi=180)
    plt.close(g.fig)


def plot_behavior_heatmap(core: pd.DataFrame, out: Path) -> None:
    metrics = (
        core.groupby("model_variant")
        .agg(
            no_submit=("no_submit", "mean"),
            tool_error=("has_tool_error", "mean"),
            runtime_error=("has_any_runtime_error", "mean"),
            hit_limit=("hit_limit", "mean"),
            truncated=("is_truncated", "mean"),
            used_pandas=("uses_pandas", "mean"),
            needed_guidance=("guidance_count", lambda x: (x > 0).mean()),
            submit_correction=("submit_correction_count", lambda x: (x > 0).mean()),
        )
        .reindex(sorted(core["model_variant"].unique(), key=order_index))
    )
    fig, ax = plt.subplots(figsize=(14, max(6, 0.58 * len(metrics))))
    sns.heatmap(
        metrics,
        cmap="rocket_r",
        vmin=0,
        vmax=max(0.25, metrics.max().max()),
        annot=True,
        fmt=".1%",
        linewidths=0.5,
        ax=ax,
    )
    ax.set_xlabel("behavioral rate")
    ax.set_ylabel("")
    ax.set_title("Behavioral and Failure Signatures")
    savefig(fig, out / "07_behavioral_failure_heatmap.png")


def plot_tool_signature(core: pd.DataFrame, out: Path) -> None:
    columns = ["read_count", "list_count", "write_count", "run_count", "submit_count"]
    matrix = (
        core.groupby("model_variant")[columns]
        .mean()
        .reindex(sorted(core["model_variant"].unique(), key=order_index))
    )
    share = matrix.div(matrix.sum(axis=1).replace(0, np.nan), axis=0)
    fig, axes = plt.subplots(1, 2, figsize=(18, max(6, 0.62 * len(matrix))))
    sns.heatmap(matrix, cmap="Blues", annot=True, fmt=".1f", linewidths=0.5, ax=axes[0])
    sns.heatmap(share, cmap="mako", annot=True, fmt=".0%", linewidths=0.5, ax=axes[1])
    axes[0].set_title("Mean tool calls per task")
    axes[1].set_title("Tool mix share")
    for ax in axes:
        ax.set_xlabel("")
        ax.set_ylabel("")
    fig.suptitle("Tool-Use Strategy Signatures", fontweight="bold")
    savefig(fig, out / "08_tool_use_signature_heatmaps.png")


def plot_efficiency_frontier(
    scorecard: pd.DataFrame, out: Path, palette: dict[str, Any]
) -> None:
    data = scorecard.sort_values("mean_reward").copy()
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.8), sharey=True)
    layouts = {
        "mean_time_s": {
            "Qwen3.5-0.8B": (8, -18, "left"),
            "Qwen3.5-4B": (8, 12, "left"),
            "Qwen3-235B-Thinking": (8, 12, "left"),
            "GPT-5.3-Codex (high)": (-8, -18, "right"),
            "GPT-5.5 (low)": (-8, 19, "right"),
            "GPT-5.5 (medium)": (0, -22, "center"),
            "GPT-5.5 (high)": (-13, 29, "right"),
            "GLM-5.2": (13, 29, "left"),
            "Kimi-K2.7-Code": (14, -23, "left"),
        },
        "mean_output_tokens": {
            "Qwen3.5-0.8B": (8, -18, "left"),
            "Qwen3.5-4B": (8, 12, "left"),
            "Qwen3-235B-Thinking": (8, 12, "left"),
            "GPT-5.3-Codex (high)": (10, -18, "left"),
            "GPT-5.5 (low)": (-8, 19, "right"),
            "GPT-5.5 (medium)": (-10, -22, "right"),
            "GPT-5.5 (high)": (-12, 25, "right"),
            "GLM-5.2": (13, -25, "left"),
            "Kimi-K2.7-Code": (12, 18, "left"),
        },
    }
    panel_specs = (
        (axes[0], "mean_time_s", "mean wall time per task (seconds)"),
        (axes[1], "mean_output_tokens", "mean output tokens per task"),
    )
    for ax, x_column, axis_label in panel_specs:
        sns.scatterplot(
            data=data,
            x=x_column,
            y="mean_reward",
            hue="model_variant",
            palette=palette,
            s=135,
            edgecolor="white",
            linewidth=1.0,
            ax=ax,
            legend=False,
            zorder=3,
        )
        for row in data.itertuples():
            dx, dy, alignment = layouts[x_column][row.model_variant]
            ax.annotate(
                short_plot_name(row.model_variant),
                xy=(getattr(row, x_column), row.mean_reward),
                xytext=(dx, dy),
                textcoords="offset points",
                ha=alignment,
                va="center",
                fontsize=8.5,
                color="0.16",
                arrowprops={
                    "arrowstyle": "-",
                    "color": "0.45",
                    "linewidth": 0.7,
                    "shrinkA": 2,
                    "shrinkB": 7,
                },
                annotation_clip=False,
                zorder=4,
            )
        ax.set_xscale("log")
        ax.margins(x=0.13)
        ax.set_xlabel(axis_label + " (log scale)")
        ax.set_ylabel("mean reward")
        ax.set_ylim(
            data["mean_reward"].min() - 0.07,
            min(1.01, data["mean_reward"].max() + 0.075),
        )
    axes[0].set_title("Wall-time frontier", pad=12)
    axes[1].set_title("Token frontier", pad=12)
    fig.suptitle("Performance-Efficiency Frontier", fontweight="bold", y=0.985)
    savefig(fig, out / "09_efficiency_frontier.png")


def plot_reasoning_effort(core: pd.DataFrame, out: Path) -> None:
    data = core[core["model_family"].eq("GPT-5.5")].copy()
    if data.empty or data["reasoning_effort"].nunique() < 2:
        return
    order = [
        value
        for value in ("low", "medium", "high")
        if value in data["reasoning_effort"].unique()
    ]
    x = np.arange(len(order))
    overall = (
        data.groupby("reasoning_effort")["reward"]
        .agg(["mean", "std", "count"])
        .reindex(order)
    )
    overall["ci95"] = 1.96 * overall["std"] / np.sqrt(overall["count"])
    family = (
        data.groupby(["reasoning_effort", "family"])["reward"]
        .mean()
        .unstack()
        .reindex(order)
    )
    resources = (
        data.groupby("reasoning_effort")
        .agg(time_s=("time_s", "mean"), output_tokens=("output_tokens", "mean"))
        .reindex(order)
    )

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5))
    axes[0, 0].errorbar(
        x, overall["mean"], yerr=overall["ci95"], marker="o", capsize=5, linewidth=2
    )
    axes[0, 0].set_ylim(overall["mean"].min() - 0.025, overall["mean"].max() + 0.025)
    axes[0, 0].set_title("Overall reward")
    axes[0, 0].set_ylabel("mean reward")

    family_colors = sns.color_palette("colorblind", n_colors=len(family.columns))
    for color, column in zip(family_colors, family.columns):
        axes[0, 1].plot(x, family[column], marker="o", color=color)
        axes[0, 1].annotate(
            column.replace("_", " "),
            (x[-1], family[column].iloc[-1]),
            xytext=(7, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color=color,
        )
    axes[0, 1].set_xlim(-0.1, len(order) - 0.55)
    axes[0, 1].set_title("Reward by task family")
    axes[0, 1].set_ylabel("mean reward")

    bars_time = axes[1, 0].bar(
        x, resources["time_s"], color=sns.color_palette("Blues", n_colors=len(order))
    )
    axes[1, 0].bar_label(bars_time, fmt="%.1f s", padding=3, fontsize=9)
    axes[1, 0].set_title("Wall time per task")
    axes[1, 0].set_ylabel("seconds")

    bars_tokens = axes[1, 1].bar(
        x,
        resources["output_tokens"],
        color=sns.color_palette("Purples", n_colors=len(order)),
    )
    axes[1, 1].bar_label(bars_tokens, fmt="%.0f", padding=3, fontsize=9)
    axes[1, 1].set_title("Output tokens per task")
    axes[1, 1].set_ylabel("tokens")

    for ax in axes.flat:
        ax.set_xticks(x, order)
        ax.set_xlabel("reasoning effort")
    fig.suptitle("GPT-5.5 Reasoning-Effort Experiment", fontweight="bold", y=0.995)
    savefig(fig, out / "10_gpt55_reasoning_effort_experiment.png")


def plot_pairwise_advantage(pairwise: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 11))
    sns.heatmap(
        pairwise,
        cmap="vlag",
        vmin=-0.5,
        vmax=0.5,
        center=0,
        annot=True,
        fmt="+.0%",
        linewidths=0.5,
        square=True,
        ax=ax,
    )
    ax.set_title(
        "Matched-Task Pairwise Advantage\n(row win rate − row loss rate vs column; ties cancel)"
    )
    ax.set_xlabel("comparison model")
    ax.set_ylabel("focal model")
    savefig(fig, out / "11_pairwise_task_advantage_heatmap.png")


def plot_relative_specialization(core: pd.DataFrame, out: Path) -> None:
    matrix = core.pivot_table(
        index="scenario_label", columns="model_variant", values="reward", aggfunc="mean"
    )
    matrix = matrix.reindex(columns=sorted(matrix.columns, key=order_index))
    relative = matrix.subtract(matrix.mean(axis=1), axis=0)
    relative = relative.loc[relative.std(axis=1).sort_values(ascending=False).index]
    relative.index = [scenario_plot_name(value) for value in relative.index]
    relative.columns = [short_plot_name(value) for value in relative.columns]
    bound = max(0.12, float(np.nanquantile(np.abs(relative.values), 0.95)))
    fig, ax = plt.subplots(figsize=(14, max(8.5, 0.58 * len(relative))))
    sns.heatmap(
        relative,
        cmap="vlag",
        center=0,
        vmin=-bound,
        vmax=bound,
        annot=True,
        fmt="+.2f",
        linewidths=0.4,
        cbar_kws={"label": "delta from scenario mean", "shrink": 0.86},
        ax=ax,
    )
    ax.set_xlabel("model")
    ax.set_ylabel("scenario (highest disagreement at top)")
    ax.set_title("Relative Specialization by Scenario", pad=12)
    ax.tick_params(axis="x", rotation=30)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    ax.tick_params(axis="y", rotation=0)
    savefig(fig, out / "12_relative_scenario_specialization.png")


def plot_stop_conditions(stop: pd.DataFrame, out: Path) -> None:
    preferred = ["submitted/final", "max turns", "time limit", "error", "other/unknown"]
    table = stop.reindex(
        columns=[column for column in preferred if column in stop.columns], fill_value=0
    )
    fig, ax = plt.subplots(figsize=(14, 7))
    table.plot(
        kind="bar",
        stacked=True,
        width=0.82,
        color=sns.color_palette("Set2", n_colors=len(table.columns)),
        ax=ax,
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("share of tasks")
    ax.set_xlabel("")
    ax.set_title("How Runs End: Stop-Condition Composition")
    ax.tick_params(axis="x", rotation=32)
    ax.legend(title="stop condition", bbox_to_anchor=(1.01, 1), loc="upper left")
    savefig(fig, out / "13_stop_condition_composition.png")


def plot_tool_count_response(
    core: pd.DataFrame, out: Path, palette: dict[str, Any]
) -> None:
    data = core.copy()
    bin_labels = ["0-7", "8-11", "12-15", "16-20", "21-30", "31+"]
    data["tool_bin"] = pd.cut(
        data["tool_count"], bins=[-1, 7, 11, 15, 20, 30, np.inf], labels=bin_labels
    )
    summary = data.groupby(
        ["model_variant", "tool_bin"], observed=True, as_index=False
    ).agg(reward=("reward", "mean"), n=("reward", "size"))
    summary = summary[summary["n"].ge(5)]
    variants = ordered_variants(core)
    fig, axes = plt.subplots(3, 3, figsize=(14, 11), sharex=True, sharey=True)
    x = np.arange(len(bin_labels))
    for ax, model in zip(axes.flat, variants):
        model_data = (
            summary[summary["model_variant"].eq(model)]
            .set_index("tool_bin")
            .reindex(bin_labels)
        )
        rewards = model_data["reward"].to_numpy(dtype=float)
        counts = model_data["n"].fillna(0).to_numpy(dtype=float)
        color = palette.get(model, (0.35, 0.35, 0.35))
        ax.plot(x, rewards, color=color, linewidth=2, marker="o")
        present = np.isfinite(rewards)
        ax.scatter(
            x[present],
            rewards[present],
            s=28 + np.sqrt(counts[present]) * 5,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        ax.set_title(short_plot_name(model), fontsize=10)
        ax.set_ylim(0, 1.02)
        ax.set_xticks(x, bin_labels, rotation=30)
    for ax in axes[:, 0]:
        ax.set_ylabel("mean reward")
    for ax in axes[-1, :]:
        ax.set_xlabel("tool calls per task")
    fig.suptitle("Reward vs Tool-Call Budget", fontweight="bold", y=0.995)
    fig.text(
        0.5,
        0.012,
        "Missing bins are shown as gaps; marker size reflects sample count (minimum n=5).",
        ha="center",
        fontsize=9,
    )
    savefig(fig, out / "14_reward_vs_tool_count_lines.png")


def plot_consistency(core: pd.DataFrame, out: Path, palette: dict[str, Any]) -> None:
    scenario = core.groupby(["model_variant", "scenario_label"], as_index=False)[
        "reward"
    ].mean()
    order = scenario.groupby("model_variant")["reward"].mean().sort_values().index
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.boxplot(
        data=scenario,
        x="model_variant",
        y="reward",
        order=order,
        hue="model_variant",
        palette=palette,
        legend=False,
        ax=ax,
    )
    sns.stripplot(
        data=scenario,
        x="model_variant",
        y="reward",
        order=order,
        color="0.15",
        alpha=0.5,
        size=4,
        jitter=0.18,
        ax=ax,
    )
    ax.set_ylim(0.25, 1.02)
    ax.set_xlabel("")
    ax.set_ylabel("scenario mean reward")
    ax.set_title("Cross-Scenario Consistency and Weak-Tail Risk")
    ax.tick_params(axis="x", rotation=32)
    savefig(fig, out / "15_cross_scenario_consistency.png")


def plot_task_win_share(core: pd.DataFrame, out: Path, palette: dict[str, Any]) -> None:
    pivot = core.pivot_table(
        index="task_id", columns="model_variant", values="reward", aggfunc="mean"
    )
    winners = pivot.eq(pivot.max(axis=1), axis=0)
    split_credit = winners.div(winners.sum(axis=1), axis=0)
    share = split_credit.mean().sort_values()
    fig, ax = plt.subplots(figsize=(12, max(6, 0.52 * len(share))))
    colors = [palette.get(model, (0.5, 0.5, 0.5)) for model in share.index]
    ax.barh(share.index, share.values, color=colors)
    for y, value in enumerate(share.values):
        ax.text(value + 0.004, y, f"{value:.1%}", va="center", fontsize=9)
    ax.set_xlabel("share of matched tasks led (ties split)")
    ax.set_ylabel("")
    ax.set_title("Task-Level Leadership Share")
    savefig(fig, out / "16_task_leadership_share.png")


def build_insights(
    core: pd.DataFrame,
    scorecard: pd.DataFrame,
    family: pd.DataFrame,
    scenario: pd.DataFrame,
    pairwise: pd.DataFrame,
) -> list[str]:
    insights: list[str] = []
    ranked = scorecard.sort_values("mean_reward", ascending=False).reset_index(
        drop=True
    )
    best, runner = ranked.iloc[0], ranked.iloc[1]
    insights.append(
        f"{best.model_variant} leads overall at {best.mean_reward:.3f} mean reward, "
        f"{best.mean_reward - runner.mean_reward:+.3f} ahead of {runner.model_variant}."
    )
    fastest = scorecard.loc[scorecard["mean_time_s"].replace(0, np.nan).idxmin()]
    leanest = scorecard.loc[scorecard["mean_output_tokens"].replace(0, np.nan).idxmin()]
    insights.append(
        f"{fastest.model_variant} is the fastest comparable run ({fastest.mean_time_s:.1f}s/task); "
        f"{leanest.model_variant} uses the fewest output tokens ({leanest.mean_output_tokens:.0f}/task)."
    )
    for task_family, group in family.groupby("family"):
        row = group.loc[group["mean_reward"].idxmax()]
        insights.append(
            f"Best on {task_family}: {row.model_variant} ({row.mean_reward:.3f} mean reward)."
        )
    scenario_mean = scenario.groupby(["family", "scenario"], as_index=False)[
        "mean_reward"
    ].mean()
    hardest = scenario_mean.loc[scenario_mean["mean_reward"].idxmin()]
    insights.append(
        f"The hardest scenario across models is {hardest.family}/{hardest.scenario} "
        f"({hardest.mean_reward:.3f} average model reward)."
    )
    spread = (
        scenario.groupby(["family", "scenario"])["mean_reward"]
        .agg(lambda x: x.max() - x.min())
        .sort_values(ascending=False)
    )
    (spread_family, spread_scenario), spread_value = spread.index[0], spread.iloc[0]
    insights.append(
        f"Models disagree most on {spread_family}/{spread_scenario}, with a {spread_value:.3f} best-to-worst reward gap."
    )
    gpt = scorecard[scorecard["model_family"].eq("GPT-5.5")]
    if len(gpt) >= 2:
        high = gpt[gpt["model_variant"].str.contains("high")]
        low = gpt[gpt["model_variant"].str.contains("low")]
        if not high.empty and not low.empty:
            high_row, low_row = high.iloc[0], low.iloc[0]
            insights.append(
                f"GPT-5.5 high vs low reasoning changes reward by {high_row.mean_reward - low_row.mean_reward:+.3f}, "
                f"time by {high_row.mean_time_s / max(low_row.mean_time_s, 1e-9):.2f}×, and output tokens by "
                f"{high_row.mean_output_tokens / max(low_row.mean_output_tokens, 1e-9):.2f}×."
            )
    if "GLM-5.2" in pairwise.index and "Kimi-K2.7-Code" in pairwise.columns:
        advantage = pairwise.loc["GLM-5.2", "Kimi-K2.7-Code"]
        insights.append(
            f"On matched tasks, GLM-5.2 has a {advantage:+.1%} net win-rate advantage over Kimi-K2.7-Code."
        )
    cleanest = scorecard.sort_values(["no_submit_rate", "tool_error_rate"]).iloc[0]
    insights.append(
        f"{cleanest.model_variant} has the cleanest completion signature: {cleanest.no_submit_rate:.1%} no-submit and "
        f"{cleanest.tool_error_rate:.1%} tool-error incidence."
    )
    return insights


def write_report(
    output_dir: Path,
    samples: pd.DataFrame,
    runs: pd.DataFrame,
    core_runs: pd.DataFrame,
    scorecard: pd.DataFrame,
    family: pd.DataFrame,
    scenario: pd.DataFrame,
    insights: list[str],
) -> None:
    excluded = runs[~runs["run_id"].isin(core_runs["run_id"])].copy()
    weakest = scenario.nsmallest(15, "mean_reward")[
        ["model_variant", "family", "scenario", "n", "mean_reward"]
    ]
    lines = [
        "# Comprehensive Model Performance & Behavior Analysis",
        "",
        f"This pack normalizes **{len(samples):,} deduplicated samples across {runs['run_id'].nunique()} evaluation runs**. "
        f"Core matched comparisons use **{len(core_runs)} latest complete balanced runs** ({int(scorecard['n'].sum()):,} samples), "
        "one per model/reasoning-effort variant.",
        "",
        "## Key insights",
        "",
        *[f"- {insight}" for insight in insights],
        "",
        "## Core benchmark scorecard",
        "",
        scorecard[
            [
                "model_variant",
                "n",
                "mean_reward",
                "reward_ci95",
                "perfect_rate",
                "zero_rate",
                "mean_turns",
                "mean_time_s",
                "mean_output_tokens",
                "no_submit_rate",
                "tool_error_rate",
            ]
        ].to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Coverage and comparability",
        "",
        "A run enters the core comparison only when it is balanced, has at least 350 samples and 350 unique task IDs, "
        "and preserves task metadata for at least 95% of samples. Repeated or partial exports remain visible in the all-run inventory.",
        "",
        core_runs[
            [
                "run_id",
                "model_variant",
                "reasoning_effort",
                "n",
                "unique_tasks",
                "sample_strategy",
                "max_turns",
                "time_limit_seconds",
                "source",
            ]
        ].to_markdown(index=False, floatfmt=".1f"),
        "",
    ]
    if not excluded.empty:
        lines.extend(
            [
                "### Runs excluded from matched core comparisons",
                "",
                excluded[
                    [
                        "run_id",
                        "model_variant",
                        "n",
                        "unique_tasks",
                        "sample_strategy",
                        "known_task_rate",
                        "source",
                    ]
                ].to_markdown(index=False, floatfmt=".3f"),
                "",
            ]
        )
    lines.extend(
        [
            "## Weakest model–scenario slices",
            "",
            weakest.to_markdown(index=False, floatfmt=".3f"),
            "",
            "## Interpretation notes",
            "",
            "- Reward is the primary performance measure; perfect/zero rates expose distribution shape hidden by the mean.",
            "- Wall time and tokens are observational efficiency measures. Different provider infrastructure and run limits can affect them.",
            "- Behavioral flags are derived from transcripts/tool events and should be interpreted as signatures, not causal explanations.",
            "- Pairwise comparisons use identical task IDs and split tied tasks, reducing task-mix confounding.",
            "- Confidence intervals are normal approximations over tasks; they do not model dependency across related seeds/scenarios.",
            "",
            "## Figures",
            "",
        ]
    )
    descriptions = {
        "01_all_run_inventory.png": "Every available run, including partial and repeated exports.",
        "02_core_model_scorecard.png": "Comparable full-run ranking with uncertainty and perfect-rate annotations.",
        "03_reward_distribution_violin.png": "Reliability, partial-credit mass, and failure tails.",
        "04_family_performance_heatmap.png": "Mean reward and perfect rate by task family.",
        "05_scenario_performance_heatmap.png": "Fine-grained model strengths and weaknesses.",
        "06_difficulty_scaling_lines.png": "Performance degradation across difficulty levels and families.",
        "07_behavioral_failure_heatmap.png": "No-submit, error, limit, guidance, and correction signatures.",
        "08_tool_use_signature_heatmaps.png": "Absolute tool use and relative strategy mix.",
        "09_efficiency_frontier.png": "Reward versus wall time and token use.",
        "10_gpt55_reasoning_effort_experiment.png": "Controlled low/medium/high reasoning-effort comparison.",
        "11_pairwise_task_advantage_heatmap.png": "Matched-task net win rates between every model pair.",
        "12_relative_scenario_specialization.png": "Where each model over- or under-performs the scenario consensus.",
        "13_stop_condition_composition.png": "How model runs terminate.",
        "14_reward_vs_tool_count_lines.png": "Relationship between action depth and reward.",
        "15_cross_scenario_consistency.png": "Scenario-level variance and weak-tail risk.",
        "16_task_leadership_share.png": "Fraction of matched tasks each model leads.",
    }
    for figure in sorted((output_dir / "figures").glob("*.png")):
        lines.append(
            f"- [{figure.name}](figures/{figure.name}) — {descriptions.get(figure.name, '')}"
        )
    lines.extend(
        [
            "",
            "## Reproduce",
            "",
            "```powershell",
            "uv run --with seaborn --with pandas --with matplotlib --with tabulate python scripts/analyze_all_model_eval_plots.py",
            "```",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def savefig(fig: plt.Figure, path: Path) -> None:
    top = 0.96 if fig._suptitle is not None else 0.99
    fig.tight_layout(pad=1.25, rect=(0.01, 0.035, 0.99, top))
    fig.savefig(path, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)


def ordered_variants(df: pd.DataFrame) -> list[str]:
    return sorted(df["model_variant"].dropna().unique(), key=order_index)


def order_index(value: str) -> int:
    try:
        return MODEL_ORDER.index(value)
    except ValueError:
        return len(MODEL_ORDER)


def first_not_none(*values: Any) -> Any:
    return next((value for value in values if value is not None), None)


def to_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


if __name__ == "__main__":
    main()
