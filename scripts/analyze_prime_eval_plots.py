from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


TASK_RE = re.compile(r"- (task_id|split|family|scenario|difficulty|seed):\s*(.*)")
REQUIRED_RE = re.compile(r"Required final artifact:\s*(.*)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exports-dir", default="prime-eval-exports")
    parser.add_argument("--output-dir", default="analysis/eval_plots")
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
    samples.to_csv(tables_dir / "all_samples_normalized.csv", index=False)
    runs.to_csv(tables_dir / "run_scorecard.csv", index=False)

    scenario = aggregate(
        samples,
        [
            "run_id",
            "run_label",
            "model_short",
            "sample_strategy",
            "family",
            "scenario",
            "difficulty",
        ],
    )
    scenario.to_csv(tables_dir / "scenario_difficulty_metrics.csv", index=False)

    sns.set_theme(
        context="talk",
        style="whitegrid",
        rc={
            "figure.dpi": 140,
            "savefig.dpi": 180,
            "axes.titleweight": "bold",
            "axes.labelsize": 12,
            "axes.titlesize": 14,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
        },
    )
    palette = sns.color_palette(
        "tab10", n_colors=max(3, samples["model_short"].nunique())
    )
    model_palette = dict(zip(sorted(samples["model_short"].unique()), palette))

    plot_run_scorecard(runs, figures_dir, model_palette)
    plot_sampling_coverage(samples, figures_dir)
    plot_reward_distributions(samples, figures_dir, model_palette)
    plot_model_family_heatmap(samples, figures_dir)
    plot_scenario_heatmap(samples, figures_dir)
    plot_difficulty_curves(samples, figures_dir, model_palette)
    plot_scenario_difficulty_full(samples, figures_dir)
    plot_qwen_full_delta(samples, figures_dir)
    plot_efficiency_tradeoffs(samples, figures_dir, model_palette)
    plot_failure_modes(samples, figures_dir)
    plot_tool_signature(samples, figures_dir)
    plot_time_limit_behavior(samples, figures_dir, model_palette)
    plot_reward_efficiency_frontier(runs, figures_dir, model_palette)

    write_report(samples, runs, output_dir)
    print(f"Wrote normalized tables to {tables_dir}")
    print(f"Wrote figures to {figures_dir}")
    print(f"Wrote report to {output_dir / 'README.md'}")


def load_all_exports(exports_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    for run_dir in sorted(exports_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        samples_path = run_dir / "samples.json"
        if not samples_path.exists():
            continue
        metadata = (
            load_json(run_dir / "metadata.json")
            if (run_dir / "metadata.json").exists()
            else {}
        )
        sample_payload = load_json(samples_path)
        samples = sample_payload.get(
            "samples", sample_payload if isinstance(sample_payload, list) else []
        )
        if not isinstance(samples, list) or not samples:
            continue
        run_id = str(
            metadata.get("evaluation_id")
            or sample_payload.get("evaluation_id")
            or run_dir.name
        )
        run = parse_run_metadata(run_id, metadata, samples)
        run_rows.append(run)
        for sample in samples:
            sample_rows.append(parse_sample(run, sample))
    return pd.DataFrame(sample_rows), pd.DataFrame(run_rows)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def parse_run_metadata(
    run_id: str, metadata: dict[str, Any], samples: list[dict[str, Any]]
) -> dict[str, Any]:
    cfg = metadata.get("eval_config") or {}
    env_args = cfg.get("env_args") or {}
    model = (
        metadata.get("model_name")
        or metadata.get("inference_model")
        or cfg.get("model")
        or "unknown"
    )
    sample_strategy = str(env_args.get("sample_strategy") or "head/default")
    short = short_model_name(str(model))
    max_turns = env_args.get("max_turns")
    time_limit = env_args.get("time_limit_seconds")
    num_examples = int(
        metadata.get("total_samples") or cfg.get("num_examples") or len(samples)
    )
    run_label = (
        f"{short}\n"
        f"n={len(samples)}/{num_examples}, {sample_strategy}, "
        f"t={max_turns or '?'}"
        f"{', wall=' + str(time_limit) + 's' if time_limit else ''}\n"
        f"{run_id[:6]}"
    )
    return {
        "run_id": run_id,
        "run_label": run_label,
        "model": model,
        "model_short": short,
        "sample_strategy": sample_strategy,
        "num_examples_config": num_examples,
        "num_samples_loaded": len(samples),
        "created_at": metadata.get("created_at") or "",
        "avg_score_metadata": metadata.get("avg_score"),
        "max_turns": max_turns,
        "max_tool_steps": env_args.get("max_tool_steps"),
        "time_limit_seconds": time_limit,
        "shuffle": env_args.get("shuffle"),
        "shuffle_seed": env_args.get("shuffle_seed"),
        "split": env_args.get("split"),
    }


def parse_sample(run: dict[str, Any], sample: dict[str, Any]) -> dict[str, Any]:
    prompt = sample.get("prompt") or []
    user = ""
    if isinstance(prompt, list):
        user = next(
            (str(m.get("content") or "") for m in prompt if m.get("role") == "user"), ""
        )
    metadata = parse_task_metadata(user)
    completion = sample.get("completion") or []
    tools, paths = parse_tool_calls(completion)
    info = sample.get("info") or {}
    timing = info.get("timing") or {}
    usage = info.get("token_usage") or {}
    metrics = info.get("metrics") or {}
    text_blob = completion_text_blob(completion)
    reward = sample.get("reward")
    if reward is None:
        reward = sample.get("swg_reward")
    if reward is None:
        reward = metrics.get("swg_reward")
    return {
        **{
            key: run[key]
            for key in [
                "run_id",
                "run_label",
                "model",
                "model_short",
                "sample_strategy",
                "created_at",
            ]
        },
        "example_id": sample.get("example_id"),
        "reward": float(reward or 0.0),
        "family": metadata.get("family", "unknown"),
        "scenario": metadata.get("scenario", "unknown"),
        "difficulty": int(metadata.get("difficulty") or 0),
        "seed": int(metadata.get("seed") or 0),
        "task_id": metadata.get("task_id", ""),
        "required_artifact": metadata.get("required_artifact", ""),
        "num_turns": float(metrics.get("num_turns") or 0.0),
        "time_s": float(timing.get("total") or sample.get("total_time") or 0.0),
        "model_time_s": float((timing.get("model") or {}).get("duration") or 0.0),
        "env_time_s": float((timing.get("env") or {}).get("duration") or 0.0),
        "input_tokens": float(usage.get("input_tokens") or 0.0),
        "output_tokens": float(usage.get("output_tokens") or 0.0),
        "stop_condition": str(info.get("stop_condition") or "unknown"),
        "is_truncated": bool(info.get("is_truncated") or False),
        "tool_count": len(tools),
        "read_count": tools.count("read_file"),
        "list_count": tools.count("list_directory"),
        "write_count": tools.count("write_file") + tools.count("append_file"),
        "run_count": tools.count("run_shell") + tools.count("run_python"),
        "submit_count": tools.count("submit"),
        "tool_error_count": count_tool_errors(completion),
        "guidance_count": sum(
            "Guidance:" in str(m.get("content") or "")
            for m in completion
            if m.get("role") == "tool"
        ),
        "submit_correction_count": sum(
            "Submit correction" in str(m.get("content") or "")
            for m in completion
            if m.get("role") == "tool"
        ),
        "uses_pandas": bool(
            re.search(r"\b(import pandas|pd\.read_|pandas)\b", text_blob, flags=re.I)
        ),
        "has_key_error": "KeyError" in text_blob,
        "has_file_not_found": "FileNotFoundError" in text_blob,
        "has_indentation_error": "IndentationError" in text_blob,
        "has_model_error": "ModelError" in str(info.get("error") or ""),
        "tools": " ".join(tools),
        "paths_tail": " | ".join(paths[-8:]),
    }


def parse_task_metadata(user_message: str) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for line in user_message.splitlines():
        match = TASK_RE.match(line.strip())
        if match:
            metadata[match.group(1)] = match.group(2).strip()
    required = REQUIRED_RE.search(user_message)
    if required:
        metadata["required_artifact"] = required.group(1).strip()
    return metadata


def parse_tool_calls(messages: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    tools: list[str] = []
    paths: list[str] = []
    for message in messages:
        for raw_call in message.get("tool_calls") or []:
            try:
                call = json.loads(raw_call)
                args = json.loads(call.get("arguments") or "{}")
            except Exception:
                continue
            name = str(call.get("name") or "")
            path = (
                args.get("path")
                or args.get("path_or_answer")
                or args.get("command_or_script")
                or args.get("command")
            )
            tools.append(name)
            paths.append(str(path or ""))
    return tools, paths


def completion_text_blob(messages: list[dict[str, Any]]) -> str:
    chunks: list[str] = []
    for message in messages:
        chunks.append(str(message.get("content") or ""))
        chunks.append(str(message.get("reasoning_content") or ""))
        chunks.extend(str(call) for call in message.get("tool_calls") or [])
    return "\n".join(chunks)


def count_tool_errors(messages: list[dict[str, Any]]) -> int:
    count = 0
    for message in messages:
        if message.get("role") != "tool":
            continue
        content = str(message.get("content") or "").lower()
        if any(
            marker in content
            for marker in (
                "error:",
                "failed",
                "rejected",
                "traceback",
                "filenotfounderror",
            )
        ):
            count += 1
    return count


def short_model_name(model: str) -> str:
    lowered = model.lower()
    if "qwen3.5-4b" in lowered or "qwen3-5-4b" in lowered:
        return "Qwen3.5-4B"
    if "235b" in lowered and "thinking" in lowered:
        return "Qwen3-235B-Thinking"
    if "235b" in lowered and "instruct" in lowered:
        return "Qwen3-235B-Instruct"
    if "nemotron" in lowered:
        return "Nemotron-120B"
    return model.split("/")[-1]


def enrich_samples(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_perfect"] = df["reward"].eq(1.0)
    df["is_zero"] = df["reward"].eq(0.0)
    df["is_partial"] = df["reward"].between(0.000001, 0.999999)
    df["no_submit"] = df["submit_count"].eq(0)
    df["time_per_turn"] = df["time_s"] / df["num_turns"].replace(0, np.nan)
    df["tokens_per_turn"] = df["output_tokens"] / df["num_turns"].replace(0, np.nan)
    df["reward_per_1k_output_tokens"] = df["reward"] / (
        df["output_tokens"].replace(0, np.nan) / 1000.0
    )
    df["reward_bucket"] = pd.cut(
        df["reward"],
        bins=[-0.001, 0.0001, 0.4, 0.6, 0.8, 0.9999, 1.0001],
        labels=["0", "(0, .4]", "(.4, .6]", "(.6, .8]", "(.8, <1)", "1"],
        include_lowest=True,
    )
    df["scenario_label"] = df["family"] + "/" + df["scenario"]
    df["run_order"] = df.groupby(["created_at", "run_id"], sort=True).ngroup()
    return df


def enrich_runs(runs: pd.DataFrame, samples: pd.DataFrame) -> pd.DataFrame:
    metrics = aggregate(samples, ["run_id"])
    df = runs.merge(metrics, on="run_id", how="left")
    df["run_order"] = df.sort_values(["created_at", "run_id"]).reset_index().index
    df["reward_per_minute"] = df["mean_reward"] / (df["mean_time_s"] / 60.0)
    df["reward_per_1k_output_tokens"] = df["mean_reward"] / (
        df["mean_output_tokens"] / 1000.0
    )
    df["loaded_fraction"] = df["num_samples_loaded"] / df[
        "num_examples_config"
    ].replace(0, np.nan)
    return df


def aggregate(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    grouped = df.groupby(keys, dropna=False)
    result = grouped.agg(
        n=("reward", "size"),
        mean_reward=("reward", "mean"),
        median_reward=("reward", "median"),
        std_reward=("reward", "std"),
        perfect_rate=("is_perfect", "mean"),
        zero_rate=("is_zero", "mean"),
        mean_turns=("num_turns", "mean"),
        mean_time_s=("time_s", "mean"),
        median_time_s=("time_s", "median"),
        mean_output_tokens=("output_tokens", "mean"),
        median_output_tokens=("output_tokens", "median"),
        no_submit_rate=("no_submit", "mean"),
        tool_error_rate=("tool_error_count", lambda x: (x > 0).mean()),
        mean_tool_count=("tool_count", "mean"),
        mean_read_count=("read_count", "mean"),
        mean_write_count=("write_count", "mean"),
        mean_run_count=("run_count", "mean"),
    )
    return result.reset_index()


def savefig(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_run_scorecard(
    runs: pd.DataFrame, out: Path, palette: dict[str, tuple[float, float, float]]
) -> None:
    data = runs.sort_values(["created_at", "run_id"]).copy()
    fig, ax = plt.subplots(figsize=(16, 7))
    sns.barplot(
        data=data,
        x="run_label",
        y="mean_reward",
        hue="model_short",
        palette=palette,
        dodge=False,
        ax=ax,
    )
    ax.scatter(
        np.arange(len(data)),
        data["perfect_rate"],
        marker="D",
        s=80,
        color="black",
        label="perfect rate",
        zorder=4,
    )
    ax.scatter(
        np.arange(len(data)),
        data["zero_rate"],
        marker="X",
        s=80,
        color="#b00020",
        label="zero rate",
        zorder=4,
    )
    for i, row in enumerate(data.itertuples()):
        ax.text(
            i,
            min(1.05, row.mean_reward + 0.035),
            f"{row.mean_reward:.2f}\nn={int(row.n)}",
            ha="center",
            fontsize=8,
        )
    ax.set_ylim(0, 1.08)
    ax.set_title("Run Scorecard: Mean Reward With Perfect/Zero Rates")
    ax.set_xlabel("")
    ax.set_ylabel("reward / rate")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    savefig(fig, out / "01_run_scorecard_mean_reward.png")


def plot_sampling_coverage(samples: pd.DataFrame, out: Path) -> None:
    counts = samples.pivot_table(
        index="run_label",
        columns="family",
        values="example_id",
        aggfunc="count",
        fill_value=0,
    )
    counts = counts.loc[
        samples.drop_duplicates("run_label").sort_values(["created_at", "run_id"])[
            "run_label"
        ]
    ]
    fig, ax = plt.subplots(figsize=(11, 8))
    sns.heatmap(counts, cmap="YlGnBu", annot=True, fmt=".0f", linewidths=0.5, ax=ax)
    ax.set_title("Sampling Coverage Audit: Tasks Per Family By Run")
    ax.set_xlabel("family")
    ax.set_ylabel("")
    savefig(fig, out / "02_sampling_coverage_family_counts.png")


def plot_reward_distributions(
    samples: pd.DataFrame, out: Path, palette: dict[str, tuple[float, float, float]]
) -> None:
    data = samples.sort_values(["created_at", "run_id"]).copy()
    fig, ax = plt.subplots(figsize=(17, 8))
    sns.violinplot(
        data=data,
        x="run_label",
        y="reward",
        hue="model_short",
        palette=palette,
        inner=None,
        cut=0,
        dodge=False,
        ax=ax,
    )
    sns.stripplot(
        data=data,
        x="run_label",
        y="reward",
        color="black",
        alpha=0.18,
        size=2.5,
        jitter=0.28,
        ax=ax,
    )
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("Reward Distribution Per Evaluation Run")
    ax.set_xlabel("")
    ax.set_ylabel("reward")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    savefig(fig, out / "03_reward_distribution_violin_strip.png")


def plot_model_family_heatmap(samples: pd.DataFrame, out: Path) -> None:
    data = samples.copy()
    mean = data.pivot_table(
        index="run_label", columns="family", values="reward", aggfunc="mean"
    )
    order = samples.drop_duplicates("run_label").sort_values(["created_at", "run_id"])[
        "run_label"
    ]
    mean = mean.loc[order]
    annot = data.pivot_table(
        index="run_label",
        columns="family",
        values="is_perfect",
        aggfunc=lambda x: f"{x.mean():.0%}",
    )
    annot = annot.reindex_like(mean)
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(
        mean, cmap="RdYlGn", vmin=0, vmax=1, annot=annot, fmt="", linewidths=0.5, ax=ax
    )
    ax.set_title("Family Robustness Heatmap: Color=Mean Reward, Text=Perfect Rate")
    ax.set_xlabel("family")
    ax.set_ylabel("")
    savefig(fig, out / "04_family_reward_heatmap_per_run.png")


def plot_scenario_heatmap(samples: pd.DataFrame, out: Path) -> None:
    full = samples.groupby("run_id").filter(lambda x: len(x) >= 300).copy()
    if full.empty:
        full = samples.copy()
    full = full[~full["family"].eq("unknown")].copy()
    if full.empty:
        return
    mean = full.pivot_table(
        index="run_label", columns="scenario_label", values="reward", aggfunc="mean"
    )
    order = full.drop_duplicates("run_label").sort_values(["created_at", "run_id"])[
        "run_label"
    ]
    mean = mean.loc[order]
    fig, ax = plt.subplots(figsize=(18, max(6, 0.55 * len(mean))))
    sns.heatmap(
        mean,
        cmap="vlag",
        vmin=0,
        vmax=1,
        center=0.65,
        annot=True,
        fmt=".2f",
        linewidths=0.4,
        ax=ax,
    )
    ax.set_title("Scenario-Level Reward Heatmap For Full/Long Runs")
    ax.set_xlabel("scenario")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=45)
    savefig(fig, out / "05_full_run_scenario_reward_heatmap.png")


def plot_difficulty_curves(
    samples: pd.DataFrame, out: Path, palette: dict[str, tuple[float, float, float]]
) -> None:
    full = samples.groupby("run_id").filter(lambda x: len(x) >= 300).copy()
    if full.empty:
        full = samples.copy()
    g = sns.relplot(
        data=full,
        x="difficulty",
        y="reward",
        hue="run_label",
        col="family",
        kind="line",
        errorbar=("ci", 68),
        marker="o",
        facet_kws={"sharey": True, "sharex": True},
        height=4,
        aspect=1.05,
    )
    g.set(ylim=(-0.02, 1.02), xticks=sorted(full["difficulty"].dropna().unique()))
    g.set_axis_labels("difficulty", "mean reward")
    g.fig.suptitle(
        "Difficulty Scaling By Family: Full/Long Runs", y=1.05, fontweight="bold"
    )
    g._legend.set_title("run")
    g.fig.savefig(
        out / "06_difficulty_scaling_by_family.png", bbox_inches="tight", dpi=180
    )
    plt.close(g.fig)


def plot_scenario_difficulty_full(samples: pd.DataFrame, out: Path) -> None:
    for run_id, data in samples.groupby("run_id"):
        if len(data) < 300:
            continue
        data = data[~data["family"].eq("unknown")].copy()
        if data.empty:
            continue
        label = sanitize_filename(str(data["model_short"].iloc[0]) + "_" + run_id[:6])
        matrix = data.pivot_table(
            index="scenario_label",
            columns="difficulty",
            values="reward",
            aggfunc="mean",
        )
        fig, ax = plt.subplots(figsize=(7, max(7, 0.42 * len(matrix))))
        sns.heatmap(
            matrix,
            cmap="RdYlGn",
            vmin=0,
            vmax=1,
            annot=True,
            fmt=".2f",
            linewidths=0.4,
            ax=ax,
        )
        ax.set_title(f"Scenario x Difficulty Matrix\n{data['run_label'].iloc[0]}")
        ax.set_xlabel("difficulty")
        ax.set_ylabel("")
        savefig(fig, out / f"07_scenario_difficulty_matrix_{label}.png")


def plot_qwen_full_delta(samples: pd.DataFrame, out: Path) -> None:
    full = samples.groupby("run_id").filter(lambda x: len(x) >= 300).copy()
    q35 = full[full["model_short"].eq("Qwen3.5-4B")]
    thinking = full[full["model_short"].eq("Qwen3-235B-Thinking")]
    if q35.empty or thinking.empty:
        return
    # Use the latest balanced/full Qwen3.5-4B run as the baseline.
    q35_run = q35.sort_values(["created_at", "run_id"])["run_id"].iloc[-1]
    thinking_run = thinking.sort_values(["created_at", "run_id"])["run_id"].iloc[-1]
    base = (
        q35[q35["run_id"].eq(q35_run)]
        .groupby(["scenario_label", "difficulty"])["reward"]
        .mean()
    )
    comp = (
        thinking[thinking["run_id"].eq(thinking_run)]
        .groupby(["scenario_label", "difficulty"])["reward"]
        .mean()
    )
    delta = (comp - base).reset_index(name="delta_reward")
    matrix = delta.pivot(
        index="scenario_label", columns="difficulty", values="delta_reward"
    )
    fig, ax = plt.subplots(figsize=(8, max(7, 0.45 * len(matrix))))
    sns.heatmap(
        matrix, cmap="coolwarm", center=0, annot=True, fmt="+.2f", linewidths=0.4, ax=ax
    )
    ax.set_title(
        "Delta Heatmap: Qwen3-235B Thinking Full Run Minus Qwen3.5-4B Full Run"
    )
    ax.set_xlabel("difficulty")
    ax.set_ylabel("")
    savefig(fig, out / "08_delta_qwen235b_thinking_minus_qwen35_4b_full.png")


def plot_efficiency_tradeoffs(
    samples: pd.DataFrame, out: Path, palette: dict[str, tuple[float, float, float]]
) -> None:
    full = samples.groupby("run_id").filter(lambda x: len(x) >= 300).copy()
    if full.empty:
        full = samples.copy()
    fig, axes = plt.subplots(1, 2, figsize=(17, 7), sharey=True)
    sns.scatterplot(
        data=full,
        x="output_tokens",
        y="reward",
        hue="model_short",
        style="family",
        size="num_turns",
        sizes=(20, 150),
        alpha=0.62,
        palette=palette,
        ax=axes[0],
    )
    axes[0].set_xscale("log")
    axes[0].set_title("Reward vs Output Tokens")
    axes[0].set_xlabel("output tokens (log)")
    sns.scatterplot(
        data=full,
        x="time_s",
        y="reward",
        hue="model_short",
        style="family",
        size="num_turns",
        sizes=(20, 150),
        alpha=0.62,
        palette=palette,
        legend=False,
        ax=axes[1],
    )
    axes[1].set_xscale("log")
    axes[1].set_title("Reward vs Wall Time")
    axes[1].set_xlabel("seconds (log)")
    axes[0].set_ylabel("reward")
    axes[0].legend(loc="upper left", bbox_to_anchor=(2.22, 1))
    fig.suptitle(
        "Efficiency Tradeoff: Thinking Tokens And Wall Time Do Not Monotonically Buy Reward",
        fontweight="bold",
    )
    savefig(fig, out / "09_efficiency_tradeoff_reward_vs_tokens_time.png")


def plot_failure_modes(samples: pd.DataFrame, out: Path) -> None:
    data = samples.copy()
    bucket = (
        data.groupby(["run_label", "reward_bucket"], observed=False)
        .size()
        .reset_index(name="count")
    )
    bucket["fraction"] = bucket["count"] / bucket.groupby("run_label")[
        "count"
    ].transform("sum")
    order = (
        data.drop_duplicates("run_label")
        .sort_values(["created_at", "run_id"])["run_label"]
        .tolist()
    )
    pivot = (
        bucket.pivot(index="run_label", columns="reward_bucket", values="fraction")
        .fillna(0)
        .loc[order]
    )
    fig, ax = plt.subplots(figsize=(16, 7))
    pivot.plot(kind="bar", stacked=True, colormap="Spectral", ax=ax, width=0.82)
    ax.set_title("Reward Bucket Composition By Run")
    ax.set_xlabel("")
    ax.set_ylabel("fraction of rollouts")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(title="reward bucket", bbox_to_anchor=(1.01, 1), loc="upper left")
    savefig(fig, out / "10_reward_bucket_stacked_bars.png")

    flags = data.groupby("run_label").agg(
        no_submit=("no_submit", "mean"),
        tool_error=("tool_error_count", lambda x: (x > 0).mean()),
        pandas_usage=("uses_pandas", "mean"),
        key_error=("has_key_error", "mean"),
        file_not_found=("has_file_not_found", "mean"),
    )
    flags = flags.loc[order]
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.heatmap(
        flags,
        cmap="mako_r",
        vmin=0,
        vmax=1,
        annot=True,
        fmt=".0%",
        linewidths=0.4,
        ax=ax,
    )
    ax.set_title("Behavioral Failure/Overhead Flags By Run")
    ax.set_xlabel("flag")
    ax.set_ylabel("")
    savefig(fig, out / "11_behavioral_failure_flags_heatmap.png")


def plot_tool_signature(samples: pd.DataFrame, out: Path) -> None:
    full = samples.groupby("run_id").filter(lambda x: len(x) >= 300).copy()
    if full.empty:
        full = samples.copy()
    tool_cols = [
        "read_count",
        "list_count",
        "write_count",
        "run_count",
        "submit_count",
        "tool_error_count",
    ]
    matrix = full.groupby(["model_short", "family"])[tool_cols].mean()
    display = matrix.copy()
    display.index = [f"{model}\n{family}" for model, family in display.index]
    fig, ax = plt.subplots(figsize=(10, max(6, 0.45 * len(display))))
    sns.heatmap(display, cmap="crest", annot=True, fmt=".1f", linewidths=0.4, ax=ax)
    ax.set_title("Tool-Use Signature: Mean Tool Counts Per Rollout")
    ax.set_xlabel("tool/action count")
    ax.set_ylabel("")
    savefig(fig, out / "12_tool_use_signature_heatmap.png")


def plot_time_limit_behavior(
    samples: pd.DataFrame, out: Path, palette: dict[str, tuple[float, float, float]]
) -> None:
    data = samples.copy()
    data["near_420s"] = data["time_s"].between(390, 460)
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.scatterplot(
        data=data,
        x="time_s",
        y="reward",
        hue="model_short",
        style="no_submit",
        size="output_tokens",
        sizes=(20, 180),
        alpha=0.65,
        palette=palette,
        ax=ax,
    )
    ax.axvline(
        420, color="#b00020", linestyle="--", linewidth=1.5, label="420s task budget"
    )
    ax.set_xscale("log")
    ax.set_title(
        "Wall-Clock Budget Effects: No-Submit Rollouts Cluster Near Time Limit"
    )
    ax.set_xlabel("seconds (log)")
    ax.set_ylabel("reward")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    savefig(fig, out / "13_time_limit_no_submit_behavior.png")


def plot_reward_efficiency_frontier(
    runs: pd.DataFrame, out: Path, palette: dict[str, tuple[float, float, float]]
) -> None:
    data = runs.copy()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.scatterplot(
        data=data,
        x="mean_time_s",
        y="mean_reward",
        hue="model_short",
        size="n",
        sizes=(80, 500),
        palette=palette,
        ax=axes[0],
    )
    axes[0].set_xscale("log")
    axes[0].set_title("Run-Level Reward vs Mean Seconds")
    axes[0].set_xlabel("mean seconds/sample (log)")
    axes[0].set_ylabel("mean reward")
    sns.scatterplot(
        data=data,
        x="mean_output_tokens",
        y="mean_reward",
        hue="model_short",
        size="n",
        sizes=(80, 500),
        palette=palette,
        legend=False,
        ax=axes[1],
    )
    axes[1].set_xscale("log")
    axes[1].set_title("Run-Level Reward vs Mean Output Tokens")
    axes[1].set_xlabel("mean output tokens/sample (log)")
    axes[1].set_ylabel("")
    for ax in axes:
        ax.set_ylim(-0.02, 1.04)
    axes[0].legend(loc="upper left", bbox_to_anchor=(2.23, 1))
    fig.suptitle("Run Efficiency Frontier", fontweight="bold")
    savefig(fig, out / "14_run_efficiency_frontier.png")


def write_report(samples: pd.DataFrame, runs: pd.DataFrame, output_dir: Path) -> None:
    full = samples.groupby("run_id").filter(lambda x: len(x) >= 300)
    full_known = full[~full["family"].eq("unknown")].copy()
    weakest = (
        full_known.groupby(["model_short", "family", "scenario"])["reward"]
        .mean()
        .reset_index()
        .sort_values("reward")
        .head(15)
    )
    strongest = (
        full_known.groupby(["model_short", "family", "scenario"])["reward"]
        .mean()
        .reset_index()
        .sort_values("reward", ascending=False)
        .head(15)
    )
    no_submit = (
        samples[samples["no_submit"]]
        .groupby(["model_short", "family", "scenario"])
        .size()
        .reset_index(name="count")
    )
    lines = [
        "# Prime Eval Plot Pack",
        "",
        f"Normalized samples: `{len(samples)}` across `{samples['run_id'].nunique()}` runs.",
        "",
        "Caveat: runs with `unknown/unknown` task metadata are retained for run-level and efficiency plots, "
        "but excluded from scenario/family rankings because their hosted export did not preserve task metadata.",
        "",
        "## Run Scorecard",
        "",
        runs.sort_values(["created_at", "run_id"])[
            [
                "run_id",
                "model_short",
                "sample_strategy",
                "n",
                "mean_reward",
                "perfect_rate",
                "zero_rate",
                "mean_turns",
                "mean_time_s",
                "mean_output_tokens",
                "no_submit_rate",
            ]
        ].to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Weakest Full-Run Scenario Slices",
        "",
        weakest.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Strongest Full-Run Scenario Slices",
        "",
        strongest.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## No-Submit Concentrations",
        "",
        no_submit.sort_values("count", ascending=False)
        .head(20)
        .to_markdown(index=False),
        "",
        "## Figures",
        "",
    ]
    for figure in sorted((output_dir / "figures").glob("*.png")):
        lines.append(f"- [{figure.name}](figures/{figure.name})")
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def sanitize_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


if __name__ == "__main__":
    main()
