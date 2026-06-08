"""Plot learning curves for all tasks.

Run from repo root:
    uv run python scripts/plot/learning_curves.py

One figure per (task, metric), saved to figures/.
"""

import matplotlib.pyplot as plt
import pandas as pd

from utils import STEP_COL, add_variant_label, load_history, save_figure

PROJECTS = {
    "push":  "multi-robot-rl-push-2026-06-06_12-22-30",
    "type":  "multi-robot-rl-type-2026-06-05_13-48-30",
    # "reach": "multi-robot-rl-euler-reach",
}

# Metrics to plot per task and their y-axis labels.
PLOTS = {
    "push": [
        ("Episode_Metrics/targets_reached_fraction", "Success rate"),
    ],
    "type": [
        ("Episode_Metrics/throughput",             "Correct keys per episode"),
        ("Episode_Metrics/wrong_keys_per_episode", "Wrong keys per episode"),
    ],
    "reach": [
        ("Episode_Metrics/goal_reached_fraction",  "Average fraction of goals reached"),
    ],
}

# Individual seed lines for step-like curves (reach); std band for smooth ones.
INDIVIDUAL_SEEDS = {"reach": True, "push": False, "type": False}

ALPHA_SEED = 0.25
ALPHA_BAND = 0.2


def plot_metric(ax: plt.Axes, df: pd.DataFrame, task: str,
                metric: str, ylabel: str) -> None:
    for variant in sorted(df["variant"].unique()):
        vdf = df[df["variant"] == variant]

        seed_series = []
        for run in vdf["run_name"].unique():
            s = vdf[vdf["run_name"] == run][[STEP_COL, metric]].dropna()
            s = s.sort_values(STEP_COL).set_index(STEP_COL)[metric]
            seed_series.append(s)

        common = seed_series[0].index
        aligned = pd.concat(
            [s.reindex(common, method="nearest") for s in seed_series], axis=1
        )
        mean = aligned.mean(axis=1)
        std  = aligned.std(axis=1)

        if INDIVIDUAL_SEEDS[task]:
            for col in aligned.columns:
                line, = ax.plot(aligned.index, aligned[col],
                                alpha=ALPHA_SEED, linewidth=0.8)
            ax.plot(aligned.index, mean, color=line.get_color(),
                    linewidth=1.8, label=variant)
        else:
            line, = ax.plot(aligned.index, mean, linewidth=1.8, label=variant)
            ax.fill_between(aligned.index, mean - std, mean + std,
                            alpha=ALPHA_BAND, color=line.get_color())

    ax.set_xlabel("Timesteps")
    ax.set_ylabel(ylabel)
    ax.legend()


def plot_task(task: str, project: str) -> None:
    df = load_history(project)
    df = add_variant_label(df)

    for metric, ylabel in PLOTS[task]:
        if metric not in df.columns:
            print(f"[{task}] '{metric}' not found in data, skipping.")
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        plot_metric(ax, df, task, metric, ylabel)
        fig.tight_layout()
        save_figure(fig, f"learning_curves_{task}_{metric.split('/')[-1]}")
        plt.close(fig)


if __name__ == "__main__":
    for task, project in PROJECTS.items():
        try:
            plot_task(task, project)
        except FileNotFoundError as e:
            print(e)
