"""Plot learning curves for all tasks.

Run from repo root:
    uv run python scripts/plot/learning_curves.py

One figure per entry in FIGURES, saved to figures/.
"""

import matplotlib.pyplot as plt
import pandas as pd

from utils import PROJECTS, STEP_COL, add_variant_label, load_history, save_figure

ALPHA_SEED = 0.25
ALPHA_BAND = 0.2
SMOOTH_SPAN = 30  # EMA half-life in steps

# Each dict defines one output figure.
#   name            — output filename ({name}.pdf)
#   projects        — list of project keys from PROJECTS; rows are concatenated
#   filter          — optional callable (df) -> bool Series applied after variant labelling
#   metric          — W&B metric column to plot
#   ylabel          — y-axis label
#   individual_seeds — True: thin per-seed lines + dashed mean; False: mean + std band
FIGURES = [
    dict(
        name="reach_success_mp",
        projects=["reach"],
        filter=lambda df: ~df["variant"].str.contains("UR10"),
        metric="Episode_Metrics/goal_reached_fraction",
        ylabel="Success rate",
        individual_seeds=True,
    ),
    dict(
        name="reach_success_ur10",
        projects=["reach"],
        filter=lambda df: df["variant"].str.contains("UR10"),
        metric="Episode_Metrics/goal_reached_fraction",
        ylabel="Success rate",
        individual_seeds=True,
    ),
        dict(
        name="reach_throughput_mp",
        projects=["reach"],
        filter=lambda df: ~df["variant"].str.contains("UR10"),
        metric="Episode_Metrics/goals_per_second",
        ylabel="Throughput (goals/s)",
        individual_seeds=True,
    ),
    dict(
        name="reach_throughput_ur10",
        projects=["reach"],
        filter=lambda df: df["variant"].str.contains("UR10"),
        metric="Episode_Metrics/goals_per_second",
        ylabel="Throughput (goals/s)",
        individual_seeds=True,
    ),
    dict(
        name="push_success",
        projects=["push"],
        metric="Episode_Metrics/targets_reached_fraction",
        ylabel="Success rate",
        individual_seeds=True,
    ),
        dict(
        name="push_throughput",
        projects=["push"],
        metric="Episode_Metrics/targets_per_second",
        ylabel="Throughput (goals/s)",
        individual_seeds=True,
    ),
    dict(
        name="type_throughput",
        projects=["type"],
        metric="Episode_Metrics/throughput",
        ylabel="Throughput (keys/episode)",
        individual_seeds=False,
    ),
    dict(
        name="type_wrong_keys",
        projects=["type"],
        metric="Episode_Metrics/wrong_keys_per_episode",
        ylabel="Mistakes (keys/episode)",
        individual_seeds=False,
    ),
    dict(
        name="push_no_curriculum",
        projects=["push-no-curriculum"],
        metric="Episode_Metrics/targets_reached_fraction",
        ylabel="Success rate",
        individual_seeds=True,
    ),
    # dict(
    #     name="type_no_curriculum_throughput",
    #     projects=["type-no-curriculum"],
    #     metric="Episode_Metrics/throughput",
    #     ylabel="Throughput (keys/episode)",
    #     individual_seeds=False,
    # ),
    # dict(
    #     name="type_1mp_curriculum_vs_no_curriculum",
    #     projects=["type", "type-no-curriculum"],
    #     project_labels={"type": "with curriculum", "type-no-curriculum": "no curriculum"},
    #     filter=lambda df: df["run_name"].str.contains("1mp"),
    #     metric="Episode_Metrics/throughput",
    #     ylabel="Throughput (keys/episode)",
    #     individual_seeds=True,
    # ),
    # dict(
    #     name="type_no_curriculum_wrong_keys",
    #     projects=["type-no-curriculum"],
    #     metric="Episode_Metrics/wrong_keys_per_episode",
    #     ylabel="Mistakes (keys/episode)",
    #     individual_seeds=False,
    # ),
]


def _ema(s: pd.Series) -> pd.Series:
    return s.ewm(span=SMOOTH_SPAN, adjust=False).mean()


def plot_metric(ax: plt.Axes, df: pd.DataFrame, metric: str, ylabel: str,
                individual_seeds: bool) -> None:
    for variant in sorted(df["variant"].unique()):
        vdf = df[df["variant"] == variant]

        seed_series = []
        for run in vdf["run_name"].unique():
            s = vdf[vdf["run_name"] == run][[STEP_COL, metric]].dropna()
            s = s.sort_values(STEP_COL).set_index(STEP_COL)[metric]
            seed_series.append(_ema(s))

        common = seed_series[0].index
        aligned = pd.concat(
            [s.reindex(common, method="nearest") for s in seed_series], axis=1
        )
        mean = aligned.mean(axis=1)
        std  = aligned.std(axis=1)

        if individual_seeds:
            color = ax._get_lines.get_next_color()
            for i in range(aligned.shape[1]):
                ax.plot(aligned.index, aligned.iloc[:, i],
                        color=color, alpha=ALPHA_SEED, linewidth=0.8)
            ax.plot(aligned.index, _ema(mean), color=color,
                    linewidth=1.8, linestyle="--", label=variant)
        else:
            line, = ax.plot(aligned.index, mean, linewidth=1.8, label=variant)
            ax.fill_between(aligned.index, mean - std, mean + std,
                            alpha=ALPHA_BAND, color=line.get_color())

    ax.set_xlabel("Timesteps")
    ax.set_ylabel(ylabel)
    ax.legend()


def plot_figure(cfg: dict) -> None:
    frames = []
    for key in cfg["projects"]:
        try:
            frames.append(load_history(PROJECTS[key]))
        except FileNotFoundError as e:
            print(e)
    if not frames:
        return

    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]

    if "project_labels" in cfg:
        df["variant"] = df["task"].map(cfg["project_labels"])
    else:
        df = add_variant_label(df)

    if "filter" in cfg:
        df = df[cfg["filter"](df)]

    metric = cfg["metric"]
    if metric not in df.columns:
        print(f"[{cfg['name']}] '{metric}' not found in data, skipping.")
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    plot_metric(ax, df, metric, cfg["ylabel"], cfg["individual_seeds"])
    fig.tight_layout()
    save_figure(fig, f"{cfg['name']}")
    plt.close(fig)


if __name__ == "__main__":
    for cfg in FIGURES:
        plot_figure(cfg)
