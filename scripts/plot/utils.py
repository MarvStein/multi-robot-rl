"""Shared utilities for plotting: data loading, variant labelling, figure saving."""

import re
from pathlib import Path

import pandas as pd
import seaborn as sns

sns.set_theme(style="darkgrid")

DATA_DIR = Path(__file__).parent.parent.parent / "data"
FIGURES_DIR = Path(__file__).parent.parent.parent / "figures"

STEP_COL = "Train/env_steps"

PROJECTS = {
    "reach-old":                "multi-robot-rl-euler-reach",
    "push-old":                 "multi-robot-rl-push-2026-06-06_12-22-30",
    "reach":                    "multi-robot-rl-euler-reach-v2",
    "push":                     "multi-robot-rl-euler-push-v2",
    "type":                     "multi-robot-rl-type-2026-06-05_13-48-30",
    "type-no-curriculum":       "multi-robot-rl-type-no-curriculum-2026-06-08_12-34-39",
    "push-no-curriculum":       "multi-robot-rl-push-no-curriculum-2026-06-08_15-35-00",
}


def load_history(project: str) -> pd.DataFrame:
    path = DATA_DIR / f"{project}_histories.csv"
    if not path.exists():
        raise FileNotFoundError(f"No data found at {path}. Run download_runs.py first.")
    return pd.read_csv(path)


def variant_label(row: pd.Series) -> str:
    """Extract robot composition from the run name (e.g. 'ppo_2mp_1cube_seed0' → '2 MPs')."""
    name = str(row.get("run_name", ""))
    parts = []
    mp = re.search(r"(\d+)mp", name)
    ur = re.search(r"(\d+)ur10", name)
    if mp:
        n = int(mp.group(1))
        parts.append(f"{n} MP{'s' if n > 1 else ''}")
    if ur:
        n = int(ur.group(1))
        parts.append(f"{n} UR10{'s' if n > 1 else ''}")
    return " + ".join(parts) if parts else name


def add_variant_label(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["variant"] = df.apply(variant_label, axis=1)
    return df


def save_figure(fig, name: str) -> None:
    FIGURES_DIR.mkdir(exist_ok=True)
    path = FIGURES_DIR / f"{name}.pdf"
    fig.savefig(path, bbox_inches="tight")
    print(f"Saved {path}")
