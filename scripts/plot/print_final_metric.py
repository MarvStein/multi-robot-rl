"""Print final mean throughput for the type task variants.

Run from repo root:
    uv run python scripts/plot/print_final_throughput.py
"""

import pandas as pd
from utils import PROJECTS, STEP_COL, add_variant_label, load_history

METRIC = "Episode_Metrics/wrong_keys_per_episode"
SMOOTH_SPAN = 30
FINAL_FRAC = 0.1  # average over the last 10% of steps


def _ema(s: pd.Series) -> pd.Series:
    return s.ewm(span=SMOOTH_SPAN, adjust=False).mean()


df = load_history(PROJECTS["type"])
df = add_variant_label(df)

for variant in sorted(df["variant"].unique()):
    vdf = df[df["variant"] == variant]
    seed_finals = []
    for run in vdf["run_name"].unique():
        s = vdf[vdf["run_name"] == run][[STEP_COL, METRIC]].dropna()
        s = s.sort_values(STEP_COL).set_index(STEP_COL)[METRIC]
        s = _ema(s)
        cutoff = s.index[-1] * (1 - FINAL_FRAC)
        seed_finals.append(s[s.index >= cutoff].mean())
    print(f"{variant}: {pd.Series(seed_finals).mean():.4f} ± {pd.Series(seed_finals).std():.4f}")
