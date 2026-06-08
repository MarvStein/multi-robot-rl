"""Download all W&B run data for multi-robot-rl experiments and save to data/.

Run from repo root:
    python scripts/plot/download_runs.py

Output:
    data/<project>_histories.csv   — one row per (run, step), metrics over time
"""

from pathlib import Path

import pandas as pd
import wandb

ENTITY = "marvin-k-steinkellner-eth"
PROJECTS = {
    # "reach": "multi-robot-rl-euler-reach",
    "push":  "multi-robot-rl-push-2026-06-06_12-22-30",
    "type":  "multi-robot-rl-type-2026-06-05_13-48-30",
}
DATA_DIR = Path(__file__).parent.parent.parent / "data"

VARIANT_KEYS = (
    "NUM_MASSPOINTS", "NUM_UR10S", "NUM_GOALS",
    "NUM_CUBOIDS", "NUM_ACTIVE_KEYS",
)


def download_project(task: str, project: str) -> None:
    print(f"\n=== {task} ({ENTITY}/{project}) ===")

    api = wandb.Api()
    try:
        runs = api.runs(f"{ENTITY}/{project}")
    except Exception as e:
        print(f"  Could not fetch runs: {e}")
        return

    runs = [r for r in runs if r.state == "finished"]
    if not runs:
        print("  No finished runs found.")
        return

    print(f"  Found {len(runs)} finished run(s).")

    histories = []

    for run in runs:
        meta = {
            "project":  project,
            "task":     task,
            "run_id":   run.id,
            "run_name": run.name,
            "seed":     run.config.get("agent", {}).get("seed"),
            **{k: v for k, v in run.config.items() if k in VARIANT_KEYS},
        }

        try:
            print(f"  Downloading {run.name} ...", end=" ", flush=True)
            rows = list(run.scan_history())
            print(f"{len(rows)} steps")
            hist = pd.DataFrame(rows)
            for col, val in meta.items():
                hist[col] = val
            histories.append(hist)
        except Exception as e:
            print(f"  Warning: could not fetch history for {run.name}: {e}")

    DATA_DIR.mkdir(exist_ok=True)

    if histories:
        df = pd.concat(histories, ignore_index=True)
        df.to_csv(DATA_DIR / f"{project}_histories.csv", index=False)
        print(f"  Saved {project}_histories.csv")
        metric_cols = [c for c in df.columns if c not in meta]
        print(f"  Metrics: {metric_cols}")


if __name__ == "__main__":
    for task, project in PROJECTS.items():
        download_project(task, project)

    print("\nDone. Files written to data/")
