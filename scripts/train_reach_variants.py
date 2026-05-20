"""Train the reach task across multiple configurations sequentially.

Each variant patches reach_constants.py, runs `uv run train reach` for
up to 3 hours, then restores the file regardless of outcome.

After all variants finish, logs are archived and metadata is written.

Run from the repo root:
    uv run python scripts/train_reach_variants.py
"""

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent.resolve()
CONSTANTS_FILE = REPO_ROOT / "src" / "multi_robot_rl" / "configs" / "reach_constants.py"
LOGS_DIR = REPO_ROOT / "logs" / "rsl_rl" / "reach_task"

# TRAIN_TIMEOUT_S = 3 * 60 * 60  # 3 hours
TRAIN_TIMEOUT_S = 60 # 1 minute for testing

VARIANTS = [
    # (num_masspoints, num_ur10s, num_goals)
    (1, 0, 1),
    (1, 0, 5),
    (2, 0, 1),
#     (2, 0, 5),
#     (0, 1, 1),
#     (0, 1, 5),
#     (0, 2, 1),
#     (0, 2, 5),
]


def variant_label(num_mp: int, num_ur10: int, num_goals: int) -> str:
    robot_part = []
    if num_mp > 0:
        robot_part.append(f"{num_mp}mp")
    if num_ur10 > 0:
        robot_part.append(f"{num_ur10}ur10")
    if not robot_part:
        robot_part = ["no-robots"]
    return "_".join(robot_part) + f"_{num_goals}goal{'s' if num_goals > 1 else ''}"


# ---------------------------------------------------------------------------
# Patch / restore reach_constants.py
# ---------------------------------------------------------------------------

_PATCH_TARGETS = {
    "NUM_MASSPOINTS": r"^NUM_MASSPOINTS\s*=\s*\d+",
    "NUM_UR10S": r"^NUM_UR10S\s*=\s*\d+",
    "NUM_GOALS": r"^NUM_GOALS\s*=\s*\d+",
}


def patch_constants(num_mp: int, num_ur10: int, num_goals: int) -> str:
    """Overwrite reach_constants.py with new robot/goal counts. Returns original text."""
    original = CONSTANTS_FILE.read_text()
    patched = original
    replacements = {
        "NUM_MASSPOINTS": num_mp,
        "NUM_UR10S": num_ur10,
        "NUM_GOALS": num_goals,
    }
    for name, value in replacements.items():
        pattern = _PATCH_TARGETS[name]
        patched = re.sub(pattern, f"{name} = {value}", patched, flags=re.MULTILINE)
    CONSTANTS_FILE.write_text(patched)
    return original


def restore_constants(original_text: str) -> None:
    CONSTANTS_FILE.write_text(original_text)


# ---------------------------------------------------------------------------
# Log staging helpers
# ---------------------------------------------------------------------------

def snapshot_subdirs(path: Path) -> set[str]:
    """Return names of direct child directories, or empty set if path doesn't exist."""
    if not path.exists():
        return set()
    return {child.name for child in path.iterdir() if child.is_dir()}


def stage_new_variant_logs(label: str, sweep_staging: Path, before: set[str]) -> None:
    """Move only directories created during this variant's run into the sweep staging area.

    The training framework saves runs as {timestamp}_{label} directly under LOGS_DIR.
    We compare LOGS_DIR children against the snapshot taken before the run and move
    only new directories whose names contain the label.
    """
    if not LOGS_DIR.exists():
        print(f"[train_reach_variants] No logs found for {label}.")
        return

    new_dirs = [
        d for d in LOGS_DIR.iterdir()
        if d.is_dir() and d.name not in before and label in d.name
    ]
    if not new_dirs:
        print(f"[train_reach_variants] No new log directories found for {label}.")
        return

    dest = sweep_staging / label
    dest.mkdir(parents=True, exist_ok=True)
    for d in new_dirs:
        shutil.move(str(d), str(dest / d.name))
    print(f"[train_reach_variants] Staged {len(new_dirs)} run(s) for {label}.")


# ---------------------------------------------------------------------------
# Signal handling
# ---------------------------------------------------------------------------

_current_proc: subprocess.Popen | None = None
_interrupted: bool = False


def _sigint_handler(signum, frame) -> None:
    global _interrupted
    _interrupted = True
    print("\n[train_reach_variants] Ctrl+C — stopping current variant, will archive results.")
    if _current_proc is not None:
        try:
            os.killpg(os.getpgid(_current_proc.pid), signal.SIGTERM)
        except (ProcessLookupError, OSError):
            pass


# ---------------------------------------------------------------------------
# Run one variant
# ---------------------------------------------------------------------------

@dataclass
class VariantResult:
    label: str
    num_masspoints: int
    num_ur10s: int
    num_goals: int
    wall_time_s: float
    exit_code: int | None  # None means timed out
    timed_out: bool
    interrupted: bool
    error: str | None


def run_variant(
    num_mp: int,
    num_ur10: int,
    num_goals: int,
    wandb_project: str,
    sweep_staging: Path,
) -> VariantResult:
    label = variant_label(num_mp, num_ur10, num_goals)
    tags = f"('reach-sweep','{label}')"
    cmd = [
        "uv", "run", "train", "reach",
        "--agent.run-name", label,
        "--agent.logger", "wandb",
        "--agent.wandb-project", wandb_project,
        "--agent.wandb-tags", tags,
    ]

    print(f"\n{'='*60}")
    print(f"  Variant: {label}")
    print(f"  NUM_MASSPOINTS={num_mp}  NUM_UR10S={num_ur10}  NUM_GOALS={num_goals}")
    print(f"  W&B project: {wandb_project}")
    print(f"  Timeout: {TRAIN_TIMEOUT_S // 3600}h")
    print(f"{'='*60}\n")

    before = snapshot_subdirs(LOGS_DIR)
    original = patch_constants(num_mp, num_ur10, num_goals)
    t0 = time.monotonic()
    timed_out = False
    exit_code = None
    error = None

    global _current_proc
    try:
        proc = subprocess.Popen(cmd, cwd=REPO_ROOT, start_new_session=True)
        _current_proc = proc
        try:
            proc.wait(timeout=TRAIN_TIMEOUT_S)
            exit_code = proc.returncode
            if exit_code != 0 and not _interrupted:
                error = f"non-zero exit code: {exit_code}"
        except subprocess.TimeoutExpired:
            timed_out = True
            print(f"\n[train_reach_variants] Timeout reached for {label} — killing process group.")
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                proc.wait()
            except ProcessLookupError:
                pass  # process group already gone
    except Exception as exc:
        error = str(exc)
        print(f"\n[train_reach_variants] Unexpected error for {label}: {exc}")
    finally:
        _current_proc = None
        restore_constants(original)

    wall_time = time.monotonic() - t0
    stage_new_variant_logs(label, sweep_staging, before)

    return VariantResult(
        label=label,
        num_masspoints=num_mp,
        num_ur10s=num_ur10,
        num_goals=num_goals,
        wall_time_s=round(wall_time, 1),
        exit_code=exit_code,
        timed_out=timed_out,
        interrupted=_interrupted,
        error="interrupted by user" if _interrupted else error,
    )


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

def collect_and_archive(sweep_staging: Path, sweep_ts: str, results: list[VariantResult]) -> None:
    """Write metadata into the sweep staging dir and archive it."""
    archive_path = REPO_ROOT / "logs" / f"{sweep_staging.name}.tar.gz"

    git_commit = _git_commit()
    metadata = {
        "sweep_name": sweep_staging.name,
        "sweep_start_local": sweep_ts,
        "git_commit": git_commit,
        "train_timeout_s": TRAIN_TIMEOUT_S,
        "variants": [asdict(r) for r in results],
    }
    metadata_file = sweep_staging / "metadata.json"
    metadata_file.write_text(json.dumps(metadata, indent=2))
    print(f"[cleanup] Wrote metadata: {metadata_file.relative_to(REPO_ROOT)}")

    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(sweep_staging, arcname=sweep_staging.name)
    size_mb = archive_path.stat().st_size / 1024 / 1024
    print(f"[cleanup] Archive: {archive_path.relative_to(REPO_ROOT)} ({size_mb:.1f} MB)")


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not CONSTANTS_FILE.exists():
        sys.exit(f"ERROR: constants file not found: {CONSTANTS_FILE}")

    sweep_ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    wandb_project = f"multi-robot-rl-reach-sweep-{sweep_ts}"
    sweep_staging = REPO_ROOT / "logs" / "rsl_rl" / f"reach_sweep_{sweep_ts}"
    sweep_staging.mkdir(parents=True, exist_ok=True)

    signal.signal(signal.SIGINT, _sigint_handler)

    print(f"[train_reach_variants] Starting sweep at {sweep_ts}")
    print(f"[train_reach_variants] W&B project: {wandb_project}")
    print(f"[train_reach_variants] Sweep staging: {sweep_staging.relative_to(REPO_ROOT)}")
    print(f"[train_reach_variants] {len(VARIANTS)} variants × {TRAIN_TIMEOUT_S // 3600}h = up to {len(VARIANTS) * TRAIN_TIMEOUT_S // 3600}h total\n")

    results: list[VariantResult] = []
    for num_mp, num_ur10, num_goals in VARIANTS:
        result = run_variant(num_mp, num_ur10, num_goals, wandb_project, sweep_staging)
        results.append(result)
        status = "INTERRUPTED" if result.interrupted else ("TIMEOUT" if result.timed_out else (f"exit {result.exit_code}" if result.exit_code else "OK"))
        print(f"[train_reach_variants] {result.label}: {status} in {result.wall_time_s / 60:.1f} min")
        if _interrupted:
            break

    print("\n[train_reach_variants] All variants done. Running cleanup..." if not _interrupted else "\n[train_reach_variants] Interrupted. Archiving completed results...")
    collect_and_archive(sweep_staging, sweep_ts, results)

    print("\n[train_reach_variants] Summary:")
    for r in results:
        status = "INTERRUPTED" if r.interrupted else ("TIMEOUT" if r.timed_out else (f"FAIL (exit {r.exit_code})" if r.exit_code else "OK"))
        print(f"  {r.label:35s}  {status}  ({r.wall_time_s / 60:.1f} min)")


if __name__ == "__main__":
    main()
