"""Generic benchmark sweep runner for multi-robot-rl training."""

import json
import os
import shutil
import signal
import subprocess
import sys
import tarfile
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.resolve()


@dataclass
class AlgorithmSpec:
    """Algorithm identity pairing a short display label with a registered task id."""

    name: str     # short label, e.g. "ppo", "fast-sac"
    task_id: str  # registered task id, e.g. "reach", "reach-fast-sac"


@dataclass
class RunResult:
    """Outcome record for a single (algorithm, variant, seed) training run."""

    label: str
    algorithm: str
    variant: dict
    seed: int
    wall_time_s: float
    exit_code: int | None  # None = timed out
    timed_out: bool
    interrupted: bool
    error: str | None


_current_proc: subprocess.Popen | None = None
_interrupted: bool = False


def _sigint_handler(signum, frame) -> None:
    """Handle SIGINT (Ctrl+C) by setting the interrupted flag and terminating the current subprocess.

    Side Effects:
        - Sets the module-level `_interrupted` flag to True.
        - Sends SIGTERM to the process group of `_current_proc` if one is running.
    """
    global _interrupted
    _interrupted = True
    print("\n[benchmark] Ctrl+C — stopping current run, will archive results.")
    if _current_proc is not None:
        try:
            os.killpg(os.getpgid(_current_proc.pid), signal.SIGTERM)
        except (ProcessLookupError, OSError):
            pass


def _git_commit() -> str:
    """Return the current HEAD commit SHA, or 'unknown' if the lookup fails.

    Returns:
        The full 40-character SHA of the current HEAD, or the string "unknown".
    """
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def _status_str(r: RunResult) -> str:
    """Format a RunResult into a human-readable status string.

    Args:
        r: The run result to describe.

    Returns:
        One of "INTERRUPTED", "TIMEOUT", "FAIL (exit <code>)", or "OK".
    """
    if r.interrupted:
        return "INTERRUPTED"
    if r.timed_out:
        return "TIMEOUT"
    if r.exit_code:
        return f"FAIL (exit {r.exit_code})"
    return "OK"


class BenchmarkRunner(ABC):
    """Base class for task-specific benchmark runners.

    Subclasses define task_name and call self.run(algorithms, variants, timeout_s)
    from their __main__ block. Variant dicts map env var names to values (e.g.
    {"NUM_MASSPOINTS": 2, "NUM_GOALS": 5}); these are passed to each training
    subprocess as environment variables so runs are fully independent and
    parallelisable across Slurm jobs.
    Override _variant_label for task-specific label formatting.
    """

    @property
    @abstractmethod
    def task_name(self) -> str:
        """Short task name, e.g. 'reach'. Used for log dirs and W&B project."""

    @property
    def logs_dir(self) -> Path:
        """Absolute path to the rsl_rl log directory for this task."""
        return REPO_ROOT / "logs" / "rsl_rl" / f"{self.task_name}_task"

    # ------------------------------------------------------------------
    # Label
    # ------------------------------------------------------------------

    def _variant_label(self, algo_name: str, variant: dict) -> str:
        """Build a run label from algorithm name and variant fields.

        Override in subclasses for task-specific formatting.
        Default: joins algo_name with each field value + lowercase field name.
        """
        parts = [algo_name] + [
            f"{v}{k.lower().removeprefix('num_')}"
            for k, v in variant.items()
        ]
        return "_".join(parts)

    # ------------------------------------------------------------------
    # Log staging
    # ------------------------------------------------------------------

    def _snapshot_subdirs(self) -> set[str]:
        """Return the names of all subdirectories currently present in logs_dir.

        Returns:
            A set of directory name strings, or an empty set if logs_dir does not exist.
        """
        if not self.logs_dir.exists():
            return set()
        return {d.name for d in self.logs_dir.iterdir() if d.is_dir()}

    def _latest_checkpoint(self, run_dir: Path) -> Path | None:
        """Return the highest-numbered model_*.pt checkpoint file in run_dir.

        Args:
            run_dir: Directory to search for checkpoint files.

        Returns:
            Path to the checkpoint with the largest step number, or None if
            no model_*.pt files are found.
        """
        pts = list(run_dir.glob("model_*.pt"))
        if not pts:
            return None
        def _step(p: Path) -> int:
            try:
                return int(p.stem.split("_", 1)[1])
            except (IndexError, ValueError):
                return -1
        return max(pts, key=_step)

    def _record_videos(self, algo: AlgorithmSpec, label: str, before: set[str], env: dict) -> None:
        """Record evaluation videos for all new run directories produced by a training run.

        Args:
            algo: The algorithm whose task_id is used to invoke the `record` command.
            label: Run label used to identify which new directories belong to this run.
            before: Snapshot of subdirectory names that existed before the run started,
                used to isolate newly created directories.
            env: Environment variables to pass to the record subprocess (must match
                the variant used during training so the env config is identical).

        Side Effects:
            - Launches a `uv run record` subprocess per new run directory.
            - Writes video files into each run directory's videos/ subfolder.
        """
        if not self.logs_dir.exists():
            return
        new_dirs = [
            d for d in self.logs_dir.iterdir()
            if d.is_dir() and d.name not in before and label in d.name
        ]
        for run_dir in new_dirs:
            ckpt = self._latest_checkpoint(run_dir)
            if ckpt is None:
                print(f"[benchmark] No checkpoint in {run_dir.name}, skipping video.")
                continue
            cmd = [
                "uv", "run", "record", algo.task_id,
                "--checkpoint-file", str(ckpt),
                "--video-length", "600",
            ]
            print(f"[benchmark] Recording video: {ckpt.name} → {run_dir.name}/videos/")
            try:
                subprocess.run(cmd, cwd=REPO_ROOT, timeout=300, env=env)
            except subprocess.TimeoutExpired:
                print(f"[benchmark] Video recording timed out for {run_dir.name}.")
            except Exception as exc:
                print(f"[benchmark] Video recording failed for {run_dir.name}: {exc}")

    def _stage_logs(self, label: str, staging: Path, before: set[str]) -> None:
        """Move new log directories for a completed run into the staging area.

        Args:
            label: Run label used to identify which new directories belong to this run.
            staging: Destination parent directory under which a label-named subdirectory
                is created to hold the moved logs.
            before: Snapshot of subdirectory names that existed before the run started,
                used to isolate newly created directories.

        Side Effects:
            - Creates a subdirectory under staging on disk.
            - Moves matching new directories from logs_dir into staging/label/.
        """
        if not self.logs_dir.exists():
            print(f"[benchmark] No logs found for {label}.")
            return
        new_dirs = [
            d for d in self.logs_dir.iterdir()
            if d.is_dir() and d.name not in before and label in d.name
        ]
        if not new_dirs:
            print(f"[benchmark] No new log directories for {label}.")
            return
        dest = staging / label
        dest.mkdir(parents=True, exist_ok=True)
        for d in new_dirs:
            shutil.move(str(d), str(dest / d.name))
        print(f"[benchmark] Staged {len(new_dirs)} run(s) for {label}.")

    # ------------------------------------------------------------------
    # Run one (algo, variant) pair
    # ------------------------------------------------------------------

    def _run_one(
        self,
        algo: AlgorithmSpec,
        variant: dict,
        timeout_s: int,
        wandb_project: str,
        staging: Path,
        seed: int = 0,
    ) -> RunResult:
        """Execute one (algorithm, variant, seed) training run and return its outcome.

        Variant values are passed to the training subprocess as environment
        variables, so multiple runs can execute concurrently without file conflicts.

        Args:
            algo: Algorithm to train, providing the task_id for the `train` command.
            variant: Mapping of env var names to values (e.g. {"NUM_MASSPOINTS": 2}).
            timeout_s: Maximum wall-clock seconds to allow the training process to run.
            wandb_project: W&B project name passed to the training command.
            staging: Directory into which completed run logs are moved.
            seed: Random seed passed to --agent.seed; appended to the run label.

        Returns:
            A RunResult capturing the label, exit code, wall time, and any error.

        Side Effects:
            - Sets the module-level `_current_proc` to the training subprocess while running.
            - Launches a `uv run train` subprocess with variant env vars set.
            - Calls `_record_videos` and `_stage_logs`, which write and move files on disk.
        """
        global _current_proc
        label = f"{self._variant_label(algo.name, variant)}_seed{seed}"
        tags = f"('{self.task_name}-sweep','{label}')"
        cmd = [
            "uv", "run", "train", algo.task_id,
            "--agent.run-name", label,
            "--agent.logger", "wandb",
            "--agent.wandb-project", wandb_project,
            "--agent.wandb-tags", tags,
            "--agent.seed", str(seed),
        ]

        print(f"\n{'='*60}")
        print(f"  Run:      {label}")
        print(f"  Task:     {algo.task_id}  ({algo.name})")
        print(f"  Variant:  {variant}")
        print(f"  Seed:     {seed}")
        print(f"  Timeout:  {timeout_s // 3600}h")
        print(f"{'='*60}\n")

        before = self._snapshot_subdirs()
        env = {**os.environ, **{k: str(v) for k, v in variant.items()}}
        t0 = time.monotonic()
        timed_out = False
        exit_code = None
        error = None

        try:
            proc = subprocess.Popen(cmd, cwd=REPO_ROOT, start_new_session=True, env=env)
            _current_proc = proc
            try:
                proc.wait(timeout=timeout_s)
                exit_code = proc.returncode
                if exit_code != 0 and not _interrupted:
                    error = f"non-zero exit code: {exit_code}"
            except subprocess.TimeoutExpired:
                timed_out = True
                print(f"\n[benchmark] Timeout for {label} — terminating.")
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    proc.wait()
                except ProcessLookupError:
                    pass
        except Exception as exc:
            error = str(exc)
            print(f"\n[benchmark] Unexpected error for {label}: {exc}")
        finally:
            _current_proc = None
            self._record_videos(algo, label, before, env)

        self._stage_logs(label, staging, before)

        return RunResult(
            label=label,
            algorithm=algo.name,
            variant=variant,
            seed=seed,
            wall_time_s=round(time.monotonic() - t0, 1),
            exit_code=exit_code,
            timed_out=timed_out,
            interrupted=_interrupted,
            error="interrupted by user" if _interrupted else error,
        )

    # ------------------------------------------------------------------
    # Archive
    # ------------------------------------------------------------------

    def _archive(
        self,
        staging: Path,
        sweep_ts: str,
        results: list[RunResult],
        timeout_s: int,
    ) -> None:
        """Write a gzipped tar archive of the staging directory and a metadata JSON file.

        Args:
            staging: Directory containing all staged run logs to archive.
            sweep_ts: Timestamp string (YYYY-MM-DD_HH-MM-SS) recorded in metadata.
            results: List of RunResult objects for all completed runs.
            timeout_s: Per-run timeout that was in effect, recorded in metadata.

        Side Effects:
            - Writes metadata.json into the staging directory on disk.
            - Creates a .tar.gz archive at logs/<staging.name>.tar.gz on disk.
        """
        archive_path = REPO_ROOT / "logs" / f"{staging.name}.tar.gz"
        metadata = {
            "sweep_name": staging.name,
            "sweep_start_local": sweep_ts,
            "git_commit": _git_commit(),
            "train_timeout_s": timeout_s,
            "runs": [asdict(r) for r in results],
        }
        (staging / "metadata.json").write_text(json.dumps(metadata, indent=2))
        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(staging, arcname=staging.name)
        size_mb = archive_path.stat().st_size / 1024 / 1024
        print(f"[benchmark] Archive: {archive_path.relative_to(REPO_ROOT)} ({size_mb:.1f} MB)")

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(
        self,
        algorithms: list[AlgorithmSpec],
        variants: list[dict],
        timeout_s: int,
        seeds: list[int] | None = None,
    ) -> None:
        """Execute the full benchmark sweep over all algorithm/variant/seed combinations.

        Iterates over every (algorithm, variant, seed) triple in order, runs each
        training job, then archives all staged logs. The sweep stops early if the
        user interrupts with Ctrl+C; already-completed runs are still archived.

        Args:
            algorithms: Ordered list of algorithms to benchmark.
            variants: Ordered list of hyperparameter variant dicts to apply per algorithm.
            timeout_s: Maximum wall-clock seconds allowed per individual training run.
            seeds: List of integer seeds to run for each (algorithm, variant) pair.
                Defaults to [0] (single seed) if not provided.

        Side Effects:
            - Installs a SIGINT handler (restores after sweep).
            - Creates a timestamped staging directory under logs/rsl_rl/ on disk.
            - Calls `_run_one` for each triple, which launches subprocesses with variant env vars.
            - Calls `_archive`, which writes the final .tar.gz to disk.
        """
        if seeds is None:
            seeds = [42]

        sweep_ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        wandb_project = f"multi-robot-rl-{self.task_name}-{sweep_ts}"
        staging = REPO_ROOT / "logs" / "rsl_rl" / f"{self.task_name}_sweep_{sweep_ts}"
        staging.mkdir(parents=True, exist_ok=True)

        signal.signal(signal.SIGINT, _sigint_handler)

        total = len(algorithms) * len(variants) * len(seeds)
        print(f"[benchmark] {self.task_name} — {sweep_ts}")
        print(f"[benchmark] {len(algorithms)} algo(s) × {len(variants)} variant(s) × {len(seeds)} seed(s) = {total} run(s), {timeout_s // 3600}h max each\n")

        results: list[RunResult] = []
        for algo in algorithms:
            for variant in variants:
                for seed in seeds:
                    result = self._run_one(algo, variant, timeout_s, wandb_project, staging, seed=seed)
                    results.append(result)
                    print(f"[benchmark] {result.label}: {_status_str(result)} in {result.wall_time_s / 60:.1f} min")
                    if _interrupted:
                        break
                if _interrupted:
                    break
            if _interrupted:
                break

        self._archive(staging, sweep_ts, results, timeout_s)

        print(f"\n[benchmark] Summary ({self.task_name}):")
        for r in results:
            print(f"  {r.label:45s}  {_status_str(r):15s}  {r.wall_time_s / 60:.1f} min")
