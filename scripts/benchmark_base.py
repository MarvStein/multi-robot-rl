"""Generic benchmark sweep runner for multi-robot-rl training."""

import json
import os
import re
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
    name: str     # short label, e.g. "ppo", "fast-sac"
    task_id: str  # registered task id, e.g. "reach", "reach-fast-sac"


@dataclass
class RunResult:
    label: str
    algorithm: str
    variant: dict
    wall_time_s: float
    exit_code: int | None  # None = timed out
    timed_out: bool
    interrupted: bool
    error: str | None


_current_proc: subprocess.Popen | None = None
_interrupted: bool = False


def _sigint_handler(signum, frame) -> None:
    global _interrupted
    _interrupted = True
    print("\n[benchmark] Ctrl+C — stopping current run, will archive results.")
    if _current_proc is not None:
        try:
            os.killpg(os.getpgid(_current_proc.pid), signal.SIGTERM)
        except (ProcessLookupError, OSError):
            pass


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def _status_str(r: RunResult) -> str:
    if r.interrupted:
        return "INTERRUPTED"
    if r.timed_out:
        return "TIMEOUT"
    if r.exit_code:
        return f"FAIL (exit {r.exit_code})"
    return "OK"


class BenchmarkRunner(ABC):
    """Base class for task-specific benchmark runners.

    Subclasses define task_name, constants_file, and patch_targets as class
    attributes, then call self.run(algorithms, variants, timeout_s) from their
    __main__ block. Override _variant_label for task-specific label formatting.
    """

    @property
    @abstractmethod
    def task_name(self) -> str:
        """Short task name, e.g. 'reach'. Used for log dirs and W&B project."""

    @property
    @abstractmethod
    def constants_file(self) -> Path:
        """Path to the *_constants.py file to patch for each variant."""

    @property
    @abstractmethod
    def patch_targets(self) -> dict[str, str]:
        """Maps constant field name -> regex matching its assignment line.

        Example: {"NUM_MASSPOINTS": r"^NUM_MASSPOINTS\\s*=\\s*\\d+"}
        """

    @property
    def logs_dir(self) -> Path:
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
    # Constants patching
    # ------------------------------------------------------------------

    def _patch_constants(self, variant: dict) -> str:
        """Write variant values into constants_file. Returns original text."""
        original = self.constants_file.read_text()
        patched = original
        for field, value in variant.items():
            if field in self.patch_targets:
                patched = re.sub(
                    self.patch_targets[field],
                    f"{field} = {value}",
                    patched,
                    flags=re.MULTILINE,
                )
        self.constants_file.write_text(patched)
        return original

    def _restore_constants(self, original: str) -> None:
        self.constants_file.write_text(original)

    # ------------------------------------------------------------------
    # Log staging
    # ------------------------------------------------------------------

    def _snapshot_subdirs(self) -> set[str]:
        if not self.logs_dir.exists():
            return set()
        return {d.name for d in self.logs_dir.iterdir() if d.is_dir()}

    def _latest_checkpoint(self, run_dir: Path) -> Path | None:
        """Return the highest-numbered model_*.pt in run_dir."""
        pts = list(run_dir.glob("model_*.pt"))
        if not pts:
            return None
        def _step(p: Path) -> int:
            try:
                return int(p.stem.split("_", 1)[1])
            except (IndexError, ValueError):
                return -1
        return max(pts, key=_step)

    def _record_videos(self, algo: AlgorithmSpec, label: str, before: set[str]) -> None:
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
            cmd = ["uv", "run", "record", algo.task_id, "--checkpoint-file", str(ckpt)]
            print(f"[benchmark] Recording video: {ckpt.name} → {run_dir.name}/videos/")
            try:
                subprocess.run(cmd, cwd=REPO_ROOT, timeout=300)
            except subprocess.TimeoutExpired:
                print(f"[benchmark] Video recording timed out for {run_dir.name}.")
            except Exception as exc:
                print(f"[benchmark] Video recording failed for {run_dir.name}: {exc}")

    def _stage_logs(self, label: str, staging: Path, before: set[str]) -> None:
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
    ) -> RunResult:
        global _current_proc
        label = self._variant_label(algo.name, variant)
        tags = f"('{self.task_name}-sweep','{label}')"
        cmd = [
            "uv", "run", "train", algo.task_id,
            "--agent.run-name", label,
            "--agent.logger", "wandb",
            "--agent.wandb-project", wandb_project,
            "--agent.wandb-tags", tags,
        ]

        print(f"\n{'='*60}")
        print(f"  Run:      {label}")
        print(f"  Task:     {algo.task_id}  ({algo.name})")
        print(f"  Variant:  {variant}")
        print(f"  Timeout:  {timeout_s // 3600}h")
        print(f"{'='*60}\n")

        before = self._snapshot_subdirs()
        original = self._patch_constants(variant)
        t0 = time.monotonic()
        timed_out = False
        exit_code = None
        error = None

        try:
            proc = subprocess.Popen(cmd, cwd=REPO_ROOT, start_new_session=True)
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
            self._restore_constants(original)

        self._record_videos(algo, label, before)
        self._stage_logs(label, staging, before)

        return RunResult(
            label=label,
            algorithm=algo.name,
            variant=variant,
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
    ) -> None:
        if not self.constants_file.exists():
            sys.exit(f"ERROR: constants file not found: {self.constants_file}")

        sweep_ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        wandb_project = f"multi-robot-rl-{self.task_name}-{sweep_ts}"
        staging = REPO_ROOT / "logs" / "rsl_rl" / f"{self.task_name}_sweep_{sweep_ts}"
        staging.mkdir(parents=True, exist_ok=True)

        signal.signal(signal.SIGINT, _sigint_handler)

        total = len(algorithms) * len(variants)
        print(f"[benchmark] {self.task_name} — {sweep_ts}")
        print(f"[benchmark] {len(algorithms)} algo(s) × {len(variants)} variant(s) = {total} run(s), {timeout_s // 3600}h max each\n")

        results: list[RunResult] = []
        for algo in algorithms:
            for variant in variants:
                result = self._run_one(algo, variant, timeout_s, wandb_project, staging)
                results.append(result)
                print(f"[benchmark] {result.label}: {_status_str(result)} in {result.wall_time_s / 60:.1f} min")
                if _interrupted:
                    break
            if _interrupted:
                break

        self._archive(staging, sweep_ts, results, timeout_s)

        print(f"\n[benchmark] Summary ({self.task_name}):")
        for r in results:
            print(f"  {r.label:45s}  {_status_str(r):15s}  {r.wall_time_s / 60:.1f} min")
