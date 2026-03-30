"""Thin CLI shims for mjlab train/play with project-specific convenience flags."""

from __future__ import annotations

import argparse
import os
import sys


def _apply_project_overrides() -> None:
    """Map convenience flags to environment variables then remove them from argv."""
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--num-masspoints", type=int, default=None)
    parser.add_argument("--num-goals", type=int, default=None)
    known, unknown = parser.parse_known_args(sys.argv[1:])

    if known.num_masspoints is not None:
        os.environ["MRRL_NUM_MASSPOINTS"] = str(known.num_masspoints)
    if known.num_goals is not None:
        os.environ["MRRL_NUM_GOALS"] = str(known.num_goals)

    sys.argv = [sys.argv[0], *unknown]


def train_main() -> None:
    """Entrypoint that preserves mjlab train behavior plus convenience flags."""
    _apply_project_overrides()
    from mjlab.scripts.train import main as mjlab_train_main

    mjlab_train_main()


def play_main() -> None:
    """Entrypoint that preserves mjlab play behavior plus convenience flags."""
    _apply_project_overrides()
    from mjlab.scripts.play import main as mjlab_play_main

    mjlab_play_main()
