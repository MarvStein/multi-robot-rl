"""Package entrypoint and task-registration side effects."""


def main() -> None:
    print("Use 'uv run train ...' or 'uv run play ...' to run tasks.")


# Expose all tasks via module import.
from .masspoints import *
