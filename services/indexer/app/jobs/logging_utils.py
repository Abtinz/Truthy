from __future__ import annotations

from pathlib import Path


def reset_log_file(log_path: Path) -> None:
    """Delete an existing log file before a fresh indexing run starts.

    The indexer runner scripts use one persistent log file per workflow. This
    helper centralizes the reset behavior so both scripts follow the same file
    lifecycle instead of keeping duplicated unlink logic.

    Args:
        log_path: Absolute log file path for the indexing workflow.

    Returns:
        None.
    """

    if log_path.exists():
        log_path.unlink()


def write_log_message(message: str, log_path: Path) -> None:
    """Write one progress line to stdout and the target log file.

    This helper is shared by both the visitor and study-permit indexing
    runners. It ensures log files are created consistently and removes the
    need for duplicated script-local `log()` functions.

    Args:
        message: Human-readable progress message to record.
        log_path: Absolute log file path for the indexing workflow.

    Returns:
        None.
    """

    print(message, flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"{message}\n")
