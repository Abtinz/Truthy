from __future__ import annotations

from pathlib import Path

from app.jobs.logging_utils import reset_log_file
from app.jobs.logging_utils import write_log_message


def test_write_log_message_persists_output(tmp_path: Path) -> None:
    """Verify the shared logging helper writes the message to disk.

    Args:
        tmp_path: Temporary directory provided by pytest.

    Returns:
        None.
    """

    log_path = tmp_path / "indexer.log"

    write_log_message("indexer_started", log_path)

    print("\n=== LOG FILE CONTENT ===")
    print(log_path.read_text(encoding="utf-8"))

    assert log_path.read_text(encoding="utf-8") == "indexer_started\n"


def test_reset_log_file_removes_existing_log(tmp_path: Path) -> None:
    """Verify the shared logging helper removes an existing log file.

    Args:
        tmp_path: Temporary directory provided by pytest.

    Returns:
        None.
    """

    log_path = tmp_path / "indexer.log"
    log_path.write_text("stale line\n", encoding="utf-8")

    reset_log_file(log_path)

    print("\n=== LOG FILE EXISTS AFTER RESET ===")
    print(log_path.exists())

    assert log_path.exists() is False
