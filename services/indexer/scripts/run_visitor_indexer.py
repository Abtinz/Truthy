from __future__ import annotations

from pathlib import Path

from app.core.config import IndexerSettings
from app.vectorstore.index_manager import VisitorProgramIndexer

LOG_PATH = Path("/workspace/services/indexer/log/indexer_run.log")


def log(message: str) -> None:
    """Write a progress message to stdout and the shared log file.

    Args:
        message: Progress message to record.

    Returns:
        None.
    """
    print(message, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as log_file:
        log_file.write(f"{message}\n")


def main() -> None:
    """Run the visitor-program indexing workflow once and print the summary.

    Args:
        None.

    Returns:
        None.
    """
    if LOG_PATH.exists():
        LOG_PATH.unlink()

    settings = IndexerSettings.from_env()
    log("loaded_settings")
    indexer = VisitorProgramIndexer(settings)
    log("indexer_initialized")
    summary = indexer.index_all_sources().to_dict()

    log("=== INDEXING SUMMARY ===")
    log(str(summary))


if __name__ == "__main__":
    main()
