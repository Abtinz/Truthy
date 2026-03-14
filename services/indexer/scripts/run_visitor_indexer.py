from __future__ import annotations

from pathlib import Path

from app.core.config import IndexerSettings
from app.jobs.logging_utils import reset_log_file
from app.jobs.logging_utils import write_log_message
from app.vectorstore.index_manager import VisitorProgramIndexer

LOG_PATH = Path("/workspace/services/indexer/log/indexer_run.log")


def main() -> None:
    """Run the visitor-program indexing workflow once and print the summary.

    Args:
        None.

    Returns:
        None.
    """
    reset_log_file(LOG_PATH)

    settings = IndexerSettings.from_env()
    write_log_message("loaded_settings", LOG_PATH)
    indexer = VisitorProgramIndexer(settings)
    write_log_message("indexer_initialized", LOG_PATH)
    summary = indexer.index_all_sources().to_dict()

    write_log_message("=== INDEXING SUMMARY ===", LOG_PATH)
    write_log_message(str(summary), LOG_PATH)


if __name__ == "__main__":
    main()
