from __future__ import annotations

from pathlib import Path

from app.core.config import IndexerSettings
from app.ingestion.crawler import VisitorProgramCrawler
from app.ingestion.crawler import build_study_permit_sources
from app.jobs.logging_utils import reset_log_file
from app.jobs.logging_utils import write_log_message
from app.vectorstore.index_manager import VisitorProgramIndexer

LOG_PATH = Path("/workspace/services/indexer/log/study_permit_indexer_run.log")


def main() -> None:
    """Run the study-permit indexing workflow once and print the summary.

    Args:
        None.

    Returns:
        None.
    """
    reset_log_file(LOG_PATH)

    settings = IndexerSettings.from_env()
    write_log_message("loaded_settings", LOG_PATH)
    crawler = VisitorProgramCrawler(sources=build_study_permit_sources())
    write_log_message("study_permit_crawler_initialized", LOG_PATH)
    indexer = VisitorProgramIndexer(settings, crawler=crawler)
    write_log_message("study_permit_indexer_initialized", LOG_PATH)
    summary = indexer.index_all_sources().to_dict()

    write_log_message("=== STUDY PERMIT INDEXING SUMMARY ===", LOG_PATH)
    write_log_message(str(summary), LOG_PATH)


if __name__ == "__main__":
    main()
