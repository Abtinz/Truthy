from __future__ import annotations

from pathlib import Path

from app.core.config import IndexerSettings
from app.vectorstore.index_manager import VisitorProgramIndexer

LOG_PATH = Path("/workspace/services/indexer/indexer_run.log")


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

    log("crawl_all_start")
    documents = indexer.crawler.crawl_all()
    log(f"crawl_all_done documents={len(documents)}")

    records_by_index = {
        "operational_guidelines": [],
        "document_checklist_pdf": [],
    }

    for document in documents:
        log(
            f"build_records_start kind={document.source.kind} "
            f"title={document.document_title}"
        )
        records = indexer._build_records_for_document(document)
        log(
            f"build_records_done kind={document.source.kind} "
            f"records={len(records)}"
        )
        records_by_index[document.source.kind].extend(records)

    log("ensure_indexes_start")
    indexer.pinecone_client.ensure_required_indexes_exist()
    log("ensure_indexes_done")

    if records_by_index["operational_guidelines"]:
        log(
            "upsert_guidelines_start "
            f"count={len(records_by_index['operational_guidelines'])}"
        )
        indexer.pinecone_client.upsert_operational_guidelines(
            records_by_index["operational_guidelines"]
        )
        log("upsert_guidelines_done")

    if records_by_index["document_checklist_pdf"]:
        log(
            "upsert_checklists_start "
            f"count={len(records_by_index['document_checklist_pdf'])}"
        )
        indexer.pinecone_client.upsert_document_checklists(
            records_by_index["document_checklist_pdf"]
        )
        log("upsert_checklists_done")

    summary = {
        "crawled_documents": len(documents),
        "generated_chunks": (
            len(records_by_index["operational_guidelines"])
            + len(records_by_index["document_checklist_pdf"])
        ),
        "operational_guidelines_upserts": len(
            records_by_index["operational_guidelines"]
        ),
        "document_checklist_upserts": len(
            records_by_index["document_checklist_pdf"]
        ),
    }

    log("=== INDEXING SUMMARY ===")
    log(str(summary))


if __name__ == "__main__":
    main()
