from __future__ import annotations

from dataclasses import dataclass

from app.chunking.text_chunker import ChunkingConfig, TextChunker
from app.core.config import IndexerSettings
from app.ingestion.crawler import CrawledDocument, CrawlerSource, HierarchicalSection
from app.vectorstore.index_manager import VisitorProgramIndexer
from app.vectorstore.pinecone_client import PineconeVectorRecord


class FakeCrawler:
    """Fake crawler that returns deterministic visitor-program documents.

    Args:
        None.

    Returns:
        FakeCrawler: Test double for the crawler dependency.
    """

    def crawl_all(self) -> list[CrawledDocument]:
        """Return deterministic structured documents for indexing tests.

        Args:
            None.

        Returns:
            list[CrawledDocument]: Fixed document set for tests.
        """
        return [
            CrawledDocument(
                source=CrawlerSource(
                    url="https://example.com/guidelines",
                    kind="operational_guidelines",
                    title="Guidelines",
                ),
                document_title="Guidelines",
                sections=[
                    HierarchicalSection(
                        title="Reviewing documents",
                        level=2,
                        path=["Guidelines", "Reviewing documents"],
                        content="Applicants must provide a passport and proof of ties.",
                    )
                ],
                full_text="Guidelines full text",
            ),
            CrawledDocument(
                source=CrawlerSource(
                    url="https://example.com/checklist.pdf",
                    kind="document_checklist_pdf",
                    title="Checklist PDF",
                ),
                document_title="Checklist PDF",
                sections=[
                    HierarchicalSection(
                        title="Page 1",
                        level=2,
                        path=["Checklist PDF", "Page 1"],
                        content="Passport copy and fee receipt.",
                    )
                ],
                full_text="Checklist full text",
            ),
        ]


class FakePineconeClient:
    """Fake Pinecone client capturing index bootstrap and upsert calls.

    Args:
        None.

    Returns:
        FakePineconeClient: Test double for Pinecone upsert operations.
    """

    def __init__(self) -> None:
        """Initialize the fake Pinecone client.

        Args:
            None.

        Returns:
            None.
        """
        self.bootstrap_called = False
        self.guideline_records: list[PineconeVectorRecord] = []
        self.checklist_records: list[PineconeVectorRecord] = []

    def ensure_required_indexes_exist(self) -> list[dict[str, str]]:
        """Record that bootstrap was called.

        Args:
            None.

        Returns:
            list[dict[str, str]]: Fake bootstrap summary.
        """
        self.bootstrap_called = True
        return [
            {"index_name": "guidelines-index", "status": "created"},
            {"index_name": "checklist-index", "status": "created"},
        ]

    def upsert_operational_guidelines(
        self,
        records: list[PineconeVectorRecord],
    ) -> dict[str, int]:
        """Capture guideline upsert records.

        Args:
            records: Guideline vector records.

        Returns:
            dict[str, int]: Fake upsert summary.
        """
        self.guideline_records.extend(records)
        return {"upserted_count": len(records)}

    def upsert_document_checklists(
        self,
        records: list[PineconeVectorRecord],
    ) -> dict[str, int]:
        """Capture checklist upsert records.

        Args:
            records: Checklist vector records.

        Returns:
            dict[str, int]: Fake upsert summary.
        """
        self.checklist_records.extend(records)
        return {"upserted_count": len(records)}


def test_index_manager_builds_and_routes_records(monkeypatch) -> None:
    """Verify the index manager chunks, embeds, and routes records correctly.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    fake_pinecone = FakePineconeClient()
    settings = _build_settings()

    monkeypatch.setattr(
        "app.vectorstore.index_manager.embed_texts",
        lambda texts: [[float(index + 1), float(len(text))] for index, text in enumerate(texts)],
    )

    indexer = VisitorProgramIndexer(
        settings,
        crawler=FakeCrawler(),
        pinecone_client=fake_pinecone,
        chunker=TextChunker(ChunkingConfig(chunk_size=80, chunk_overlap=10)),
    )

    summary = indexer.index_all_sources()

    print("\n=== INDEXING SUMMARY ===")
    print(summary.to_dict())
    print("\n=== GUIDELINE RECORDS ===")
    print([record.to_payload() for record in fake_pinecone.guideline_records])
    print("\n=== CHECKLIST RECORDS ===")
    print([record.to_payload() for record in fake_pinecone.checklist_records])

    assert fake_pinecone.bootstrap_called is True
    assert summary.crawled_documents == 2
    assert summary.generated_chunks >= 2
    assert len(fake_pinecone.guideline_records) >= 1
    assert len(fake_pinecone.checklist_records) >= 1


def _build_settings() -> IndexerSettings:
    """Construct stable settings for index-manager tests.

    Args:
        None.

    Returns:
        IndexerSettings: Test settings object.
    """
    return IndexerSettings(
        pinecone_api_key="test-key",
        pinecone_operational_guidelines_index_name="guidelines-index",
        pinecone_document_checklist_index_name="checklist-index",
        pinecone_namespace="truthy-dev",
        pinecone_top_k=5,
        pinecone_dimension=1536,
        pinecone_metric="cosine",
        pinecone_cloud="aws",
        pinecone_region="us-east-1",
        pinecone_deletion_protection="disabled",
    )
