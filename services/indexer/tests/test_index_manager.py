from __future__ import annotations

from app.cache.policy_freshness_cache import PolicyFreshnessResult
from app.chunking.text_chunker import ChunkingConfig, TextChunker
from app.core.config import IndexerSettings
from app.ingestion.crawler import CrawledDocument, CrawlerSource, HierarchicalSection
from app.ingestion.crawler import VisitorProgramCrawler
from app.ingestion.crawler import build_study_permit_sources
from app.ingestion.pdf_to_text import ExtractedPdf, TextChunk
from app.vectorstore.index_manager import VisitorProgramIndexer
from app.vectorstore.pinecone_client import PineconeVectorRecord


class FakeCrawler:
    """Fake crawler that returns deterministic visitor-program documents.

    Args:
        None.

    Returns:
        FakeCrawler: Test double for the crawler dependency.
    """

    def __init__(self) -> None:
        """Initialize the fake crawler with guideline and checklist sources.

        Args:
            None.

        Returns:
            None.
        """
        self.sources = [
            CrawlerSource(
                kind="operational_guidelines",
                title="Guidelines",
                url="https://example.com/guidelines",
            ),
            CrawlerSource(
                kind="document_checklist_pdf",
                title="Checklist PDF",
                url="",
                file_path="/workspace/services/data/forms/imm5484e.pdf",
            ),
        ]

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
                    kind="operational_guidelines",
                    title="Guidelines",
                    url="https://example.com/guidelines",
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
                modified_date="2026-03-03",
            ),
        ]

    def crawl_source(self, source: CrawlerSource) -> CrawledDocument:
        """Return the deterministic guideline document for direct source calls.

        Args:
            source: Source requested by the index manager.

        Returns:
            CrawledDocument: Fixed guideline document for the requested source.
        """
        return self.crawl_all()[0]

    def fetch_source_modified_date(self, source: CrawlerSource) -> str | None:
        """Return a deterministic modified date for guideline sources.

        Args:
            source: Source requested by the index manager.

        Returns:
            str | None: Fixed modified date string for guideline sources.
        """

        if source.kind == "operational_guidelines":
            return "2026-03-03"
        return None


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


class FakePolicyCache:
    """Fake freshness cache used by index-manager tests.

    Args:
        has_changed_by_url: Optional per-URL freshness decisions.

    Returns:
        FakePolicyCache: Test double for the Redis freshness cache.
    """

    def __init__(self, has_changed_by_url: dict[str, bool] | None = None) -> None:
        """Initialize the fake freshness cache.

        Args:
            has_changed_by_url: Optional per-URL freshness decisions.

        Returns:
            None.
        """

        self.has_changed_by_url = has_changed_by_url or {}
        self.stored_dates: list[tuple[str, str]] = []

    def compare_modified_date(
        self,
        source_url: str,
        modified_date: str,
    ) -> PolicyFreshnessResult:
        """Return the configured freshness decision for one source URL.

        Args:
            source_url: Policy source URL.
            modified_date: Current page-level modified date.

        Returns:
            PolicyFreshnessResult: Fake freshness comparison result.
        """

        has_changed = self.has_changed_by_url.get(source_url, True)
        return PolicyFreshnessResult(
            source_url=source_url,
            modified_date=modified_date,
            cached_modified_date=None if has_changed else modified_date,
            has_changed=has_changed,
        )

    def store_modified_date(self, source_url: str, modified_date: str) -> None:
        """Capture persisted modified dates for assertions.

        Args:
            source_url: Policy source URL.
            modified_date: Current page-level modified date.

        Returns:
            None.
        """

        self.stored_dates.append((source_url, modified_date))


def test_index_manager_builds_and_routes_records(monkeypatch) -> None:
    """Verify the index manager chunks, embeds, and routes records correctly.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    fake_pinecone = FakePineconeClient()
    fake_policy_cache = FakePolicyCache()
    settings = _build_settings()

    monkeypatch.setattr(
        "app.vectorstore.index_manager.embed_texts",
        lambda texts: [[float(index + 1), float(len(text))] for index, text in enumerate(texts)],
    )
    monkeypatch.setattr(
        "app.vectorstore.index_manager.extract_pdf_to_text_chunks",
        lambda pdf_bytes, **kwargs: ExtractedPdf(
            full_text="Passport copy and fee receipt.",
            chunks=[
                TextChunk(
                    chunk_id="p1-page_text-1",
                    page_number=1,
                    source_type="page_text",
                    text="Passport copy and fee receipt.",
                    char_count=30,
                    metadata={"page_number": 1, "method": "pymupdf_text"},
                )
            ],
            pages=[
                {
                    "page_number": 1,
                    "text": "Passport copy and fee receipt.",
                    "entries": [],
                }
            ],
        ),
    )

    indexer = VisitorProgramIndexer(
        settings,
        crawler=FakeCrawler(),
        pinecone_client=fake_pinecone,
        chunker=TextChunker(ChunkingConfig(chunk_size=80, chunk_overlap=10)),
        policy_cache=fake_policy_cache,
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
    assert (
        fake_pinecone.checklist_records[0].metadata["source_reference"]
        == "/workspace/services/data/forms/imm5484e.pdf"
    )
    assert fake_policy_cache.stored_dates == [
        ("https://example.com/guidelines", "2026-03-03")
    ]


def test_index_manager_skips_unchanged_operational_guidelines(monkeypatch) -> None:
    """Verify unchanged IRCC pages are skipped before embedding.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """

    fake_pinecone = FakePineconeClient()
    fake_policy_cache = FakePolicyCache(
        has_changed_by_url={"https://example.com/guidelines": False}
    )
    settings = _build_settings()

    monkeypatch.setattr(
        "app.vectorstore.index_manager.embed_texts",
        lambda texts: [[float(index + 1), float(len(text))] for index, text in enumerate(texts)],
    )
    monkeypatch.setattr(
        "app.vectorstore.index_manager.extract_pdf_to_text_chunks",
        lambda pdf_bytes, **kwargs: ExtractedPdf(
            full_text="Passport copy and fee receipt.",
            chunks=[
                TextChunk(
                    chunk_id="p1-page_text-1",
                    page_number=1,
                    source_type="page_text",
                    text="Passport copy and fee receipt.",
                    char_count=30,
                    metadata={"page_number": 1, "method": "pymupdf_text"},
                )
            ],
            pages=[
                {
                    "page_number": 1,
                    "text": "Passport copy and fee receipt.",
                    "entries": [],
                }
            ],
        ),
    )

    indexer = VisitorProgramIndexer(
        settings,
        crawler=FakeCrawler(),
        pinecone_client=fake_pinecone,
        chunker=TextChunker(ChunkingConfig(chunk_size=80, chunk_overlap=10)),
        policy_cache=fake_policy_cache,
    )

    summary = indexer.index_all_sources()

    print("\n=== SKIPPED GUIDELINE SUMMARY ===")
    print(summary.to_dict())
    print("\n=== SKIPPED GUIDELINE RECORD COUNTS ===")
    print(
        {
            "guideline_records": len(fake_pinecone.guideline_records),
            "checklist_records": len(fake_pinecone.checklist_records),
        }
    )

    assert summary.crawled_documents == 1
    assert len(fake_pinecone.guideline_records) == 0
    assert len(fake_pinecone.checklist_records) == 1
    assert fake_policy_cache.stored_dates == []


def test_study_permit_indexer_uses_operational_guidelines_and_checklist_pdf(
    monkeypatch,
) -> None:
    """Verify the study-permit configuration indexes both configured sources.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """

    fake_pinecone = FakePineconeClient()
    fake_policy_cache = FakePolicyCache()
    settings = _build_settings()

    monkeypatch.setattr(
        "app.vectorstore.index_manager.embed_texts",
        lambda texts: [[float(index + 1), float(len(text))] for index, text in enumerate(texts)],
    )
    monkeypatch.setattr(
        "app.vectorstore.index_manager.extract_pdf_to_text_chunks",
        lambda pdf_bytes, **kwargs: ExtractedPdf(
            full_text="Study permit checklist content.",
            chunks=[
                TextChunk(
                    chunk_id="p1-page_text-1",
                    page_number=1,
                    source_type="page_text",
                    text="Study permit checklist content.",
                    char_count=31,
                    metadata={"page_number": 1, "method": "pymupdf_text"},
                )
            ],
            pages=[
                {
                    "page_number": 1,
                    "text": "Study permit checklist content.",
                    "entries": [],
                }
            ],
        ),
    )

    indexer = VisitorProgramIndexer(
        settings,
        crawler=VisitorProgramCrawler(sources=build_study_permit_sources()),
        pinecone_client=fake_pinecone,
        chunker=TextChunker(ChunkingConfig(chunk_size=80, chunk_overlap=10)),
        policy_cache=fake_policy_cache,
    )

    summary = indexer.index_all_sources()

    print("\n=== STUDY PERMIT SUMMARY ===")
    print(summary.to_dict())

    assert summary.crawled_documents == 2
    assert summary.operational_guidelines_upserts > 0
    assert summary.document_checklist_upserts == 1
    assert len(fake_pinecone.guideline_records) > 0
    assert len(fake_pinecone.checklist_records) == 1
    assert fake_policy_cache.stored_dates
    assert fake_pinecone.checklist_records[0].metadata["document_title"] == (
        "Study permit document checklist PDF"
    )


def test_index_single_source_skips_unchanged_guideline_page(monkeypatch) -> None:
    """Verify direct indexing skips unchanged operational-guidelines pages.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """

    fake_pinecone = FakePineconeClient()
    fake_policy_cache = FakePolicyCache(
        has_changed_by_url={"https://example.com/guidelines": False}
    )
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
        policy_cache=fake_policy_cache,
    )

    result = indexer.index_single_source(
        CrawlerSource(
            kind="operational_guidelines",
            title="Guidelines",
            url="https://example.com/guidelines",
        )
    )

    print("\n=== SINGLE SOURCE SKIP RESULT ===")
    print(result.to_dict())

    assert result.status == "skipped_up_to_date"
    assert result.records_upserted == 0
    assert len(fake_pinecone.guideline_records) == 0
    assert any("policy_source_skipped" in line for line in result.logs)


def test_index_single_source_indexes_local_pdf(monkeypatch) -> None:
    """Verify direct indexing reads a local PDF and upserts checklist vectors.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """

    fake_pinecone = FakePineconeClient()
    fake_policy_cache = FakePolicyCache()
    settings = _build_settings()

    monkeypatch.setattr(
        "app.vectorstore.index_manager.embed_texts",
        lambda texts: [[float(index + 1), float(len(text))] for index, text in enumerate(texts)],
    )
    monkeypatch.setattr(
        "app.vectorstore.index_manager.extract_pdf_to_text_chunks",
        lambda pdf_bytes, **kwargs: ExtractedPdf(
            full_text="Checklist content.",
            chunks=[
                TextChunk(
                    chunk_id="p1-page_text-1",
                    page_number=1,
                    source_type="page_text",
                    text="Checklist content.",
                    char_count=18,
                    metadata={"page_number": 1, "method": "pymupdf_text"},
                )
            ],
            pages=[
                {
                    "page_number": 1,
                    "text": "Checklist content.",
                    "entries": [],
                }
            ],
        ),
    )

    indexer = VisitorProgramIndexer(
        settings,
        crawler=FakeCrawler(),
        pinecone_client=fake_pinecone,
        chunker=TextChunker(ChunkingConfig(chunk_size=80, chunk_overlap=10)),
        policy_cache=fake_policy_cache,
    )

    result = indexer.index_single_source(
        CrawlerSource(
            kind="document_checklist_pdf",
            title="Checklist PDF",
            url="",
            file_path="/workspace/services/data/forms/imm5484e.pdf",
        )
    )

    print("\n=== SINGLE SOURCE PDF RESULT ===")
    print(result.to_dict())

    assert result.status == "indexed"
    assert result.records_upserted == 1
    assert len(fake_pinecone.checklist_records) == 1
    assert any("upsert_checklists_done count=1" in line for line in result.logs)


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
        redis_url="redis://redis:6379/0",
        redis_policy_cache_prefix="truthy:test-policy-modified",
    )
