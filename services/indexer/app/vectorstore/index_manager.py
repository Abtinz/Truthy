from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from app.cache.policy_freshness_cache import PolicyFreshnessCache
from app.chunking.text_chunker import ChunkingConfig, TextChunker
from app.core.config import IndexerSettings
from app.embeddings.embedder import embed_texts
from app.ingestion.crawler import (
    CrawledDocument,
    CrawlerSource,
    HierarchicalSection,
    VisitorProgramCrawler,
)
from app.ingestion.pdf_to_text import ExtractedPdf, extract_pdf_to_text_chunks
from app.vectorstore.pinecone_client import PineconeIndexerClient, PineconeVectorRecord


@dataclass(frozen=True)
class IndexingSummary:
    """Summary of one indexer run across all configured visitor-program sources.

    Args:
        crawled_documents: Number of documents crawled.
        generated_chunks: Number of chunks produced for embedding.
        operational_guidelines_upserts: Number of vectors sent to the
            operational guidelines index.
        document_checklist_upserts: Number of vectors sent to the document
            checklist index.

    Returns:
        IndexingSummary: Immutable indexing run summary.
    """

    crawled_documents: int
    generated_chunks: int
    operational_guidelines_upserts: int
    document_checklist_upserts: int

    def to_dict(self) -> dict[str, int]:
        """Convert the summary to a serializable dictionary.

        Args:
            None.

        Returns:
            dict[str, int]: Plain dictionary representation of the summary.
        """
        return asdict(self)


class VisitorProgramIndexer:
    """End-to-end indexer for the visitor-program seed sources.

    This manager ties together the crawler, chunker, embedder, and Pinecone
    upsert client. It is intentionally limited to the currently supported
    visitor-program sources and the two Pinecone indexes already configured in
    this repository.

    Args:
        settings: Environment-backed indexer settings.
        crawler: Optional crawler override for tests or custom runs.
        pinecone_client: Optional Pinecone client override for tests.
        chunker: Optional text chunker override for tests.
        policy_cache: Optional Redis freshness cache override for tests.

    Returns:
        VisitorProgramIndexer: Configured indexing workflow manager.
    """

    def __init__(
        self,
        settings: IndexerSettings,
        *,
        crawler: VisitorProgramCrawler | None = None,
        pinecone_client: PineconeIndexerClient | None = None,
        chunker: TextChunker | None = None,
        policy_cache: PolicyFreshnessCache | None = None,
    ) -> None:
        """Initialize the indexer manager.

        Args:
            settings: Runtime indexer settings.
            crawler: Optional crawler override.
            pinecone_client: Optional Pinecone client override.
            chunker: Optional text chunker override.
            policy_cache: Optional freshness cache override.

        Returns:
            None.
        """
        self.settings = settings
        self.crawler = crawler or VisitorProgramCrawler()
        self.pinecone_client = pinecone_client or PineconeIndexerClient(settings)
        self.chunker = chunker or TextChunker(
            ChunkingConfig(chunk_size=1200, chunk_overlap=150)
        )
        self.policy_cache = policy_cache or PolicyFreshnessCache(settings)

    def index_all_sources(self) -> IndexingSummary:
        """Index the visitor-program sources into the two Pinecone indexes.

        The indexing path is intentionally split by source type:
        - operational guidelines continue to use the hierarchy-preserving
          crawler output
        - document checklist PDF bypasses the crawler entirely and is read from
          the configured local file through the PDF-to-text extraction pipeline

        Args:
            None.

        Returns:
            IndexingSummary: Summary of the indexing run.
        """
        print("crawl_all_start", flush=True)
        documents = self._crawl_operational_guidelines_documents()
        print(f"crawl_all_done documents={len(documents)}", flush=True)
        records_by_index = {
            "operational_guidelines": [],
            "document_checklist_pdf": [],
        }

        total_chunks = 0
        for document in documents:
            print(
                f"build_records_start kind={document.source.kind} title={document.document_title}",
                flush=True,
            )
            chunk_records = self._build_records_for_document(document)
            print(
                f"build_records_done kind={document.source.kind} records={len(chunk_records)}",
                flush=True,
            )
            total_chunks += len(chunk_records)
            records_by_index[document.source.kind].extend(chunk_records)

        checklist_source = self._get_document_checklist_source()
        if checklist_source is not None:
            print(
                "build_records_start "
                f"kind=document_checklist_pdf title={checklist_source.title}",
                flush=True,
            )
            checklist_records = self._build_records_for_document_checklist_pdf(
                checklist_source
            )
            print(
                "build_records_done "
                f"kind=document_checklist_pdf records={len(checklist_records)}",
                flush=True,
            )
            total_chunks += len(checklist_records)
            records_by_index["document_checklist_pdf"].extend(checklist_records)

        print("ensure_indexes_start", flush=True)
        self.pinecone_client.ensure_required_indexes_exist()
        print("ensure_indexes_done", flush=True)

        if records_by_index["operational_guidelines"]:
            print(
                f"upsert_guidelines_start count={len(records_by_index['operational_guidelines'])}",
                flush=True,
            )
            self.pinecone_client.upsert_operational_guidelines(
                records_by_index["operational_guidelines"],
            )
            for document in documents:
                if document.source.url and document.modified_date:
                    self.policy_cache.store_modified_date(
                        document.source.url,
                        document.modified_date,
                    )
            print("upsert_guidelines_done", flush=True)

        if records_by_index["document_checklist_pdf"]:
            print(
                f"upsert_checklists_start count={len(records_by_index['document_checklist_pdf'])}",
                flush=True,
            )
            self.pinecone_client.upsert_document_checklists(
                records_by_index["document_checklist_pdf"],
            )
            print("upsert_checklists_done", flush=True)

        return IndexingSummary(
            crawled_documents=len(documents) + (1 if checklist_source else 0),
            generated_chunks=total_chunks,
            operational_guidelines_upserts=len(
                records_by_index["operational_guidelines"]
            ),
            document_checklist_upserts=len(
                records_by_index["document_checklist_pdf"]
            ),
        )

    def _build_records_for_document(
        self,
        document: CrawledDocument,
    ) -> list[PineconeVectorRecord]:
        """Convert one crawled document into Pinecone vector records.

        Args:
            document: Structured crawled document.

        Returns:
            list[PineconeVectorRecord]: Vector records ready for Pinecone upsert.
        """
        prepared_chunks: list[dict[str, Any]] = []

        for section in document.sections:
            prepared_chunks.extend(self._chunk_section(document, section))

        return self._embed_prepared_chunks(prepared_chunks)

    def _build_records_for_document_checklist_pdf(
        self,
        source: CrawlerSource,
    ) -> list[PineconeVectorRecord]:
        """Read the local checklist PDF and convert extracted chunks to vectors.

        This method intentionally bypasses the crawler for the checklist index.
        The PDF is read directly from disk, converted to text through the
        repository's PDF extraction function, and each extracted chunk is sent
        to the embedder and Pinecone.

        Args:
            source: Checklist source containing the local PDF path.

        Returns:
            list[PineconeVectorRecord]: Vector records ready for checklist
            index upsert.
        """
        extracted_pdf = self._extract_document_checklist_pdf(source)
        prepared_chunks: list[dict[str, Any]] = []

        for chunk in extracted_pdf.chunks:
            digest = sha256(
                "|".join(
                    [
                        source.source_reference(),
                        str(chunk.page_number),
                        chunk.source_type,
                        chunk.text[:200],
                    ]
                ).encode("utf-8")
            ).hexdigest()[:16]
            prepared_chunks.append(
                {
                    "record_id": f"{source.kind}-{digest}",
                    "text": chunk.text,
                    "metadata": {
                        "source_reference": source.source_reference(),
                        "source_kind": source.kind,
                        "document_title": source.title,
                        "page_number": chunk.page_number,
                        "source_type": chunk.source_type,
                        "text": chunk.text,
                        "char_count": chunk.char_count,
                        **chunk.metadata,
                    },
                }
            )

        return self._embed_prepared_chunks(prepared_chunks)

    def _extract_document_checklist_pdf(self, source: CrawlerSource) -> ExtractedPdf:
        """Load the local checklist PDF and run the PDF-to-text extractor.

        Args:
            source: Checklist source containing the local PDF path.

        Returns:
            ExtractedPdf: Structured extraction output containing full text,
            page summaries, and chunked text records.

        Raises:
            ValueError: If the checklist source has no local file path.
            FileNotFoundError: If the local checklist file does not exist.
        """
        if not source.file_path:
            raise ValueError("document_checklist_pdf source requires file_path")

        pdf_path = Path(source.file_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file does not exist: {pdf_path}")

        return extract_pdf_to_text_chunks(
            pdf_path.read_bytes(),
            chunk_size=1200,
            chunk_overlap=150,
            ocr_images=True,
        )

    def _embed_prepared_chunks(
        self,
        prepared_chunks: list[dict[str, Any]],
    ) -> list[PineconeVectorRecord]:
        """Embed prepared chunk payloads and package them for Pinecone.

        Args:
            prepared_chunks: Chunk payloads containing `record_id`, `text`, and
                metadata fields.

        Returns:
            list[PineconeVectorRecord]: Fully embedded vector records.
        """
        if not prepared_chunks:
            return []

        vectors = embed_texts([chunk["text"] for chunk in prepared_chunks])
        return [
            PineconeVectorRecord(
                record_id=chunk["record_id"],
                values=vector,
                metadata=chunk["metadata"],
            )
            for chunk, vector in zip(prepared_chunks, vectors, strict=True)
        ]

    def _crawl_operational_guidelines_documents(self) -> list[CrawledDocument]:
        """Crawl only the operational-guidelines sources.

        Args:
            None.

        Returns:
            list[CrawledDocument]: Crawled guideline documents ready for the
            section-based indexing path.
        """
        documents: list[CrawledDocument] = []

        for source in self.crawler.sources:
            if source.kind != "operational_guidelines":
                continue

            modified_date = self.crawler.fetch_source_modified_date(source)
            if modified_date:
                freshness_result = self.policy_cache.compare_modified_date(
                    source.url,
                    modified_date,
                )
                print(
                    "policy_freshness_check "
                    f"url={source.url} modified_date={modified_date} "
                    f"cached_modified_date={freshness_result.cached_modified_date} "
                    f"has_changed={freshness_result.has_changed}",
                    flush=True,
                )
                if not freshness_result.has_changed:
                    print(
                        f"policy_source_skipped url={source.url} modified_date={modified_date}",
                        flush=True,
                    )
                    continue

            document = self.crawler.crawl_source(source)
            if modified_date and document.modified_date is None:
                document = CrawledDocument(
                    source=document.source,
                    document_title=document.document_title,
                    sections=document.sections,
                    full_text=document.full_text,
                    modified_date=modified_date,
                )
            documents.append(document)

        return documents

    def _get_document_checklist_source(self) -> CrawlerSource | None:
        """Return the configured checklist source used for direct PDF indexing.

        Args:
            None.

        Returns:
            CrawlerSource | None: Checklist source when configured, otherwise
            `None`.
        """
        for source in self.crawler.sources:
            if source.kind == "document_checklist_pdf":
                return source
        return None

    def _chunk_section(
        self,
        document: CrawledDocument,
        section: HierarchicalSection,
    ) -> list[dict[str, Any]]:
        """Chunk one structured section and attach indexing metadata.

        Args:
            document: Parent crawled document.
            section: Structured section to chunk.

        Returns:
            list[dict[str, Any]]: Chunk payloads ready for embedding.
        """
        section_text = self._render_section_for_embedding(section)
        chunk_prefix = self._build_record_prefix(document, section)
        chunk_metadata = {
            "source_reference": document.source.source_reference(),
            "source_kind": document.source.kind,
            "document_title": document.document_title,
            "section_title": section.title,
            "section_level": section.level,
            "section_path": " > ".join(section.path),
        }

        text_chunks = self.chunker.chunk_text(
            section_text,
            chunk_id_prefix=chunk_prefix,
            metadata=chunk_metadata,
        )

        return [
            {
                "record_id": text_chunk.chunk_id,
                "text": text_chunk.text,
                "metadata": {
                    **text_chunk.metadata,
                    "text": text_chunk.text,
                    "char_count": text_chunk.char_count,
                },
            }
            for text_chunk in text_chunks
        ]

    def _render_section_for_embedding(self, section: HierarchicalSection) -> str:
        """Render a hierarchical section into embedding-ready text.

        Args:
            section: Structured section to render.

        Returns:
            str: Embedding-ready text preserving hierarchy context.
        """
        return (
            f"Section path: {' > '.join(section.path)}\n"
            f"Section title: {section.title}\n"
            f"Content:\n{section.content}"
        ).strip()

    def _build_record_prefix(
        self,
        document: CrawledDocument,
        section: HierarchicalSection,
    ) -> str:
        """Build a stable record-id prefix for one section.

        Args:
            document: Parent crawled document.
            section: Section being chunked.

        Returns:
            str: Stable chunk-id prefix.
        """
        digest_source = "|".join(
            [
                document.source.kind,
                document.source.source_reference(),
                " > ".join(section.path),
                section.content[:200],
            ]
        )
        digest = sha256(digest_source.encode("utf-8")).hexdigest()[:16]
        return f"{document.source.kind}-{digest}"
