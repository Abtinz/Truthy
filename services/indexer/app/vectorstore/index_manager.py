from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Any

from app.chunking.text_chunker import ChunkingConfig, TextChunker
from app.core.config import IndexerSettings
from app.embeddings.embedder import embed_texts
from app.ingestion.crawler import CrawledDocument, HierarchicalSection, VisitorProgramCrawler
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
    ) -> None:
        """Initialize the indexer manager.

        Args:
            settings: Runtime indexer settings.
            crawler: Optional crawler override.
            pinecone_client: Optional Pinecone client override.
            chunker: Optional text chunker override.

        Returns:
            None.
        """
        self.settings = settings
        self.crawler = crawler or VisitorProgramCrawler()
        self.pinecone_client = pinecone_client or PineconeIndexerClient(settings)
        self.chunker = chunker or TextChunker(
            ChunkingConfig(chunk_size=1200, chunk_overlap=150)
        )

    def index_all_sources(self) -> IndexingSummary:
        """Crawl, chunk, embed, and upsert all configured visitor sources.

        Args:
            None.

        Returns:
            IndexingSummary: Summary of the indexing run.
        """
        print("crawl_all_start", flush=True)
        documents = self.crawler.crawl_all()
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
            crawled_documents=len(documents),
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
            "source_url": document.source.url,
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
                document.source.url,
                " > ".join(section.path),
                section.content[:200],
            ]
        )
        digest = sha256(digest_source.encode("utf-8")).hexdigest()[:16]
        return f"{document.source.kind}-{digest}"
