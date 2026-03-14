from __future__ import annotations

import io
from dataclasses import asdict, dataclass
from typing import Any

import fitz
import pytesseract
from PIL import Image, ImageOps

from app.chunking.text_chunker import (
    ChunkingConfig,
    TextChunker,
    normalize_text,
)


@dataclass(frozen=True)
class TextChunk:
    """Structured text segment extracted from a PDF.

    This object represents one chunk of text that is ready for downstream
    indexing, retrieval, or reasoning workflows. Chunks retain source context
    so later services can trace findings back to the originating page and
    extraction method.

    Args:
        chunk_id: Stable identifier for the chunk within one extraction run.
        page_number: One-based page number from the source PDF.
        source_type: Extraction origin such as native page text or image OCR.
        text: Final chunk text after normalization and chunking.
        char_count: Character count for the chunk text.
        metadata: Additional structured context about extraction provenance.

    Returns:
        TextChunk: Immutable chunk record for downstream processing.
    """

    chunk_id: str
    page_number: int
    source_type: str
    text: str
    char_count: int
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert the chunk to a plain dictionary.

        Args:
            None.

        Returns:
            dict[str, Any]: Serializable dictionary representation of the chunk.
        """
        return asdict(self)


@dataclass(frozen=True)
class ExtractedPdf:
    """Structured result of a PDF extraction pass.

    This object contains the full normalized text, chunked text payloads, and
    page-level summaries. It is designed to be the main transport object between
    the ingestion layer and downstream indexing or reasoning components.

    Args:
        full_text: Concatenated normalized text across all extracted sources.
        chunks: Chunked text records prepared for downstream processing.
        pages: Page-level extraction summaries including source entries.

    Returns:
        ExtractedPdf: Immutable extraction result container.
    """

    full_text: str
    chunks: list[TextChunk]
    pages: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        """Convert the extraction result to a serializable dictionary.

        Args:
            None.

        Returns:
            dict[str, Any]: Dictionary form containing full text, chunks, and
            page summaries.
        """
        return {
            "full_text": self.full_text,
            "chunks": [chunk.to_dict() for chunk in self.chunks],
            "pages": self.pages,
        }


def extract_pdf_to_text_chunks(
    pdf_bytes: bytes,
    *,
    chunk_size: int = 1200,
    chunk_overlap: int = 150,
    ocr_images: bool = True,
) -> ExtractedPdf:
    """Extract native text and OCR text from a PDF, then chunk the result.

    The function reads a PDF from bytes, extracts machine-readable page text,
    optionally OCRs embedded page images, normalizes the result, and splits
    extracted content into chunks sized for indexing and retrieval workloads.

    Args:
        pdf_bytes: Raw PDF content as bytes.
        chunk_size: Maximum target size for each chunk in characters.
        chunk_overlap: Character overlap to preserve context between chunks.
        ocr_images: Whether embedded images should be OCR processed.

    Returns:
        ExtractedPdf: Full extraction result with concatenated text, chunks,
        and page-level details.

    Raises:
        ValueError: If the PDF content is empty or chunking parameters are
        invalid.
    """
    if not pdf_bytes:
        raise ValueError("pdf_bytes must not be empty")

    chunker = TextChunker(
        ChunkingConfig(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    )

    document = fitz.open(stream=pdf_bytes, filetype="pdf")
    page_summaries: list[dict[str, Any]] = []
    chunks: list[TextChunk] = []

    try:
        for page_index, page in enumerate(document):
            page_number = page_index + 1
            page_entries = _extract_page_entries(page, page_number, ocr_images=ocr_images)
            page_text = "\n\n".join(entry["text"] for entry in page_entries if entry["text"])

            page_summaries.append(
                {
                    "page_number": page_number,
                    "text": page_text,
                    "entries": page_entries,
                }
            )

            for entry in page_entries:
                # Chunk each extraction source independently so downstream
                # consumers can still distinguish between native PDF text and
                # OCR-derived text.
                chunks.extend(
                    _chunk_entry(
                        chunker=chunker,
                        page_number=page_number,
                        source_type=entry["source_type"],
                        text=entry["text"],
                        metadata=entry["metadata"],
                    )
                )
    finally:
        document.close()

    full_text = "\n\n".join(page["text"] for page in page_summaries if page["text"])
    return ExtractedPdf(full_text=full_text, chunks=chunks, pages=page_summaries)


def _extract_page_entries(page: fitz.Page, page_number: int, *, ocr_images: bool) -> list[dict[str, Any]]:
    """Extract all text-bearing entries from a single PDF page.

    A page can contribute multiple entries. Native PDF text is extracted first.
    If OCR is enabled, embedded images are also processed and added as separate
    entries with their own metadata.

    Args:
        page: PyMuPDF page object to inspect.
        page_number: One-based page number used for metadata.
        ocr_images: Whether embedded images should be OCR processed.

    Returns:
        list[dict[str, Any]]: Ordered extraction entries for the page.
    """
    entries: list[dict[str, Any]] = []

    text = normalize_text(page.get_text("text"))
    if text:
        entries.append(
            {
                "source_type": "page_text",
                "text": text,
                "metadata": {
                    "page_number": page_number,
                    "method": "pymupdf_text",
                },
            }
        )

    if not ocr_images:
        return entries

    # Pages can reference the same underlying image multiple times. Tracking
    # xrefs prevents duplicate OCR work and duplicate text in the output.
    seen_xrefs: set[int] = set()
    for image_index, image_info in enumerate(page.get_images(full=True), start=1):
        xref = image_info[0]
        if xref in seen_xrefs:
            continue
        seen_xrefs.add(xref)

        extracted = page.parent.extract_image(xref)
        image_bytes = extracted.get("image")
        if not image_bytes:
            continue

        ocr_text = _extract_text_from_image_bytes(image_bytes)
        if not ocr_text:
            continue

        entries.append(
            {
                "source_type": "image_ocr",
                "text": ocr_text,
                "metadata": {
                    "page_number": page_number,
                    "image_index": image_index,
                    "xref": xref,
                    "method": "tesseract_ocr",
                    "extension": extracted.get("ext"),
                },
            }
        )

    return entries


def _extract_text_from_image_bytes(image_bytes: bytes) -> str:
    """Run OCR against raw image bytes and normalize the result.

    The preprocessing step intentionally stays lightweight: grayscale conversion
    and autocontrast are usually enough to improve OCR quality without making
    the extraction path brittle.

    Args:
        image_bytes: Raw bytes for one embedded image.

    Returns:
        str: Normalized OCR text. Empty string if OCR yields no useful text.
    """
    with Image.open(io.BytesIO(image_bytes)) as image:
        grayscale = ImageOps.grayscale(image)
        normalized = ImageOps.autocontrast(grayscale)
        text = pytesseract.image_to_string(normalized, config="--psm 6")
    return normalize_text(text)


def _chunk_entry(
    *,
    chunker: TextChunker,
    page_number: int,
    source_type: str,
    text: str,
    metadata: dict[str, Any],
) -> list[TextChunk]:
    """Split one extracted text entry into retrieval-friendly chunks.

    The function preserves paragraph boundaries when possible. If a paragraph is
    too large for the configured chunk size, it falls back to word-based
    splitting with overlap preservation.

    Args:
        page_number: Source page number.
        source_type: Origin of the text such as native text or OCR.
        text: Text to split into chunks.
        metadata: Provenance metadata that should be copied to each chunk.
        chunker: Reusable text chunker instance.

    Returns:
        list[TextChunk]: Final chunk objects for the input entry.
    """
    base_chunks = chunker.chunk_text(
        text,
        chunk_id_prefix=f"p{page_number}-{source_type}",
        metadata=metadata,
    )

    finalized = [
        TextChunk(
            chunk_id=chunk.chunk_id,
            page_number=page_number,
            source_type=source_type,
            text=chunk.text,
            char_count=chunk.char_count,
            metadata=chunk.metadata,
        )
        for chunk in base_chunks
    ]
    return finalized
