from __future__ import annotations

import io
import re
from dataclasses import asdict, dataclass
from typing import Any

import fitz
import pytesseract
from PIL import Image, ImageOps


@dataclass(frozen=True)
class TextChunk:
    chunk_id: str
    page_number: int
    source_type: str
    text: str
    char_count: int
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExtractedPdf:
    full_text: str
    chunks: list[TextChunk]
    pages: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
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
    if not pdf_bytes:
        raise ValueError("pdf_bytes must not be empty")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must not be negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

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
                chunks.extend(
                    _chunk_entry(
                        page_number=page_number,
                        source_type=entry["source_type"],
                        text=entry["text"],
                        metadata=entry["metadata"],
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                    )
                )
    finally:
        document.close()

    full_text = "\n\n".join(page["text"] for page in page_summaries if page["text"])
    return ExtractedPdf(full_text=full_text, chunks=chunks, pages=page_summaries)


def _extract_page_entries(page: fitz.Page, page_number: int, *, ocr_images: bool) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []

    text = _normalize_text(page.get_text("text"))
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
    with Image.open(io.BytesIO(image_bytes)) as image:
        grayscale = ImageOps.grayscale(image)
        normalized = ImageOps.autocontrast(grayscale)
        text = pytesseract.image_to_string(normalized, config="--psm 6")
    return _normalize_text(text)


def _chunk_entry(
    *,
    page_number: int,
    source_type: str,
    text: str,
    metadata: dict[str, Any],
    chunk_size: int,
    chunk_overlap: int,
) -> list[TextChunk]:
    normalized_text = _normalize_text(text)
    if not normalized_text:
        return []

    paragraphs = [part.strip() for part in re.split(r"\n{2,}", normalized_text) if part.strip()]
    if not paragraphs:
        paragraphs = [normalized_text]

    raw_chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        candidate = paragraph if not current else f"{current}\n\n{paragraph}"
        if len(candidate) <= chunk_size:
            current = candidate
            continue

        if current:
            raw_chunks.append(current)
            current = paragraph
        else:
            raw_chunks.extend(_split_long_text(paragraph, chunk_size=chunk_size, chunk_overlap=chunk_overlap))
            current = ""

    if current:
        raw_chunks.append(current)

    finalized = [
        TextChunk(
            chunk_id=f"p{page_number}-{source_type}-{index}",
            page_number=page_number,
            source_type=source_type,
            text=chunk_text,
            char_count=len(chunk_text),
            metadata=metadata,
        )
        for index, chunk_text in enumerate(
            _apply_overlap(raw_chunks, chunk_overlap=chunk_overlap),
            start=1,
        )
        if chunk_text
    ]
    return finalized


def _split_long_text(text: str, *, chunk_size: int, chunk_overlap: int) -> list[str]:
    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    current_words: list[str] = []

    for word in words:
        candidate = " ".join(current_words + [word])
        if len(candidate) <= chunk_size:
            current_words.append(word)
            continue

        if current_words:
            chunks.append(" ".join(current_words))
            overlap_words = _take_overlap_words(current_words, chunk_overlap)
            current_words = overlap_words + [word]
        else:
            chunks.append(word[:chunk_size])
            remainder = word[chunk_size - chunk_overlap :] if chunk_overlap else word[chunk_size:]
            current_words = [remainder] if remainder else []

    if current_words:
        chunks.append(" ".join(current_words))
    return chunks


def _apply_overlap(chunks: list[str], *, chunk_overlap: int) -> list[str]:
    if not chunks or chunk_overlap == 0:
        return chunks

    overlapped: list[str] = []
    previous = ""
    for chunk in chunks:
        if previous:
            prefix = previous[-chunk_overlap:].strip()
            if prefix and not chunk.startswith(prefix):
                chunk = f"{prefix} {chunk}".strip()
        overlapped.append(chunk)
        previous = chunk
    return overlapped


def _take_overlap_words(words: list[str], overlap_chars: int) -> list[str]:
    if overlap_chars == 0:
        return []

    selected: list[str] = []
    total = 0
    for word in reversed(words):
        projected = len(word) if total == 0 else total + 1 + len(word)
        if projected > overlap_chars and selected:
            break
        selected.insert(0, word)
        total = projected
        if total >= overlap_chars:
            break
    return selected


def _normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
