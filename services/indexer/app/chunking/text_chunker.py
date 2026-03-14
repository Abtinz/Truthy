from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ChunkingConfig:
    """Configuration for paragraph-aware text chunking.

    This configuration controls the size and overlap characteristics of chunks
    generated from extracted source text. It is intended to be shared across
    ingestion workflows so the repository has one consistent chunking policy.

    Args:
        chunk_size: Maximum number of characters allowed in one chunk.
        chunk_overlap: Number of trailing characters from the previous chunk to
            preserve as context in the next chunk.
        separator_pattern: Regular expression used to split text into
            higher-level segments before chunking.

    Returns:
        ChunkingConfig: Immutable configuration object for the chunker.
    """

    chunk_size: int = 1200
    chunk_overlap: int = 150
    separator_pattern: str = r"\n{2,}"

    def validate(self) -> None:
        """Validate chunking configuration values.

        Args:
            None.

        Returns:
            None.

        Raises:
            ValueError: If chunk size or overlap values are invalid.
        """
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must not be negative")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")


@dataclass(frozen=True)
class TextChunk:
    """Structured chunk ready for indexing or downstream reasoning.

    Args:
        chunk_id: Stable identifier for the chunk within the source context.
        text: Final chunk text after normalization and overlap handling.
        char_count: Character count for the chunk content.
        metadata: Structured provenance metadata copied from the source.

    Returns:
        TextChunk: Immutable chunk object.
    """

    chunk_id: str
    text: str
    char_count: int
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert the chunk to a serializable dictionary.

        Args:
            None.

        Returns:
            dict[str, Any]: Plain dictionary representation of the chunk.
        """
        return asdict(self)


class TextChunker:
    """Reusable paragraph-aware text chunker for ingestion pipelines.

    The chunker preserves paragraph boundaries when possible and falls back to
    word-aware splitting when a single segment exceeds the target chunk size.
    Overlap is applied with whole-word prefixes so chunks retain context
    without producing broken partial-word fragments.

    Args:
        config: Chunking configuration for size and overlap behavior.

    Returns:
        TextChunker: Configured reusable chunker instance.
    """

    def __init__(self, config: ChunkingConfig | None = None) -> None:
        """Initialize a chunker instance.

        Args:
            config: Optional custom chunking configuration. Default values are
                used if omitted.

        Returns:
            None.
        """
        self.config = config or ChunkingConfig()
        self.config.validate()

    def chunk_text(
        self,
        text: str,
        *,
        chunk_id_prefix: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[TextChunk]:
        """Split normalized source text into reusable chunks.

        Args:
            text: Source text to segment.
            chunk_id_prefix: Prefix used to generate stable chunk identifiers.
            metadata: Optional source metadata copied into every chunk.

        Returns:
            list[TextChunk]: Chunked text records. Empty list if the input has
            no meaningful content.
        """
        normalized_text = normalize_text(text)
        if not normalized_text:
            return []

        segments = [
            part.strip()
            for part in re.split(self.config.separator_pattern, normalized_text)
            if part.strip()
        ]
        if not segments:
            segments = [normalized_text]

        base_chunks = self._build_base_chunks(segments)
        overlapped_chunks = self._apply_overlap(base_chunks)
        final_metadata = dict(metadata or {})

        return [
            TextChunk(
                chunk_id=f"{chunk_id_prefix}-{index}",
                text=chunk_text,
                char_count=len(chunk_text),
                metadata=final_metadata,
            )
            for index, chunk_text in enumerate(overlapped_chunks, start=1)
            if chunk_text
        ]

    def _build_base_chunks(self, segments: list[str]) -> list[str]:
        """Combine segments into chunk-sized blocks before overlap is applied.

        Args:
            segments: Paragraph or segment strings prepared for chunking.

        Returns:
            list[str]: Base chunk strings without overlap prefixes.
        """
        raw_chunks: list[str] = []
        current = ""

        for segment in segments:
            candidate = segment if not current else f"{current}\n\n{segment}"
            if len(candidate) <= self.config.chunk_size:
                current = candidate
                continue

            if current:
                raw_chunks.append(current)
                current = segment
                continue

            raw_chunks.extend(
                split_long_text(
                    segment,
                    chunk_size=self.config.chunk_size,
                    chunk_overlap=self.config.chunk_overlap,
                )
            )

        if current:
            raw_chunks.append(current)

        return raw_chunks

    def _apply_overlap(self, chunks: list[str]) -> list[str]:
        """Apply word-safe overlap between consecutive chunks.

        Args:
            chunks: Base chunk strings without overlap.

        Returns:
            list[str]: Chunk strings with contextual overlap prefixes.
        """
        if not chunks or self.config.chunk_overlap == 0:
            return chunks

        overlapped: list[str] = []
        previous = ""

        for chunk in chunks:
            if previous:
                prefix = self._build_overlap_prefix(previous)
                if prefix and not chunk.startswith(prefix):
                    candidate = f"{prefix} {chunk}".strip()
                    if len(candidate) <= self.config.chunk_size:
                        chunk = candidate
            overlapped.append(chunk)
            previous = chunk

        return overlapped

    def _build_overlap_prefix(self, previous_chunk: str) -> str:
        """Build a whole-word overlap prefix from the previous chunk.

        Args:
            previous_chunk: The chunk that precedes the current one.

        Returns:
            str: Word-safe overlap prefix.
        """
        words = previous_chunk.split()
        overlap_words = take_overlap_words(words, self.config.chunk_overlap)
        return " ".join(overlap_words).strip()


def split_long_text(text: str, *, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split oversized text into word-aware chunks.

    This helper is used when a single paragraph or segment is too large for the
    target chunk size. It preserves whole words when possible and carries
    trailing words forward to preserve context.

    Args:
        text: Oversized text to split.
        chunk_size: Maximum characters allowed in each chunk.
        chunk_overlap: Target overlap size in characters.

    Returns:
        list[str]: Word-aware chunk strings.
    """
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
            overlap_words = take_overlap_words(current_words, chunk_overlap)
            current_words = overlap_words + [word]
            continue

        # A single very long token is hard-split because there is no sensible
        # word boundary available. This avoids infinite loops on pathological
        # inputs such as IDs or base64-like strings.
        chunks.append(word[:chunk_size])
        remainder = word[chunk_size:]
        current_words = [remainder] if remainder else []

    if current_words:
        chunks.append(" ".join(current_words))

    return chunks


def take_overlap_words(words: list[str], overlap_chars: int) -> list[str]:
    """Select trailing words that fit inside the overlap window.

    Args:
        words: Existing words from the previous chunk.
        overlap_chars: Maximum overlap size in characters.

    Returns:
        list[str]: Trailing whole words to reuse in the next chunk.
    """
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


def normalize_text(text: str) -> str:
    """Normalize extracted text for consistent chunking.

    Args:
        text: Raw text from OCR, PDF extraction, or parsers.

    Returns:
        str: Text with nulls removed and whitespace normalized.
    """
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
