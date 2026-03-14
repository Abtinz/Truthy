from __future__ import annotations

from app.chunking.text_chunker import (
    ChunkingConfig,
    TextChunker,
    normalize_text,
    split_long_text,
    take_overlap_words,
)


def test_text_chunker_prints_chunks_for_paragraph_input() -> None:
    """Verify that the chunker splits paragraph input into reusable chunks.

    Args:
        None.

    Returns:
        None.
    """
    chunker = TextChunker(ChunkingConfig(chunk_size=80, chunk_overlap=20))
    text = (
        "First paragraph about applicant identity.\n\n"
        "Second paragraph about passport information and program eligibility.\n\n"
        "Third paragraph about signatures and supporting documents."
    )

    chunks = chunker.chunk_text(
        text,
        chunk_id_prefix="policy-doc",
        metadata={"source": "unit-test"},
    )

    print("\n=== CHUNKER PARAGRAPH OUTPUT ===")
    for chunk in chunks:
        print(chunk.to_dict())

    assert len(chunks) >= 2
    assert chunks[0].chunk_id == "policy-doc-1"
    assert all(chunk.metadata["source"] == "unit-test" for chunk in chunks)


def test_text_chunker_preserves_word_safe_overlap() -> None:
    """Verify that overlap prefixes remain word-safe.

    Args:
        None.

    Returns:
        None.
    """
    chunker = TextChunker(ChunkingConfig(chunk_size=35, chunk_overlap=12))
    text = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda"

    chunks = chunker.chunk_text(text, chunk_id_prefix="overlap")

    print("\n=== WORD-SAFE OVERLAP ===")
    for chunk in chunks:
        print(chunk.to_dict())

    assert len(chunks) >= 2
    assert chunks[1].text.split()[0] == "epsilon"


def test_split_long_text_prints_chunk_boundaries() -> None:
    """Verify that oversized text is split into bounded chunks.

    Args:
        None.

    Returns:
        None.
    """
    text = (
        "This is a longer paragraph intended to verify that oversized text is "
        "split into multiple chunks while preserving readability and overlap."
    )

    chunks = split_long_text(text, chunk_size=55, chunk_overlap=12)

    print("\n=== SPLIT LONG TEXT ===")
    print(chunks)

    assert len(chunks) >= 2
    assert all(len(chunk) <= 55 for chunk in chunks)


def test_take_overlap_words_prints_selection() -> None:
    """Verify trailing-word overlap selection behavior.

    Args:
        None.

    Returns:
        None.
    """
    words = ["minor", "citizenship", "application", "package"]

    selected = take_overlap_words(words, overlap_chars=18)

    print("\n=== OVERLAP WORDS ===")
    print(selected)

    assert selected
    assert "package" in selected


def test_normalize_text_prints_normalized_result() -> None:
    """Verify shared text normalization behavior used by the chunker.

    Args:
        None.

    Returns:
        None.
    """
    raw = "Applicant\x00   Name:\t Jane Doe\n\n\nReference   Number:   TEST-001"

    normalized = normalize_text(raw)

    print("\n=== CHUNKER NORMALIZED TEXT ===")
    print(normalized)

    assert normalized == "Applicant Name: Jane Doe\n\nReference Number: TEST-001"


def test_chunking_config_validation_rejects_invalid_values() -> None:
    """Verify invalid chunker configuration is rejected.

    Args:
        None.

    Returns:
        None.
    """
    try:
        ChunkingConfig(chunk_size=50, chunk_overlap=50).validate()
    except ValueError as exc:
        print("\n=== CONFIG VALIDATION ERROR ===")
        print(str(exc))
        assert "chunk_overlap must be smaller than chunk_size" in str(exc)
    else:
        raise AssertionError("Expected invalid chunking configuration error")
