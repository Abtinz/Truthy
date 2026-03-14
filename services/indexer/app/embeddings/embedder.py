"""Embedding creation pipeline using the configured OpenAI embedding model."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable

from app.embeddings.openai_client import (
    OpenAIEmbeddingClientManager,
    get_default_openai_embedding_manager,
)


@dataclass(frozen=True)
class EmbeddingConfig:
    """Configuration for embedding generation.

    Args:
        batch_size: Number of texts to embed in one API call.

    Returns:
        EmbeddingConfig: Immutable embedding pipeline configuration.
    """

    batch_size: int = 32


def load_embedding_config() -> EmbeddingConfig:
    """Load embedding config from environment variables.

    Args:
        None.

    Returns:
        EmbeddingConfig: Parsed embedding configuration.
    """
    batch_size = int(os.getenv("EMBED_BATCH_SIZE", "32"))
    return EmbeddingConfig(batch_size=batch_size)


def _batch(iterable: Iterable[str], size: int) -> Iterable[list[str]]:
    """Yield fixed-size batches from an iterable of texts.

    Args:
        iterable: Source iterable of strings.
        size: Maximum batch size.

    Returns:
        Iterable[list[str]]: Sequential string batches.
    """
    batch_list: list[str] = []
    for item in iterable:
        batch_list.append(item)
        if len(batch_list) >= size:
            yield batch_list
            batch_list = []
    if batch_list:
        yield batch_list


def embed_texts(
    texts: list[str],
    config: EmbeddingConfig | None = None,
    manager: OpenAIEmbeddingClientManager | None = None,
) -> list[list[float]]:
    """Embed a list of texts using the configured OpenAI embedding model.

    Args:
        texts: Input texts to embed.
        config: Optional embedding configuration override.
        manager: Optional embedding client manager override.

    Returns:
        list[list[float]]: Embedding vectors in input order.
    """
    cfg = config or load_embedding_config()
    client = manager or get_default_openai_embedding_manager()

    embeddings: list[list[float]] = []
    for batch in _batch(texts, cfg.batch_size):
        embeddings.extend(client.embed_documents(batch))

    return embeddings


def embed_query(
    text: str,
    manager: OpenAIEmbeddingClientManager | None = None,
) -> list[float]:
    """Embed a single query text for retrieval.

    Args:
        text: Query text to embed.
        manager: Optional embedding client manager override.

    Returns:
        list[float]: Query embedding vector.

    Raises:
        RuntimeError: If the embedding API returns no vector.
    """
    client = manager or get_default_openai_embedding_manager()
    vector = client.embed_query(text)
    if not vector:
        raise RuntimeError("Embedding API returned no vector.")
    return vector
