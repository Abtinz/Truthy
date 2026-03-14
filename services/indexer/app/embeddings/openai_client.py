from __future__ import annotations

import os
from dataclasses import dataclass

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - runtime dependency guard only.
    OpenAI = None  # type: ignore[assignment]


@dataclass(frozen=True)
class OpenAIEmbeddingClientConfig:
    """Configuration for OpenAI embedding client creation.

    Args:
        api_key: OpenAI API key used for authentication.
        model: Embedding model identifier.
        dimensions: Optional output dimension override for supported models.

    Returns:
        OpenAIEmbeddingClientConfig: Immutable client configuration.
    """

    api_key: str
    model: str = "text-embedding-3-small"
    dimensions: int | None = None


class OpenAIEmbeddingClientManager:
    """Small wrapper around the OpenAI embeddings API.

    This manager exposes two narrow methods that mirror the usage pattern from
    the prior project reference:
    - `embed_documents` for batch document embedding
    - `embed_query` for single-query embedding

    Args:
        config: OpenAI embedding client configuration.

    Returns:
        OpenAIEmbeddingClientManager: Configured embeddings client wrapper.
    """

    def __init__(self, config: OpenAIEmbeddingClientConfig) -> None:
        """Initialize the OpenAI embedding client manager.

        Args:
            config: Runtime configuration for the OpenAI embeddings client.

        Returns:
            None.

        Raises:
            RuntimeError: If the OpenAI SDK is not installed.
        """
        if OpenAI is None:
            raise RuntimeError(
                "The OpenAI SDK is not installed. Install the `openai` package "
                "before using OpenAIEmbeddingClientManager."
            )

        self.config = config
        self._client = OpenAI(api_key=config.api_key)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of document texts using the OpenAI embeddings API.

        Args:
            texts: Batch of document texts to embed.

        Returns:
            list[list[float]]: Embedding vectors in input order.
        """
        request_kwargs = {
            "model": self.config.model,
            "input": texts,
            "encoding_format": "float",
        }
        if self.config.dimensions is not None:
            request_kwargs["dimensions"] = self.config.dimensions

        response = self._client.embeddings.create(**request_kwargs)
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string for retrieval workflows.

        Args:
            text: Query text to embed.

        Returns:
            list[float]: Query embedding vector.
        """
        vectors = self.embed_documents([text])
        return vectors[0] if vectors else []


def get_default_openai_embedding_manager() -> OpenAIEmbeddingClientManager:
    """Build the default OpenAI embedding manager from environment settings.

    Args:
        None.

    Returns:
        OpenAIEmbeddingClientManager: Default runtime embedding manager.

    Raises:
        ValueError: If the OpenAI API key is missing.
    """
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small").strip()
    dimensions_raw = os.getenv("OPENAI_EMBED_DIMENSIONS", "").strip()

    if not api_key:
        raise ValueError("Missing required environment variable: OPENAI_API_KEY")

    dimensions = int(dimensions_raw) if dimensions_raw else None
    return OpenAIEmbeddingClientManager(
        OpenAIEmbeddingClientConfig(
            api_key=api_key,
            model=model,
            dimensions=dimensions,
        )
    )
