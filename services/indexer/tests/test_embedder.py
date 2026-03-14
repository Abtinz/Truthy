from __future__ import annotations

from unittest.mock import patch

from app.embeddings.embedder import (
    EmbeddingConfig,
    _batch,
    embed_query,
    embed_texts,
    load_embedding_config,
)
from app.embeddings.openai_client import (
    OpenAIEmbeddingClientConfig,
    OpenAIEmbeddingClientManager,
)


class FakeEmbeddingItem:
    """Simple embedding item stub matching the OpenAI SDK response shape.

    Args:
        embedding: Embedding vector to expose.

    Returns:
        FakeEmbeddingItem: Test embedding response item.
    """

    def __init__(self, embedding: list[float]) -> None:
        """Initialize the fake embedding item.

        Args:
            embedding: Embedding vector to expose.

        Returns:
            None.
        """
        self.embedding = embedding


class FakeEmbeddingsAPI:
    """Fake embeddings API used to test the OpenAI client wrapper.

    Args:
        None.

    Returns:
        FakeEmbeddingsAPI: Test double for OpenAI embeddings endpoint.
    """

    def __init__(self) -> None:
        """Initialize the fake embeddings endpoint.

        Args:
            None.

        Returns:
            None.
        """
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object):
        """Record a create call and return deterministic vectors.

        Args:
            **kwargs: Request payload for embeddings creation.

        Returns:
            object: Simple response object with `data` items.
        """
        self.calls.append(kwargs)
        inputs = kwargs["input"]
        if isinstance(inputs, str):
            inputs = [inputs]

        response_items = [
            FakeEmbeddingItem([float(index + 1), float(len(text))])
            for index, text in enumerate(inputs)
        ]
        return type("EmbeddingResponse", (), {"data": response_items})()


class FakeOpenAI:
    """Fake OpenAI client exposing only the embeddings interface.

    Args:
        api_key: API key passed by the caller.

    Returns:
        FakeOpenAI: Test double for the OpenAI SDK client.
    """

    def __init__(self, api_key: str) -> None:
        """Initialize the fake OpenAI client.

        Args:
            api_key: API key passed by the caller.

        Returns:
            None.
        """
        self.api_key = api_key
        self.embeddings = FakeEmbeddingsAPI()


def test_load_embedding_config_reads_batch_size(monkeypatch) -> None:
    """Verify embedding config is loaded from environment variables.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    monkeypatch.setenv("EMBED_BATCH_SIZE", "4")

    config = load_embedding_config()

    print("\n=== EMBEDDING CONFIG ===")
    print(config)

    assert config.batch_size == 4


def test_batch_prints_expected_groups() -> None:
    """Verify the batching helper yields fixed-size groups.

    Args:
        None.

    Returns:
        None.
    """
    batches = list(_batch(["a", "b", "c", "d", "e"], 2))

    print("\n=== TEXT BATCHES ===")
    print(batches)

    assert batches == [["a", "b"], ["c", "d"], ["e"]]


def test_openai_embedding_client_manager_embeds_documents() -> None:
    """Verify the OpenAI client wrapper returns document embeddings.

    Args:
        None.

    Returns:
        None.
    """
    config = OpenAIEmbeddingClientConfig(
        api_key="test-key",
        model="text-embedding-3-small",
        dimensions=1536,
    )

    with patch("app.embeddings.openai_client.OpenAI", FakeOpenAI):
        manager = OpenAIEmbeddingClientManager(config)
        vectors = manager.embed_documents(["alpha", "beta"])
        calls = manager._client.embeddings.calls

    print("\n=== DOCUMENT EMBEDDINGS ===")
    print(vectors)
    print(calls)

    assert len(vectors) == 2
    assert calls[0]["model"] == "text-embedding-3-small"
    assert calls[0]["dimensions"] == 1536


def test_embed_texts_batches_requests_and_prints_vectors() -> None:
    """Verify `embed_texts` batches input texts before embedding.

    Args:
        None.

    Returns:
        None.
    """
    with patch("app.embeddings.openai_client.OpenAI", FakeOpenAI):
        manager = OpenAIEmbeddingClientManager(
            OpenAIEmbeddingClientConfig(api_key="test-key")
        )
        vectors = embed_texts(
            ["one", "two", "three"],
            config=EmbeddingConfig(batch_size=2),
            manager=manager,
        )
        calls = manager._client.embeddings.calls

    print("\n=== BATCHED TEXT EMBEDDINGS ===")
    print(vectors)
    print(calls)

    assert len(vectors) == 3
    assert len(calls) == 2


def test_embed_query_returns_single_vector_and_prints_result() -> None:
    """Verify `embed_query` returns the first query embedding vector.

    Args:
        None.

    Returns:
        None.
    """
    with patch("app.embeddings.openai_client.OpenAI", FakeOpenAI):
        manager = OpenAIEmbeddingClientManager(
            OpenAIEmbeddingClientConfig(api_key="test-key")
        )
        vector = embed_query("visitor visa checklist", manager=manager)

    print("\n=== QUERY EMBEDDING ===")
    print(vector)

    assert vector
    assert isinstance(vector[0], float)
