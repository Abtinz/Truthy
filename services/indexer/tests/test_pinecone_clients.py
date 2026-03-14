from __future__ import annotations

from unittest.mock import patch

from app.core.config import IndexerSettings
from app.vectorstore.pinecone_client import PineconeIndexerClient, PineconeVectorRecord
from app.vectorstore.pinecone_retriever import PineconeRetrieverClient


class FakePineconeIndex:
    """Simple in-memory Pinecone index stub used in unit tests.

    Args:
        name: Logical index name for assertion visibility.

    Returns:
        FakePineconeIndex: Test double for Pinecone index operations.
    """

    def __init__(self, name: str) -> None:
        """Initialize the fake index.

        Args:
            name: Logical index name for the stub.

        Returns:
            None.
        """
        self.name = name
        self.upsert_calls: list[dict[str, object]] = []
        self.query_calls: list[dict[str, object]] = []

    def upsert(self, *, vectors: list[dict[str, object]], namespace: str | None = None) -> dict[str, object]:
        """Record an upsert call and return a predictable fake response.

        Args:
            vectors: Upsert payload.
            namespace: Optional namespace value.

        Returns:
            dict[str, object]: Fake upsert result.
        """
        call = {"vectors": vectors, "namespace": namespace}
        self.upsert_calls.append(call)
        return {"upserted_count": len(vectors), "index_name": self.name}

    def query(self, **kwargs: object) -> dict[str, object]:
        """Record a query call and return predictable fake matches.

        Args:
            **kwargs: Query parameters from the client.

        Returns:
            dict[str, object]: Fake query result payload.
        """
        self.query_calls.append(kwargs)
        return {
            "matches": [
                {
                    "id": f"{self.name}-match-1",
                    "score": 0.91,
                    "metadata": {"source": self.name, "rank": 1},
                },
                {
                    "id": f"{self.name}-match-2",
                    "score": 0.72,
                    "metadata": {"source": self.name, "rank": 2},
                },
            ]
        }


class FakePinecone:
    """Simple Pinecone SDK stub that exposes named fake indexes.

    Args:
        api_key: API key passed by the client under test.

    Returns:
        FakePinecone: Test double for the Pinecone SDK entrypoint.
    """

    def __init__(self, api_key: str) -> None:
        """Initialize the fake Pinecone SDK object.

        Args:
            api_key: API key passed by the client under test.

        Returns:
            None.
        """
        self.api_key = api_key
        self.indexes: dict[str, FakePineconeIndex] = {}

    def Index(self, name: str) -> FakePineconeIndex:
        """Return a stable fake index object for the requested name.

        Args:
            name: Pinecone index name.

        Returns:
            FakePineconeIndex: Reused fake index handle.
        """
        if name not in self.indexes:
            self.indexes[name] = FakePineconeIndex(name)
        return self.indexes[name]


def test_indexer_settings_from_env_reads_pinecone_values(monkeypatch) -> None:
    """Verify Pinecone settings are loaded correctly from environment values.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    monkeypatch.setenv("PINECONE_API_KEY", "test-key")
    monkeypatch.setenv("PINECONE_OPERATIONAL_GUIDELINES_INDEX_NAME", "guidelines-index")
    monkeypatch.setenv("PINECONE_DOCUMENT_CHECKLIST_INDEX_NAME", "checklist-index")
    monkeypatch.setenv("PINECONE_NAMESPACE", "truthy-dev")
    monkeypatch.setenv("PINECONE_TOP_K", "7")

    settings = IndexerSettings.from_env()

    print("\n=== INDEXER SETTINGS ===")
    print(settings)

    assert settings.pinecone_api_key == "test-key"
    assert settings.pinecone_operational_guidelines_index_name == "guidelines-index"
    assert settings.pinecone_document_checklist_index_name == "checklist-index"
    assert settings.pinecone_namespace == "truthy-dev"
    assert settings.pinecone_top_k == 7


def test_pinecone_indexer_client_upserts_into_operational_guidelines_index() -> None:
    """Verify the write client targets the operational guidelines index.

    Args:
        None.

    Returns:
        None.
    """
    settings = _build_settings()

    with patch("app.vectorstore.pinecone_client.Pinecone", FakePinecone):
        client = PineconeIndexerClient(settings)
        response = client.upsert_operational_guidelines(
            [
                PineconeVectorRecord(
                    record_id="guide-1",
                    values=[0.1, 0.2, 0.3],
                    metadata={"document_type": "guideline"},
                )
            ]
        )

        fake_index = client._pinecone.Index("guidelines-index")

    print("\n=== GUIDELINES UPSERT RESPONSE ===")
    print(response)
    print(fake_index.upsert_calls)

    assert response["upserted_count"] == 1
    assert fake_index.upsert_calls[0]["namespace"] == "truthy-dev"


def test_pinecone_indexer_client_upserts_into_document_checklist_index() -> None:
    """Verify the write client targets the checklist index.

    Args:
        None.

    Returns:
        None.
    """
    settings = _build_settings()

    with patch("app.vectorstore.pinecone_client.Pinecone", FakePinecone):
        client = PineconeIndexerClient(settings)
        response = client.upsert_document_checklists(
            [
                PineconeVectorRecord(
                    record_id="checklist-1",
                    values=[0.4, 0.5],
                    metadata={"document_type": "checklist"},
                )
            ],
            namespace="custom-space",
        )

        fake_index = client._pinecone.Index("checklist-index")

    print("\n=== CHECKLIST UPSERT RESPONSE ===")
    print(response)
    print(fake_index.upsert_calls)

    assert response["index_name"] == "checklist-index"
    assert fake_index.upsert_calls[0]["namespace"] == "custom-space"


def test_pinecone_retriever_client_queries_guidelines_index() -> None:
    """Verify the retriever queries the operational guidelines index.

    Args:
        None.

    Returns:
        None.
    """
    settings = _build_settings()

    with patch("app.vectorstore.pinecone_client.Pinecone", FakePinecone):
        retriever = PineconeRetrieverClient(settings)
        matches = retriever.search_operational_guidelines([0.1, 0.2, 0.3])
        fake_index = retriever._indexer_client._pinecone.Index("guidelines-index")

    print("\n=== GUIDELINES SEARCH MATCHES ===")
    print([match.to_dict() for match in matches])
    print(fake_index.query_calls)

    assert len(matches) == 2
    assert fake_index.query_calls[0]["top_k"] == 5
    assert fake_index.query_calls[0]["namespace"] == "truthy-dev"


def test_pinecone_retriever_client_queries_checklist_index() -> None:
    """Verify the retriever queries the document checklist index.

    Args:
        None.

    Returns:
        None.
    """
    settings = _build_settings()

    with patch("app.vectorstore.pinecone_client.Pinecone", FakePinecone):
        retriever = PineconeRetrieverClient(settings)
        matches = retriever.search_document_checklists(
            [0.9, 0.8],
            top_k=3,
            namespace="retrieval-space",
        )
        fake_index = retriever._indexer_client._pinecone.Index("checklist-index")

    print("\n=== CHECKLIST SEARCH MATCHES ===")
    print([match.to_dict() for match in matches])
    print(fake_index.query_calls)

    assert len(matches) == 2
    assert fake_index.query_calls[0]["top_k"] == 3
    assert fake_index.query_calls[0]["namespace"] == "retrieval-space"


def _build_settings() -> IndexerSettings:
    """Construct a stable settings object for unit tests.

    Args:
        None.

    Returns:
        IndexerSettings: Reusable test settings.
    """
    return IndexerSettings(
        pinecone_api_key="test-key",
        pinecone_operational_guidelines_index_name="guidelines-index",
        pinecone_document_checklist_index_name="checklist-index",
        pinecone_namespace="truthy-dev",
        pinecone_top_k=5,
    )
