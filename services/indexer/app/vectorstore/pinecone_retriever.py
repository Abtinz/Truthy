from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.core.config import IndexerSettings
from app.vectorstore.pinecone_client import PineconeIndexKey, PineconeIndexerClient


@dataclass(frozen=True)
class PineconeSearchMatch:
    """Structured Pinecone retrieval result.

    Args:
        record_id: Unique record identifier returned by Pinecone.
        score: Similarity score returned for the match.
        metadata: Metadata attached to the matched vector.

    Returns:
        PineconeSearchMatch: Immutable retrieval result object.
    """

    record_id: str
    score: float
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert the match to a serializable dictionary.

        Args:
            None.

        Returns:
            dict[str, Any]: Plain dictionary representation of the match.
        """
        return {
            "record_id": self.record_id,
            "score": self.score,
            "metadata": self.metadata,
        }


class PineconeRetrieverClient:
    """Client responsible for querying the configured Pinecone indexes.

    The retriever is intentionally separated from the write client to keep the
    API explicit: indexing code upserts vectors, retrieval code queries them.

    Args:
        settings: Environment-backed indexer settings.

    Returns:
        PineconeRetrieverClient: Configured Pinecone read client.
    """

    def __init__(self, settings: IndexerSettings) -> None:
        """Initialize the Pinecone retriever client.

        Args:
            settings: Runtime settings including API key and index names.

        Returns:
            None.
        """
        self.settings = settings
        self._indexer_client = PineconeIndexerClient(settings)

    def search_operational_guidelines(
        self,
        vector: list[float],
        *,
        top_k: int | None = None,
        namespace: str | None = None,
        include_metadata: bool = True,
    ) -> list[PineconeSearchMatch]:
        """Query the operational guidelines index.

        Args:
            vector: Query vector used for semantic similarity search.
            top_k: Optional override for number of matches to return.
            namespace: Optional namespace override.
            include_metadata: Whether metadata should be included in results.

        Returns:
            list[PineconeSearchMatch]: Structured search results.
        """
        return self._search_index(
            index_key="operational_guidelines",
            vector=vector,
            top_k=top_k,
            namespace=namespace,
            include_metadata=include_metadata,
        )

    def search_document_checklists(
        self,
        vector: list[float],
        *,
        top_k: int | None = None,
        namespace: str | None = None,
        include_metadata: bool = True,
    ) -> list[PineconeSearchMatch]:
        """Query the document checklist index.

        Args:
            vector: Query vector used for semantic similarity search.
            top_k: Optional override for number of matches to return.
            namespace: Optional namespace override.
            include_metadata: Whether metadata should be included in results.

        Returns:
            list[PineconeSearchMatch]: Structured search results.
        """
        return self._search_index(
            index_key="document_checklist",
            vector=vector,
            top_k=top_k,
            namespace=namespace,
            include_metadata=include_metadata,
        )

    def _search_index(
        self,
        *,
        index_key: PineconeIndexKey,
        vector: list[float],
        top_k: int | None,
        namespace: str | None,
        include_metadata: bool,
    ) -> list[PineconeSearchMatch]:
        """Execute a similarity query against a selected Pinecone index.

        Args:
            index_key: Internal logical index identifier.
            vector: Query vector used for similarity search.
            top_k: Optional result count override.
            namespace: Optional namespace override.
            include_metadata: Whether metadata should be included.

        Returns:
            list[PineconeSearchMatch]: Structured match list.
        """
        index = self._indexer_client._get_index(index_key)
        effective_top_k = top_k or self.settings.pinecone_top_k
        effective_namespace = namespace or self.settings.pinecone_namespace

        query_kwargs: dict[str, Any] = {
            "vector": vector,
            "top_k": effective_top_k,
            "include_metadata": include_metadata,
        }
        if effective_namespace:
            query_kwargs["namespace"] = effective_namespace

        response = index.query(**query_kwargs)
        return [
            PineconeSearchMatch(
                record_id=match["id"],
                score=match["score"],
                metadata=match.get("metadata"),
            )
            for match in response.get("matches", [])
        ]
