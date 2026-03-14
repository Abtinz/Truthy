from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.core.config import AgenticRagSettings

try:
    from pinecone import Pinecone
except ImportError:  # pragma: no cover - runtime dependency guard only.
    Pinecone = None  # type: ignore[assignment]


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


class PineconeRetrieverClient:
    """Client responsible for querying the configured Pinecone indexes.

    Args:
        settings: Environment-backed agentic-RAG settings.

    Returns:
        PineconeRetrieverClient: Configured Pinecone read client.
    """

    def __init__(self, settings: AgenticRagSettings) -> None:
        """Initialize the Pinecone retriever client.

        Args:
            settings: Runtime settings including API key and index names.

        Returns:
            None.
        """

        if Pinecone is None:
            raise RuntimeError(
                "The Pinecone SDK is not installed. Install the `pinecone` package "
                "before using PineconeRetrieverClient."
            )

        self.settings = settings
        self._pinecone = Pinecone(api_key=settings.pinecone_api_key)

    def search_operational_guidelines(
        self,
        vector: list[float],
        *,
        top_k: int | None = None,
    ) -> list[PineconeSearchMatch]:
        """Query the operational-guidelines index.

        Args:
            vector: Query vector used for semantic similarity search.
            top_k: Optional override for number of matches to return.

        Returns:
            list[PineconeSearchMatch]: Structured search results.
        """

        return self._search_index(
            index_name=self.settings.pinecone_operational_guidelines_index_name,
            vector=vector,
            top_k=top_k,
        )

    def search_document_checklists(
        self,
        vector: list[float],
        *,
        top_k: int | None = None,
    ) -> list[PineconeSearchMatch]:
        """Query the document-checklist index.

        Args:
            vector: Query vector used for semantic similarity search.
            top_k: Optional override for number of matches to return.

        Returns:
            list[PineconeSearchMatch]: Structured search results.
        """

        return self._search_index(
            index_name=self.settings.pinecone_document_checklist_index_name,
            vector=vector,
            top_k=top_k,
        )

    def _search_index(
        self,
        *,
        index_name: str,
        vector: list[float],
        top_k: int | None,
    ) -> list[PineconeSearchMatch]:
        """Execute one similarity query against the selected Pinecone index.

        Args:
            index_name: Target Pinecone index name.
            vector: Query vector used for similarity search.
            top_k: Optional result count override.

        Returns:
            list[PineconeSearchMatch]: Structured match list.
        """

        index = self._pinecone.Index(index_name)
        query_kwargs: dict[str, Any] = {
            "vector": vector,
            "top_k": top_k or self.settings.pinecone_top_k,
            "include_metadata": True,
        }
        if self.settings.pinecone_namespace:
            query_kwargs["namespace"] = self.settings.pinecone_namespace

        response = index.query(**query_kwargs)
        return [
            PineconeSearchMatch(
                record_id=match["id"],
                score=match["score"],
                metadata=match.get("metadata"),
            )
            for match in response.get("matches", [])
        ]
