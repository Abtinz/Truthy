from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from app.core.config import IndexerSettings

try:
    from pinecone import Pinecone
except ImportError:  # pragma: no cover - exercised indirectly in runtime only.
    Pinecone = None  # type: ignore[assignment]


PineconeIndexKey = Literal["operational_guidelines", "document_checklist"]


@dataclass(frozen=True)
class PineconeVectorRecord:
    """Vector record to be written into a Pinecone index.

    Args:
        record_id: Unique identifier for the record inside Pinecone.
        values: Dense vector values associated with the record.
        metadata: Optional metadata stored alongside the vector.

    Returns:
        PineconeVectorRecord: Immutable record container for upsert operations.
    """

    record_id: str
    values: list[float]
    metadata: dict[str, Any] | None = None

    def to_payload(self) -> dict[str, Any]:
        """Convert the record to the structure expected by Pinecone.

        Args:
            None.

        Returns:
            dict[str, Any]: Pinecone-compatible vector payload.
        """
        payload: dict[str, Any] = {
            "id": self.record_id,
            "values": self.values,
        }
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload


class PineconeIndexerClient:
    """Client responsible for writing vectors into Pinecone indexes.

    This client manages both repository-level Pinecone indexes used by the
    indexer service:
    - operational guidelines instructions
    - document checklist PDF

    Args:
        settings: Environment-backed indexer settings.

    Returns:
        PineconeIndexerClient: Configured Pinecone write client.
    """

    def __init__(self, settings: IndexerSettings) -> None:
        """Initialize the Pinecone indexer client.

        Args:
            settings: Runtime settings including API key and index names.

        Returns:
            None.

        Raises:
            RuntimeError: If the Pinecone SDK is not installed.
        """
        if Pinecone is None:
            raise RuntimeError(
                "The Pinecone SDK is not installed. Install the `pinecone` package "
                "before using PineconeIndexerClient."
            )

        self.settings = settings
        self._pinecone = Pinecone(api_key=settings.pinecone_api_key)

    def upsert_operational_guidelines(
        self,
        records: list[PineconeVectorRecord],
        *,
        namespace: str | None = None,
    ) -> dict[str, Any]:
        """Upsert vectors into the operational guidelines index.

        Args:
            records: Vector records to upsert.
            namespace: Optional namespace override for this operation.

        Returns:
            dict[str, Any]: Raw Pinecone upsert response.
        """
        return self._upsert_records(
            index_key="operational_guidelines",
            records=records,
            namespace=namespace,
        )

    def upsert_document_checklists(
        self,
        records: list[PineconeVectorRecord],
        *,
        namespace: str | None = None,
    ) -> dict[str, Any]:
        """Upsert vectors into the document checklist index.

        Args:
            records: Vector records to upsert.
            namespace: Optional namespace override for this operation.

        Returns:
            dict[str, Any]: Raw Pinecone upsert response.
        """
        return self._upsert_records(
            index_key="document_checklist",
            records=records,
            namespace=namespace,
        )

    def _upsert_records(
        self,
        *,
        index_key: PineconeIndexKey,
        records: list[PineconeVectorRecord],
        namespace: str | None,
    ) -> dict[str, Any]:
        """Perform the actual upsert call against a selected Pinecone index.

        Args:
            index_key: Internal logical index identifier.
            records: Records to convert and upsert.
            namespace: Optional namespace override.

        Returns:
            dict[str, Any]: Raw response returned by Pinecone.
        """
        index = self._get_index(index_key)
        payload = [record.to_payload() for record in records]
        effective_namespace = namespace or self.settings.pinecone_namespace

        if effective_namespace:
            return index.upsert(vectors=payload, namespace=effective_namespace)
        return index.upsert(vectors=payload)

    def _get_index(self, index_key: PineconeIndexKey) -> Any:
        """Resolve one of the configured Pinecone index handles.

        Args:
            index_key: Internal logical index identifier.

        Returns:
            Any: Pinecone index handle from the SDK.
        """
        if index_key == "operational_guidelines":
            return self._pinecone.Index(
                self.settings.pinecone_operational_guidelines_index_name
            )
        return self._pinecone.Index(
            self.settings.pinecone_document_checklist_index_name
        )
