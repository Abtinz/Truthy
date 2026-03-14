from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class IndexerSettings:
    """Environment-backed configuration for the indexer service.

    The settings object centralizes all Pinecone-related configuration so the
    rest of the indexer code can depend on one stable source of truth. Values
    are loaded from environment variables and kept immutable after creation.

    Args:
        pinecone_api_key: API key used to authenticate with Pinecone.
        pinecone_operational_guidelines_index_name: Index name for operational
            guidelines and instructions.
        pinecone_document_checklist_index_name: Index name for document
            checklist PDFs.
        pinecone_namespace: Optional namespace shared across indexing and
            retrieval calls.
        pinecone_top_k: Default number of matches to request during retrieval.

    Returns:
        IndexerSettings: Immutable runtime configuration for the indexer.
    """

    pinecone_api_key: str
    pinecone_operational_guidelines_index_name: str
    pinecone_document_checklist_index_name: str
    pinecone_namespace: str | None = None
    pinecone_top_k: int = 5

    @classmethod
    def from_env(cls) -> "IndexerSettings":
        """Build settings from environment variables.

        Args:
            None.

        Returns:
            IndexerSettings: Parsed settings populated from the process
            environment.

        Raises:
            ValueError: If required environment variables are missing.
        """
        pinecone_api_key = os.getenv("PINECONE_API_KEY", "").strip()
        operational_index_name = os.getenv(
            "PINECONE_OPERATIONAL_GUIDELINES_INDEX_NAME",
            "",
        ).strip()
        checklist_index_name = os.getenv(
            "PINECONE_DOCUMENT_CHECKLIST_INDEX_NAME",
            "",
        ).strip()
        pinecone_namespace = os.getenv("PINECONE_NAMESPACE", "").strip() or None
        pinecone_top_k_raw = os.getenv("PINECONE_TOP_K", "5").strip()

        missing_variables = [
            name
            for name, value in [
                ("PINECONE_API_KEY", pinecone_api_key),
                (
                    "PINECONE_OPERATIONAL_GUIDELINES_INDEX_NAME",
                    operational_index_name,
                ),
                (
                    "PINECONE_DOCUMENT_CHECKLIST_INDEX_NAME",
                    checklist_index_name,
                ),
            ]
            if not value
        ]
        if missing_variables:
            joined_names = ", ".join(missing_variables)
            raise ValueError(f"Missing required environment variables: {joined_names}")

        return cls(
            pinecone_api_key=pinecone_api_key,
            pinecone_operational_guidelines_index_name=operational_index_name,
            pinecone_document_checklist_index_name=checklist_index_name,
            pinecone_namespace=pinecone_namespace,
            pinecone_top_k=int(pinecone_top_k_raw),
        )
