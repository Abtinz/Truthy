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
        pinecone_dimension: Vector dimension used when creating dense indexes.
        pinecone_metric: Similarity metric used when creating dense indexes.
        pinecone_cloud: Serverless cloud provider for Pinecone indexes.
        pinecone_region: Serverless deployment region for Pinecone indexes.
        pinecone_deletion_protection: Pinecone deletion protection mode for
            created indexes.

    Returns:
        IndexerSettings: Immutable runtime configuration for the indexer.
    """

    pinecone_api_key: str
    pinecone_operational_guidelines_index_name: str
    pinecone_document_checklist_index_name: str
    pinecone_namespace: str | None = None
    pinecone_top_k: int = 5
    pinecone_dimension: int = 1536
    pinecone_metric: str = "cosine"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    pinecone_deletion_protection: str = "disabled"

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
        pinecone_dimension_raw = os.getenv("PINECONE_DIMENSION", "1536").strip()
        pinecone_metric = os.getenv("PINECONE_METRIC", "cosine").strip() or "cosine"
        pinecone_cloud = os.getenv("PINECONE_CLOUD", "aws").strip() or "aws"
        pinecone_region = os.getenv("PINECONE_REGION", "us-east-1").strip() or "us-east-1"
        pinecone_deletion_protection = (
            os.getenv("PINECONE_DELETION_PROTECTION", "disabled").strip()
            or "disabled"
        )

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
            pinecone_dimension=int(pinecone_dimension_raw),
            pinecone_metric=pinecone_metric,
            pinecone_cloud=pinecone_cloud,
            pinecone_region=pinecone_region,
            pinecone_deletion_protection=pinecone_deletion_protection,
        )
