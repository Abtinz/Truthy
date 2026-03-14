from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class AgenticRagSettings:
    """Environment-backed configuration for the agentic-RAG service.

    Args:
        pinecone_api_key: API key used to authenticate Pinecone retrieval.
        pinecone_operational_guidelines_index_name: Index used for operational
            policy and guidance chunks.
        pinecone_document_checklist_index_name: Index used for checklist PDF
            chunks.
        pinecone_namespace: Optional shared namespace for all vector calls.
        pinecone_top_k: Default number of nearest-neighbour matches per index.
        pinecone_min_similarity_score: Minimum score required for a retrieved
            chunk to be returned to the chain.
        openai_api_key: OpenAI API key used for embedding query text.
        openai_embed_model: Embedding model identifier.
        openai_embed_dimensions: Optional embedding dimension override.
        max_file_text_characters: Maximum characters contributed by any one
            file to the retrieval query.
        max_total_query_characters: Maximum total retrieval query length.

    Returns:
        AgenticRagSettings: Immutable runtime settings container.
    """

    pinecone_api_key: str
    pinecone_operational_guidelines_index_name: str
    pinecone_document_checklist_index_name: str
    pinecone_namespace: str | None
    pinecone_top_k: int
    pinecone_min_similarity_score: float
    openai_api_key: str
    openai_embed_model: str
    openai_embed_dimensions: int | None
    max_file_text_characters: int
    max_total_query_characters: int

    @classmethod
    def from_env(cls) -> "AgenticRagSettings":
        """Build runtime settings from environment variables.

        Args:
            None.

        Returns:
            AgenticRagSettings: Parsed settings populated from environment
            variables.
        """

        dimensions_raw = os.getenv("OPENAI_EMBED_DIMENSIONS", "").strip()
        namespace = os.getenv("PINECONE_NAMESPACE", "").strip() or None

        return cls(
            pinecone_api_key=os.getenv("PINECONE_API_KEY", "").strip(),
            pinecone_operational_guidelines_index_name=os.getenv(
                "PINECONE_OPERATIONAL_GUIDELINES_INDEX_NAME",
                "",
            ).strip(),
            pinecone_document_checklist_index_name=os.getenv(
                "PINECONE_DOCUMENT_CHECKLIST_INDEX_NAME",
                "",
            ).strip(),
            pinecone_namespace=namespace,
            pinecone_top_k=int(os.getenv("PINECONE_TOP_K", "3").strip() or "3"),
            pinecone_min_similarity_score=float(
                os.getenv("PINECONE_MIN_SIMILARITY_SCORE", "0.25").strip() or "0.25"
            ),
            openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
            openai_embed_model=os.getenv(
                "OPENAI_EMBED_MODEL",
                "text-embedding-3-small",
            ).strip()
            or "text-embedding-3-small",
            openai_embed_dimensions=int(dimensions_raw) if dimensions_raw else None,
            max_file_text_characters=int(
                os.getenv("AGENTIC_RAG_MAX_FILE_TEXT_CHARACTERS", "1500").strip()
                or "1500"
            ),
            max_total_query_characters=int(
                os.getenv("AGENTIC_RAG_MAX_TOTAL_QUERY_CHARACTERS", "6000").strip()
                or "6000"
            ),
        )
