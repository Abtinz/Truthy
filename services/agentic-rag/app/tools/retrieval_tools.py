from __future__ import annotations

from typing import Any

from langchain_core.documents import Document
from langchain_core.tools import tool

from app.core.config import AgenticRagSettings
from app.retrieval.pinecone_retriever import PineconeRetrieverClient

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - runtime dependency guard only.
    OpenAI = None  # type: ignore[assignment]


@tool
def retrieve_contexts(
    application_name: str,
    normalized_file_texts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Retrieve relevant vector context for the application package.

    Args:
        application_name: Program name under review.
        normalized_file_texts: Normalized uploaded file records containing
            extracted text.

    Returns:
        list[dict[str, Any]]: Retrieved context chunks from the operational
        guidelines and checklist indexes after score filtering.
    """

    try:
        settings = AgenticRagSettings.from_env()
        if (
            not settings.pinecone_api_key
            or not settings.openai_api_key
            or not settings.pinecone_operational_guidelines_index_name
            or not settings.pinecone_document_checklist_index_name
        ):
            return []

        query_text = build_retrieval_query.invoke(
            {
                "application_name": application_name,
                "normalized_file_texts": normalized_file_texts,
                "max_file_text_characters": settings.max_file_text_characters,
                "max_total_query_characters": settings.max_total_query_characters,
            }
        )
        if not query_text.strip():
            return []

        query_vector = embed_query_text.invoke(
            {
                "query_text": query_text,
                "settings_dict": settings.__dict__,
            }
        )
        retriever = PineconeRetrieverClient(settings)
        operational_matches = retriever.search_operational_guidelines(query_vector)
        checklist_matches = retriever.search_document_checklists(query_vector)

        return [
            *normalize_retrieval_matches.invoke(
                {
                    "index_name": "operational-guidelines-instructions",
                    "matches": [
                        {
                            "record_id": match.record_id,
                            "score": match.score,
                            "metadata": match.metadata,
                        }
                        for match in operational_matches
                    ],
                    "minimum_score": settings.pinecone_min_similarity_score,
                }
            ),
            *normalize_retrieval_matches.invoke(
                {
                    "index_name": "document-checklist-pdf",
                    "matches": [
                        {
                            "record_id": match.record_id,
                            "score": match.score,
                            "metadata": match.metadata,
                        }
                        for match in checklist_matches
                    ],
                    "minimum_score": settings.pinecone_min_similarity_score,
                }
            ),
        ]
    except Exception:
        return []


@tool
def build_retrieval_query(
    application_name: str,
    normalized_file_texts: list[dict[str, Any]],
    max_file_text_characters: int,
    max_total_query_characters: int,
) -> str:
    """Build a bounded retrieval query from app name and extracted file text.

    Args:
        application_name: Program name under review.
        normalized_file_texts: Normalized uploaded file records.
        max_file_text_characters: Maximum contribution per file.
        max_total_query_characters: Maximum total query length.

    Returns:
        str: Combined query text for embedding and retrieval.
    """

    file_documents = build_query_documents.invoke(
        {
            "normalized_file_texts": normalized_file_texts,
            "max_file_text_characters": max_file_text_characters,
        }
    )

    query_parts = [f"Application name: {application_name.strip()}"]
    for index, document in enumerate(file_documents, start=1):
        query_parts.append(f"File {index}:\n{document.page_content}")

    combined_query = "\n\n".join(part for part in query_parts if part.strip())
    if len(combined_query) <= max_total_query_characters:
        return combined_query

    truncated_query = combined_query[: max_total_query_characters - 16].rstrip()
    return f"{truncated_query}\n[TRUNCATED]"


@tool
def build_query_documents(
    normalized_file_texts: list[dict[str, Any]],
    max_file_text_characters: int,
) -> list[Document]:
    """Convert normalized file texts into bounded LangChain documents.

    Args:
        normalized_file_texts: Normalized uploaded file records.
        max_file_text_characters: Maximum text contribution for one file.

    Returns:
        list[Document]: LangChain documents carrying file metadata.
    """

    documents: list[Document] = []
    for index, file_item in enumerate(normalized_file_texts, start=1):
        cleaned_text = str(file_item.get("text", "")).strip()
        if not cleaned_text:
            continue
        bounded_text = cleaned_text[:max_file_text_characters].rstrip()
        if len(cleaned_text) > max_file_text_characters:
            bounded_text = f"{bounded_text}\n[TRUNCATED]"
        documents.append(
            Document(
                page_content=bounded_text,
                metadata={
                    "file_index": index,
                    "file_name": file_item.get("file_name"),
                },
            )
        )
    return documents


@tool
def embed_query_text(query_text: str, settings_dict: dict[str, Any]) -> list[float]:
    """Embed one retrieval query string using OpenAI embeddings.

    Args:
        query_text: Query text built for vector retrieval.
        settings_dict: Serialized runtime settings.

    Returns:
        list[float]: Query embedding vector.
    """

    if OpenAI is None:
        raise RuntimeError(
            "The OpenAI SDK is not installed. Install the `openai` package before "
            "using embed_query_text."
        )

    client = OpenAI(api_key=settings_dict["openai_api_key"])
    request_kwargs: dict[str, Any] = {
        "model": settings_dict["openai_embed_model"],
        "input": [query_text],
        "encoding_format": "float",
    }
    if settings_dict.get("openai_embed_dimensions") is not None:
        request_kwargs["dimensions"] = settings_dict["openai_embed_dimensions"]

    response = client.embeddings.create(**request_kwargs)
    if not response.data:
        raise RuntimeError("Embedding API returned no vector.")
    return response.data[0].embedding


@tool
def normalize_retrieval_matches(
    index_name: str,
    matches: list[dict[str, Any]],
    minimum_score: float,
) -> list[dict[str, Any]]:
    """Normalize and filter raw Pinecone matches for downstream prompts.

    Args:
        index_name: Logical index name used in the output.
        matches: Raw retrieval match payloads.
        minimum_score: Minimum similarity score required for retention.

    Returns:
        list[dict[str, Any]]: Filtered retrieval results for one index.
    """

    normalized_results: list[dict[str, Any]] = []

    for match in matches:
        metadata = dict(match.get("metadata") or {})
        text = str(metadata.get("text", "")).strip()
        score = float(match.get("score", 0.0))
        if score < minimum_score or not text:
            continue
        normalized_results.append(
            {
                "index_name": index_name,
                "record_id": str(match.get("record_id") or ""),
                "score": score,
                "metadata": metadata,
                "text": text,
            }
        )

    return normalized_results
