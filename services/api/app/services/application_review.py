from __future__ import annotations

from app.clients.agentic_rag_client import AgenticRagClient
from app.schemas.application import ApplicationReviewRequest


class ApplicationReviewService:
    """Gateway orchestration service for application completeness reviews.

    Args:
        agentic_rag_client: Client used to call the downstream agentic-RAG
            service.

    Returns:
        ApplicationReviewService: Configured review orchestration service.
    """

    def __init__(self, agentic_rag_client: AgenticRagClient) -> None:
        """Initialize the application review service.

        Args:
            agentic_rag_client: Downstream agentic-RAG client.

        Returns:
            None.
        """
        self.agentic_rag_client = agentic_rag_client

    def create_review(self, payload: ApplicationReviewRequest) -> dict[str, object]:
        """Submit the review request to the downstream agentic-RAG service.

        Args:
            payload: Validated review request payload.

        Returns:
            dict[str, object]: Review response returned by the downstream
            service.
        """
        return self.agentic_rag_client.create_review(payload.model_dump())
