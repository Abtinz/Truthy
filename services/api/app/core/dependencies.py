from __future__ import annotations

from app.clients.agentic_rag_client import AgenticRagClient
from app.core.config import ApiSettings
from app.services.application_review import ApplicationReviewService


def get_application_review_service() -> ApplicationReviewService:
    """Build the API-layer application review service dependency.

    This dependency function centralizes service construction so the FastAPI
    routes can inject a configured review service and tests can override it
    cleanly.

    Args:
        None.

    Returns:
        ApplicationReviewService: Configured review service instance.
    """

    settings = ApiSettings.from_env()
    client = AgenticRagClient(settings)
    return ApplicationReviewService(client)
