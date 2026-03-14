from __future__ import annotations

from typing import Any

import httpx

from app.core.config import ApiSettings


class AgenticRagClient:
    """HTTP client used by the gateway service to call agentic-RAG review APIs.

    Args:
        settings: Runtime settings containing the agentic-RAG base URL.

    Returns:
        AgenticRagClient: Configured HTTP client wrapper.
    """

    def __init__(self, settings: ApiSettings) -> None:
        """Initialize the agentic-RAG client.

        Args:
            settings: Runtime settings object.

        Returns:
            None.
        """
        self.settings = settings

    def create_review(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Submit a review request to the agentic-RAG service.

        Args:
            payload: Serialized review request body.

        Returns:
            dict[str, Any]: Review response payload from the downstream service.
        """
        response = httpx.post(
            f"{self.settings.agentic_rag_base_url}/review",
            json=payload,
            timeout=60.0,
        )
        response.raise_for_status()
        return dict(response.json())
