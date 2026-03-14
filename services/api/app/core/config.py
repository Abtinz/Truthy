from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ApiSettings:
    """Environment-backed runtime settings for the FastAPI gateway service.

    Args:
        agentic_rag_base_url: Base URL used to call the agentic-RAG service.

    Returns:
        ApiSettings: Immutable runtime configuration.
    """

    agentic_rag_base_url: str

    @classmethod
    def from_env(cls) -> "ApiSettings":
        """Load API settings from environment variables.

        Args:
            None.

        Returns:
            ApiSettings: Parsed settings object.
        """
        return cls(
            agentic_rag_base_url=(
                os.getenv("AGENTIC_RAG_BASE_URL", "http://agentic-rag:8002").strip()
                or "http://agentic-rag:8002"
            )
        )
