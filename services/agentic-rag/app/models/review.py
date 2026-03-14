from __future__ import annotations

from typing import Any

from pydantic import BaseModel
from pydantic import Field


class ReviewFileInput(BaseModel):
    """Incoming file payload accepted by the agentic-RAG review endpoint.

    Args:
        file_name: Optional file name label.
        content_type: Optional MIME type declared by the caller.
        text: Optional direct text content.
        base64_data: Optional base64-encoded file content.
        byte_values: Optional integer byte array representation.

    Returns:
        ReviewFileInput: Validated file payload model.
    """

    file_name: str | None = None
    content_type: str | None = None
    text: str | None = None
    base64_data: str | None = None
    byte_values: list[int] | None = None


class ReviewRequest(BaseModel):
    """Incoming review request accepted by the agentic-RAG service.

    Args:
        application_name: Program name under review.
        files: Submitted application files.

    Returns:
        ReviewRequest: Validated review request model.
    """

    application_name: str = Field(min_length=1)
    files: list[ReviewFileInput] = Field(default_factory=list)


class ReviewResponse(BaseModel):
    """Structured response returned by the review chain.

    Args:
        application_name: Program name under review.
        normalized_file_texts: Normalized extracted file texts.
        retrieved_contexts: Retrieved context chunks returned by the RAG layer.
        stage_outcomes: Ordered stage-by-stage review outcomes.
        final_report_text: Final strict officer-facing report text.

    Returns:
        ReviewResponse: Validated response payload.
    """

    application_name: str
    normalized_file_texts: list[dict[str, Any]]
    retrieved_contexts: list[dict[str, Any]]
    stage_outcomes: list[dict[str, Any]]
    final_report_text: str
