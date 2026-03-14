from __future__ import annotations

from pydantic import BaseModel, Field


class ApplicationFileInput(BaseModel):
    """Incoming file payload accepted by the FastAPI review endpoint.

    Args:
        file_name: Optional file name label.
        content_type: Optional declared MIME type.
        text: Optional direct plain-text payload.
        base64_data: Optional base64-encoded file content.
        byte_values: Optional integer byte array representation.

    Returns:
        ApplicationFileInput: Validated API file payload.
    """

    file_name: str | None = None
    content_type: str | None = None
    text: str | None = None
    base64_data: str | None = None
    byte_values: list[int] | None = None


class ApplicationReviewRequest(BaseModel):
    """Incoming review request accepted by the FastAPI gateway.

    Args:
        application_name: Program name under review.
        files: Raw file payloads forwarded to the agentic-RAG service.

    Returns:
        ApplicationReviewRequest: Validated review request.
    """

    application_name: str = Field(min_length=1)
    files: list[ApplicationFileInput] = Field(default_factory=list)
