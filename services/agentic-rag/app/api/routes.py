from __future__ import annotations

import base64
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel
from pydantic import Field


router = APIRouter()


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


def _decode_file_to_text(file_input: ReviewFileInput) -> str:
    """Convert one incoming file payload into best-effort plain text.

    The current implementation favors resilience over deep parsing so the
    service can keep running even after placeholder regressions. Direct text is
    preferred, then base64-decoded bytes, then integer byte arrays.

    Args:
        file_input: One validated incoming file payload.

    Returns:
        str: Best-effort normalized text extracted from the incoming payload.
    """

    if file_input.text:
        return file_input.text.strip()

    if file_input.base64_data:
        try:
            decoded_bytes = base64.b64decode(file_input.base64_data)
            return decoded_bytes.decode("utf-8", errors="ignore").strip()
        except Exception:
            return ""

    if file_input.byte_values:
        try:
            raw_bytes = bytes(file_input.byte_values)
            return raw_bytes.decode("utf-8", errors="ignore").strip()
        except Exception:
            return ""

    return ""


def _build_stage_outcomes(normalized_file_texts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Create the three conditional review outcomes for the strict report.

    Args:
        normalized_file_texts: Normalized file texts prepared from the request.

    Returns:
        list[dict[str, Any]]: Ordered stage outcomes used by the final report.
    """

    has_any_text = any(item.get("text", "").strip() for item in normalized_file_texts)

    document_presence_status = "passed" if normalized_file_texts else "failed"
    document_presence_explanation = (
        "Application files were provided for review."
        if normalized_file_texts
        else "No application files were provided for review."
    )

    form_completion_status = "passed" if has_any_text else "manual_review"
    form_completion_explanation = (
        "Readable textual content was identified in the submitted materials."
        if has_any_text
        else "No readable textual content was identified in the submitted materials."
    )

    content_status = "manual_review" if has_any_text else "skipped"
    content_explanation = (
        "Content review requires officer confirmation against the retrieved guidance."
        if has_any_text
        else "Content review was not completed because no readable textual content was available."
    )

    return [
        {
            "stage_name": "Document Presence",
            "status": document_presence_status,
            "explanation": document_presence_explanation,
            "evidence": [item.get("file_name", "unnamed-file") for item in normalized_file_texts],
            "rendered_prompt": "Check whether the user has provided all application files.",
        },
        {
            "stage_name": "Form Completion",
            "status": form_completion_status,
            "explanation": form_completion_explanation,
            "evidence": [item.get("text", "")[:160] for item in normalized_file_texts if item.get("text")],
            "rendered_prompt": "Check whether the submitted forms appear complete and readable.",
        },
        {
            "stage_name": "Content Sufficiency",
            "status": content_status,
            "explanation": content_explanation,
            "evidence": [],
            "rendered_prompt": "Check whether the submitted content is consistent with the governing completeness guidance.",
        },
    ]


@router.post("/review")
def create_review(payload: ReviewRequest) -> dict[str, Any]:
    """Generate a strict, structured review payload for the API gateway.

    Args:
        payload: Validated incoming review request.

    Returns:
        dict[str, Any]: Structured review response consumed by the FastAPI
        gateway and Streamlit frontend.
    """

    normalized_file_texts = []
    for file_input in payload.files:
        normalized_file_texts.append(
            {
                "file_name": file_input.file_name,
                "content_type": file_input.content_type,
                "text": _decode_file_to_text(file_input),
            }
        )

    stage_outcomes = _build_stage_outcomes(normalized_file_texts)
    retrieved_contexts = [
        {
            "index_name": "operational-guidelines-instructions",
            "record_id": "guideline-placeholder",
            "score": 1.0,
            "metadata": {"source": "local-fallback"},
            "text": "All the questions on the application form are answered. Proof of payment has been submitted. All required forms are signed. All documents have been submitted.",
        },
        {
            "index_name": "document-checklist-pdf",
            "record_id": "checklist-placeholder",
            "score": 1.0,
            "metadata": {"source": "local-fallback"},
            "text": "Document checklist reference loaded for officer review.",
        },
    ]

    final_report_text = (
        f"Application Name: {payload.application_name}\n\n"
        "This is a strict completeness review report.\n"
        f"Document Presence: {stage_outcomes[0]['status']}\n"
        f"Form Completion: {stage_outcomes[1]['status']}\n"
        f"Content Sufficiency: {stage_outcomes[2]['status']}\n"
        "Officer review is still required before any final decision."
    )

    return {
        "application_name": payload.application_name,
        "normalized_file_texts": normalized_file_texts,
        "retrieved_contexts": retrieved_contexts,
        "stage_outcomes": stage_outcomes,
        "final_report_text": final_report_text,
    }
