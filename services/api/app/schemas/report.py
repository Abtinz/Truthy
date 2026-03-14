from __future__ import annotations

from pydantic import BaseModel, Field


class ReviewStageOutcomeResponse(BaseModel):
    """Officer-facing response model for one review stage outcome.

    Args:
        stage_name: Human-readable stage label.
        status: Stage result status.
        explanation: Exact stage explanation.
        evidence: Supporting evidence lines.
        rendered_prompt: Rendered prompt text used for the stage.

    Returns:
        ReviewStageOutcomeResponse: Validated stage response model.
    """

    stage_name: str
    status: str
    explanation: str
    evidence: list[str] = Field(default_factory=list)
    rendered_prompt: str


class ApplicationReviewResponse(BaseModel):
    """Officer-facing response model for the application review endpoint.

    Args:
        application_name: Program name under review.
        normalized_file_texts: Normalized input texts used by the chain.
        retrieved_contexts: Retrieved knowledge contexts.
        stage_outcomes: Conditional stage outcomes.
        final_report_text: Final strict report text.

    Returns:
        ApplicationReviewResponse: Validated endpoint response model.
    """

    application_name: str
    normalized_file_texts: list[dict] = Field(default_factory=list)
    retrieved_contexts: list[dict] = Field(default_factory=list)
    stage_outcomes: list[ReviewStageOutcomeResponse] = Field(default_factory=list)
    final_report_text: str
