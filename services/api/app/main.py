from __future__ import annotations

from fastapi import Depends
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.dependencies import get_application_review_service
from app.schemas.application import ApplicationReviewRequest
from app.schemas.report import ApplicationReviewResponse
from app.services.application_review import ApplicationReviewService

app = FastAPI(title="Truthy API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/review", response_model=ApplicationReviewResponse)
def create_review(
    payload: ApplicationReviewRequest,
    review_service: ApplicationReviewService = Depends(get_application_review_service),
) -> dict[str, object]:
    """Submit one application package for completeness review.

    Args:
        payload: Validated incoming review request.
        review_service: Injected review orchestration service.

    Returns:
        dict[str, object]: Structured completeness review response.
    """
    return review_service.create_review(payload)
