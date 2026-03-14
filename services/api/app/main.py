from __future__ import annotations

from fastapi import Depends
from fastapi import FastAPI
from fastapi import HTTPException
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


@app.get("/")
def read_root() -> dict[str, str]:
    """Return the root status payload for the FastAPI service.

    Args:
        None.

    Returns:
        dict[str, str]: Basic service status payload.
    """
    return {
        "service": "truthy-api",
        "status": "ok",
    }


@app.get("/health")
def read_health() -> dict[str, str]:
    """Return the FastAPI service health status.

    Args:
        None.

    Returns:
        dict[str, str]: Health-check payload.
    """
    return {
        "status": "ok",
    }


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


@app.get("/review/{review_id}")
def get_review(review_id: str) -> dict[str, str]:
    """Return the placeholder status for a review lookup request.

    Args:
        review_id: Review identifier from the path.

    Returns:
        dict[str, str]: Placeholder review-status payload.
    """
    raise HTTPException(
        status_code=501,
        detail=f"Review lookup is not implemented for review_id={review_id}.",
    )


@app.post("/policy/refresh")
def refresh_policy() -> dict[str, str]:
    """Return the placeholder policy-refresh trigger response.

    Args:
        None.

    Returns:
        dict[str, str]: Placeholder refresh acknowledgement.
    """
    raise HTTPException(
        status_code=501,
        detail="Policy refresh is not implemented.",
    )
