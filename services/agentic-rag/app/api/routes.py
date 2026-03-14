from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from app.chains.review_chain import LangChainReviewChain
from app.models.review import ReviewRequest
from app.models.review import ReviewResponse


router = APIRouter()
REVIEW_CHAIN = LangChainReviewChain()


@router.post("/review", response_model=ReviewResponse)
def create_review(payload: ReviewRequest) -> dict[str, Any]:
    """Generate a strict structured review payload through the LangChain flow.

    Args:
        payload: Validated incoming review request.

    Returns:
        dict[str, Any]: Structured review response consumed by the FastAPI
        gateway and frontend clients.
    """

    return REVIEW_CHAIN.review(
        payload.application_name,
        [file_input.model_dump() for file_input in payload.files],
    )
