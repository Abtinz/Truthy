from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app
from app.main import get_application_review_service
from app.schemas.application import ApplicationReviewRequest


class FakeApplicationReviewService:
    """Minimal fake review service used by the API endpoint tests.

    The fake service returns a deterministic payload so the tests can verify
    the request/response contract without depending on the downstream
    agentic-RAG service.

    Args:
        None.

    Returns:
        FakeApplicationReviewService: Test double for the review service.
    """

    def create_review(self, payload: ApplicationReviewRequest) -> dict:
        """Return a deterministic review response for API contract tests.

        Args:
            payload: Validated review request submitted to the API.

        Returns:
            dict: Mocked response payload matching the endpoint schema.
        """

        return {
            "application_name": payload.application_name,
            "normalized_file_texts": [item.model_dump() for item in payload.files],
            "retrieved_contexts": [
                {
                    "index_name": "operational-guidelines-instructions",
                    "record_id": "guideline-1",
                    "score": 0.99,
                    "metadata": {"section_title": "Document requirements"},
                    "text": "All required forms are signed.",
                }
            ],
            "stage_outcomes": [
                {
                    "stage_name": "Document Presence",
                    "status": "passed",
                    "explanation": "All required documents were provided.",
                    "evidence": ["IMM 5257 identified", "Passport identified"],
                    "rendered_prompt": "prompt text",
                }
            ],
            "final_report_text": "Strict completeness report.",
        }


client = TestClient(app)


def override_application_review_service() -> FakeApplicationReviewService:
    """Provide a fake review service for endpoint tests.

    Args:
        None.

    Returns:
        FakeApplicationReviewService: Fake service injected into FastAPI.
    """

    return FakeApplicationReviewService()


app.dependency_overrides[get_application_review_service] = (
    override_application_review_service
)


def test_root_endpoint_returns_service_status() -> None:
    """Verify the API root endpoint returns the expected status payload.

    Args:
        None.

    Returns:
        None: Assertions validate the endpoint response.
    """

    response = client.get("/")

    print("=== ROOT RESPONSE ===")
    print(response.json())

    assert response.status_code == 200
    assert response.json() == {"service": "truthy-api", "status": "ok"}


def test_review_endpoint_returns_review_payload() -> None:
    """Verify the review endpoint returns the structured review response.

    Args:
        None.

    Returns:
        None: Assertions validate the endpoint response.
    """

    response = client.post(
        "/review",
        json={
            "application_name": "visitor visa",
            "files": [
                {
                    "file_name": "passport.txt",
                    "content_type": "text/plain",
                    "text": "Passport for Jane Doe",
                }
            ],
        },
    )

    print("=== REVIEW RESPONSE ===")
    print(response.json())

    assert response.status_code == 200
    assert response.json()["application_name"] == "visitor visa"
    assert response.json()["final_report_text"] == "Strict completeness report."
    assert response.json()["stage_outcomes"][0]["status"] == "passed"


def test_review_endpoint_validates_application_name() -> None:
    """Verify the review endpoint rejects an empty application name.

    Args:
        None.

    Returns:
        None: Assertions validate the request validation behavior.
    """

    response = client.post(
        "/review",
        json={
            "application_name": "",
            "files": [],
        },
    )

    print("=== REVIEW VALIDATION ERROR ===")
    print(response.json())

    assert response.status_code == 422


def test_review_lookup_endpoint_returns_not_implemented() -> None:
    """Verify the review lookup placeholder endpoint returns 501.

    Args:
        None.

    Returns:
        None: Assertions validate the placeholder behavior.
    """

    response = client.get("/review/example-review-id")

    print("=== REVIEW LOOKUP RESPONSE ===")
    print(response.json())

    assert response.status_code == 501


def test_policy_refresh_endpoint_returns_not_implemented() -> None:
    """Verify the policy refresh placeholder endpoint returns 501.

    Args:
        None.

    Returns:
        None: Assertions validate the placeholder behavior.
    """

    response = client.post("/policy/refresh")

    print("=== POLICY REFRESH RESPONSE ===")
    print(response.json())

    assert response.status_code == 501
