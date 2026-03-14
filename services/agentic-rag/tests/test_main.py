from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_root_endpoint_returns_service_status() -> None:
    """Verify the agentic-RAG service root endpoint is available.

    Args:
        None.

    Returns:
        None: Assertions validate the root response payload.
    """

    response = client.get("/")

    print("=== AGENTIC RAG ROOT ===")
    print(response.json())

    assert response.status_code == 200
    assert response.json() == {"service": "truthy-agentic-rag", "status": "ok"}


def test_review_endpoint_returns_structured_response() -> None:
    """Verify the review endpoint returns the expected structured payload.

    Args:
        None.

    Returns:
        None: Assertions validate the review response payload.
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

    print("=== AGENTIC RAG REVIEW ===")
    print(response.json())

    assert response.status_code == 200
    assert response.json()["application_name"] == "visitor visa"
    assert response.json()["stage_outcomes"][0]["stage_name"] == "Document Presence"
    assert "Officer review is still required" in response.json()["final_report_text"]
