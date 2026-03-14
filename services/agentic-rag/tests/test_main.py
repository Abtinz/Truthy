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

    print(response.json())

    assert response.status_code == 200
    assert response.json()["application_name"] == "visitor visa"
    assert response.json()["stage_outcomes"][0]["stage_name"] == "Document Presence"
    assert "Officer review is still required" in response.json()["final_report_text"]


def test_review_endpoint_reports_specific_file_errors() -> None:
    """Verify the review endpoint names the problematic file and excerpt.

    Args:
        None.

    Returns:
        None: Assertions validate the specific evidence payload.
    """

    response = client.post(
        "/review",
        json={
            "application_name": "visitor visa",
            "files": [
                {
                    "file_name": "fee_receipt.pdf",
                    "content_type": "text/plain",
                    "text": "Case 7 Fee Receipt. No receipt enclosed. Proof of payment is missing.",
                },
                {
                    "file_name": "supplementary_notes.pdf",
                    "content_type": "text/plain",
                    "text": "Outcome: FAIL. The application does not pass completeness because proof of payment is missing.",
                },
                {
                    "file_name": "document_checklist.pdf",
                    "content_type": "text/plain",
                    "text": "Fee receipt. No receipt enclosed. Missing.",
                },
                {
                    "file_name": "imm5257.pdf",
                    "content_type": "text/plain",
                    "text": "Completed and signed.",
                },
                {
                    "file_name": "imm5707.pdf",
                    "content_type": "text/plain",
                    "text": "Completed and signed.",
                },
                {
                    "file_name": "passport_information_page.pdf",
                    "content_type": "text/plain",
                    "text": "Passport info page enclosed.",
                },
                {
                    "file_name": "passport_photos.pdf",
                    "content_type": "text/plain",
                    "text": "Passport photos enclosed.",
                },
                {
                    "file_name": "proof_of_financial_support.pdf",
                    "content_type": "text/plain",
                    "text": "Funds evidence enclosed.",
                },
                {
                    "file_name": "purpose_of_travel.pdf",
                    "content_type": "text/plain",
                    "text": "Visit explanation enclosed.",
                },
            ],
        },
    )

    print(response.json())

    assert response.status_code == 200
    assert response.json()["stage_outcomes"][2]["status"] == "failed"
    assert any(
        "fee_receipt.pdf" in line and "Proof of payment for the applicable fee is missing." in line
        for line in response.json()["stage_outcomes"][2]["evidence"]
    )
    assert "fee_receipt.pdf" in response.json()["stage_outcomes"][2]["explanation"]
