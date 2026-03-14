from __future__ import annotations

from app.chains.review_chain import LangChainReviewChain


def test_langchain_review_chain_returns_structured_result() -> None:
    """Verify the LangChain review chain returns the expected payload shape.

    Args:
        None.

    Returns:
        None: Assertions validate the runnable chain output.
    """

    chain = LangChainReviewChain()
    result = chain.review(
        "visitor visa",
        [
            {
                "file_name": "imm5257.pdf",
                "content_type": "text/plain",
                "text": "Completed and signed form.",
            },
            {
                "file_name": "fee_receipt.pdf",
                "content_type": "text/plain",
                "text": "Proof of payment enclosed.",
            },
            {
                "file_name": "document_checklist.pdf",
                "content_type": "text/plain",
                "text": "Checklist included.",
            },
            {
                "file_name": "passport_information_page.pdf",
                "content_type": "text/plain",
                "text": "Passport information page enclosed.",
            },
            {
                "file_name": "passport_photos.pdf",
                "content_type": "text/plain",
                "text": "Passport photos enclosed.",
            },
            {
                "file_name": "proof_of_financial_support.pdf",
                "content_type": "text/plain",
                "text": "Financial support evidence enclosed.",
            },
            {
                "file_name": "purpose_of_travel.pdf",
                "content_type": "text/plain",
                "text": "Travel purpose enclosed.",
            },
            {
                "file_name": "imm5707.pdf",
                "content_type": "text/plain",
                "text": "Family information enclosed.",
            },
        ],
    )

    print("=== LANGCHAIN REVIEW CHAIN RESULT ===")
    print(result)

    assert result["application_name"] == "visitor visa"
    assert len(result["normalized_file_texts"]) == 8
    assert len(result["stage_outcomes"]) == 3
    assert "Officer review is still required" in result["final_report_text"]


def test_langchain_review_chain_reports_specific_deficiency() -> None:
    """Verify the LangChain review chain preserves file-specific rule findings.

    Args:
        None.

    Returns:
        None: Assertions validate file-specific deficiency reporting.
    """

    chain = LangChainReviewChain()
    result = chain.review(
        "visitor visa",
        [
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
    )

    print("=== LANGCHAIN REVIEW CHAIN DEFICIENCY RESULT ===")
    print(result)

    assert result["stage_outcomes"][2]["status"] == "failed"
    assert "fee_receipt.pdf" in result["stage_outcomes"][2]["explanation"]
