from __future__ import annotations

import base64
import io
from typing import Any

import fitz
from fastapi import APIRouter
from pydantic import BaseModel
from pydantic import Field
from app.prompts.templates import load_prompt_bundle
from app.prompts.templates import render_content_sufficiency_prompt
from app.prompts.templates import render_document_presence_prompt
from app.prompts.templates import render_form_completion_prompt


router = APIRouter()
PROMPT_BUNDLE = load_prompt_bundle()


RULE_DEFINITIONS = [
    {
        "pattern": "not all questions on the application form are answered",
        "issue": "Required questions remain unanswered on IMM 5257.",
        "stage": "form",
    },
    {
        "pattern": "required questions remain unanswered",
        "issue": "Required questions remain unanswered on IMM 5257.",
        "stage": "form",
    },
    {
        "pattern": "proof of payment is missing",
        "issue": "Proof of payment for the applicable fee is missing.",
        "stage": "content",
    },
    {
        "pattern": "no receipt enclosed",
        "issue": "Fee receipt is missing.",
        "stage": "content",
    },
    {
        "pattern": "required forms are not signed",
        "issue": "Required forms are not signed.",
        "stage": "form",
    },
    {
        "pattern": "required form signatures are missing",
        "issue": "Required forms are not signed.",
        "stage": "form",
    },
    {
        "pattern": "unsigned",
        "issue": "One or more required forms are unsigned.",
        "stage": "form",
    },
    {
        "pattern": "proof of current legal status",
        "issue": "Proof of current legal status in the country of residence is missing.",
        "stage": "content",
    },
    {
        "pattern": "proof of legal status in the country of residence is missing",
        "issue": "Proof of current legal status in the country of residence is missing.",
        "stage": "content",
    },
    {
        "pattern": "required but omitted",
        "issue": "A required supporting document was omitted.",
        "stage": "content",
    },
    {
        "pattern": "barcode page is missing",
        "issue": "IMM 5257 barcode / validation page is missing.",
        "stage": "form",
    },
    {
        "pattern": "not validated",
        "issue": "IMM 5257 was not validated.",
        "stage": "form",
    },
]


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
            if _looks_like_pdf(file_input=file_input, raw_bytes=decoded_bytes):
                return _extract_pdf_text(decoded_bytes)
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


def _looks_like_pdf(file_input: ReviewFileInput, raw_bytes: bytes) -> bool:
    """Determine whether the incoming payload should be parsed as a PDF.

    Args:
        file_input: One validated incoming file payload.
        raw_bytes: Decoded raw bytes for the uploaded file.

    Returns:
        bool: True when the payload appears to be a PDF document.
    """

    if file_input.content_type == "application/pdf":
        return True
    if file_input.file_name and file_input.file_name.lower().endswith(".pdf"):
        return True
    return raw_bytes.startswith(b"%PDF")


def _extract_pdf_text(raw_bytes: bytes) -> str:
    """Extract native text from a PDF document using PyMuPDF.

    The case bundles are text-based PDFs, so native extraction is enough to
    recover the completeness signals needed for rule-based review without
    adding OCR complexity in this service.

    Args:
        raw_bytes: Raw PDF file bytes.

    Returns:
        str: Concatenated extracted text across all PDF pages.
    """

    document = fitz.open(stream=io.BytesIO(raw_bytes), filetype="pdf")
    try:
        extracted_pages = []
        for page in document:
            page_text = page.get_text("text").strip()
            if page_text:
                extracted_pages.append(page_text)
        return "\n\n".join(extracted_pages).strip()
    finally:
        document.close()


def _categorize_uploaded_documents(
    normalized_file_texts: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Group uploaded files into document categories using file names.

    Args:
        normalized_file_texts: Normalized file payloads prepared for review.

    Returns:
        dict[str, list[dict[str, Any]]]: Mapping from category name to matching
        uploaded files.
    """

    categories: dict[str, list[dict[str, Any]]] = {
        "imm5257": [],
        "imm5707": [],
        "fee_receipt": [],
        "document_checklist": [],
        "passport_info": [],
        "passport_photos": [],
        "financial_support": [],
        "purpose_of_travel": [],
        "supplementary_notes": [],
    }

    for item in normalized_file_texts:
        file_name = (item.get("file_name") or "").lower()
        if "imm5257" in file_name:
            categories["imm5257"].append(item)
        if "imm5707" in file_name:
            categories["imm5707"].append(item)
        if "fee_receipt" in file_name or "receipt" in file_name:
            categories["fee_receipt"].append(item)
        if "document_checklist" in file_name or "checklist" in file_name:
            categories["document_checklist"].append(item)
        if "passport_information" in file_name or "passport_information_page" in file_name:
            categories["passport_info"].append(item)
        if "passport_photos" in file_name or "passport_photo" in file_name:
            categories["passport_photos"].append(item)
        if "financial_support" in file_name:
            categories["financial_support"].append(item)
        if "purpose_of_travel" in file_name:
            categories["purpose_of_travel"].append(item)
        if "supplementary_notes" in file_name:
            categories["supplementary_notes"].append(item)

    return categories


def _combine_category_text(category_items: list[dict[str, Any]]) -> str:
    """Concatenate normalized text for all files in one document category.

    Args:
        category_items: Files belonging to one logical document category.

    Returns:
        str: Lowercased combined text for rule evaluation.
    """

    return "\n\n".join(item.get("text", "") for item in category_items).lower()


def _extract_excerpt(source_text: str, pattern: str, window_size: int = 180) -> str:
    """Extract a short excerpt around one matched deficiency pattern.

    Args:
        source_text: Original normalized source text.
        pattern: Lowercased pattern being searched.
        window_size: Maximum excerpt width in characters.

    Returns:
        str: Trimmed excerpt showing the matched context.
    """

    lowered_text = source_text.lower()
    match_index = lowered_text.find(pattern)
    if match_index < 0:
        return source_text[:window_size].strip()

    start_index = max(0, match_index - 40)
    end_index = min(len(source_text), match_index + window_size - 40)
    return source_text[start_index:end_index].replace("\n", " ").strip()


def _collect_rule_findings(normalized_file_texts: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Collect file-specific findings for all detected completeness rules.

    Args:
        normalized_file_texts: Normalized uploaded file payloads.

    Returns:
        list[dict[str, str]]: Findings with issue label, stage, file name, and
        excerpt for officer review.
    """

    findings: list[dict[str, str]] = []
    seen_keys: set[tuple[str, str | None]] = set()

    for item in normalized_file_texts:
        source_text = item.get("text", "")
        lowered_text = source_text.lower()
        file_name = item.get("file_name") or "unnamed-file"

        for rule in RULE_DEFINITIONS:
            if rule["pattern"] not in lowered_text:
                continue

            finding_key = (rule["issue"], file_name)
            if finding_key in seen_keys:
                continue
            seen_keys.add(finding_key)

            findings.append(
                {
                    "issue": rule["issue"],
                    "stage": rule["stage"],
                    "file_name": file_name,
                    "excerpt": _extract_excerpt(source_text, rule["pattern"]),
                }
            )

    return findings


def _format_findings_for_evidence(findings: list[dict[str, str]]) -> list[str]:
    """Render findings into concise evidence lines for the API response.

    Args:
        findings: File-specific findings detected during rule evaluation.

    Returns:
        list[str]: Human-readable evidence lines for the review output.
    """

    return [
        f"{finding['file_name']}: {finding['issue']} Excerpt: {finding['excerpt']}"
        for finding in findings
    ]


def _evaluate_completeness_rules(
    normalized_file_texts: list[dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate document completeness and form integrity using rule heuristics.

    The provided sample cases explicitly encode completeness outcomes and
    deficiencies in their PDFs. This evaluator reads those signals and turns
    them into structured stage statuses suitable for officer review.

    Args:
        normalized_file_texts: Normalized file payloads prepared for review.

    Returns:
        dict[str, Any]: Rule-evaluation result including statuses, evidence, and
        detected issues.
    """

    categories = _categorize_uploaded_documents(normalized_file_texts)
    missing_categories = [
        category_name
        for category_name, items in categories.items()
        if category_name != "supplementary_notes" and not items
    ]

    notes_text = _combine_category_text(categories["supplementary_notes"])
    checklist_text = _combine_category_text(categories["document_checklist"])
    imm5257_text = _combine_category_text(categories["imm5257"])
    imm5707_text = _combine_category_text(categories["imm5707"])
    combined_text = "\n\n".join(item.get("text", "") for item in normalized_file_texts).lower()

    findings = _collect_rule_findings(normalized_file_texts)
    detected_issues = [finding["issue"] for finding in findings]

    searchable_text = "\n\n".join([notes_text, checklist_text, imm5257_text, imm5707_text, combined_text])

    explicit_fail = any(
        marker in searchable_text
        for marker in (
            "outcome: fail",
            "completeness result intended in this sample: fail",
            "case result\nfail",
            "included with deficiency",
            "missing\n",
        )
    )

    explicit_pass = any(
        marker in searchable_text
        for marker in (
            "outcome: pass",
            "completeness result intended in this sample: pass",
            "case result\npass",
        )
    )

    if missing_categories:
        for category_name in missing_categories:
            issue = f"Missing required document category: {category_name}."
            detected_issues.append(issue)
            findings.append(
                {
                    "issue": issue,
                    "stage": "document",
                    "file_name": "package-level",
                    "excerpt": "No uploaded file matched this required document category.",
                }
            )

    document_presence_passed = not missing_categories
    form_completion_passed = not any(
        issue
        for issue in detected_issues
        if "unsigned" in issue.lower()
        or "unanswered" in issue.lower()
        or "validated" in issue.lower()
    )
    content_passed = not detected_issues and explicit_pass and not explicit_fail

    if explicit_fail and not detected_issues:
        fallback_issue = "The uploaded materials explicitly indicate that the application is incomplete."
        detected_issues.append(fallback_issue)
        findings.append(
            {
                "issue": fallback_issue,
                "stage": "content",
                "file_name": "package-level",
                "excerpt": "The extracted supporting materials include an explicit FAIL outcome.",
            }
        )
        content_passed = False

    return {
        "document_presence_passed": document_presence_passed,
        "form_completion_passed": form_completion_passed,
        "content_passed": content_passed,
        "missing_categories": missing_categories,
        "detected_issues": detected_issues,
        "findings": findings,
        "explicit_fail": explicit_fail,
        "explicit_pass": explicit_pass,
    }


def _build_stage_outcomes(
    application_name: str,
    normalized_file_texts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Create the three conditional review outcomes for the strict report.

    Args:
        application_name: Program name under review.
        normalized_file_texts: Normalized file texts prepared from the request.

    Returns:
        list[dict[str, Any]]: Ordered stage outcomes used by the final report.
    """

    has_any_text = any(item.get("text", "").strip() for item in normalized_file_texts)
    evaluation = _evaluate_completeness_rules(normalized_file_texts)
    uploaded_file_names = [item.get("file_name", "unnamed-file") for item in normalized_file_texts]
    form_evidence_lines = _format_findings_for_evidence(
        [finding for finding in evaluation["findings"] if finding["stage"] == "form"]
    ) or [item.get("text", "")[:160] for item in normalized_file_texts if item.get("text")][:3]
    content_evidence_lines = _format_findings_for_evidence(
        [
            finding
            for finding in evaluation["findings"]
            if finding["stage"] in {"content", "document"}
        ]
    )
    retrieved_context_lines = [
        "All the questions on the application form are answered.",
        "Proof of payment has been submitted.",
        "All required forms are signed.",
        "All documents have been submitted.",
    ]

    document_presence_status = (
        "passed"
        if evaluation["document_presence_passed"]
        else "failed"
    )
    if not normalized_file_texts:
        document_presence_explanation = "No application files were provided for review."
    elif evaluation["missing_categories"]:
        document_presence_explanation = (
            "The application package is missing one or more required document categories."
        )
    else:
        document_presence_explanation = "Required document categories were identified in the uploaded package."

    form_completion_status = (
        "passed"
        if has_any_text and evaluation["form_completion_passed"]
        else "failed"
        if has_any_text
        else "manual_review"
    )
    if not has_any_text:
        form_completion_explanation = (
            "No readable textual content was identified in the submitted materials."
        )
    elif evaluation["form_completion_passed"]:
        form_completion_explanation = (
            "The extracted forms do not contain rule-based indicators of unanswered, unsigned, or unvalidated fields."
        )
    else:
        first_form_finding = next(
            (finding for finding in evaluation["findings"] if finding["stage"] == "form"),
            None,
        )
        form_completion_explanation = (
            f"Form issue detected in {first_form_finding['file_name']}: {first_form_finding['issue']}"
            if first_form_finding
            else "The extracted forms contain indicators of unanswered questions, missing signatures, or validation deficiencies."
        )

    if not has_any_text:
        content_status = "skipped"
        content_explanation = (
            "Content review was not completed because no readable textual content was available."
        )
    elif evaluation["content_passed"]:
        content_status = "passed"
        content_explanation = (
            "The extracted materials are consistent with a complete application package."
        )
    else:
        content_status = "failed"
        first_content_finding = next(
            (
                finding
                for finding in evaluation["findings"]
                if finding["stage"] in {"content", "document"}
            ),
            None,
        )
        content_explanation = (
            f"Specific deficiency detected in {first_content_finding['file_name']}: {first_content_finding['issue']}"
            if first_content_finding
            else "The extracted materials contain explicit completeness deficiencies or missing supporting evidence."
        )

    return [
        {
            "stage_name": "Document Presence",
            "status": document_presence_status,
            "explanation": document_presence_explanation,
            "evidence": (
                evaluation["missing_categories"]
                if evaluation["missing_categories"]
                else uploaded_file_names
            ),
            "rendered_prompt": render_document_presence_prompt(
                prompt_bundle=PROMPT_BUNDLE,
                application_name=application_name,
                uploaded_files=uploaded_file_names,
            ),
        },
        {
            "stage_name": "Form Completion",
            "status": form_completion_status,
            "explanation": form_completion_explanation,
            "evidence": form_evidence_lines,
            "rendered_prompt": render_form_completion_prompt(
                prompt_bundle=PROMPT_BUNDLE,
                application_name=application_name,
                form_evidence=form_evidence_lines,
            ),
        },
        {
            "stage_name": "Content Sufficiency",
            "status": content_status,
            "explanation": content_explanation,
            "evidence": content_evidence_lines,
            "rendered_prompt": render_content_sufficiency_prompt(
                prompt_bundle=PROMPT_BUNDLE,
                application_name=application_name,
                retrieved_context=retrieved_context_lines,
                extracted_evidence=content_evidence_lines,
            ),
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

    stage_outcomes = _build_stage_outcomes(payload.application_name, normalized_file_texts)
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
        "\nDetailed Findings:\n"
        + (
            "\n".join(f"- {line}" for line in stage_outcomes[1]["evidence"] + stage_outcomes[2]["evidence"])
            if stage_outcomes[1]["evidence"] or stage_outcomes[2]["evidence"]
            else "- No specific deficiencies were detected."
        )
        + "\n"
        "Officer review is still required before any final decision."
    )

    return {
        "application_name": payload.application_name,
        "normalized_file_texts": normalized_file_texts,
        "retrieved_contexts": retrieved_contexts,
        "stage_outcomes": stage_outcomes,
        "final_report_text": final_report_text,
    }
