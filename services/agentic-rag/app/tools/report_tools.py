from __future__ import annotations

from typing import Any

from langchain_core.tools import tool

from app.prompts.templates import load_prompt_bundle
from app.prompts.templates import render_content_sufficiency_prompt
from app.prompts.templates import render_document_presence_prompt
from app.prompts.templates import render_form_completion_prompt
from app.tools.rule_tools import format_findings_for_evidence


PROMPT_BUNDLE = load_prompt_bundle()


@tool
def build_stage_outcomes(
    application_name: str,
    normalized_file_texts: list[dict[str, Any]],
    evaluation: dict[str, Any],
    retrieved_contexts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Create the conditional review outcomes for the strict report.

    Args:
        application_name: Program name under review.
        normalized_file_texts: Normalized uploaded file records.
        evaluation: Rule-evaluation result produced by the rule tool.
        retrieved_contexts: Retrieved context chunks from the RAG layer.

    Returns:
        list[dict[str, Any]]: Ordered stage outcomes used by the final report.
    """

    has_any_text = any(str(item.get("text", "")).strip() for item in normalized_file_texts)
    uploaded_file_names = [
        str(item.get("file_name") or "unnamed-file") for item in normalized_file_texts
    ]
    form_evidence_lines = format_findings_for_evidence.invoke(
        {
            "findings": [
                finding
                for finding in evaluation["findings"]
                if finding["stage"] == "form"
            ]
        }
    ) or [
        str(item.get("text", ""))[:160]
        for item in normalized_file_texts
        if str(item.get("text", "")).strip()
    ][:3]
    content_evidence_lines = format_findings_for_evidence.invoke(
        {
            "findings": [
                finding
                for finding in evaluation["findings"]
                if finding["stage"] in {"content", "document"}
            ]
        }
    )
    retrieved_context_lines = [
        context["text"] for context in retrieved_contexts if str(context.get("text", "")).strip()
    ] or [
        "All the questions on the application form are answered.",
        "Proof of payment has been submitted.",
        "All required forms are signed.",
        "All documents have been submitted.",
    ]

    document_presence_status = (
        "passed" if evaluation["document_presence_passed"] else "failed"
    )
    if not normalized_file_texts:
        document_presence_explanation = "No application files were provided for review."
    elif evaluation["missing_categories"]:
        document_presence_explanation = (
            "The application package is missing one or more required document categories."
        )
    else:
        document_presence_explanation = (
            "Required document categories were identified in the uploaded package."
        )

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


@tool
def synthesize_final_report(
    application_name: str,
    stage_outcomes: list[dict[str, Any]],
) -> str:
    """Create the final strict officer-facing completeness report.

    Args:
        application_name: Program name under review.
        stage_outcomes: Ordered stage-by-stage review outcomes.

    Returns:
        str: Final strict report text for officers.
    """

    return (
        f"Application Name: {application_name}\n\n"
        "This is a strict completeness review report.\n"
        f"Document Presence: {stage_outcomes[0]['status']}\n"
        f"Form Completion: {stage_outcomes[1]['status']}\n"
        f"Content Sufficiency: {stage_outcomes[2]['status']}\n"
        "\nDetailed Findings:\n"
        + (
            "\n".join(
                f"- {line}"
                for line in stage_outcomes[1]["evidence"] + stage_outcomes[2]["evidence"]
            )
            if stage_outcomes[1]["evidence"] or stage_outcomes[2]["evidence"]
            else "- No specific deficiencies were detected."
        )
        + "\n"
        "Officer review is still required before any final decision."
    )
