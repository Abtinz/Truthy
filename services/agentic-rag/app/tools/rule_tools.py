from __future__ import annotations

from typing import Any

from langchain_core.tools import tool


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


@tool
def categorize_uploaded_documents(
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
        file_name = str(item.get("file_name") or "").lower()
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


@tool
def collect_rule_findings(
    normalized_file_texts: list[dict[str, Any]],
) -> list[dict[str, str]]:
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
        source_text = str(item.get("text", ""))
        lowered_text = source_text.lower()
        file_name = str(item.get("file_name") or "unnamed-file")

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
                    "excerpt": extract_excerpt.invoke(
                        {"source_text": source_text, "pattern": rule["pattern"]}
                    ),
                }
            )

    return findings


@tool
def extract_excerpt(source_text: str, pattern: str, window_size: int = 180) -> str:
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


@tool
def format_findings_for_evidence(findings: list[dict[str, str]]) -> list[str]:
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


@tool
def evaluate_completeness_rules(
    normalized_file_texts: list[dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate document completeness and form integrity using rule heuristics.

    Args:
        normalized_file_texts: Normalized file payloads prepared for review.

    Returns:
        dict[str, Any]: Rule-evaluation result including statuses, evidence,
        and detected issues.
    """

    categories = categorize_uploaded_documents.invoke(
        {"normalized_file_texts": normalized_file_texts}
    )
    missing_categories = [
        category_name
        for category_name, items in categories.items()
        if category_name != "supplementary_notes" and not items
    ]

    notes_text = combine_category_text.invoke(
        {"category_items": categories["supplementary_notes"]}
    )
    checklist_text = combine_category_text.invoke(
        {"category_items": categories["document_checklist"]}
    )
    imm5257_text = combine_category_text.invoke({"category_items": categories["imm5257"]})
    imm5707_text = combine_category_text.invoke({"category_items": categories["imm5707"]})
    combined_text = "\n\n".join(
        str(item.get("text", "")) for item in normalized_file_texts
    ).lower()

    findings = collect_rule_findings.invoke(
        {"normalized_file_texts": normalized_file_texts}
    )
    detected_issues = [finding["issue"] for finding in findings]

    searchable_text = "\n\n".join(
        [notes_text, checklist_text, imm5257_text, imm5707_text, combined_text]
    )

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
        fallback_issue = (
            "The uploaded materials explicitly indicate that the application is incomplete."
        )
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


@tool
def combine_category_text(category_items: list[dict[str, Any]]) -> str:
    """Concatenate normalized text for all files in one document category.

    Args:
        category_items: Files belonging to one logical document category.

    Returns:
        str: Lowercased combined text for rule evaluation.
    """

    return "\n\n".join(str(item.get("text", "")) for item in category_items).lower()
