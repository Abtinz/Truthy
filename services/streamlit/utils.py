from __future__ import annotations

import base64
from pathlib import Path
from typing import Iterable


def compute_overall_result(response_payload: dict) -> str:
    """Derive a single overall completeness result from stage outcomes.

    Args:
        response_payload: JSON response payload returned by the API review
            endpoint.

    Returns:
        str: Overall result label, either `PASS`, `FAIL`, or `MANUAL REVIEW`.
    """

    stage_outcomes = response_payload.get("stage_outcomes", [])
    if not stage_outcomes:
        return "MANUAL REVIEW"

    statuses = [stage.get("status", "").lower() for stage in stage_outcomes]
    if all(status == "passed" for status in statuses):
        return "PASS"
    if any(status == "failed" for status in statuses):
        return "FAIL"
    return "MANUAL REVIEW"


def build_review_payload(application_name: str, uploaded_files: list) -> dict:
    """Convert uploaded Streamlit files into the API request payload.

    Each uploaded file is encoded into base64 so the FastAPI gateway can pass
    the raw file contents downstream without depending on the browser session.

    Args:
        application_name: Human-entered immigration application name.
        uploaded_files: Streamlit uploaded file objects exposing `name`,
            `type`, and `getvalue()`.

    Returns:
        dict: JSON-serializable review payload for the FastAPI gateway.
    """

    files_payload = []
    for uploaded_file in uploaded_files:
        file_bytes = uploaded_file.getvalue()
        files_payload.append(
            {
                "file_name": uploaded_file.name,
                "content_type": getattr(uploaded_file, "type", None),
                "base64_data": base64.b64encode(file_bytes).decode("ascii"),
            }
        )

    return {
        "application_name": application_name,
        "files": files_payload,
    }


def collect_backend_logs(ui_logs: list[str], log_paths: Iterable[Path]) -> tuple[str, str]:
    """Build the text shown in the Streamlit logs panel.

    Args:
        ui_logs: In-memory UI activity log messages.
        log_paths: Backend log file paths that should be displayed in the UI.

    Returns:
        tuple[str, str]: Two strings containing UI logs and backend log text.
    """

    ui_logs_text = "\n".join(ui_logs) if ui_logs else "No UI log entries yet."
    backend_sections = []

    for log_path in log_paths:
        if log_path.exists():
            backend_sections.append(
                f"[{log_path.name}]\n{log_path.read_text(encoding='utf-8')}"
            )
        else:
            backend_sections.append(f"[{log_path.name}]\nLog file not found.")

    return ui_logs_text, "\n\n".join(backend_sections)


def format_review_response(response_payload: dict) -> str:
    """Render the review response into a readable plain-text panel.

    Args:
        response_payload: JSON response payload returned by the API review
            endpoint.

    Returns:
        str: Human-readable multiline summary for the Streamlit results panel.
    """

    lines = [
        f"Application Name: {response_payload.get('application_name', '')}",
        f"Overall Result: {compute_overall_result(response_payload)}",
        "",
        "Stage Outcomes:",
    ]

    for stage in response_payload.get("stage_outcomes", []):
        lines.extend(
            [
                f"- {stage.get('stage_name', '')}: {stage.get('status', '')}",
                f"  Explanation: {stage.get('explanation', '')}",
            ]
        )

    lines.extend(
        [
            "",
            "Final Report:",
            response_payload.get("final_report_text", ""),
        ]
    )
    return "\n".join(lines).strip()
