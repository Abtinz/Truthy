from __future__ import annotations

import base64
from pathlib import Path

from utils import build_review_payload
from utils import collect_backend_logs
from utils import compute_overall_result
from utils import format_review_response


class FakeUploadedFile:
    """Simple Streamlit upload test double.

    Args:
        name: File name reported by the upload widget.
        content_type: MIME type associated with the uploaded file.
        file_bytes: Raw file content returned by `getvalue()`.

    Returns:
        FakeUploadedFile: Test helper object compatible with the payload
        builder.
    """

    def __init__(self, name: str, content_type: str, file_bytes: bytes) -> None:
        """Initialize the upload-file test double.

        Args:
            name: File name reported by the upload widget.
            content_type: MIME type associated with the uploaded file.
            file_bytes: Raw file content returned by `getvalue()`.

        Returns:
            None.
        """

        self.name = name
        self.type = content_type
        self._file_bytes = file_bytes

    def getvalue(self) -> bytes:
        """Return the raw uploaded file bytes.

        Args:
            None.

        Returns:
            bytes: Raw uploaded file bytes.
        """

        return self._file_bytes


def test_build_review_payload_encodes_files() -> None:
    """Verify uploaded files are serialized into the expected API payload.

    Args:
        None.

    Returns:
        None: Assertions validate the generated request payload.
    """

    payload = build_review_payload(
        "visitor visa",
        [
            FakeUploadedFile("passport.txt", "text/plain", b"passport content"),
            FakeUploadedFile("form.pdf", "application/pdf", b"%PDF-1.4 fake"),
        ],
    )

    print("=== STREAMLIT REVIEW PAYLOAD ===")
    print(payload)

    assert payload["application_name"] == "visitor visa"
    assert len(payload["files"]) == 2
    assert payload["files"][0]["base64_data"] == base64.b64encode(
        b"passport content"
    ).decode("ascii")


def test_collect_backend_logs_reads_existing_files(tmp_path: Path) -> None:
    """Verify the logs panel reads backend log files and formats missing ones.

    Args:
        tmp_path: Pytest temporary path fixture.

    Returns:
        None: Assertions validate the collected log text.
    """

    existing_log = tmp_path / "indexer_run.log"
    existing_log.write_text("indexing complete", encoding="utf-8")
    missing_log = tmp_path / "missing.log"

    ui_logs, backend_logs = collect_backend_logs(
        ui_logs=["submitted request", "received response"],
        log_paths=[existing_log, missing_log],
    )

    print("=== STREAMLIT UI LOGS ===")
    print(ui_logs)
    print("=== STREAMLIT BACKEND LOGS ===")
    print(backend_logs)

    assert "submitted request" in ui_logs
    assert "[indexer_run.log]" in backend_logs
    assert "indexing complete" in backend_logs
    assert "[missing.log]" in backend_logs
    assert "Log file not found." in backend_logs


def test_format_review_response_renders_stage_and_report() -> None:
    """Verify the formatted results panel renders key response sections.

    Args:
        None.

    Returns:
        None: Assertions validate the formatted response text.
    """

    formatted_text = format_review_response(
        {
            "application_name": "visitor visa",
            "stage_outcomes": [
                {
                    "stage_name": "Document Presence",
                    "status": "passed",
                    "explanation": "All required documents were provided.",
                }
            ],
            "final_report_text": "Strict completeness report.",
        }
    )

    print("=== STREAMLIT FORMATTED REPORT ===")
    print(formatted_text)

    assert "Application Name: visitor visa" in formatted_text
    assert "Overall Result: PASS" in formatted_text
    assert "- Document Presence: passed" in formatted_text
    assert "Strict completeness report." in formatted_text


def test_compute_overall_result_detects_failure() -> None:
    """Verify the overall-result helper maps failed stages to `FAIL`.

    Args:
        None.

    Returns:
        None: Assertions validate the computed overall result.
    """

    overall_result = compute_overall_result(
        {
            "stage_outcomes": [
                {"stage_name": "Document Presence", "status": "passed"},
                {"stage_name": "Form Completion", "status": "passed"},
                {"stage_name": "Content Sufficiency", "status": "failed"},
            ]
        }
    )

    print("=== STREAMLIT OVERALL RESULT ===")
    print(overall_result)

    assert overall_result == "FAIL"
