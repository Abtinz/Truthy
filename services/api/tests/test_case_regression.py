from __future__ import annotations

import base64
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app


TEST_CASES_ROOT = Path("/workspace/docs/Test Cases")


def _build_case_payload(case_directory: Path) -> dict:
    """Build one API payload from all files found inside a case directory.

    The regression suite mirrors the frontend/API contract by uploading the raw
    file bytes as base64 strings instead of pre-extracted text. This keeps the
    test aligned with the current end-to-end application path.

    Args:
        case_directory: Directory containing all files for one test case.

    Returns:
        dict: JSON payload compatible with the `/review` API endpoint.
    """

    files_payload = []
    for file_path in sorted(path for path in case_directory.iterdir() if path.is_file()):
        files_payload.append(
            {
                "file_name": file_path.name,
                "content_type": _guess_content_type(file_path),
                "base64_data": base64.b64encode(file_path.read_bytes()).decode("ascii"),
            }
        )

    return {
        "application_name": "visitor visa",
        "files": files_payload,
    }


def _guess_content_type(file_path: Path) -> str:
    """Map common case-file extensions to a reasonable MIME type.

    Args:
        file_path: Case-file path being serialized into the API payload.

    Returns:
        str: Best-effort MIME type string for the uploaded file.
    """

    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return "application/pdf"
    if suffix == ".png":
        return "image/png"
    if suffix == ".jpg" or suffix == ".jpeg":
        return "image/jpeg"
    return "application/octet-stream"


def _iter_cases() -> list[tuple[str, Path, bool]]:
    """Discover the 10 regression cases and their expected outcomes.

    Args:
        None.

    Returns:
        list[tuple[str, Path, bool]]: Case label, directory path, and whether
        the case is expected to pass.
    """

    discovered_cases: list[tuple[str, Path, bool]] = []
    for root_name, expected_pass in (("Complete", True), ("Incomplete", False)):
        root_directory = TEST_CASES_ROOT / root_name
        for case_directory in sorted(path for path in root_directory.iterdir() if path.is_dir()):
            discovered_cases.append(
                (
                    f"{root_name}/{case_directory.name}",
                    case_directory,
                    expected_pass,
                )
            )
    return discovered_cases


def _case_passed(response_json: dict) -> bool:
    """Determine whether the current system treated a case as passed.

    The regression definition treats a case as passed only when every review
    stage is explicitly marked as `passed`. A package should not be considered
    complete if any downstream stage still flags a failure.

    Args:
        response_json: Structured JSON response returned by the review API.

    Returns:
        bool: True when the case is treated as passed by the current system.
    """

    stage_outcomes = response_json.get("stage_outcomes", [])
    if not stage_outcomes:
        return False
    return all(stage.get("status") == "passed" for stage in stage_outcomes)


def test_case_regression_suite() -> None:
    """Run the 10 provided case folders against the review API.

    The test prints a compact result line for every case so the user can see
    which cases matched or missed the expected complete/incomplete behavior.

    Args:
        None.

    Returns:
        None: Assertions validate the suite against the expected outcomes.
    """

    client = TestClient(app)
    mismatches: list[str] = []

    for case_label, case_directory, expected_pass in _iter_cases():
        response = client.post("/review", json=_build_case_payload(case_directory))
        assert response.status_code == 200

        response_json = response.json()
        actual_pass = _case_passed(response_json)
        stage_statuses = [stage["status"] for stage in response_json.get("stage_outcomes", [])]

        print("=== CASE RESULT ===")
        print(
            {
                "case": case_label,
                "expected_pass": expected_pass,
                "actual_pass": actual_pass,
                "stage_statuses": stage_statuses,
            }
        )

        if actual_pass != expected_pass:
            mismatches.append(
                f"{case_label}: expected_pass={expected_pass}, actual_pass={actual_pass}, "
                f"stage_statuses={stage_statuses}"
            )

    assert not mismatches, "\n".join(mismatches)
