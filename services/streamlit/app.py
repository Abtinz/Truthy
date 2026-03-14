from __future__ import annotations

import os
from pathlib import Path

import requests
import streamlit as st

from utils import build_review_payload
from utils import collect_backend_logs
from utils import format_review_response


API_BASE_URL = os.getenv("API_BASE_URL", "http://api:8000").rstrip("/")
INDEXER_LOG_PATH = Path("/workspace/services/indexer/log/indexer_run.log")
CRAWLER_LOG_PATH = Path("/workspace/services/indexer/log/crawler_live_results.txt")


def submit_review_request(application_name: str, uploaded_files: list) -> dict:
    """Submit the review request to the FastAPI gateway.

    Args:
        application_name: Human-entered immigration application name.
        uploaded_files: Streamlit uploaded file objects selected by the user.

    Returns:
        dict: Parsed JSON response returned by the API service.
    """

    payload = build_review_payload(application_name, uploaded_files)
    response = requests.post(
        f"{API_BASE_URL}/review",
        json=payload,
        timeout=120,
    )
    response.raise_for_status()
    return dict(response.json())


def main() -> None:
    """Render the Streamlit officer-facing review interface.

    Args:
        None.

    Returns:
        None: Streamlit renders the application UI directly.
    """

    st.set_page_config(page_title="Truthy Review Console", layout="wide")
    st.title("Truthy Review Console")
    st.caption("Upload application files, submit a review request, inspect logs, and read the generated report.")

    if "ui_logs" not in st.session_state:
        st.session_state.ui_logs = []
    if "review_result" not in st.session_state:
        st.session_state.review_result = None

    left_column, right_column = st.columns([1, 1])

    with left_column:
        st.subheader("Submission")
        application_name = st.text_input(
            "Application Name",
            placeholder="Visitor visa",
        )
        uploaded_files = st.file_uploader(
            "Upload Files",
            accept_multiple_files=True,
            type=None,
        )

        if st.button("Run Review", type="primary", use_container_width=True):
            try:
                st.session_state.ui_logs.append(
                    f"Submitting review for '{application_name}' with {len(uploaded_files or [])} file(s)."
                )
                st.session_state.review_result = submit_review_request(
                    application_name,
                    uploaded_files or [],
                )
                st.session_state.ui_logs.append("Review request completed successfully.")
            except Exception as exc:  # pragma: no cover - Streamlit runtime path
                st.session_state.ui_logs.append(f"Review request failed: {exc}")
                st.error(str(exc))

        st.subheader("Logs")
        ui_logs, backend_logs = collect_backend_logs(
            ui_logs=st.session_state.ui_logs,
            log_paths=[INDEXER_LOG_PATH, CRAWLER_LOG_PATH],
        )
        logs_tab, backend_tab = st.tabs(["UI Logs", "Backend Logs"])
        with logs_tab:
            st.text_area(
                "UI Activity",
                value=ui_logs,
                height=260,
            )
        with backend_tab:
            st.text_area(
                "Backend Files",
                value=backend_logs,
                height=260,
            )

    with right_column:
        st.subheader("Results")
        if st.session_state.review_result:
            st.text_area(
                "Formatted Report",
                value=format_review_response(st.session_state.review_result),
                height=420,
            )
            st.json(st.session_state.review_result)
        else:
            st.info("Submit an application package to see the generated review output.")


if __name__ == "__main__":
    main()
