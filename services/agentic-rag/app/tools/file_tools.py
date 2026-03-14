from __future__ import annotations

import base64
import io
from typing import Any

import fitz
from langchain_core.tools import tool


@tool
def normalize_uploaded_files(files: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize incoming uploaded files into text-bearing records.

    This tool converts the mixed file payload formats accepted by the API into
    one stable structure consumed by retrieval, rule evaluation, and report
    synthesis. PDF files are parsed with PyMuPDF, while plain text payloads are
    decoded directly.

    Args:
        files: Uploaded file payloads containing file metadata and one of
            direct text, base64 content, or byte values.

    Returns:
        list[dict[str, Any]]: Normalized file records containing file name,
        content type, and extracted text.
    """

    normalized_file_texts: list[dict[str, Any]] = []
    for file_input in files:
        normalized_file_texts.append(
            {
                "file_name": file_input.get("file_name"),
                "content_type": file_input.get("content_type"),
                "text": decode_uploaded_file.invoke({"file_input": file_input}),
            }
        )
    return normalized_file_texts


@tool
def decode_uploaded_file(file_input: dict[str, Any]) -> str:
    """Convert one incoming file payload into best-effort plain text.

    Args:
        file_input: One uploaded file payload as a plain dictionary.

    Returns:
        str: Best-effort normalized text extracted from the incoming payload.
    """

    direct_text = file_input.get("text")
    if isinstance(direct_text, str) and direct_text.strip():
        return direct_text.strip()

    base64_data = file_input.get("base64_data")
    if isinstance(base64_data, str) and base64_data.strip():
        try:
            decoded_bytes = base64.b64decode(base64_data)
            if looks_like_pdf.invoke(
                {"file_input": file_input, "raw_bytes": decoded_bytes}
            ):
                return extract_pdf_text.invoke({"raw_bytes": decoded_bytes})
            return decoded_bytes.decode("utf-8", errors="ignore").strip()
        except Exception:
            return ""

    byte_values = file_input.get("byte_values")
    if isinstance(byte_values, list) and byte_values:
        try:
            raw_bytes = bytes(byte_values)
            if looks_like_pdf.invoke({"file_input": file_input, "raw_bytes": raw_bytes}):
                return extract_pdf_text.invoke({"raw_bytes": raw_bytes})
            return raw_bytes.decode("utf-8", errors="ignore").strip()
        except Exception:
            return ""

    return ""


@tool
def looks_like_pdf(file_input: dict[str, Any], raw_bytes: bytes) -> bool:
    """Determine whether an uploaded payload should be parsed as a PDF.

    Args:
        file_input: Uploaded file payload as a plain dictionary.
        raw_bytes: Raw decoded bytes for the uploaded file.

    Returns:
        bool: True when the payload appears to be a PDF document.
    """

    content_type = str(file_input.get("content_type") or "")
    file_name = str(file_input.get("file_name") or "")

    if content_type == "application/pdf":
        return True
    if file_name.lower().endswith(".pdf"):
        return True
    return raw_bytes.startswith(b"%PDF")


@tool
def extract_pdf_text(raw_bytes: bytes) -> str:
    """Extract native text from a PDF document using PyMuPDF.

    Args:
        raw_bytes: Raw PDF file bytes.

    Returns:
        str: Concatenated extracted text across all PDF pages.
    """

    document = fitz.open(stream=io.BytesIO(raw_bytes), filetype="pdf")
    try:
        extracted_pages: list[str] = []
        for page in document:
            page_text = page.get_text("text").strip()
            if page_text:
                extracted_pages.append(page_text)
        return "\n\n".join(extracted_pages).strip()
    finally:
        document.close()
