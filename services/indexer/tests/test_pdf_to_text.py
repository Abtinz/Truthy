from __future__ import annotations

import io

from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from app.ingestion.pdf_to_text import extract_pdf_to_text_chunks


def test_extract_pdf_to_text_chunks_prints_result_for_simple_pdf() -> None:
    pdf_bytes = _build_simple_pdf()

    result = extract_pdf_to_text_chunks(pdf_bytes, chunk_size=200, chunk_overlap=20)

    print("\n=== FULL TEXT ===")
    print(result.full_text)
    print("\n=== CHUNKS ===")
    for chunk in result.chunks:
        print(chunk.to_dict())

    assert "Simple PDF Example" in result.full_text
    assert "Reference Number: TEST-001" in result.full_text


def test_extract_pdf_to_text_chunks_extracts_text_and_image_ocr() -> None:
    pdf_bytes = _build_sample_pdf()

    result = extract_pdf_to_text_chunks(pdf_bytes, chunk_size=120, chunk_overlap=20)

    assert "Applicant Name: John Doe" in result.full_text
    assert "Application Program: Minor Citizenship" in result.full_text
    assert "Passport Number: ABC12345" in result.full_text

    source_types = {chunk.source_type for chunk in result.chunks}
    assert "page_text" in source_types
    assert "image_ocr" in source_types
    assert all(chunk.char_count <= 140 for chunk in result.chunks)


def test_extract_pdf_to_text_chunks_preserves_page_metadata() -> None:
    pdf_bytes = _build_sample_pdf()

    result = extract_pdf_to_text_chunks(pdf_bytes, chunk_size=90, chunk_overlap=10)

    assert len(result.pages) == 2
    assert result.pages[0]["page_number"] == 1
    assert result.pages[1]["page_number"] == 2
    assert any(entry["source_type"] == "image_ocr" for entry in result.pages[1]["entries"])


def test_extract_pdf_to_text_chunks_validates_chunk_arguments() -> None:
    pdf_bytes = _build_sample_pdf()

    try:
        extract_pdf_to_text_chunks(pdf_bytes, chunk_size=50, chunk_overlap=50)
    except ValueError as exc:
        assert "chunk_overlap must be smaller than chunk_size" in str(exc)
    else:
        raise AssertionError("Expected chunk overlap validation error")


def _build_sample_pdf() -> bytes:
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer)

    pdf.setFont("Helvetica", 12)
    pdf.drawString(72, 760, "Applicant Name: John Doe")
    pdf.drawString(72, 740, "Application Program: Minor Citizenship")
    pdf.drawString(72, 720, "Date of Birth: 2010-05-04")
    pdf.showPage()

    image = _build_text_image("Passport Number: ABC12345")
    pdf.drawImage(ImageReader(image), 72, 620, width=420, height=120)
    pdf.drawString(72, 580, "Supporting document image appears above.")
    pdf.save()

    return buffer.getvalue()


def _build_simple_pdf() -> bytes:
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer)

    pdf.setFont("Helvetica", 12)
    pdf.drawString(72, 760, "Simple PDF Example")
    pdf.drawString(72, 740, "Reference Number: TEST-001")
    pdf.drawString(72, 720, "Applicant Name: Jane Doe")
    pdf.save()

    return buffer.getvalue()


def _build_text_image(text: str) -> Image.Image:
    image = Image.new("RGB", (1600, 400), color="white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 72)
    draw.text((60, 130), text, fill="black", font=font)
    return image
