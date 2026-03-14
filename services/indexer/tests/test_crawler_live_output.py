from __future__ import annotations

from pathlib import Path

from app.ingestion.crawler import CrawledDocument, HierarchicalSection, VisitorProgramCrawler


OUTPUT_PATH = Path("/workspace/services/indexer/log/crawler_live_results.txt")


def test_live_crawler_writes_results_to_txt() -> None:
    """Run the live guideline crawler and persist its output to a text file.

    This test is intentionally integration-style. It fetches the live
    operational-guidelines sources owned by the crawler, renders their
    hierarchical sections into a readable text report, writes that report into
    the repository workspace, and verifies the result is non-empty.

    Args:
        None.

    Returns:
        None.
    """
    crawler = VisitorProgramCrawler(
        sources=[
            source
            for source in VisitorProgramCrawler().sources
            if source.kind == "operational_guidelines"
        ]
    )
    documents = crawler.crawl_all()

    rendered_output = _render_documents(documents)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(rendered_output, encoding="utf-8")

    print("\n=== LIVE CRAWLER OUTPUT PATH ===")
    print(str(OUTPUT_PATH))
    print("\n=== LIVE CRAWLER OUTPUT PREVIEW ===")
    print(rendered_output[:3000])

    assert len(documents) == 2
    assert OUTPUT_PATH.exists()
    assert OUTPUT_PATH.stat().st_size > 0
    assert "Temporary residents: Document requirements" in rendered_output
    assert "Temporary resident: Application intake assessment" in rendered_output


def _render_documents(documents: list[CrawledDocument]) -> str:
    """Render crawled documents into a readable integration-test artifact.

    Args:
        documents: Structured crawler output to serialize into plain text.

    Returns:
        str: Human-readable text report containing document metadata, section
        hierarchy, and content previews.
    """
    rendered_documents: list[str] = []

    for index, document in enumerate(documents, start=1):
        section_blocks = [_render_section(section) for section in document.sections]
        rendered_documents.append(
            "\n".join(
                [
                    f"DOCUMENT {index}",
                    f"Source URL: {document.source.url}",
                    f"Source Kind: {document.source.kind}",
                    f"Document Title: {document.document_title}",
                    f"Section Count: {len(document.sections)}",
                    "",
                    *section_blocks,
                ]
            ).strip()
        )

    return "\n\n" + ("\n\n" + ("=" * 80) + "\n\n").join(rendered_documents) + "\n"


def _render_section(section: HierarchicalSection) -> str:
    """Render one hierarchical section into a readable text block.

    Args:
        section: Structured section extracted by the crawler.

    Returns:
        str: Plain-text block containing heading path and content.
    """
    return "\n".join(
        [
            f"Section Title: {section.title}",
            f"Section Level: {section.level}",
            f"Section Path: {' > '.join(section.path)}",
            "Section Content:",
            section.content,
            "-" * 40,
        ]
    )
