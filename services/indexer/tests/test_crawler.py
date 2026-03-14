from __future__ import annotations

from types import SimpleNamespace

from bs4 import BeautifulSoup

from app.ingestion.crawler import (
    CrawlerSource,
    HierarchicalSection,
    VisitorProgramCrawler,
    build_study_permit_sources,
    build_visitor_program_sources,
)


def test_extract_html_sections_preserves_heading_hierarchy() -> None:
    """Verify the crawler preserves heading paths for HTML content.

    Args:
        None.

    Returns:
        None.
    """
    crawler = VisitorProgramCrawler(sources=[])
    soup = BeautifulSoup(
        """
        <html>
          <body>
            <main>
              <h1>Visitor document requirements</h1>
              <p>Introductory summary.</p>
              <h2>Required documents</h2>
              <p>Applicants must include a passport.</p>
              <ul>
                <li>Passport copy</li>
                <li>Proof of funds</li>
              </ul>
              <h3>Additional notes</h3>
              <p>Some applicants may need biometrics.</p>
            </main>
          </body>
        </html>
        """,
        "html.parser",
    )

    sections = crawler._extract_html_sections(soup, "Visitor document requirements")

    print("\n=== HTML HIERARCHY SECTIONS ===")
    for section in sections:
        print(section.to_dict())

    assert len(sections) >= 3
    assert sections[1].path == [
        "Visitor document requirements",
        "Required documents",
    ]
    assert "- Passport copy" in sections[1].content


def test_extract_pdf_sections_preserves_page_hierarchy() -> None:
    """Verify the crawler preserves page hierarchy for PDF content.

    Args:
        None.

    Returns:
        None.
    """
    crawler = VisitorProgramCrawler(sources=[])
    documents = [
        SimpleNamespace(page_content="Checklist title\nPassport", metadata={"page": 0}),
        SimpleNamespace(page_content="Second page\nSignature", metadata={"page": 1}),
    ]

    sections = crawler._extract_pdf_sections(documents, "Visitor document checklist PDF")

    print("\n=== PDF PAGE SECTIONS ===")
    for section in sections:
        print(section.to_dict())

    assert len(sections) == 2
    assert sections[0].path == ["Visitor document checklist PDF", "Page 1"]
    assert sections[1].title == "Page 2"


def test_crawl_source_uses_expected_source_types() -> None:
    """Verify source definitions remain aligned with visitor-program inputs.

    Args:
        None.

    Returns:
        None.
    """
    crawler = VisitorProgramCrawler()

    print("\n=== VISITOR PROGRAM SOURCES ===")
    for source in crawler.sources:
        print(source)

    assert len(crawler.sources) == 3
    assert crawler.sources[0].kind == "operational_guidelines"
    assert crawler.sources[2].kind == "document_checklist_pdf"


def test_program_source_builders_match_visitor_and_study_permit_scope() -> None:
    """Verify the source builders reflect each program's configured scope.

    Args:
        None.

    Returns:
        None.
    """

    visitor_sources = build_visitor_program_sources()
    study_permit_sources = build_study_permit_sources()

    print("\n=== VISITOR SOURCES ===")
    print(visitor_sources)
    print("\n=== STUDY PERMIT SOURCES ===")
    print(study_permit_sources)

    assert len(visitor_sources) == 3
    assert sum(source.kind == "operational_guidelines" for source in visitor_sources) == 2
    assert len(study_permit_sources) == 2
    assert (
        sum(source.kind == "operational_guidelines" for source in study_permit_sources)
        == 1
    )
    assert study_permit_sources[0].kind == "operational_guidelines"
    assert "study-permits/assessing-application.html#s1" in study_permit_sources[0].url
    assert study_permit_sources[1].kind == "document_checklist_pdf"
    assert study_permit_sources[1].file_path.endswith("IMM5483.pdf")


def test_render_sections_to_text_prints_hierarchical_output() -> None:
    """Verify rendered text keeps a readable heading path structure.

    Args:
        None.

    Returns:
        None.
    """
    crawler = VisitorProgramCrawler(
        sources=[
            CrawlerSource(
                url="https://example.com",
                kind="operational_guidelines",
                title="Example",
            )
        ]
    )
    sections = [
        HierarchicalSection(
            title="Visitor guide",
            level=1,
            path=["Visitor guide"],
            content="Intro",
        ),
        HierarchicalSection(
            title="Documents",
            level=2,
            path=["Visitor guide", "Documents"],
            content="Passport",
        ),
    ]

    text = crawler._render_sections_to_text(sections)

    print("\n=== RENDERED HIERARCHICAL TEXT ===")
    print(text)

    assert "# Visitor guide" in text
    assert "## Visitor guide > Documents" in text


def test_extract_modified_date_reads_ircc_style_date() -> None:
    """Verify the crawler extracts the `Date modified` value from HTML.

    Args:
        None.

    Returns:
        None.
    """

    crawler = VisitorProgramCrawler(sources=[])
    soup = BeautifulSoup(
        """
        <html>
          <body>
            <main>
              <h1>Visitor document requirements</h1>
            </main>
            <section>
              <dt>Date modified:</dt>
              <dd><time datetime="2026-03-03">2026-03-03</time></dd>
            </section>
          </body>
        </html>
        """,
        "html.parser",
    )

    modified_date = crawler._extract_modified_date(soup)

    print("\n=== EXTRACTED MODIFIED DATE ===")
    print(modified_date)

    assert modified_date == "2026-03-03"


def test_extract_modified_date_reads_dcterms_modified_meta() -> None:
    """Verify the crawler extracts the `dcterms.modified` meta value.

    Args:
        None.

    Returns:
        None.
    """

    crawler = VisitorProgramCrawler(sources=[])
    soup = BeautifulSoup(
        """
        <html>
          <head>
            <meta name="dcterms.modified" content="2025-11-25" />
          </head>
          <body>
            <main>
              <h1>Study permits: Assessing the application</h1>
            </main>
          </body>
        </html>
        """,
        "html.parser",
    )

    modified_date = crawler._extract_modified_date(soup)

    print("\n=== EXTRACTED META MODIFIED DATE ===")
    print(modified_date)

    assert modified_date == "2025-11-25"
