from __future__ import annotations

import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Literal

import requests
from bs4 import BeautifulSoup, Tag
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader


SourceKind = Literal["operational_guidelines", "document_checklist_pdf"]


@dataclass(frozen=True)
class CrawlerSource:
    """Source definition for a visitor-program crawl target.

    Args:
        url: Remote URL to crawl.
        kind: Logical source type used by downstream indexing code.
        title: Human-readable label for the source.

    Returns:
        CrawlerSource: Immutable crawl target definition.
    """

    url: str
    kind: SourceKind
    title: str


@dataclass(frozen=True)
class HierarchicalSection:
    """Structured section extracted from a crawled source.

    Args:
        title: Section heading title.
        level: Heading depth. Lower values indicate higher-level headings.
        path: Ordered heading path from the document root to the section.
        content: Section body text associated with the heading path.

    Returns:
        HierarchicalSection: Immutable hierarchical section record.
    """

    title: str
    level: int
    path: list[str]
    content: str

    def to_dict(self) -> dict[str, object]:
        """Convert the section to a serializable dictionary.

        Args:
            None.

        Returns:
            dict[str, object]: Plain dictionary representation of the section.
        """
        return asdict(self)


@dataclass(frozen=True)
class CrawledDocument:
    """Final structured result returned by the visitor-program crawler.

    Args:
        source: Source definition describing the crawled document.
        document_title: Top-level resolved document title.
        sections: Hierarchical sections extracted from the source.
        full_text: Text rendition of the hierarchical structure for inspection
            or downstream indexing.

    Returns:
        CrawledDocument: Immutable crawled document record.
    """

    source: CrawlerSource
    document_title: str
    sections: list[HierarchicalSection]
    full_text: str

    def to_dict(self) -> dict[str, object]:
        """Convert the crawled document to a serializable dictionary.

        Args:
            None.

        Returns:
            dict[str, object]: Plain dictionary representation of the document.
        """
        return {
            "source": asdict(self.source),
            "document_title": self.document_title,
            "sections": [section.to_dict() for section in self.sections],
            "full_text": self.full_text,
        }


class VisitorProgramCrawler:
    """LangChain-based crawler for the visitor-program seed sources.

    The crawler intentionally stays narrow in scope for this stage:
    - two operational guideline pages
    - one document checklist PDF

    Web pages are fetched with LangChain's `WebBaseLoader`, then parsed into a
    heading-aware structure. The checklist PDF is loaded with LangChain's
    `PyPDFLoader` and preserved as page-level sections.

    Args:
        sources: Optional custom source list. Defaults to the three visitor
            program sources requested for this stage.

    Returns:
        VisitorProgramCrawler: Configured crawler for the visitor-program set.
    """

    DEFAULT_SOURCES = [
        CrawlerSource(
            url=(
                "https://www.canada.ca/en/immigration-refugees-citizenship/"
                "corporate/publications-manuals/operational-bulletins-manuals/"
                "temporary-residents/visitors/document-requirements.html"
            ),
            kind="operational_guidelines",
            title="Visitor document requirements",
        ),
        CrawlerSource(
            url=(
                "https://www.canada.ca/en/immigration-refugees-citizenship/"
                "corporate/publications-manuals/operational-bulletins-manuals/"
                "temporary-residents/visitors/application-intake-assessment.html"
            ),
            kind="operational_guidelines",
            title="Visitor application intake assessment",
        ),
        CrawlerSource(
            url=(
                "https://www.canada.ca/content/dam/ircc/documents/pdf/english/"
                "kits/forms/imm5484/08-07-2024/imm5484e.pdf"
            ),
            kind="document_checklist_pdf",
            title="Visitor document checklist PDF",
        ),
    ]

    def __init__(self, sources: list[CrawlerSource] | None = None) -> None:
        """Initialize the crawler with a source list.

        Args:
            sources: Optional crawl targets. Defaults to the visitor-program
                sources defined by the class.

        Returns:
            None.
        """
        os.environ.setdefault("USER_AGENT", "TruthyIndexer/0.1")
        self.sources = sources or list(self.DEFAULT_SOURCES)

    def crawl_all(self) -> list[CrawledDocument]:
        """Crawl every configured source and return structured documents.

        Args:
            None.

        Returns:
            list[CrawledDocument]: Structured results for all configured sources.
        """
        return [self.crawl_source(source) for source in self.sources]

    def crawl_source(self, source: CrawlerSource) -> CrawledDocument:
        """Crawl one source and return a hierarchy-preserving document.

        Args:
            source: Crawl target definition.

        Returns:
            CrawledDocument: Structured crawl result for the source.
        """
        if source.kind == "document_checklist_pdf":
            return self._crawl_pdf_source(source)
        return self._crawl_html_source(source)

    def _crawl_html_source(self, source: CrawlerSource) -> CrawledDocument:
        """Fetch and parse one HTML source with LangChain's web loader.

        Args:
            source: HTML crawl target definition.

        Returns:
            CrawledDocument: Structured hierarchy-preserving HTML crawl result.
        """
        loader = self._build_web_loader(source.url)
        soup = self._load_soup_from_loader(loader)
        document_title = self._extract_document_title(soup, source.title)
        sections = self._extract_html_sections(soup, document_title)
        full_text = self._render_sections_to_text(sections)

        return CrawledDocument(
            source=source,
            document_title=document_title,
            sections=sections,
            full_text=full_text,
        )

    def _crawl_pdf_source(self, source: CrawlerSource) -> CrawledDocument:
        """Fetch and parse one PDF source with LangChain's PDF loader.

        Args:
            source: PDF crawl target definition.

        Returns:
            CrawledDocument: Page-structured PDF crawl result.
        """
        loader = self._build_web_loader(source.url)
        response = loader.session.get(source.url, timeout=30)
        response.raise_for_status()

        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_path = Path(temp_dir) / "visitor_document_checklist.pdf"
            pdf_path.write_bytes(response.content)

            pdf_loader = PyPDFLoader(str(pdf_path))
            documents = pdf_loader.load()

        sections = self._extract_pdf_sections(documents, source.title)
        full_text = self._render_sections_to_text(sections)

        return CrawledDocument(
            source=source,
            document_title=source.title,
            sections=sections,
            full_text=full_text,
        )

    def _build_web_loader(self, url: str) -> WebBaseLoader:
        """Construct a LangChain web loader with stable request settings.

        Args:
            url: Target URL to load.

        Returns:
            WebBaseLoader: Configured LangChain web loader.
        """
        loader = WebBaseLoader(web_paths=(url,), requests_per_second=1)
        loader.session = requests.Session()
        loader.session.trust_env = False
        loader.session.headers.update({"User-Agent": "TruthyIndexer/0.1"})
        return loader

    def _load_soup_from_loader(self, loader: WebBaseLoader) -> BeautifulSoup:
        """Load a BeautifulSoup object using the LangChain web loader.

        Args:
            loader: Configured LangChain web loader instance.

        Returns:
            BeautifulSoup: Parsed soup for the remote page.
        """
        response = loader.session.get(loader.web_paths[0], timeout=30)
        response.raise_for_status()
        return BeautifulSoup(response.text, "html.parser")

    def _extract_document_title(self, soup: BeautifulSoup, fallback_title: str) -> str:
        """Resolve the best available title for an HTML document.

        Args:
            soup: Parsed HTML document.
            fallback_title: Default title to use if the page lacks one.

        Returns:
            str: Resolved document title.
        """
        h1 = soup.find("h1")
        if h1:
            text = normalize_text(h1.get_text(" ", strip=True))
            if text:
                return text

        if soup.title:
            text = normalize_text(soup.title.get_text(" ", strip=True))
            if text:
                return text

        return fallback_title

    def _extract_html_sections(
        self,
        soup: BeautifulSoup,
        document_title: str,
    ) -> list[HierarchicalSection]:
        """Extract heading-aware sections from an HTML page.

        Args:
            soup: Parsed HTML page.
            document_title: Document title used as the root heading.

        Returns:
            list[HierarchicalSection]: Structured section list preserving
            heading hierarchy.
        """
        root = soup.find("main") or soup.find("article") or soup.body
        if root is None:
            return [
                HierarchicalSection(
                    title=document_title,
                    level=1,
                    path=[document_title],
                    content="",
                )
            ]

        heading_stack: dict[int, str] = {1: document_title}
        sections: list[HierarchicalSection] = []
        section_buffers: dict[tuple[str, ...], list[str]] = {}

        for element in self._iter_structural_elements(root):
            if element.name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
                level = int(element.name[1])
                heading_text = normalize_text(element.get_text(" ", strip=True))
                if not heading_text:
                    continue

                # Lower-level headings belong under the newest heading at the
                # same or higher hierarchy level, so deeper stale headings are
                # cleared whenever a new heading appears.
                heading_stack = {
                    existing_level: existing_title
                    for existing_level, existing_title in heading_stack.items()
                    if existing_level < level
                }
                heading_stack[level] = heading_text

                path = [heading_stack[key] for key in sorted(heading_stack)]
                sections.append(
                    HierarchicalSection(
                        title=heading_text,
                        level=level,
                        path=path,
                        content="",
                    )
                )
                section_buffers[tuple(path)] = []
                continue

            text = self._extract_element_text(element)
            if not text:
                continue

            current_path = tuple(
                sections[-1].path if sections else [document_title]
            )
            if current_path not in section_buffers:
                sections.append(
                    HierarchicalSection(
                        title=document_title,
                        level=1,
                        path=list(current_path),
                        content="",
                    )
                )
                section_buffers[current_path] = []
            section_buffers[current_path].append(text)

        finalized_sections: list[HierarchicalSection] = []
        for section in sections:
            content = "\n".join(section_buffers.get(tuple(section.path), [])).strip()
            finalized_sections.append(
                HierarchicalSection(
                    title=section.title,
                    level=section.level,
                    path=section.path,
                    content=content,
                )
            )

        return [section for section in finalized_sections if section.content or section.level == 1]

    def _extract_pdf_sections(
        self,
        documents: Iterable[object],
        document_title: str,
    ) -> list[HierarchicalSection]:
        """Convert LangChain PDF page documents into page-level sections.

        Args:
            documents: LangChain page documents returned by `PyPDFLoader`.
            document_title: Top-level title for the PDF.

        Returns:
            list[HierarchicalSection]: Page-level PDF sections.
        """
        sections: list[HierarchicalSection] = []

        for index, document in enumerate(documents, start=1):
            page_number = getattr(document, "metadata", {}).get("page", index - 1) + 1
            page_title = f"Page {page_number}"
            page_text = normalize_text(getattr(document, "page_content", ""))

            sections.append(
                HierarchicalSection(
                    title=page_title,
                    level=2,
                    path=[document_title, page_title],
                    content=page_text,
                )
            )

        return sections

    def _render_sections_to_text(self, sections: list[HierarchicalSection]) -> str:
        """Render hierarchical sections into a readable text representation.

        Args:
            sections: Structured sections to render.

        Returns:
            str: Combined text representation of all sections.
        """
        rendered_sections: list[str] = []

        for section in sections:
            heading_prefix = "#" * max(section.level, 1)
            rendered_sections.append(
                f"{heading_prefix} {' > '.join(section.path)}\n{section.content}".strip()
            )

        return "\n\n".join(part for part in rendered_sections if part.strip())

    def _iter_structural_elements(self, root: Tag) -> Iterable[Tag]:
        """Iterate meaningful structural tags from the HTML content root.

        Args:
            root: Root element from which structure should be extracted.

        Returns:
            Iterable[Tag]: Sequence of structural tags in document order.
        """
        allowed_tags = {"h1", "h2", "h3", "h4", "h5", "h6", "p", "li"}

        for element in root.find_all(allowed_tags):
            if element.name == "li":
                parent = element.find_parent(["ul", "ol"])
                if parent and parent.find_parent("li") is not None:
                    continue
            yield element

    def _extract_element_text(self, element: Tag) -> str:
        """Normalize text from one structural HTML element.

        Args:
            element: Structural HTML tag such as paragraph or list item.

        Returns:
            str: Normalized text ready for section content.
        """
        text = normalize_text(element.get_text(" ", strip=True))
        if not text:
            return ""
        if element.name == "li":
            return f"- {text}"
        return text


def normalize_text(text: str) -> str:
    """Normalize crawler text while preserving section readability.

    Args:
        text: Raw text extracted from HTML or PDF sources.

    Returns:
        str: Whitespace-normalized text.
    """
    return " ".join(text.split())
