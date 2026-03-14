from __future__ import annotations

from pathlib import Path
from typing import Literal

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from pydantic import BaseModel
from pydantic import Field

from app.cache.policy_freshness_cache import PolicyFreshnessCache
from app.core.dependencies import get_indexer_manager
from app.core.dependencies import get_policy_freshness_cache
from app.core.config import IndexerSettings
from app.ingestion.crawler import CrawlerSource
from app.vectorstore.index_manager import VisitorProgramIndexer

router = APIRouter()


class SingleSourceIndexRequest(BaseModel):
    """Request payload for one direct source-indexing operation.

    Args:
        source_value: Source URL for crawling or local filesystem path for
            direct PDF indexing.
        index_name: Target Pinecone index name selected by the operator.
        ingestion_mode: Whether to crawl a remote page or read a local PDF.
        source_title: Optional human-readable label used in vector metadata.

    Returns:
        SingleSourceIndexRequest: Validated API request payload.
    """

    source_value: str = Field(..., min_length=1)
    index_name: str = Field(..., min_length=1)
    ingestion_mode: Literal["crawling", "local_pdf"]
    source_title: str | None = None


def _build_source_from_request(
    request: SingleSourceIndexRequest,
    settings: IndexerSettings,
) -> CrawlerSource:
    """Convert the direct-indexing request into one concrete crawler source.

    Args:
        request: Validated indexing request payload.
        settings: Environment-backed indexer settings used for index routing.

    Returns:
        CrawlerSource: Source definition ready for the index manager.

    Raises:
        ValueError: If the request mode and target index are incompatible.
    """

    cleaned_value = request.source_value.strip()
    cleaned_title = (request.source_title or "").strip()

    if request.ingestion_mode == "crawling":
        if request.index_name != settings.pinecone_operational_guidelines_index_name:
            raise ValueError(
                "Crawling mode must target the operational guidelines index."
            )
        return CrawlerSource(
            kind="operational_guidelines",
            title=cleaned_title or "Direct operational-guidelines source",
            url=cleaned_value,
        )

    if request.index_name != settings.pinecone_document_checklist_index_name:
        raise ValueError("Local PDF mode must target the document checklist index.")

    pdf_path = Path(cleaned_value)
    return CrawlerSource(
        kind="document_checklist_pdf",
        title=cleaned_title or pdf_path.name or "Direct checklist PDF source",
        url="",
        file_path=str(pdf_path),
    )


@router.get("/cache/policy-freshness")
def read_policy_freshness_cache(
    policy_cache: PolicyFreshnessCache = Depends(get_policy_freshness_cache),
) -> dict[str, object]:
    """Return the Redis-tracked policy freshness records.

    This endpoint acts as a lightweight log view for the Redis optimization
    layer. It exposes only the minimal cache contents requested by the product
    design: the tracked source URL and its latest stored `Date modified` value.

    Args:
        policy_cache: Redis freshness cache dependency.

    Returns:
        dict[str, object]: Structured cache-log payload for operator review.
    """

    entries = policy_cache.list_entries()
    return {
        "cache_name": "policy_freshness",
        "entry_count": len(entries),
        "entries": [
            {
                "source_url": entry.source_url,
                "modified_date": entry.modified_date,
            }
            for entry in entries
        ],
    }


@router.post("/index")
def index_single_source(
    request: SingleSourceIndexRequest,
    indexer: VisitorProgramIndexer = Depends(get_indexer_manager),
) -> dict[str, object]:
    """Index one operator-supplied source and return structured logs.

    The route supports two direct indexing modes:
    - `crawling` for remote operational-guidelines pages
    - `local_pdf` for local checklist PDFs

    Operational-guidelines requests are checked against Redis freshness state
    before indexing. If the cached `Date modified` value matches the current
    page, the route skips embedding and Pinecone upserts and returns a
    `skipped_up_to_date` result.

    Args:
        request: Direct source-indexing request payload.
        indexer: Shared index manager dependency.

    Returns:
        dict[str, object]: Structured indexing result including workflow logs.
    """

    try:
        source = _build_source_from_request(request, indexer.settings)
        result = indexer.index_single_source(source)
        return result.to_dict()
    except (ValueError, FileNotFoundError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
