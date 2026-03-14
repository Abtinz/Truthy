from __future__ import annotations

from fastapi.testclient import TestClient

from app.cache.policy_freshness_cache import PolicyFreshnessCacheEntry
from app.core.dependencies import get_indexer_manager
from app.core.dependencies import get_policy_freshness_cache
from app.ingestion.crawler import CrawlerSource
from app.main import app


class FakePolicyFreshnessCache:
    """Small fake cache used to test the indexer Redis log endpoint.

    Args:
        None.

    Returns:
        FakePolicyFreshnessCache: Deterministic cache dependency for tests.
    """

    def list_entries(self) -> list[PolicyFreshnessCacheEntry]:
        """Return stable cache entries for endpoint verification.

        Args:
            None.

        Returns:
            list[PolicyFreshnessCacheEntry]: Mocked Redis freshness entries.
        """

        return [
            PolicyFreshnessCacheEntry(
                source_url="https://example.com/policy-a",
                modified_date="2026-03-03",
            ),
            PolicyFreshnessCacheEntry(
                source_url="https://example.com/policy-b",
                modified_date="2026-03-04",
            ),
        ]


class FakeIndexerManager:
    """Small fake indexer manager for direct indexing endpoint tests.

    Args:
        None.

    Returns:
        FakeIndexerManager: Deterministic indexer dependency for route tests.
    """

    def __init__(self) -> None:
        """Initialize fake settings and capture list.

        Args:
            None.

        Returns:
            None.
        """
        self.settings = type(
            "FakeSettings",
            (),
            {
                "pinecone_operational_guidelines_index_name": (
                    "operational-guidelines-instructions"
                ),
                "pinecone_document_checklist_index_name": "document-checklist-pdf",
            },
        )()
        self.indexed_sources: list[CrawlerSource] = []

    def index_single_source(self, source: CrawlerSource):
        """Return a deterministic single-source indexing result.

        Args:
            source: Source requested by the API route.

        Returns:
            object: Simple object exposing a `to_dict()` method.
        """

        self.indexed_sources.append(source)

        if source.kind == "operational_guidelines":
            result = {
                "source_reference": source.source_reference(),
                "source_kind": source.kind,
                "index_name": "operational-guidelines-instructions",
                "status": "skipped_up_to_date",
                "modified_date": "2025-11-25",
                "records_upserted": 0,
                "logs": [
                    "single_source_index_start kind=operational_guidelines",
                    "policy_source_skipped url=https://example.com/policy modified_date=2025-11-25",
                ],
            }
        else:
            result = {
                "source_reference": source.source_reference(),
                "source_kind": source.kind,
                "index_name": "document-checklist-pdf",
                "status": "indexed",
                "modified_date": None,
                "records_upserted": 3,
                "logs": [
                    "single_source_index_start kind=document_checklist_pdf",
                    "upsert_checklists_done count=3",
                ],
            }

        return type("FakeIndexResult", (), {"to_dict": lambda self: result})()


client = TestClient(app)


def override_policy_freshness_cache() -> FakePolicyFreshnessCache:
    """Provide a fake Redis freshness cache for endpoint tests.

    Args:
        None.

    Returns:
        FakePolicyFreshnessCache: Fake dependency used by TestClient.
    """

    return FakePolicyFreshnessCache()


app.dependency_overrides[get_policy_freshness_cache] = (
    override_policy_freshness_cache
)


def override_indexer_manager() -> FakeIndexerManager:
    """Provide a fake direct-indexing dependency for route tests.

    Args:
        None.

    Returns:
        FakeIndexerManager: Fake dependency used by TestClient.
    """

    return FakeIndexerManager()


app.dependency_overrides[get_indexer_manager] = override_indexer_manager


def test_root_endpoint_returns_indexer_status() -> None:
    """Verify the indexer root endpoint returns the expected payload.

    Args:
        None.

    Returns:
        None.
    """

    response = client.get("/")

    print("=== INDEXER ROOT RESPONSE ===")
    print(response.json())

    assert response.status_code == 200
    assert response.json() == {"service": "truthy-indexer", "status": "ok"}


def test_health_endpoint_returns_ok_when_cache_is_reachable() -> None:
    """Verify the health endpoint succeeds when the cache dependency works.

    Args:
        None.

    Returns:
        None.
    """

    response = client.get("/health")

    print("=== INDEXER HEALTH RESPONSE ===")
    print(response.json())

    assert response.status_code == 200
    assert response.json() == {"service": "truthy-indexer", "status": "ok"}


def test_policy_freshness_cache_endpoint_returns_entries() -> None:
    """Verify the Redis cache-log endpoint returns URL/date pairs.

    Args:
        None.

    Returns:
        None.
    """

    response = client.get("/cache/policy-freshness")

    print("=== INDEXER POLICY CACHE RESPONSE ===")
    print(response.json())

    assert response.status_code == 200
    assert response.json()["cache_name"] == "policy_freshness"
    assert response.json()["entry_count"] == 2
    assert response.json()["entries"] == [
        {
            "source_url": "https://example.com/policy-a",
            "modified_date": "2026-03-03",
        },
        {
            "source_url": "https://example.com/policy-b",
            "modified_date": "2026-03-04",
        },
    ]


def test_direct_indexing_endpoint_skips_up_to_date_policy_sources() -> None:
    """Verify crawling requests can return a cache-based skip result.

    Args:
        None.

    Returns:
        None.
    """

    response = client.post(
        "/index",
        json={
            "source_value": "https://example.com/policy",
            "index_name": "operational-guidelines-instructions",
            "ingestion_mode": "crawling",
            "source_title": "Example policy",
        },
    )

    print("=== INDEXER DIRECT CRAWL RESPONSE ===")
    print(response.json())

    assert response.status_code == 200
    assert response.json()["status"] == "skipped_up_to_date"
    assert response.json()["source_kind"] == "operational_guidelines"
    assert response.json()["modified_date"] == "2025-11-25"
    assert "policy_source_skipped" in "\n".join(response.json()["logs"])


def test_direct_indexing_endpoint_indexes_local_pdf_sources() -> None:
    """Verify local PDF requests route to the checklist indexing flow.

    Args:
        None.

    Returns:
        None.
    """

    response = client.post(
        "/index",
        json={
            "source_value": "/workspace/services/data/forms/IMM5483.pdf",
            "index_name": "document-checklist-pdf",
            "ingestion_mode": "local_pdf",
            "source_title": "Study permit checklist",
        },
    )

    print("=== INDEXER DIRECT PDF RESPONSE ===")
    print(response.json())

    assert response.status_code == 200
    assert response.json()["status"] == "indexed"
    assert response.json()["source_kind"] == "document_checklist_pdf"
    assert response.json()["records_upserted"] == 3
    assert "upsert_checklists_done count=3" in response.json()["logs"]
