from __future__ import annotations

from fastapi.testclient import TestClient

from app.cache.policy_freshness_cache import PolicyFreshnessCacheEntry
from app.core.dependencies import get_policy_freshness_cache
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
