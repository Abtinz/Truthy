from __future__ import annotations

from app.cache.policy_freshness_cache import PolicyFreshnessCache
from app.core.config import IndexerSettings


class FakeRedisClient:
    """Minimal in-memory Redis client for freshness-cache tests.

    Args:
        None.

    Returns:
        FakeRedisClient: Test double implementing `get` and `set`.
    """

    def __init__(self) -> None:
        """Initialize the in-memory storage.

        Args:
            None.

        Returns:
            None.
        """

        self.storage: dict[str, str] = {}
        self.hash_storage: dict[str, dict[str, str]] = {}

    def get(self, key: str) -> str | None:
        """Return the cached value for one key.

        Args:
            key: Redis key string.

        Returns:
            str | None: Stored value when present.
        """

        return self.storage.get(key)

    def set(self, key: str, value: str) -> None:
        """Persist one key-value pair.

        Args:
            key: Redis key string.
            value: Value to store.

        Returns:
            None.
        """

        self.storage[key] = value

    def hget(self, name: str, key: str) -> str | None:
        """Return one value from a named Redis hash.

        Args:
            name: Redis hash key.
            key: Field name inside the Redis hash.

        Returns:
            str | None: Stored hash value when present.
        """

        return self.hash_storage.get(name, {}).get(key)

    def hset(self, name: str, key: str, value: str) -> None:
        """Store one field inside a named Redis hash.

        Args:
            name: Redis hash key.
            key: Field name inside the Redis hash.
            value: Value to store for the field.

        Returns:
            None.
        """

        if name not in self.hash_storage:
            self.hash_storage[name] = {}
        self.hash_storage[name][key] = value

    def hgetall(self, name: str) -> dict[str, str]:
        """Return all field/value pairs from a named Redis hash.

        Args:
            name: Redis hash key.

        Returns:
            dict[str, str]: Stored hash field/value pairs.
        """

        return dict(self.hash_storage.get(name, {}))


def test_policy_freshness_cache_detects_change_and_reuse() -> None:
    """Verify the cache reports changed and unchanged source states.

    Args:
        None.

    Returns:
        None.
    """

    cache = PolicyFreshnessCache(_build_settings(), redis_client=FakeRedisClient())
    source_url = "https://example.com/policy"

    first_result = cache.compare_modified_date(source_url, "2026-03-03")
    cache.store_modified_date(source_url, "2026-03-03")
    second_result = cache.compare_modified_date(source_url, "2026-03-03")
    third_result = cache.compare_modified_date(source_url, "2026-03-04")

    print("\n=== POLICY FRESHNESS RESULTS ===")
    print(first_result)
    print(second_result)
    print(third_result)

    assert first_result.has_changed is True
    assert second_result.has_changed is False
    assert third_result.has_changed is True


def test_policy_freshness_cache_lists_saved_entries() -> None:
    """Verify the cache can expose stored URL/date pairs for inspection.

    Args:
        None.

    Returns:
        None.
    """

    cache = PolicyFreshnessCache(_build_settings(), redis_client=FakeRedisClient())
    cache.store_modified_date("https://example.com/policy-b", "2026-03-04")
    cache.store_modified_date("https://example.com/policy-a", "2026-03-03")

    entries = cache.list_entries()

    print("\n=== POLICY FRESHNESS CACHE ENTRIES ===")
    for entry in entries:
        print(entry)

    assert [entry.source_url for entry in entries] == [
        "https://example.com/policy-a",
        "https://example.com/policy-b",
    ]
    assert [entry.modified_date for entry in entries] == [
        "2026-03-03",
        "2026-03-04",
    ]


def _build_settings() -> IndexerSettings:
    """Construct stable settings for freshness-cache tests.

    Args:
        None.

    Returns:
        IndexerSettings: Test settings object.
    """

    return IndexerSettings(
        pinecone_api_key="test-key",
        pinecone_operational_guidelines_index_name="guidelines-index",
        pinecone_document_checklist_index_name="checklist-index",
        pinecone_namespace="truthy-dev",
        pinecone_top_k=5,
        pinecone_dimension=1536,
        pinecone_metric="cosine",
        pinecone_cloud="aws",
        pinecone_region="us-east-1",
        pinecone_deletion_protection="disabled",
        redis_url="redis://redis:6379/0",
        redis_policy_cache_prefix="truthy:test-policy-modified",
    )
