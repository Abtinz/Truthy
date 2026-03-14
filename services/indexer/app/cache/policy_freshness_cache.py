from __future__ import annotations

from dataclasses import dataclass

import redis

from app.core.config import IndexerSettings


@dataclass(frozen=True)
class PolicyFreshnessResult:
    """Result of one Redis freshness comparison for a policy source.

    Args:
        source_url: Policy source URL being checked.
        modified_date: Newly observed `Date modified` value.
        cached_modified_date: Previously cached date from Redis, if any.
        has_changed: Whether the source should be re-indexed.

    Returns:
        PolicyFreshnessResult: Immutable freshness comparison result.
    """

    source_url: str
    modified_date: str
    cached_modified_date: str | None
    has_changed: bool


@dataclass(frozen=True)
class PolicyFreshnessCacheEntry:
    """One cached IRCC freshness record stored in Redis.

    This lightweight record intentionally stores only the minimum metadata
    needed for operator visibility and refresh optimization: the source URL and
    the latest observed `Date modified` value from that page.

    Args:
        source_url: Policy source URL tracked by the freshness cache.
        modified_date: Last stored `Date modified` value for the source.

    Returns:
        PolicyFreshnessCacheEntry: Immutable Redis cache entry projection.
    """

    source_url: str
    modified_date: str


class PolicyFreshnessCache:
    """Redis-backed cache for IRCC source freshness checks.

    The cache stores only the minimum signal required by the indexer
    optimization: source URL and its last observed `Date modified` value.

    Args:
        settings: Environment-backed indexer settings.
        redis_client: Optional Redis client override for tests.

    Returns:
        PolicyFreshnessCache: Configured freshness cache client.
    """

    def __init__(
        self,
        settings: IndexerSettings,
        redis_client: redis.Redis | None = None,
    ) -> None:
        """Initialize the policy freshness cache.

        Args:
            settings: Environment-backed indexer settings.
            redis_client: Optional Redis client override for tests.

        Returns:
            None.
        """

        self.settings = settings
        self.redis_client = redis_client or redis.Redis.from_url(
            settings.redis_url,
            decode_responses=True,
        )
        self.registry_key = f"{self.settings.redis_policy_cache_prefix}:entries"

    def compare_modified_date(
        self,
        source_url: str,
        modified_date: str,
    ) -> PolicyFreshnessResult:
        """Compare a source modified date against the cached Redis value.

        Args:
            source_url: Policy source URL.
            modified_date: Current page-level `Date modified` value.

        Returns:
            PolicyFreshnessResult: Result showing whether the source changed.
        """

        cached_modified_date = self.redis_client.hget(self.registry_key, source_url)
        return PolicyFreshnessResult(
            source_url=source_url,
            modified_date=modified_date,
            cached_modified_date=cached_modified_date,
            has_changed=cached_modified_date != modified_date,
        )

    def store_modified_date(self, source_url: str, modified_date: str) -> None:
        """Persist the latest modified date for one source URL.

        Args:
            source_url: Policy source URL.
            modified_date: Current page-level `Date modified` value.

        Returns:
            None.
        """

        self.redis_client.hset(self.registry_key, source_url, modified_date)

    def list_entries(self) -> list[PolicyFreshnessCacheEntry]:
        """Return the currently tracked Redis freshness records.

        Args:
            None.

        Returns:
            list[PolicyFreshnessCacheEntry]: Sorted cache entries for operator
            inspection and debugging.
        """

        raw_entries = self.redis_client.hgetall(self.registry_key)
        return [
            PolicyFreshnessCacheEntry(
                source_url=source_url,
                modified_date=modified_date,
            )
            for source_url, modified_date in sorted(raw_entries.items())
        ]
