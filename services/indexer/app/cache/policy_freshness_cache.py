from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

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

        cached_modified_date = self.redis_client.get(self._build_key(source_url))
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

        self.redis_client.set(self._build_key(source_url), modified_date)

    def _build_key(self, source_url: str) -> str:
        """Build the Redis key used for one source URL.

        Args:
            source_url: Policy source URL.

        Returns:
            str: Redis key string for the cached freshness record.
        """

        digest = sha256(source_url.encode("utf-8")).hexdigest()
        return f"{self.settings.redis_policy_cache_prefix}:{digest}"
